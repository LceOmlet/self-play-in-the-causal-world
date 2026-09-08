"""Exercise official DAPO fit/save/load with the actual journaled stream dataset.

Only task contents and GPU RPCs are CPU fixtures. The runner is reused from
the earlier progress probe; official filtering, group repetition, advantages,
TIS, checkpoint control, sampler and StatefulDataLoader remain executable.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
import sys
import textwrap
from pathlib import Path
from unittest.mock import Mock, patch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reuse_gpu_fixture_runner(path, namespace):
    """Reuse the prior RPC harness, adapting only how fixture row IDs are read."""
    source = path.read_text(encoding="utf-8")
    parsed = ast.parse(source)
    main = next(
        node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    runner = next(
        node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    code = textwrap.dedent(ast.get_source_segment(source, runner))
    replacements = [
        (
            'row = int(batch.non_tensor_batch["raw_prompt"][0])',
            "row = fixture_generated_row(batch, trainer)",
        ),
        (
            'row = int(batch.batch["fixture_row"][0].item())',
            "row = fixture_retained_row(batch, trainer)",
        ),
    ]
    for before, after in replacements:
        if code.count(before) != 1:
            raise ValueError("The reused CPU fixture runner changed")
        code = code.replace(before, after)
    exec(compile(code, str(path) + ":stream-fixture-row-ids", "exec"), namespace)
    return namespace["run"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verl-root", type=Path, required=True)
    parser.add_argument("--recipe-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    sys.path[:0] = [str(args.recipe_root), str(args.verl_root), str(project / "src"), str(project)]
    import numpy as np
    import torch
    from dapo import dapo_ray_trainer as module
    from dapo.dapo_ray_trainer import RayDAPOTrainer
    from omegaconf import OmegaConf
    from torch.utils.data import SequentialSampler
    from torchdata.stateful_dataloader import StatefulDataLoader
    from verl import DataProto
    from verl.trainer.ppo.utils import create_rl_sampler
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

    from cpt_world import verl_streaming_dataset as stream_module
    from cpt_world.identification import INTERACTION_SURFACE_VERSION

    native_cursor = load_module(
        "continuous_data_migration", project / "scripts/migrate_dapo_continuous_data.py"
    ).native_cursor

    assert (
        Path(RayDAPOTrainer.fit.__code__.co_filename).resolve()
        == (args.recipe_root / "dapo/dapo_ray_trainer.py").resolve()
    )
    assert (
        Path(create_rl_sampler.__code__.co_filename).resolve()
        == (args.verl_root / "verl/trainer/ppo/utils.py").resolve()
    )
    assert stream_module.CPTWorldStreamingDataset.__getitem__ is RLHFDataset.__getitem__
    fixture_path = project / "tests/test_verl_streaming_dataset.py"
    fixture = load_module("continuous_dataset_cpu_fixture", fixture_path)
    original = OmegaConf.load(args.config)
    original.data.shuffle = False
    original.data.dataloader_num_workers = 0
    original.data.train_batch_size = original.data.gen_batch_size = 1
    original.data.filter_overlong_prompts = False
    assert original.actor_rollout_ref.rollout.n == 4
    assert original.algorithm.filter_groups.enable
    assert original.algorithm.filter_groups.max_num_gen_batches == 10
    active = {}
    source_calls = []

    def converted_row(row, index):
        result = fixture.fake_convert(row, index)
        result["agent_name"] = "tool_agent"
        result["extra_info"]["need_tools_kwargs"] = True
        result["extra_info"]["tools_kwargs"] = {
            "act": {"create_kwargs": {"row_json": json.dumps(row, sort_keys=True)}}
        }
        return result

    def descriptor(name, start=0):
        folder = args.output / name
        folder.mkdir()
        path = folder / "continuous.json"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "kind": "cpt_world_continuous",
                    "environment_version": INTERACTION_SURFACE_VERSION,
                    "start_seed": 1000 + 2 * start,
                    "stream_start_index": start,
                    "journal_dir": str(folder / "journal"),
                    "source_fingerprints": stream_module.current_source_fingerprints(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def dataset(path):
        return stream_module.CPTWorldStreamingDataset(
            data_files=str(path), tokenizer=None, config=original.data
        )

    def loader(_n):
        data = dataset(active["descriptor"])
        sampler = create_rl_sampler(original.data, data)
        assert type(sampler) is SequentialSampler
        dl = StatefulDataLoader(
            data,
            batch_size=1,
            num_workers=0,
            drop_last=True,
            sampler=sampler,
            collate_fn=collate_fn,
        )
        assert len(dl) == len(data) == sys.maxsize
        active["dataset"] = data
        return dl

    def shared_identity(batch):
        extras = batch.non_tensor_batch["extra_info"]
        indices = [int(extra["index"]) for extra in extras]
        tapes = [extra["tape_key"] for extra in extras]
        rows = [extra["tools_kwargs"]["act"]["create_kwargs"]["row_json"] for extra in extras]
        assert len(batch) == len(indices) == 4
        assert len(set(indices)) == len(set(tapes)) == len(set(rows)) == 1
        assert tapes[0] == f"fixture:{1001 + 2 * indices[0]}"
        return indices[0], tapes[0]

    def fixture_generated_row(batch, trainer):
        row, tape = shared_identity(batch)
        active["generated_groups"].append(
            {
                "index": row,
                "tape_key": tape,
                "group_size": len(batch),
                "upcoming_update": trainer.global_steps,
                "generated_counter": trainer.gen_steps,
            }
        )
        return row

    def fixture_retained_row(batch, trainer):
        row, tape = shared_identity(batch)
        assert len(set(batch.non_tensor_batch["uid"])) == 1
        active["retained_groups"].append(
            {
                "index": row,
                "tape_key": tape,
                "update": trainer.global_steps,
            }
        )
        return row

    runner_path = project / "research/dapo_progress_20260908/probe_progress.py"
    namespace = {
        "args": args,
        "original": original,
        "copy": copy,
        "Path": Path,
        "Mock": Mock,
        "patch": patch,
        "json": json,
        "torch": torch,
        "np": np,
        "DataProto": DataProto,
        "RayDAPOTrainer": RayDAPOTrainer,
        "module": module,
        "loader": loader,
        "fixture_generated_row": fixture_generated_row,
        "fixture_retained_row": fixture_retained_row,
    }
    run_rpc_fixture = reuse_gpu_fixture_runner(runner_path, namespace)

    def run(name, path, **kwargs):
        active.clear()
        active.update(descriptor=path, generated_groups=[], retained_groups=[])
        before = len(source_calls)
        result = run_rpc_fixture(name, n=sys.maxsize, **kwargs)
        result.update(
            descriptor=str(path),
            new_fixture_worlds=len(source_calls) - before,
            generated_groups=active["generated_groups"],
            retained_groups=active["retained_groups"],
            journal_state=active["dataset"].state_dict(),
        )
        result["saved_data_cursors"] = []
        for checkpoint in sorted(Path(result["checkpoint_root"]).glob("global_step_*")):
            state = torch.load(checkpoint / "data.pt", map_location="cpu", weights_only=False)
            progress = json.loads((checkpoint / "dapo_progress.json").read_text())
            assert progress["batches_per_epoch"] == sys.maxsize
            assert progress["generated_batches"] == state["_num_yielded"]
            assert state["dataset_state"]["kind"] == "cpt_world_continuous_journal"
            result["saved_data_cursors"].append(
                {
                    "update": progress["completed_updates"],
                    "generated": progress["generated_batches"],
                    "dataset_state": state["dataset_state"],
                }
            )
        (args.output / name / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    results = []
    with (
        patch.object(
            stream_module,
            "_make_stream",
            side_effect=lambda seed: fixture.FakeStream(seed, source_calls),
        ),
        patch.object(stream_module, "_convert_row", side_effect=converted_row),
    ):
        full_path = descriptor("continuous-journal")
        full = run("continuous_filtering", full_path, target=8)
        results.append(full)
        assert full["error"] is None
        assert full["rows_generated"] == list(range(15))
        assert [r["row"] for r in full["actor_rpcs"]] == list(range(0, 15, 2))
        assert full["generated_counter"] == 15 and full["reported_global_steps"] == 8
        assert (
            full["generated_groups"][1]["upcoming_update"]
            == full["generated_groups"][2]["upcoming_update"]
        )
        assert (
            full["generated_groups"][1]["generated_counter"] + 1
            == full["generated_groups"][2]["generated_counter"]
        )

        split_path = descriptor("split-journal")
        prefix = run("filtered_prefix", split_path, target=5)
        results.append(prefix)
        checkpoint = Path(prefix["checkpoint_root"]) / "global_step_5"
        # Journaled work beyond the checkpoint must replay rather than regenerate.
        prepared = dataset(split_path)
        prepared[9]
        prepared[10]
        continuation = run("filtered_continuation", split_path, target=8, resume=checkpoint)
        results.append(continuation)
        assert continuation["error"] is None
        assert prefix["rows_generated"] + continuation["rows_generated"] == full["rows_generated"]
        assert prefix["actor_rpcs"] + continuation["actor_rpcs"] == full["actor_rpcs"]
        assert continuation["generated_counter"] == full["generated_counter"]
        assert continuation["new_fixture_worlds"] == 4
        assert (
            prefix["generated_groups"] + continuation["generated_groups"]
            == full["generated_groups"]
        )

        completed = run(
            "completed_target",
            full_path,
            target=8,
            resume=Path(full["checkpoint_root"]) / "global_step_8",
        )
        results.append(completed)
        assert completed["error"] is None
        assert not completed["rows_generated"] and not completed["actor_rpcs"]
        assert completed["new_fixture_worlds"] == 0 and not completed["save_rpcs"]

        limit = run(
            "constant_group_limit", descriptor("constant-journal"), target=8, vary_none=True
        )
        results.append(limit)
        assert limit["error"]["type"] == "ValueError"
        assert "num_gen_batches=10" in limit["error"]["message"]
        assert limit["rows_generated"] == list(range(10)) and limit["generated_counter"] == 10
        assert not limit["actor_rpcs"] and not limit["save_rpcs"]

        boundary_path = descriptor("boundary-journal", start=499)
        boundary_data = dataset(boundary_path)
        initial = args.output / "explicit-cursor-fixture" / "global_step_249"
        (initial / "actor").mkdir(parents=True)
        before = len(source_calls)
        torch.save(native_cursor(boundary_data, 499), initial / "data.pt")
        assert len(source_calls) == before
        (initial / "dapo_progress.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "completed_updates": 249,
                    "generated_batches": 499,
                    "data_epoch": 0,
                    "batches_per_epoch": sys.maxsize,
                }
            )
        )
        (initial / "actor/fixture-only.json").write_text(
            json.dumps(
                {
                    "fixture_only": True,
                    "scope": "Explicit data cursor fixture; no model/Adam checkpoint.",
                }
            )
        )
        boundary = run("cross_old_500_boundary", boundary_path, target=251, resume=initial)
        results.append(boundary)
        assert boundary["error"] is None
        assert boundary["rows_generated"] == [499, 500, 501, 502]
        assert boundary["actor_rpcs"] == [{"step": 250, "row": 500}, {"step": 251, "row": 502}]
        assert [r["tape_key"] for r in boundary["generated_groups"]] == [
            "fixture:1999",
            "fixture:2001",
            "fixture:2003",
            "fixture:2005",
        ]
        assert boundary["generated_counter"] == 503 and boundary["data_epoch"] == 0

    assert not torch.cuda.is_initialized()
    sources = [
        Path(__file__),
        runner_path,
        fixture_path,
        project / "src/cpt_world/verl_streaming_dataset.py",
        project / "scripts/migrate_dapo_continuous_data.py",
        args.verl_root / "verl/trainer/ppo/utils.py",
        args.verl_root / "verl/trainer/ppo/ray_trainer.py",
        args.verl_root / "verl/utils/dataset/rl_dataset.py",
        args.recipe_root / "dapo/dapo_ray_trainer.py",
    ]
    report = {
        "passed": True,
        "cases": results,
        "cuda_initialized": False,
        "source_sha256": {str(path): digest(path) for path in sources},
        "config_sha256": digest(args.config),
        "scope": "Actual official fit/save/load, CPTWorldStreamingDataset, journal, official "
        "getitem/collate/sampler and StatefulDataLoader. Task contents and GPU RPCs are "
        "explicit fixtures; no model learning or GPU restoration claim.",
        "runner_reuse": "The old progress probe's run function is reused with only two fixture "
        "row-ID reads adapted. No official loop or filtering implementation is copied.",
        "boundary_fixture": "IndexOnlyDataset/native_cursor primes logical position 499 without "
        "generating past tasks; U249/G499 is an explicit synthetic checkpoint.",
    }
    (args.output / "official-stream-probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "cases": len(results), "output": str(args.output)}))


if __name__ == "__main__":
    main()
