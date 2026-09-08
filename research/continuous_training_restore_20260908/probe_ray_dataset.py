"""Exercise the real continuous dataset inside an isolated Ray CPU worker.

Use the production import order and official working directory. No environment,
converter, tokenizer, or counterfactual solver is mocked. No model is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker_probe(options):
    import importlib

    import ray
    import torch
    from omegaconf import OmegaConf
    from verl.trainer.ppo.utils import create_rl_dataset
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.tokenizer import normalize_token_ids

    from cpt_world import verl_streaming_dataset as module
    from cpt_world.identification import INTERACTION_SURFACE_VERSION
    from cpt_world.registry import TASK_FAMILY_QUERY_TYPES

    started = time.perf_counter()
    project, output = Path(options["project"]), Path(options["output"])
    assert Path.cwd() == Path(options["verl_root"])
    assert os.environ["PYTHONPATH"] == options["pythonpath"]
    assert not ray.get_gpu_ids() and not torch.cuda.is_initialized()
    legacy_import = {}
    try:
        imported = importlib.import_module("scripts.prepare_verl_cpt_data")
        legacy_import["file"] = imported.__file__
    except ModuleNotFoundError as error:
        legacy_import["error"] = str(error)
    scripts = sys.modules.get("scripts")
    legacy_import["scripts_file"] = getattr(scripts, "__file__", None)
    legacy_import["scripts_path"] = list(getattr(scripts, "__path__", []))

    cfg = OmegaConf.load(options["config"])
    model_path = Path(cfg.actor_rollout_ref.model.path)
    assert model_path.is_dir()
    tokenizer = hf_tokenizer(
        str(model_path), trust_remote_code=cfg.data.trust_remote_code, local_files_only=True
    )
    processor = hf_processor(
        str(model_path),
        trust_remote_code=cfg.data.trust_remote_code,
        use_fast=True,
        local_files_only=True,
    )
    descriptor = output / "stream.json"
    descriptor.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "cpt_world_continuous",
                "environment_version": INTERACTION_SURFACE_VERSION,
                "start_seed": options["start_seed"],
                "stream_start_index": 0,
                "journal_dir": str(output / "journal"),
                "source_fingerprints": module.current_source_fingerprints(),
            },
            indent=2,
        )
        + "\n"
    )
    cfg.data.train_files = str(descriptor)
    cfg.data.custom_cls.path = str(project / "src/cpt_world/verl_streaming_dataset.py")
    cfg.data.tool_config_path = str(project / "configs/verl/cpt_world_tools.yaml")
    assert cfg.data.custom_cls.name == "CPTWorldStreamingDataset"
    assert cfg.data.shuffle is False and cfg.data.dataloader_num_workers == 0
    assert cfg.data.train_batch_size == cfg.data.gen_batch_size == 1
    assert cfg.data.filter_overlong_prompts is False

    cf_processes = []

    def audit_subprocess(event, args):
        if event == "subprocess.Popen" and any(
            "cpt_world.counterfactual_isolation" in str(part) for part in args[1]
        ):
            cf_processes.append({"executable": args[0], "command": list(args[1])})

    sys.addaudithook(audit_subprocess)
    dataset = create_rl_dataset(
        cfg.data.train_files,
        cfg.data,
        tokenizer,
        processor,
        is_train=True,
        max_samples=cfg.data.train_max_samples,
    )
    assert dataset.__class__.__getitem__ is RLHFDataset.__getitem__
    assert len(dataset) == sys.maxsize
    rows = []
    for index, query_type in enumerate(TASK_FAMILY_QUERY_TYPES):
        row_started = time.perf_counter()
        row = dataset[index]
        assert row["extra_info"]["query_type"] == query_type
        assert row["index"] == index
        raw = json.loads(row["tools_kwargs"]["act"]["create_kwargs"]["row_json"])
        assert raw["query_type"] == query_type
        assert raw["tape_key"] == row["extra_info"]["tape_key"]
        apply_kwargs = dict(cfg.data.apply_chat_template_kwargs)
        if dataset.tool_schemas is not None:
            apply_kwargs["tools"] = dataset.tool_schemas
        tokens = tokenizer.apply_chat_template(
            row["raw_prompt"], add_generation_prompt=True, tokenize=True, **apply_kwargs
        )
        tokens = normalize_token_ids(tokens)
        assert len(tokens) > 0
        record_path = output / "journal/rows" / f"{index:020d}.json"
        record = json.loads(record_path.read_text())
        assert record["row"] == raw
        evidence = {
            "index": index,
            "query_type": query_type,
            "tape_key": raw["tape_key"],
            "prompt_tokens": len(tokens),
            "seconds": time.perf_counter() - row_started,
            "last_attempts": record["last_attempts"],
            "next_generator_state": record["next_generator_state"],
            "journal_sha256": sha(record_path),
        }
        if raw.get("terminal_truth_json"):
            evidence["certified_truth"] = json.loads(raw["terminal_truth_json"])
            assert cf_processes and record["last_attempts"][-1]["status"] == "accepted"
        rows.append(evidence)
        print("REAL_RAY_DATASET_ROW=" + json.dumps(evidence), flush=True)

    assert len(cf_processes) >= 1
    assert sum("certified_truth" in row for row in rows) == 1
    namespace = dataset.dataframe.__class__.__getitem__.__globals__
    converter_file = Path(namespace["_row_converter"]().__code__.co_filename)
    assert converter_file == project / "scripts/prepare_verl_cpt_data.py"
    assert not torch.cuda.is_initialized()
    return {
        "passed": True,
        "worker_pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "pythonpath": os.environ["PYTHONPATH"],
        "sys_path": sys.path,
        "legacy_import": legacy_import,
        "tokenizer_class": type(tokenizer).__name__,
        "processor_class": type(processor).__name__,
        "model_path": str(model_path),
        "tokenizer_sha256": {p.name: sha(p) for p in model_path.glob("*tokenizer*.json")},
        "custom_class_module": dataset.__class__.__module__,
        "converter_file": str(converter_file),
        "rows": rows,
        "counterfactual_subprocesses": cf_processes,
        "journal_state": dataset.state_dict(),
        "source_fingerprints": module.current_source_fingerprints(),
        "cuda_initialized": False,
        "elapsed_seconds": time.perf_counter() - started,
        "official_dataset_sha256": sha(Path(RLHFDataset.__getitem__.__code__.co_filename)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "verl-root", "recipe-root", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pythonpath = ":".join(
        map(str, [args.project / "src", args.project, args.recipe_root, args.verl_root])
    )
    ray_tmp = "/tmp/cpt-ray-dataset-" + uuid4().hex[:10]
    env = {
        "PYTHONPATH": pythonpath,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CPT_WORLD_EXPECTED_SOURCE": str(args.project / "src/cpt_world"),
        "CPT_WORLD_VERL_AUDIT_DIR": str(args.output / "environment"),
        "CPT_WORLD_RESOURCE_DIAGNOSTIC_DIR": str(args.output / "cf-diagnostics"),
        "VERL_ROOT": str(args.verl_root),
        "VERL_RECIPE_ROOT": str(args.recipe_root),
        "CPT_WORLD_DAPO_SOURCE_PROFILE": "progress-v1",
        "RAY_TMPDIR": ray_tmp,
    }
    os.environ.update(env)
    sys.path[:0] = pythonpath.split(":")
    os.chdir(args.verl_root)
    import ray

    assert not ray.is_initialized()
    options = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    options["pythonpath"] = pythonpath
    try:
        ray.init(
            address="local",
            num_cpus=1,
            num_gpus=0,
            include_dashboard=False,
            _temp_dir=ray_tmp,
            object_store_memory=128 * 1024 * 1024,
            runtime_env={"env_vars": env},
        )
        task = ray.remote(num_cpus=1, num_gpus=0, max_calls=1)(worker_probe)
        result = ray.get(task.remote(options), timeout=180)
        result.update(
            probe_sha256=sha(Path(__file__)),
            config_sha256=sha(args.config),
            ray_tmp=ray_tmp,
            scope="Real Ray CPU worker, official dataset factory/getitem, "
            "local Qwen tokenizer/processor and five real generated task families, "
            "including the actual isolated CF truth process. No model loading or training.",
        )
    finally:
        # This driver created the local Ray cluster. Never call global `ray stop`.
        ray.shutdown()
    result["ray_shutdown_completed"] = not ray.is_initialized()
    (args.output / "ray-dataset-probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "rows": len(result["rows"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
