"""Exercise actual upstream fit/save/load control with explicitly mocked GPU RPCs.

Tiny deterministic CPU fixtures test progress, not model learning. Reward
extraction, filtering, advantage, masks and TIS still run in the upstream class;
no training loop, filter, loss, or optimizer is reimplemented in this probe.
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import Mock, patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-root", type=Path, required=True)
    parser.add_argument("--recipe-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-fixed", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    sys.path[:0] = [str(args.recipe_root), str(args.verl_root)]
    import numpy as np
    import torch
    from dapo import dapo_ray_trainer as module
    from dapo.dapo_ray_trainer import RayDAPOTrainer
    from omegaconf import OmegaConf
    from torchdata.stateful_dataloader import StatefulDataLoader
    from verl import DataProto
    from verl.trainer.ppo.utils import create_rl_sampler

    assert (
        Path(RayDAPOTrainer.fit.__code__.co_filename).resolve()
        == (args.recipe_root / "dapo/dapo_ray_trainer.py").resolve()
    )
    assert (
        Path(create_rl_sampler.__code__.co_filename).resolve()
        == (args.verl_root / "verl/trainer/ppo/utils.py").resolve()
    )
    original = OmegaConf.load(
        "/home/chen/runs/inference-efficiency-20260908/resume-compiled-02/resolved.yaml"
    )

    def collate(rows):
        return {
            "fixture_row": torch.tensor(rows).reshape(-1, 1),
            "raw_prompt": np.array(rows, dtype=object),
        }

    def loader(n):
        rows = list(range(n))
        return StatefulDataLoader(
            rows,
            batch_size=1,
            num_workers=0,
            drop_last=True,
            sampler=create_rl_sampler(original.data, rows),
            collate_fn=collate,
        )

    def expected_rows(n, target, vary_all=False):
        # Independent oracle: one continuous real sampler stream and a specified
        # fixture reward rule; no DAPO control code is copied here.
        dl = loader(n)
        result = []
        retained = 0
        while retained < target:
            for item in dl:
                row = int(item["fixture_row"].item())
                result.append(row)
                retained += int(vary_all or row % 2 == 0)
                if retained == target:
                    return result

    def run(name, *, target=8, epochs=1, n=5, vary_all=False, vary_none=False, resume=None):
        folder = args.output / name
        folder.mkdir()
        cfg = copy.deepcopy(original)
        cfg.trainer.default_local_dir = str(folder / "checkpoints")
        cfg.trainer.total_training_steps = target
        cfg.trainer.total_epochs = epochs
        cfg.trainer.resume_mode = "resume_path" if resume else "disable"
        cfg.trainer.resume_from_path = str(resume) if resume else None
        cfg.trainer.default_hdfs_dir = None
        cfg.trainer.del_local_ckpt_after_load = False
        cfg.trainer.val_before_train = False
        cfg.trainer.test_freq = -1
        cfg.trainer.save_freq = 1
        cfg.trainer.rollout_data_dir = None
        cfg.trainer.balance_batch = False
        cfg.trainer.critic_warmup = 0
        cfg.actor_rollout_ref.actor.checkpoint.async_save = False
        trainer = object.__new__(RayDAPOTrainer)
        trainer.config = cfg
        trainer.total_training_steps = n * epochs if target is None else target
        trainer.train_dataloader = loader(n)
        trainer.use_critic = trainer.use_rm = trainer.use_reference_policy = False
        trainer.resource_pool_manager = Mock()
        trainer.resource_pool_manager.get_n_gpus.return_value = 1
        trainer.checkpoint_manager = Mock()
        trainer.actor_rollout_wg = Mock()
        generated, updates, saves, loads, logged = [], [], [], [], []

        def generate(batch):
            row = int(batch.non_tensor_batch["raw_prompt"][0])
            assert len(batch) == 4
            generated.append(row)
            varying = not vary_none and (vary_all or row % 2 == 0)
            rewards = [0.0, 0.25, 0.5, 1.0] if varying else [0.0] * 4
            scores = torch.zeros((4, 3))
            scores[:, 1] = torch.tensor(rewards)
            return DataProto.from_dict(
                tensors={
                    "prompts": torch.ones((4, 2), dtype=torch.long),
                    "responses": torch.tensor([[1, 2, 0]] * 4),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 0]] * 4),
                    "response_mask": torch.tensor([[1, 1, 0]] * 4),
                    "rm_scores": scores,
                    "rollout_log_probs": torch.full((4, 3), -0.7),
                },
                non_tensors={"acc": np.array(rewards)},
                meta_info={"timing": {}, "reward_extra_keys": ["acc"]},
            )

        trainer.async_rollout_manager = Mock()
        trainer.async_rollout_manager.generate_sequences.side_effect = generate

        def old_log_prob(batch):
            return DataProto.from_dict(
                tensors={
                    "old_log_probs": torch.full((len(batch), 3), -0.7),
                    "entropys": torch.full((len(batch), 3), 0.2),
                }
            ), 0.0

        trainer._compute_old_log_prob = old_log_prob

        def update(batch):
            assert len(batch) == 4
            assert torch.isfinite(batch.batch["advantages"]).all()
            assert torch.equal(batch.batch["response_mask"], torch.tensor([[1, 1, 0]] * 4))
            row = int(batch.batch["fixture_row"][0].item())
            updates.append({"step": trainer.global_steps, "row": row})
            return DataProto(meta_info={"metrics": {"fixture/actor_rpc": 1.0}})

        trainer._update_actor = update

        def save(path, remote, step, **kwargs):
            # Deliberately no model/optimizer file: this is a mocked RPC marker.
            Path(path).mkdir(parents=True)
            record = {
                "saved_step": step,
                "last_actual_rpc_step": updates[-1]["step"] if updates else None,
                "fixture_only": True,
            }
            (Path(path) / "fixture-only.json").write_text(json.dumps(record))
            saves.append(record)

        def load(path, **kwargs):
            loads.append(path)

        trainer.actor_rollout_wg.save_checkpoint.side_effect = save
        trainer.actor_rollout_wg.load_checkpoint.side_effect = load
        logger = Mock()
        logger.log.side_effect = lambda *, data, step: logged.append(
            {"step": step, "metrics": dict(data)}
        )
        error = None
        with (
            patch("verl.utils.tracking.Tracking", return_value=logger),
            patch.object(module, "tqdm"),
        ):
            try:
                trainer.fit()
            except (RuntimeError, ValueError) as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
        result = {
            "case": name,
            "target": trainer.total_training_steps,
            "explicit_target": target,
            "epochs": epochs,
            "rows_generated": generated,
            "actor_rpcs": updates,
            "save_rpcs": saves,
            "load_rpcs": loads,
            "reported_global_steps": trainer.global_steps,
            "generated_counter": trainer.gen_steps,
            "data_epoch": getattr(trainer, "data_epoch", None),
            "error": error,
            "checkpoint_root": cfg.trainer.default_local_dir,
            "logged_gen_batches": [
                x["metrics"]["train/num_gen_batches"]
                for x in logged
                if "train/num_gen_batches" in x["metrics"]
            ],
        }
        (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    results = []
    explicit = run("explicit_target_with_filtering")
    results.append(explicit)
    implicit = run("implicit_epoch_exhaustion", target=None)
    results.append(implicit)
    if args.expect_fixed:
        assert len(explicit["actor_rpcs"]) == 8 and explicit["error"] is None
        assert explicit["rows_generated"] == expected_rows(5, 8)
        assert explicit["generated_counter"] == len(explicit["rows_generated"])
        assert implicit["error"]["type"] == "RuntimeError"
        assert len(implicit["actor_rpcs"]) == 3 and len(implicit["save_rpcs"]) == 3
        assert implicit["reported_global_steps"] == 3
    else:
        assert len(explicit["actor_rpcs"]) == len(implicit["actor_rpcs"]) == 3
        assert explicit["error"] is implicit["error"] is None
        assert explicit["save_rpcs"][-1] == {
            "saved_step": 4,
            "last_actual_rpc_step": 3,
            "fixture_only": True,
        }
        assert implicit["save_rpcs"][-1]["saved_step"] == 4

    phase = run("before_epoch_boundary", target=4, vary_all=True)
    results.append(phase)
    resumed = run(
        "resume_one_batch_before_boundary",
        target=5,
        vary_all=True,
        resume=Path(phase["checkpoint_root"]) / "global_step_4",
    )
    results.append(resumed)
    if args.expect_fixed:
        assert [u["step"] for u in resumed["actor_rpcs"]] == [5] and resumed["error"] is None
        assert phase["rows_generated"] + resumed["rows_generated"] == expected_rows(5, 5, True)
    else:
        assert not resumed["actor_rpcs"] and resumed["save_rpcs"][-1]["saved_step"] == 5

    complete = run("completed_checkpoint", target=5, vary_all=True)
    results.append(complete)
    same = run(
        "resume_completed_target",
        target=5,
        epochs=3,
        vary_all=True,
        resume=Path(complete["checkpoint_root"]) / "global_step_5",
    )
    results.append(same)
    assert same["error"] is None
    assert len(same["actor_rpcs"]) == (0 if args.expect_fixed else 1)
    if args.expect_fixed:
        assert not same["rows_generated"] and not same["save_rpcs"]
        limit = run("unchanged_filter_limit", epochs=4, vary_none=True)
        results.append(limit)
        assert limit["error"]["type"] == "ValueError"
        assert len(limit["rows_generated"]) == original.algorithm.filter_groups.max_num_gen_batches
        assert not limit["actor_rpcs"] and not limit["save_rpcs"]
        native = run(
            "real_update6_data_with_fixture_gpu_rpcs",
            target=7,
            n=500,
            resume=Path(
                "/home/chen/runs/inference-efficiency-20260908/"
                "resumed-update6-acceptance/global_step_6"
            ),
        )
        results.append(native)
        assert native["error"] is None
        assert native["rows_generated"] == [461, 178]
        assert native["generated_counter"] == 11 and native["data_epoch"] == 0
        assert [u["step"] for u in native["actor_rpcs"]] == [7]
        split = run("filtered_prefix", target=5)
        results.append(split)
        continuation = run(
            "filtered_continuation",
            target=8,
            resume=Path(split["checkpoint_root"]) / "global_step_5",
        )
        results.append(continuation)
        assert continuation["error"] is None
        assert (
            split["rows_generated"] + continuation["rows_generated"] == explicit["rows_generated"]
        )
        assert split["actor_rpcs"] + continuation["actor_rpcs"] == explicit["actor_rpcs"]
        assert continuation["generated_counter"] == explicit["generated_counter"]

        boundary = run(
            "resume_at_epoch_end",
            target=8,
            vary_all=True,
            resume=Path(complete["checkpoint_root"]) / "global_step_5",
        )
        results.append(boundary)
        assert boundary["error"] is None
        assert complete["rows_generated"] + boundary["rows_generated"] == expected_rows(5, 8, True)
        for name, mutation in [
            ("reject_ambiguous_legacy", "legacy"),
            ("reject_inconsistent_generation_count", "count"),
            ("reject_missing_data", "data"),
            ("exhausted_iterator_boundary", "finished"),
        ]:
            parent = args.output / (name + "-fixture")
            copied = parent / "global_step_5"
            shutil.copytree(Path(complete["checkpoint_root"]) / "global_step_5", copied)
            if mutation == "legacy":
                (copied / "dapo_progress.json").unlink()
            elif mutation == "count":
                data = json.loads((copied / "dapo_progress.json").read_text())
                data["generated_batches"] += 1
                (copied / "dapo_progress.json").write_text(json.dumps(data))
            elif mutation == "data":
                (copied / "data.pt").unlink()
            else:
                exhausted = loader(5)
                iterator = iter(exhausted)
                for _ in range(5):
                    next(iterator)
                try:
                    next(iterator)
                except StopIteration:
                    pass
                else:
                    raise AssertionError("The real fixture iterator did not exhaust")
                data = copy.deepcopy(exhausted.state_dict())
                assert data["_num_yielded"] == 5 and data["_iterator_finished"] is True
                torch.save(data, copied / "data.pt")
            checked = run(name, target=8, vary_all=True, resume=copied)
            results.append(checked)
            if mutation == "finished":
                assert checked["error"] is None
                assert complete["rows_generated"] + checked["rows_generated"] == expected_rows(
                    5, 8, True
                )
            else:
                assert checked["error"]["type"] == "ValueError" and not checked["actor_rpcs"]

    assert not torch.cuda.is_initialized()
    report = {
        "expect_fixed": args.expect_fixed,
        "cases": results,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                args.verl_root / "verl/trainer/ppo/ray_trainer.py",
                args.recipe_root / "dapo/dapo_ray_trainer.py",
            )
        },
        "cuda_initialized": False,
        "scope": "Actual upstream fit/save/load and StatefulDataLoader; GPU generation, logprob, "
        "actor update and model save/load RPCs mocked. No model learning or GPU restore claim.",
    }
    (args.output / "progress-probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "cases": len(results),
                "expect_fixed": args.expect_fixed,
                "output": str(args.output),
                "cuda_initialized": False,
            }
        )
    )


if __name__ == "__main__":
    main()
