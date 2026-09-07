#!/usr/bin/env python3
"""Observe one real training step without replacing any trainer or reward method.

The production entry point owns generation, environment calls, advantages,
backward and optimizer updates. This script only observes Python return frames
and callback events, then checks the tensors actually used by those owners.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import TrainerCallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    cli = parser.parse_args()
    argv = cli.training_args
    if argv and argv[0] == "--":
        argv = argv[1:]
    if "--max-steps" not in argv or argv[argv.index("--max-steps") + 1] != "1":
        raise ValueError("The execution audit permits exactly one isolated optimizer step")
    if "--resume-from-checkpoint" in argv:
        raise ValueError("Use --initial-adapter; this audit verifies a fresh optimizer")
    cli.report_dir.mkdir(parents=True, exist_ok=True)
    output = Path(argv[argv.index("--output-dir") + 1]).resolve()
    if cli.report_dir.resolve() not in output.parents:
        raise ValueError("Audit training artifacts must stay under --report-dir")
    report = {"events": [], "batches": [], "optimizer_steps": 0, "checks": {}}
    owner = None
    before_step = {}
    batch_number = 0

    def save():
        (cli.report_dir / "execution-report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def profile(frame, event, returned):
        nonlocal owner, before_step, batch_number
        if event not in ("call", "return"):
            return
        filename = frame.f_code.co_filename.replace("\\", "/")
        name = frame.f_code.co_name
        if filename.endswith("/trl/trainer/grpo_trainer.py"):
            if event == "call" and owner is None and name == "_prepare_inputs":
                owner = frame.f_locals["self"]
                report["trainer_class"] = f"{type(owner).__module__}.{type(owner).__qualname__}"
                report["config"] = {
                    key: getattr(owner.args, key)
                    for key in (
                        "loss_type",
                        "scale_rewards",
                        "epsilon",
                        "epsilon_high",
                        "beta",
                        "num_generations",
                        "generation_batch_size",
                        "steps_per_generation",
                        "gradient_accumulation_steps",
                        "max_completion_length",
                        "use_vllm",
                        "vllm_importance_sampling_mode",
                        "vllm_importance_sampling_clip_max",
                    )
                }
            elif event == "return" and name == "_generate_and_score_completions":
                if not isinstance(returned, dict):
                    return
                local = frame.f_locals
                batch_number += 1
                tensors = {
                    key: value.detach().cpu() if torch.is_tensor(value) else value
                    for key, value in returned.items()
                }
                torch.save(tensors, cli.report_dir / f"actual-batch-{batch_number}.pt")
                mask = tensors["completion_mask"] * tensors.get("tool_mask", 1)
                rewards = local["rewards"].detach().double().cpu()
                grouped = rewards.view(-1, owner.num_generations)
                means = grouped.mean(1).repeat_interleave(owner.num_generations)
                stds = grouped.std(1).repeat_interleave(owner.num_generations)
                expected = (rewards - means) / (stds + 1e-4)
                actual = tensors["advantages"].double()
                rows = local["inputs"]
                keys = [
                    [
                        row.get(key)
                        for key in ("sample_index", "query_type", "anchor_index", "tape_key")
                    ]
                    for row in rows
                ]
                checks = {
                    "same_world_and_tape_in_group": all(key == keys[0] for key in keys),
                    "owner_reward_columns_equal": bool(
                        torch.equal(
                            local["rewards_per_func"][:, 0], local["rewards_per_func"][:, 1]
                        )
                    ),
                    "reward_counted_once": owner.reward_weights.tolist() == [1.0, 0.0],
                    "global_policy_token_denominator": int(mask.sum())
                    == int(tensors["num_items_in_batch"]),
                    "standard_group_advantages": bool(
                        torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
                    ),
                    "finite_training_tensors": all(
                        bool(torch.isfinite(value).all())
                        for value in tensors.values()
                        if torch.is_tensor(value) and value.is_floating_point()
                    ),
                }
                report["batches"].append(
                    {
                        "group_keys": keys,
                        "rewards": rewards.tolist(),
                        "advantages": actual.tolist(),
                        "policy_tokens": mask.sum(1).tolist(),
                        "tool_tokens": (tensors["completion_mask"] - mask).sum(1).tolist(),
                        "zero_reward_variance": bool((stds == 0).all()),
                        "checks": checks,
                    }
                )
                report["events"].append("actual_generation_and_reward_completed")
                save()
            elif event == "return" and name == "training_step" and returned is not None:
                report["events"].append(f"backward_microstep_{owner._step}")
        elif filename.endswith("/trl/generation/vllm_generation.py"):
            if event == "return" and name == "sync_weights":
                report["events"].append("vllm_sync_weights_returned")
        elif (
            filename.endswith("/transformers/trainer_callback.py")
            and event == "call"
            and type(frame.f_locals.get("self")).__name__ == "CallbackHandler"
        ):
            if name == "on_pre_optimizer_step":
                before_step = {}
                for group in owner.optimizer.param_groups:
                    for parameter in group["params"]:
                        if parameter.grad is None:
                            continue
                        state = owner.optimizer.state.get(parameter, {})
                        if state and float(state.get("step", 0)) != 0:
                            raise AssertionError("The audit requires a fresh AdamW state")
                        before = parameter.detach().double().cpu()
                        grad = parameter.grad.detach().double().cpu()
                        # First-step AdamW equation, evaluated in float64, is only
                        # a test reference. Production uses the actual Torch optimizer.
                        expected = before * (1 - group["lr"] * group["weight_decay"])
                        expected -= group["lr"] * grad / (grad.abs() + group["eps"])
                        before_step[id(parameter)] = (before, expected)
                report["pre_optimizer_zero_variance_group"] = report["batches"][-1][
                    "zero_reward_variance"
                ]
                report["events"].append("optimizer_received_batch")
                save()
            elif name == "on_optimizer_step":
                max_error = 0.0
                update_squared = 0.0
                for group in owner.optimizer.param_groups:
                    for parameter in group["params"]:
                        if id(parameter) not in before_step:
                            continue
                        before, expected = before_step[id(parameter)]
                        actual = parameter.detach().double().cpu()
                        max_error = max(max_error, float((actual - expected).abs().max()))
                        update_squared += float((actual - before).square().sum())
                report["optimizer_steps"] += 1
                report["adamw_fp64_reference_max_absolute_error"] = max_error
                report["parameter_update_l2"] = update_squared**0.5
                report["checks"]["adamw_first_update"] = max_error <= 1e-7
                report["events"].append("actual_optimizer_step_completed")
                before_step.clear()
                save()

    entry = Path(__file__).with_name("train_grpo_resource_smoke.py")
    sys.argv = [str(entry), *argv]

    class StartObservation(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            # Observe only execution, keeping expensive model/backend construction
            # outside Python profiling. This callback never changes TrainerControl.
            sys.setprofile(profile)

    from train_grpo_resource_smoke import main as production_main

    try:
        production_main(callbacks=[StartObservation()])
        report["entry_completed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        sys.setprofile(None)
        save()
    assert report["optimizer_steps"] == 1
    assert all(report["checks"].values())
    assert report["batches"] and all(all(batch["checks"].values()) for batch in report["batches"])
    print("EXECUTION_AUDIT=" + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
