#!/usr/bin/env python3
"""Run official TRL/vLLM/LoRA GRPO training or a one-step resource validation."""

from __future__ import annotations

import argparse
import json
import os
import time
from fractions import Fraction
from pathlib import Path
from types import MethodType

import torch.distributed as dist
from datasets import IterableDataset
from grpo_kernel_check import enable_local_fla_kernels, require_gdn_kernels_active
from grpo_logprob_guard import require_finite_sampling_logprobs
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

import cpt_world
from cpt_world import (
    TASK_FAMILY_QUERY_TYPES,
    TERMINAL_QUALITY_REWARD_VERSION,
    CPTWorldEnvironment,
    build_cpt_world_advantage_utility,
    iter_random_balanced_training_rows,
    task_advantage_utility,
    terminal_quality_reward,
)

_EXPECTED_REWARD_VERSION = "terminal-quality-v9"


def require_cpt_world_training_contract() -> dict[str, object]:
    """Fail before model loading when the process imported the wrong CPT-World."""

    loaded_source = Path(cpt_world.__file__).resolve()
    expected_source_value = os.environ.get("CPT_WORLD_EXPECTED_SOURCE")
    if expected_source_value:
        expected_source = Path(expected_source_value).resolve()
        if expected_source not in loaded_source.parents:
            raise RuntimeError(
                "cpt_world import escaped the requested project: "
                f"{loaded_source} not under {expected_source}"
            )
    if TERMINAL_QUALITY_REWARD_VERSION != _EXPECTED_REWARD_VERSION:
        raise RuntimeError(
            f"training requires {_EXPECTED_REWARD_VERSION}, got {TERMINAL_QUALITY_REWARD_VERSION}"
        )
    for query_type in TASK_FAMILY_QUERY_TYPES:
        for quality in (0.0, 0.25, 0.75, 1.0):
            if task_advantage_utility(quality, query_type) != quality:
                raise RuntimeError(
                    f"training utility altered {query_type} terminal quality {quality}"
                )
    backdoor_contract = {
        "exact": terminal_quality_reward({"kind": "backadj", "edit_distance": 0}),
        "one_edit": terminal_quality_reward({"kind": "backadj", "edit_distance": 1}),
        "two_edits": terminal_quality_reward({"kind": "backadj", "edit_distance": 2}),
    }
    if backdoor_contract != {
        "exact": Fraction(1),
        "one_edit": Fraction(1, 2),
        "two_edits": Fraction(1, 3),
    }:
        raise RuntimeError(f"unexpected backdoor reward contract: {backdoor_contract}")
    return {
        "source": str(loaded_source),
        "reward_version": TERMINAL_QUALITY_REWARD_VERSION,
        "task_families": list(TASK_FAMILY_QUERY_TYPES),
        "utility": "identity",
        "backdoor_reward": {key: str(value) for key, value in backdoor_contract.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-completion-length", type=int, default=7168)
    parser.add_argument("--max-model-length", type=int, default=9216)
    parser.add_argument(
        "--vllm-mode",
        choices=("colocate", "server"),
        default="colocate",
    )
    parser.add_argument("--vllm-memory-utilization", type=float, default=0.50)
    parser.add_argument(
        "--vllm-server-base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument("--vllm-server-timeout", type=float, default=600.0)
    parser.add_argument("--vllm-group-port", type=int, default=51216)
    parser.add_argument(
        "--vllm-rollout-residency",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep colocated vLLM awake across every tool turn in one rollout.",
    )
    parser.add_argument(
        "--vllm-enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable automatic prefix caching; omit to preserve the vLLM default.",
    )
    parser.add_argument("--vllm-mtp-speculative-tokens", type=int, default=0)
    parser.add_argument("--vllm-sleep-level", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--capture-rollouts",
        action="store_true",
        help=(
            "Write exact rollout token IDs, tool traces, logprobs, and reward parquet "
            "for A/B validation."
        ),
    )
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=200)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument(
        "--use-liger-kernel",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fla-kernel-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    print(
        "CPT_WORLD_RUNTIME="
        + json.dumps(require_cpt_world_training_contract(), separators=(",", ":")),
        flush=True,
    )
    if cli.vllm_mtp_speculative_tokens < 0:
        raise ValueError("--vllm-mtp-speculative-tokens must be nonnegative")
    if cli.vllm_mtp_speculative_tokens and cli.vllm_sleep_level != 1:
        raise ValueError(
            "MTP speculative decoding requires --vllm-sleep-level 1 so its draft "
            "weights survive the generation/training sleep boundary"
        )
    if cli.vllm_mode == "server" and (
        cli.vllm_rollout_residency
        or cli.vllm_enable_prefix_caching is not None
        or cli.vllm_mtp_speculative_tokens
        or cli.vllm_sleep_level != 2
    ):
        raise ValueError(
            "colocated rollout acceleration flags do not apply in server mode; "
            "configure prefix caching and speculative decoding on `trl vllm-serve`"
        )
    output_dir = Path(cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = IterableDataset.from_generator(iter_random_balanced_training_rows)
    acceleration_config = {}
    if cli.vllm_mode == "colocate" and "vllm_rollout_residency" in GRPOConfig.__dataclass_fields__:
        acceleration_config = {
            "vllm_rollout_residency": cli.vllm_rollout_residency,
            "vllm_enable_prefix_caching": cli.vllm_enable_prefix_caching,
            "vllm_speculative_config": (
                {
                    "method": "mtp",
                    "num_speculative_tokens": cli.vllm_mtp_speculative_tokens,
                }
                if cli.vllm_mtp_speculative_tokens
                else None
            ),
            "vllm_sleep_level": cli.vllm_sleep_level,
        }
    elif cli.vllm_mode == "colocate" and (
        cli.vllm_rollout_residency
        or cli.vllm_enable_prefix_caching is not None
        or cli.vllm_mtp_speculative_tokens
        or cli.vllm_sleep_level != 2
    ):
        raise RuntimeError(
            "rollout acceleration was requested but the installed TRL owner patch is absent"
        )
    config = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=cli.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        generation_batch_size=4,
        num_generations=4,
        max_completion_length=cli.max_completion_length,
        max_tool_calling_iterations=None,
        use_vllm=True,
        vllm_mode=cli.vllm_mode,
        vllm_server_base_url=cli.vllm_server_base_url,
        vllm_server_timeout=cli.vllm_server_timeout,
        vllm_group_port=cli.vllm_group_port,
        vllm_enable_sleep_mode=cli.vllm_mode == "colocate",
        vllm_gpu_memory_utilization=cli.vllm_memory_utilization,
        vllm_max_model_length=cli.max_model_length,
        vllm_importance_sampling_correction=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=None,
        beta=0.0,
        num_iterations=1,
        scale_rewards="none",
        loss_type="dapo",
        mask_truncated_completions=False,
        learning_rate=1e-6,
        warmup_steps=int(cli.max_steps * 0.05),
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=cli.use_liger_kernel,
        model_init_kwargs={"dtype": "bfloat16"},
        chat_template_kwargs={"enable_thinking": False},
        remove_unused_columns=False,
        shuffle_dataset=False,
        logging_steps=1,
        log_completions=cli.capture_rollouts,
        num_completions_to_print=0,
        save_strategy="steps",
        save_steps=cli.save_steps,
        save_total_limit=cli.save_total_limit,
        report_to="none",
        seed=42,
        data_seed=42,
        **acceleration_config,
    )
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        target_modules="all-linear",
        exclude_modules=r".*(?:visual|vision|aligner|multi_token_predictor|mtp|lm_head).*",
        task_type="CAUSAL_LM",
    )
    trainer = GRPOTrainer(
        model=cli.model,
        args=config,
        train_dataset=dataset,
        reward_funcs=build_cpt_world_advantage_utility(),
        peft_config=peft_config,
        environment_factory=CPTWorldEnvironment,
    )
    if trainer.reward_func_names != ["CPTWorldAdvantageUtility", "CPTWorldEnvironment"]:
        raise RuntimeError(
            "TRL reward-source order changed; refusing to guess advantage utility weights"
        )
    trainer.reward_weights[0] = 1.0
    trainer.reward_weights[1] = 0.0
    original_generate = trainer._generate

    def guarded_generate(_trainer, prompts):
        result = original_generate(prompts)
        require_finite_sampling_logprobs(
            result[4],
            global_step=trainer.state.global_step,
        )
        return result

    trainer._generate = MethodType(guarded_generate, trainer)
    if cli.capture_rollouts:
        rollout_path = output_dir / "rollout-artifacts.jsonl"
        lifecycle_path = output_dir / "vllm-lifecycle.jsonl"
        guarded_generate_method = trainer._generate

        def append_jsonl(path: Path, payload: dict) -> None:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")

        def time_method(owner, method_name: str, event_name: str) -> None:
            original = getattr(owner, method_name)

            def timed(*args, **kwargs):
                started = time.perf_counter()
                try:
                    result = original(*args, **kwargs)
                finally:
                    elapsed = time.perf_counter() - started
                    append_jsonl(
                        lifecycle_path,
                        {
                            "global_step": trainer.state.global_step,
                            "event": event_name,
                            "seconds": elapsed,
                        },
                    )
                if event_name == "generate":
                    for output in result:
                        metrics = getattr(output, "metrics", None)
                        append_jsonl(
                            lifecycle_path,
                            {
                                "global_step": trainer.state.global_step,
                                "event": "request",
                                "prompt_tokens": len(output.prompt_token_ids),
                                "completion_tokens": sum(
                                    len(completion.token_ids) for completion in output.outputs
                                ),
                                "cached_tokens": getattr(output, "num_cached_tokens", 0),
                                "cache_creation_tokens": getattr(
                                    output,
                                    "num_cache_creation_tokens",
                                    0,
                                ),
                                "first_token_latency": getattr(
                                    metrics,
                                    "first_token_latency",
                                    None,
                                ),
                                "num_generation_tokens": getattr(
                                    metrics,
                                    "num_generation_tokens",
                                    None,
                                ),
                                "first_token_ts": getattr(metrics, "first_token_ts", None),
                                "last_token_ts": getattr(metrics, "last_token_ts", None),
                            },
                        )
                return result

            setattr(owner, method_name, timed)

        time_method(trainer.vllm_generation, "sync_weights", "sync_weights")
        time_method(trainer.vllm_generation.llm, "wake_up", "wake_up")
        time_method(trainer.vllm_generation.llm, "sleep", "sleep")
        time_method(trainer.vllm_generation.llm, "generate", "generate")

        def capture_generate(_trainer, prompts):
            result = guarded_generate_method(prompts)
            prompt_ids, completion_ids, tool_mask, completions, logprobs = result[:5]
            payload = {
                "global_step": trainer.state.global_step,
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids,
                "tool_mask": tool_mask,
                "tool_trace": completions,
                "sampling_logprobs": logprobs,
            }
            append_jsonl(rollout_path, payload)
            return result

        trainer._generate = MethodType(capture_generate, trainer)
    if cli.fla_kernel_dir is not None:
        enable_local_fla_kernels(trainer.model, cli.fla_kernel_dir)
        require_gdn_kernels_active(trainer.model)
    try:
        trainer.train(resume_from_checkpoint=cli.resume_from_checkpoint)
        trainer.save_model(str(output_dir / "final-adapter"))
        trainer.save_state()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
