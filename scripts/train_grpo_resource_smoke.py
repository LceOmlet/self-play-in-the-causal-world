#!/usr/bin/env python3
"""Run one official TRL/vLLM/LoRA GRPO optimizer step for resource validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch.distributed as dist
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from cpt_world import CPTWorldEnvironment, build_balanced_training_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-completion-length", type=int, default=7168)
    parser.add_argument("--max-model-length", type=int, default=9216)
    parser.add_argument("--vllm-memory-utilization", type=float, default=0.48)
    parser.add_argument(
        "--use-liger-kernel",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    output_dir = Path(cli.output_dir).expanduser().resolve()
    rows = build_balanced_training_rows(count_per_family=1)
    dataset = Dataset.from_list(list(rows))
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
        vllm_mode="colocate",
        vllm_enable_sleep_mode=True,
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
        save_strategy="no",
        report_to="none",
        seed=42,
        data_seed=42,
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
        peft_config=peft_config,
        environment_factory=CPTWorldEnvironment,
    )
    try:
        trainer.train()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
