#!/usr/bin/env python3
"""Deterministically stress the owning vLLM and TRL GRPO context paths."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset
from grpo_kernel_check import enable_local_fla_kernels, require_gdn_kernels_active
from peft import LoraConfig
from transformers.integrations.liger import apply_liger_kernel
from trl import GRPOConfig, GRPOTrainer


def reward_func(completions, **kwargs):
    return [0.0] * len(completions)


def lora_config() -> LoraConfig:
    return LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        target_modules="all-linear",
        exclude_modules=r".*(?:visual|vision|aligner|multi_token_predictor|mtp|lm_head).*",
        task_type="CAUSAL_LM",
    )


def common_config(
    output_dir: Path,
    *,
    max_completion_length: int,
    use_vllm: bool,
    max_model_length: int,
    vllm_memory_utilization: float,
) -> GRPOConfig:
    return GRPOConfig(
        output_dir=str(output_dir),
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        generation_batch_size=4,
        num_generations=4,
        max_completion_length=max_completion_length,
        use_vllm=use_vllm,
        vllm_mode="colocate",
        vllm_enable_sleep_mode=True,
        vllm_gpu_memory_utilization=vllm_memory_utilization,
        vllm_max_model_length=max_model_length,
        vllm_importance_sampling_correction=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=None,
        beta=0.0,
        num_iterations=1,
        scale_rewards="none",
        loss_type="dapo",
        learning_rate=1e-6,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=True,
        model_init_kwargs={"dtype": "bfloat16"},
        remove_unused_columns=False,
        shuffle_dataset=False,
        save_strategy="no",
        report_to="none",
        seed=42,
        data_seed=42,
    )


def build_trainer(args: argparse.Namespace, *, use_vllm: bool) -> GRPOTrainer:
    completion_length = args.total_length - args.prompt_length
    config = common_config(
        args.output_dir,
        max_completion_length=completion_length,
        use_vllm=use_vllm,
        max_model_length=args.total_length,
        vllm_memory_utilization=args.vllm_memory_utilization,
    )
    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        reward_funcs=reward_func,
        train_dataset=Dataset.from_list([{"prompt": "Context stress."}]),
        peft_config=lora_config(),
    )
    if args.fla_kernel_dir is not None:
        enable_local_fla_kernels(trainer.model, args.fla_kernel_dir)
        require_gdn_kernels_active(trainer.model)
    return trainer


def run_kv_capacity(args: argparse.Namespace) -> None:
    trainer = build_trainer(args, use_vllm=True)
    engine = trainer.vllm_generation.llm.llm_engine
    cache_config = engine.vllm_config.cache_config
    num_blocks = cache_config.num_gpu_blocks
    block_size = cache_config.block_size
    if num_blocks is None or num_blocks <= 0:
        raise RuntimeError("vLLM did not report a positive GPU KV block count")
    capacity_tokens = int(num_blocks) * int(block_size)
    print(
        "KV_CAPACITY_RESULT="
        + json.dumps(
            {
                "total_length": args.total_length,
                "num_gpu_blocks": int(num_blocks),
                "block_size": int(block_size),
                "resident_token_capacity": capacity_tokens,
                "full_length_resident_sequences": capacity_tokens // args.total_length,
                "requested_concurrency": 4,
                "all_four_fully_resident": capacity_tokens >= 4 * args.total_length,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def repeated_natural_tokens(tokenizer, length: int) -> list[int]:
    seed = tokenizer.encode(
        "Causal world context stress with observations interventions and terminal quality. ",
        add_special_tokens=False,
    )
    if not seed:
        raise RuntimeError("tokenizer produced an empty stress sequence")
    return (seed * math.ceil(length / len(seed)))[:length]


def phase_result(name: str, started: float) -> dict[str, float | str]:
    torch.cuda.synchronize()
    return {
        "phase": name,
        "seconds": time.perf_counter() - started,
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def run_train(args: argparse.Namespace) -> None:
    trainer = build_trainer(args, use_vllm=False)
    if trainer.args.use_liger_kernel:
        apply_liger_kernel(trainer.model, trainer.args.liger_kernel_config)
    if trainer.args.gradient_checkpointing:
        trainer.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=trainer.args.gradient_checkpointing_kwargs
        )
    if args.fla_kernel_dir is not None:
        require_gdn_kernels_active(trainer.model)
    trainer.model = trainer.accelerator.prepare_model(trainer.model)
    trainer.model.train()
    trainer.current_gradient_accumulation_steps = 1
    device = trainer.accelerator.device
    tokenizer = getattr(trainer.processing_class, "tokenizer", trainer.processing_class)
    token_ids = repeated_natural_tokens(tokenizer, args.total_length)
    all_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    prompt_ids = all_ids[:, : args.prompt_length]
    completion_ids = all_ids[:, args.prompt_length :]
    prompt_mask = torch.ones_like(prompt_ids)
    completion_mask = torch.ones_like(completion_ids)
    attention_mask = torch.ones_like(all_ids)
    phase = "old_logprob"
    results = []
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.no_grad():
            old_logps, entropy, aux_loss = trainer._get_per_token_logps_and_entropies(
                trainer.model,
                all_ids,
                attention_mask,
                completion_ids.size(1),
                batch_size=1,
            )
        if entropy is not None or aux_loss is not None or not torch.isfinite(old_logps).all():
            raise RuntimeError("fused old-logprob stress returned an invalid result")
        results.append(phase_result(phase, started))

        phase = "liger_loss_backward"
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        inputs = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": torch.ones(1, dtype=torch.float32, device=device),
            "old_per_token_logps": old_logps,
            "importance_sampling_ratio": torch.ones_like(old_logps),
        }
        loss = trainer.compute_loss(trainer.model, inputs)
        if not torch.isfinite(loss):
            raise RuntimeError("Liger GRPO stress loss is not finite")
        loss.backward()
        results.append(phase_result(phase, started))

        phase = "adam_step"
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        trainer.create_optimizer()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad(set_to_none=True)
        results.append(phase_result(phase, started))
    except torch.OutOfMemoryError:
        print(
            "CONTEXT_STRESS_OOM="
            + json.dumps(
                {
                    "total_length": args.total_length,
                    "prompt_length": args.prompt_length,
                    "completion_length": completion_ids.size(1),
                    "phase": phase,
                    "completed_phases": results,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise

    print(
        "TRAIN_CONTEXT_RESULT="
        + json.dumps(
            {
                "total_length": args.total_length,
                "prompt_length": args.prompt_length,
                "completion_length": completion_ids.size(1),
                "loss": float(loss.detach()),
                "phases": results,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("kv", "train"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-length", type=int, required=True)
    parser.add_argument("--prompt-length", type=int, default=1271)
    parser.add_argument("--vllm-memory-utilization", type=float, default=0.50)
    parser.add_argument("--fla-kernel-dir", type=Path)
    args = parser.parse_args()
    if args.total_length <= args.prompt_length:
        parser.error("total-length must exceed prompt-length")
    args.output_dir = args.output_dir.expanduser().resolve()
    try:
        run_kv_capacity(args) if args.mode == "kv" else run_train(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
