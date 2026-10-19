"""Official GRPO math with an explicitly registered rollout scheduling patch."""
from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
from pathlib import Path

from datasets import IterableDataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from cpt_world import (
    CPTWorldEnvironment,
    OBSERVATION_BUDGET_EXPONENTS,
    TASK_FAMILY_QUERY_TYPES,
    TERMINAL_QUALITY_REWARD_VERSION,
    build_cpt_world_advantage_utility,
    iter_random_balanced_training_rows,
    task_advantage_utility,
)

VERSIONS = {
    "trl": "1.14.0+cpt.rollout1", "transformers": "5.15.1", "accelerate": "1.14.0",
    "liger-kernel": "0.8.2", "torch": "2.11.0", "vllm": "0.26.0", "peft": "0.20.0",
    "flash-linear-attention": "0.5.2", "fla-core": "0.5.2",
}


def configuration(output_dir: str, *, use_vllm: bool = True, use_liger: bool = True) -> GRPOConfig:
    """All algorithm behavior is selected through the public official config."""
    return GRPOConfig(
        output_dir=output_dir, max_steps=10000,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        generation_batch_size=4, num_generations=4,
        max_completion_length=31497, max_tool_calling_iterations=None,
        use_vllm=use_vllm, vllm_mode="colocate", vllm_enable_sleep_mode=use_vllm,
        vllm_gpu_memory_utilization=0.50, vllm_max_model_length=32768,
        vllm_importance_sampling_correction=True,
        vllm_importance_sampling_mode="sequence_mask", vllm_importance_sampling_clip_max=3.0,
        temperature=1.0, top_p=1.0, top_k=0, min_p=None,
        beta=0.0, num_iterations=1, scale_rewards="none", loss_type="dapo",
        importance_sampling_level="token", epsilon=0.2, epsilon_high=0.2,
        mask_truncated_completions=False,
        # The existing adapter logs native task metrics and receives zero weight.
        # TRL appends the environment's get_reward with its official weight 1.
        # This avoids mutating trainer.reward_weights after construction.
        reward_weights=[0.0],
        learning_rate=1e-6, warmup_steps=500, lr_scheduler_type="linear",
        optim="adamw_torch_fused", weight_decay=0.0, max_grad_norm=1.0,
        bf16=True, tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=use_liger, model_init_kwargs={"dtype": "bfloat16"},
        chat_template_kwargs={"enable_thinking": False},
        remove_unused_columns=False, shuffle_dataset=False,
        logging_steps=1, log_completions=False, num_completions_to_print=0,
        save_strategy="steps", save_steps=50, save_total_limit=200,
        report_to="none", seed=42, data_seed=42, ignore_data_skip=False,
    )


def lora_configuration() -> LoraConfig:
    return LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.0, bias="none", target_modules="all-linear",
        exclude_modules=r".*(?:visual|vision|aligner|multi_token_predictor|mtp|lm_head).*",
        task_type="CAUSAL_LM",
    )


def require_contract() -> dict:
    import sys
    import torch
    # Official PyTorch backend setting. The actual BF16 head comparison fails
    # its upstream threshold with reduced-precision reductions enabled.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if os.environ.get("PYTHONPATH"):
        raise RuntimeError("PYTHONPATH must be unset; package overlays are not accepted")
    if not sys.flags.isolated:
        raise RuntimeError("Launch with python -I to exclude user-site/current-directory injection")
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS:
        raise RuntimeError(f"Official version lock mismatch: {versions}")
    trainer_source = Path(inspect.getfile(GRPOTrainer)).resolve()
    if not trainer_source.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Trainer imported outside isolated environment: {trainer_source}")
    assert TERMINAL_QUALITY_REWARD_VERSION == "terminal-quality-v10"
    assert OBSERVATION_BUDGET_EXPONENTS == (11, 12, 13, 14)
    for family in TASK_FAMILY_QUERY_TYPES:
        for reward in (0.0, 0.25, 0.75, 1.0):
            assert task_advantage_utility(reward, family) == reward
    return {"versions": versions, "trainer_source": str(trainer_source), "isolated": True,
            "allow_bf16_reduced_precision_reduction": False,
            "reward_source": "environment.get_reward", "metric_adapter_weight": 0.0,
            "task_source_commit": "85cb577c73b17ccc6cde3bc6944db3ce2e80376f"}


def require_official_methods(trainer: GRPOTrainer) -> dict:
    if type(trainer) is not GRPOTrainer:
        raise RuntimeError("Trainer subclasses are not accepted")
    names = ("train", "training_step", "compute_loss", "_chunked_logps", "_full_logits_logps", "_compute_loss",
             "_generate", "_generate_and_score_completions", "_calculate_rewards",
             "_get_per_token_logps_and_entropies")
    sources = {}
    for name in names:
        bound = getattr(trainer, name)
        if name in vars(trainer) or bound.__func__ is not getattr(GRPOTrainer, name):
            raise RuntimeError(f"Instance method override detected: {name}")
        sources[name] = inspect.getfile(bound)
    if trainer.reward_func_names != ["CPTWorldAdvantageUtility", "CPTWorldEnvironment"]:
        raise RuntimeError(f"Unexpected reward sources: {trainer.reward_func_names}")
    if trainer.reward_weights.tolist() != [0.0, 1.0]:
        raise RuntimeError("Expected unmodified official reward assembly [0, 1]")
    return sources


def build_trainer(*, model, args, train_dataset, peft_config=None, callbacks=None,
                  rollout_func=None, processing_class=None) -> GRPOTrainer:
    """Wire public environment/reward hooks; no owner methods are overwritten."""
    trainer = GRPOTrainer(
        model=model, args=args, train_dataset=train_dataset,
        reward_funcs=build_cpt_world_advantage_utility(), peft_config=peft_config,
        environment_factory=CPTWorldEnvironment, callbacks=callbacks,
        rollout_func=rollout_func, processing_class=processing_class,
    )
    require_official_methods(trainer)
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/chen/models/Qwen/Qwen3.5-9B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--use-liger-kernel", action=argparse.BooleanOptionalAction, default=True)
    cli = parser.parse_args()
    contract = require_contract()
    output = Path(cli.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = configuration(str(output), use_liger=cli.use_liger_kernel)
    (output / "official-config.json").write_text(config.to_json_string())
    (output / "source-contract.json").write_text(json.dumps(contract, indent=2))
    if cli.config_only:
        print(json.dumps(contract), flush=True)
        return
    trainer = build_trainer(
        model=cli.model, args=config,
        train_dataset=IterableDataset.from_generator(iter_random_balanced_training_rows),
        peft_config=lora_configuration(),
    )
    sources = require_official_methods(trainer)
    (output / "actual-method-sources.json").write_text(json.dumps(sources, indent=2))
    trainer.train(resume_from_checkpoint=cli.resume_from_checkpoint)
    trainer.save_model(str(output / "final-adapter"))
    trainer.save_state()


if __name__ == "__main__":
    main()
