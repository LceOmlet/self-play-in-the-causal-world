#!/usr/bin/env python3
"""Compare fused/unfused DAPO on the actual model without any optimizer update."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from grpo_kernel_check import enable_local_fla_kernels, require_gdn_kernels_active
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer


def zero_reward(completions, **kwargs):
    return [0.0] * len(completions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fla-kernel-dir", type=Path, required=True)
    cli = parser.parse_args()
    out = Path(cli.output_dir)
    args = GRPOConfig(
        output_dir=str(out),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        generation_batch_size=4,
        num_generations=4,
        max_completion_length=16,
        use_vllm=False,
        vllm_importance_sampling_correction=False,
        loss_type="dapo",
        scale_rewards="none",
        epsilon=0.2,
        epsilon_high=0.28,
        beta=0.0,
        use_liger_kernel=True,
        bf16=True,
        disable_dropout=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        model_init_kwargs={"dtype": "bfloat16"},
        report_to="none",
        max_steps=1,
    )
    trainer = GRPOTrainer(
        model=cli.model,
        args=args,
        reward_funcs=zero_reward,
        train_dataset=Dataset.from_dict({"prompt": ["A short causal question."]}),
        peft_config=LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            target_modules="all-linear",
            exclude_modules=r".*(?:visual|vision|aligner|multi_token_predictor|mtp|lm_head).*",
            task_type="CAUSAL_LM",
        ),
    )
    trainer.model.load_adapter(cli.adapter, adapter_name="default", is_trainable=True)
    enable_local_fla_kernels(trainer.model, cli.fla_kernel_dir)
    require_gdn_kernels_active(trainer.model)
    trainer.model.train()
    trainer.current_gradient_accumulation_steps = 4
    device = trainer.accelerator.device
    tokenizer = trainer._tokenizer
    prompt = tokenizer.encode("Given observations of X and Y, reason about their causal relation.")
    completion = tokenizer.encode(
        "We can compare the conditional probabilities using a controlled experiment."
    )[:12]
    while len(completion) < 12:
        completion.append(tokenizer.eos_token_id)
    lengths = [3, 6, 9, 12]
    masks = []
    for i, length in enumerate(lengths):
        mask = torch.arange(12, device=device) < length
        if i == 2:
            mask[4:6] = False
        masks.append(mask)
    denominator = torch.stack(masks).sum()
    advantages = [-1.0, -0.3, 0.4, 0.9]
    trainable = {
        name: parameter
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad
    }
    before = {name: parameter.detach().cpu().clone() for name, parameter in trainable.items()}
    # Replay fixed tensors while retaining the real Trainer/Accelerate backward
    # path. Generation is deliberately excluded from this normalization check.
    trainer._prepare_inputs = lambda inputs: inputs
    collected = {}
    for backend in ("fused", "unfused", "training_step"):
        trainer.model.zero_grad(set_to_none=True)
        losses = []
        for i, length in enumerate(lengths):
            inputs = {
                "prompt_ids": torch.tensor([prompt], device=device),
                "prompt_mask": torch.ones(1, len(prompt), device=device, dtype=torch.long),
                "completion_ids": torch.tensor([completion], device=device),
                "completion_mask": (torch.arange(12, device=device) < length)[None, :],
                "tool_mask": masks[i][None, :],
                "advantages": torch.tensor([advantages[i]], device=device),
                "num_items_in_batch": denominator,
            }
            with trainer.compute_loss_context_manager():
                if backend == "fused":
                    loss = trainer.compute_liger_loss(trainer.model, inputs)
                elif backend == "unfused":
                    loss = GRPOTrainer._compute_loss(trainer, trainer.model, inputs)
                else:
                    loss = trainer.training_step(trainer.model, inputs, denominator)
            losses.append(loss.detach().float().item())
            if backend != "training_step":
                loss.backward()
        collected[backend] = {
            "loss": sum(losses),
            "gradient": torch.cat(
                [
                    (
                        parameter.grad.detach().float().cpu()
                        if parameter.grad is not None
                        else torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
                    ).reshape(-1)
                    for parameter in trainable.values()
                ]
            ),
        }
    fused = collected["fused"]["gradient"].double()
    unfused = collected["unfused"]["gradient"].double()
    step_gradient = collected["training_step"]["gradient"].double()
    relative_error = (fused - unfused).norm() / unfused.norm().clamp_min(1e-30)
    step_error = (fused - step_gradient).norm() / fused.norm().clamp_min(1e-30)
    unchanged = all(
        torch.equal(before[name], parameter.detach().cpu()) for name, parameter in trainable.items()
    )
    result = {
        "model": cli.model,
        "adapter": cli.adapter,
        "optimizer_steps": 0,
        "trainable_parameters_unchanged": unchanged,
        "trainable_parameter_count": fused.numel(),
        "active_tokens": int(denominator),
        "fused_loss": collected["fused"]["loss"],
        "unfused_loss": collected["unfused"]["loss"],
        "relative_gradient_l2_error": relative_error.item(),
        "trainer_accelerate_backward_relative_error": step_error.item(),
        "gradient_cosine_similarity": torch.nn.functional.cosine_similarity(
            fused, unfused, dim=0
        ).item(),
        "max_absolute_gradient_error": (fused - unfused).abs().max().item(),
        "bf16_relative_l2_tolerance": 0.02,
        "scope": (
            "short-sequence forward/backward integration; no rollout or optimizer step; "
            "not a 32k resource certification"
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "model-gradient-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    assert unchanged
    assert torch.isfinite(fused).all() and torch.isfinite(unfused).all()
    assert step_error < 2e-4, "Trainer/Accelerate introduces unexpected accumulation scaling"

    # A float64 autograd head derivative at the SAME bf16 forward logits
    # distinguishes mathematical gradient correctness from bf16 softmax rounding.
    trainer.model.zero_grad(set_to_none=True)
    all_ids = torch.tensor([prompt + completion], device=device)
    full_attention = torch.ones_like(all_ids)
    with torch.no_grad(), trainer.compute_loss_context_manager():
        hidden = trainer._get_last_hidden_state(
            trainer.model,
            all_ids,
            full_attention,
            12,
        ).detach()
    weight = trainer.model.lm_head.weight
    target_ids = torch.tensor([completion], device=device)
    head_mask = torch.ones(1, 12, device=device)
    head_advantage = torch.tensor([0.7], device=device)
    head_inputs = {
        "prompt_ids": torch.tensor([prompt], device=device),
        "prompt_mask": torch.ones(1, len(prompt), device=device),
        "completion_ids": target_ids,
        "completion_mask": head_mask,
        "advantages": head_advantage,
        "num_items_in_batch": head_mask.sum(),
    }
    head_input = hidden.clone().requires_grad_()
    original_hidden_method = trainer._get_last_hidden_state
    trainer._get_last_hidden_state = lambda *args: head_input
    with trainer.compute_loss_context_manager():
        trainer.compute_liger_loss(trainer.model, head_inputs).backward()
    trainer._get_last_hidden_state = original_hidden_method
    with torch.no_grad():
        logits = torch.nn.functional.linear(hidden, weight).double()
    logits.requires_grad_()
    score = logits.log_softmax(-1).gather(-1, target_ids[..., None]).squeeze(-1)
    # At ratio=1, the DAPO derivative is the score-function derivative.
    # Use independent PyTorch float64 autograd to avoid a hand-coded sign oracle.
    reference_loss = -(score * head_advantage.double()[:, None]).mean()
    grad_logits = torch.autograd.grad(reference_loss, logits)[0]
    with torch.no_grad():
        oracle = (grad_logits @ weight.double()).to(hidden.dtype)
        head_actual = head_input.grad.double()
        head_error = (head_actual - oracle.double()).norm() / oracle.double().norm()
    result["head_float64_autograd_relative_error"] = head_error.item()
    result["unfused_bf16_relative_tolerance_passed"] = bool(relative_error < 0.02)
    (out / "model-gradient-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "HEAD_FLOAT64_COMPARISON=" + json.dumps({"relative_error": head_error.item()}), flush=True
    )
    assert head_error < 0.002, "Fused head gradient differs from float64 DAPO derivative"


if __name__ == "__main__":
    main()
