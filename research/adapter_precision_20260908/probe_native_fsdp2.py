"""Exercise the real candidate LoRA builder and native FSDP2/AdamW on CPU.

This checks mixed-dtype computation and checkpoint loading; it is not a DAPO
implementation or a model capability evaluation. CUDA must be hidden.
"""

# Enforce the CPU boundary before importing either framework.
# ruff: noqa: E402

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "This probe is CPU-only"

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import Qwen2Config, Qwen2ForCausalLM
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


def local(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def build(strategy):
    torch.manual_seed(42)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        attn_implementation="eager",
    )
    model = Qwen2ForCausalLM(config).to(torch.bfloat16)
    owner = SimpleNamespace(
        model_config=SimpleNamespace(
            lora_adapter_path=None,
            lora_rank=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            target_parameters=None,
            exclude_modules=None,
        ),
        engine_config=SimpleNamespace(strategy=strategy),
    )
    model = FSDPEngine._build_lora_module(owner, model)
    # Nonzero BF16-representable B weights exercise the adapter contribution;
    # comparing only zero-initialized B would merely compare frozen bases.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(float(torch.tensor(0.01, dtype=torch.bfloat16)))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    init = tempfile.mktemp(prefix="cpt-precision-gloo-")
    dist.init_process_group("gloo", init_method="file://" + init, rank=0, world_size=1)
    try:
        legacy = build("fsdp")
        assert all(p.dtype == torch.bfloat16 for p in legacy.parameters())
        corrected = build("fsdp2")
        assert all(
            p.dtype == (torch.float32 if p.requires_grad else torch.bfloat16)
            for p in corrected.parameters()
        )
        mesh = init_device_mesh("cpu", (1,))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        fully_shard(legacy, mesh=mesh, mp_policy=policy)
        fully_shard(corrected, mesh=mesh, mp_policy=policy)
        optimizer = torch.optim.AdamW(
            corrected.parameters(), lr=1e-6, betas=(0.9, 0.999), weight_decay=0.1
        )
        inputs = torch.tensor([[1, 2, 3, 4, 5]])
        old_logits = legacy(input_ids=inputs).logits
        new_logits = corrected(input_ids=inputs).logits
        torch.testing.assert_close(old_logits, new_logits, rtol=0, atol=0)
        new_logits.float().square().mean().backward()
        assert all(p.grad is None for p in corrected.parameters() if not p.requires_grad)
        assert all(p.grad.dtype == torch.float32 for p in corrected.parameters() if p.requires_grad)
        optimizer.step()
        assert all(
            v.dtype == torch.float32
            for s in optimizer.state.values()
            for k, v in s.items()
            if k != "step"
        )

        # Native model and optimizer load_state_dict must upgrade old BF16
        # checkpoint values into FP32 destinations without resetting moments.
        state = copy.deepcopy(optimizer.state_dict())
        for item in state["state"].values():
            for key in ("exp_avg", "exp_avg_sq"):
                item[key] = item[key].to(torch.bfloat16)
        optimizer.load_state_dict(state)
        reloaded = optimizer.state_dict()
        for index, item in state["state"].items():
            for key in ("exp_avg", "exp_avg_sq"):
                actual = local(reloaded["state"][index][key])
                assert actual.dtype == torch.float32
                torch.testing.assert_close(actual, local(item[key]).float(), rtol=0, atol=0)
        saved = {
            n: p.detach().clone().to(torch.bfloat16)
            for n, p in corrected.named_parameters()
            if p.requires_grad
        }
        corrected.load_state_dict(saved, strict=False)
        for name, parameter in corrected.named_parameters():
            if parameter.requires_grad:
                assert parameter.dtype == torch.float32
                torch.testing.assert_close(
                    local(parameter), local(saved[name]).float(), rtol=0, atol=0
                )
        optimizer.zero_grad()
        corrected(input_ids=inputs).logits.float().square().mean().backward()
        optimizer.step()
        assert all(float(s["step"]) == 2 for s in optimizer.state.values())
        result = {
            "cpu_only": True,
            "torch_version": torch.__version__,
            "fsdp1_existing_cast_preserved": True,
            "fsdp2_trainable_dtype": "float32",
            "frozen_base_dtype": "bfloat16",
            "compute_dtype": str(new_logits.dtype),
            "initial_logits_bitwise_equal": True,
            "gradients_and_moments_dtype": "float32",
            "bf16_checkpoint_values_preserved_in_fp32": True,
            "optimizer_steps": 2,
            "trainable_tensors": sum(p.requires_grad for p in corrected.parameters()),
            "source": FSDPEngine._build_lora_module.__code__.co_filename,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "passed": True,
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
