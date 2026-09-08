"""Measure storage rounding using saved native parameters and post-step moments.

No inference or optimizer step is run. Holding the recorded BF16 moments fixed
isolates parameter/update arithmetic; it does not reconstruct an FP32 training
history or measure learning loss. LoRA Frobenius products are evaluated through
small Gram matrices instead of materializing full dense weight updates.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def local(value):
    return value.to_local() if isinstance(value, DTensor) else value


def norm2(left, right):
    return float(((left.T @ left) * (right @ right.T)).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    p0, p1 = [
        {
            k: local(v).double()
            for k, v in load(
                args.root / f"global_step_{step}/actor/model_world_size_1_rank_0.pt"
            ).items()
        }
        for step in [57, 58]
    ]
    optimizer = load(args.root / "global_step_58/actor/optim_world_size_1_rank_0.pt")
    group = optimizer["param_groups"][0]
    lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
    beta1, beta2 = group["betas"]
    order = json.loads((args.root / "parameter-order.json").read_text())
    ideal = {}
    counts = {
        part: {
            k: 0
            for k in [
                "elements",
                "actual_nonzero",
                "desired_squared",
                "actual_squared",
                "error_squared",
                "v_positive",
                "v_decay_unchanged",
            ]
        }
        for part in ["lora_A", "lora_B"]
    }
    for row in order:
        if not row["requires_grad"]:
            assert not optimizer["state"][row["id"]]
            continue
        name = row["name"]
        state = optimizer["state"][row["id"]]
        assert list(p0[name].shape) == row["shape"] == list(state["exp_avg"].shape)
        step = float(local(state["step"]))
        assert step == 58
        m, v = [local(state[k]).double() for k in ["exp_avg", "exp_avg_sq"]]
        delta = -lr * wd * p0[name] - lr * (m / (1 - beta1**step)) / (
            torch.sqrt(v / (1 - beta2**step)) + eps
        )
        ideal[name] = p0[name] + delta
        actual = p1[name] - p0[name]
        item = counts["lora_A" if "lora_A" in name else "lora_B"]
        item["elements"] += delta.numel()
        item["actual_nonzero"] += int(torch.count_nonzero(actual))
        item["desired_squared"] += float(delta.square().sum())
        item["actual_squared"] += float(actual.square().sum())
        item["error_squared"] += float((actual - delta).square().sum())
        vb = v.bfloat16()
        item["v_positive"] += int((vb > 0).sum())
        item["v_decay_unchanged"] += int(((vb * beta2 == vb) & (vb > 0)).sum())
    effective = {
        k: 0.0
        for k in [
            "actual_squared",
            "desired_squared",
            "error_squared",
            "desired_a_term_squared",
            "desired_b_term_squared",
        ]
    }
    for a in p0:
        if "lora_A" not in a:
            continue
        b = a.replace("lora_A", "lora_B")
        a0, b0, a1, b1, af, bf = p0[a], p0[b], p1[a], p1[b], ideal[a], ideal[b]
        effective["actual_squared"] += norm2(
            torch.cat([b1 - b0, b0], 1), torch.cat([a1, a1 - a0], 0)
        )
        effective["desired_squared"] += norm2(
            torch.cat([bf - b0, b0], 1), torch.cat([af, af - a0], 0)
        )
        effective["error_squared"] += norm2(
            torch.cat([b1 - bf, bf], 1), torch.cat([a1, a1 - af], 0)
        )
        effective["desired_a_term_squared"] += norm2(b0, af - a0)
        effective["desired_b_term_squared"] += norm2(bf - b0, a0)
    actual, desired, error = [
        effective[k] for k in ["actual_squared", "desired_squared", "error_squared"]
    ]
    result = {
        "step": 58,
        "parameters": counts,
        "effective_weight_update": effective,
        "effective_relative_error": math.sqrt(error / desired),
        "update_direction_cosine": (actual + desired - error) / (2 * math.sqrt(actual * desired)),
        "update_norm_ratio": math.sqrt(actual / desired),
        "a_term_to_b_term_norm_ratio": math.sqrt(
            effective["desired_a_term_squared"] / effective["desired_b_term_squared"]
        ),
        "scope": "Recorded post-step moments held fixed. Storage/update arithmetic only; "
        "not full FP32 training or capability gain. LoRA scale 4 cancels in ratios.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
