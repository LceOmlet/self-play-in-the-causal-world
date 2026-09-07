"""Check reward-scale interpretation using real groups and official GRPO code.

This offline CPU comparison never substitutes rewards or advantages in training.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    sys.path.insert(0, str(args.verl))
    import numpy as np
    import torch
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    source = Path(compute_grpo_outcome_advantage.__code__.co_filename).resolve()
    assert source == (args.verl / "verl/trainer/ppo/core_algos.py").resolve()
    groups = defaultdict(list)
    for path in args.inputs:
        for row in json.loads(path.read_text())["retained_rollouts"]:
            if row["query_type"] == "backadj_minimal_sets":
                groups[row["step"]].append(row)
    records, skipped = [], []
    epsilon, denominator = 1e-6, 14
    for step, rows in sorted(groups.items()):
        assert len(rows) == 4
        assert len({r["request_id"] for r in rows}) == 4
        assert all(r["event_join_method"] != "unresolved" for r in rows)
        if not all(r["completed"] and r["official_overlong_reward"] == 0 for r in rows):
            skipped.append(step)
            continue
        distance = np.array([r["task_metrics"]["edit_distance"] for r in rows], dtype=np.float64)
        quality = np.array([r["raw_quality"] for r in rows], dtype=np.float64)
        assert np.max(np.abs(quality - (1 - distance / denominator))) < 1e-14
        std = distance.std(ddof=1)
        assert std > 0
        expected = -(distance - distance.mean()) / (std + denominator * epsilon)

        def official(values, dtype):
            rewards = torch.tensor(values, dtype=dtype).unsqueeze(1)
            mask = torch.ones_like(rewards)
            advantage, _ = compute_grpo_outcome_advantage(
                rewards,
                mask,
                np.array(["same_actual_task"] * 4),
                epsilon=epsilon,
                norm_adv_by_std_in_grpo=True,
            )
            return advantage[:, 0].double().numpy()

        q64 = official(quality, torch.float64)
        q32 = official(quality, torch.float32)
        neg_d32 = official(-distance, torch.float32)
        assert np.max(np.abs(q64 - expected)) < 1e-12
        assert np.max(np.abs(q32 - expected)) < 5e-6
        exact_scale_difference = np.max(
            np.abs(distance - distance.mean())
            * (denominator - 1)
            * epsilon
            / ((std + denominator * epsilon) * (std + epsilon))
        )
        assert np.max(np.abs(q32 - neg_d32)) <= exact_scale_difference + 5e-6
        records.append(
            {
                "step": step,
                "requests": [r["request_id"] for r in rows],
                "edit_distances": distance.tolist(),
                "qualities": quality.tolist(),
                "official_quality_advantage_float32": q32.tolist(),
                "official_negative_distance_advantage_float32": neg_d32.tolist(),
                "formula_max_error_float64": float(np.max(np.abs(q64 - expected))),
                "quality_vs_negative_distance_max_difference_float32": float(
                    np.max(np.abs(q32 - neg_d32))
                ),
                "exact_epsilon_scale_difference_bound": float(exact_scale_difference),
            }
        )
    assert records and not torch.cuda.is_initialized()
    report = {
        "official_source": str(source),
        "official_source_sha256": sha(source),
        "script_sha256": sha(Path(__file__)),
        "input_sha256": {str(path): sha(path) for path in args.inputs},
        "epsilon": epsilon,
        "reward_denominator": denominator,
        "groups": records,
        "incomplete_or_penalized_groups_excluded": skipped,
        "scope": "Actual retained backdoor groups with completed answers and no length penalty. "
        "Official outcome-advantage call on CPU scalar rewards, not replay of full GPU loss. "
        "Affine reward compression does not imply equally compressed standardized advantages. "
        "This does not establish task learnability or surrogate equivalence to exact-set accuracy.",
    }
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "verified_actual_groups": len(records),
                "excluded_groups": skipped,
                "maximum_float32_advantage_difference": max(
                    r["quality_vs_negative_distance_max_difference_float32"] for r in records
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
