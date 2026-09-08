"""Read trusted completed DAPO groups and call the official scalar advantage.

No action, reward, loss, or parameter is injected into training. Scalar CPU
advantages describe relative outcome signals, not a replay of GPU gradients.
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verl", type=Path, required=True)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    sys.path.insert(0, str(args.verl))
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    from verl.trainer.ppo.utils import create_rl_sampler

    official = Path(compute_grpo_outcome_advantage.__code__.co_filename).resolve()
    assert official == (args.verl / "verl/trainer/ppo/core_algos.py").resolve()
    assert sha(official) == "4ddf7d6e8b06c396f192ea537c4bd0c2d75772378a04fcec4c9646149c5ae086"
    config = OmegaConf.load(args.configuration)
    assert config.algorithm.adv_estimator == "grpo"
    assert config.algorithm.norm_adv_by_std_in_grpo
    assert config.algorithm.filter_groups.metric == "acc" and config.algorithm.filter_groups.enable
    tasks = {
        row["tape_key"]: row
        for line in args.accepted.read_bytes().splitlines()
        if (row := json.loads(line))["split"] == "train"
    }
    assert len(tasks) == 500
    groups, metrics, traces = defaultdict(list), {}, {}
    fingerprints = {}
    for path in args.inputs:
        data = json.loads(path.read_text())
        fingerprints[str(path)] = sha(path)
        assert not data["nonfinite_metrics"]
        for relative, expected in data["source_prefixes"].items():
            source = path.parent / relative
            assert source.stat().st_size == expected["bytes"] and sha(source) == expected["sha256"]
        for metric in data["official_metrics"]:
            if "actor/grad_norm" in metric:
                assert metric["step"] not in metrics
                metrics[metric["step"]] = metric
        for row in data["retained_rollouts"]:
            assert row["event_join_method"] != "unresolved" and row["tape_key"] in tasks
            groups[row["step"]].append(row)
        for trace in data["trajectories"]:
            assert trace["trajectory_id"] not in traces
            traces[trace["trajectory_id"]] = trace
    steps = sorted(groups)
    assert steps == sorted(metrics) == list(range(1, max(steps) + 1))
    generated, records, families = 0, [], defaultdict(list)
    for step in steps:
        rows = groups[step]
        assert len(rows) == 4 and len({r["tape_key"] for r in rows}) == 1
        task = tasks[rows[0]["tape_key"]]
        rewards = np.array([r["shaped_reward_from_components"] for r in rows])
        assert abs(rewards.mean() - metrics[step]["critic/score/mean"]) < 1e-6
        raw = [r["raw_quality"] for r in rows]
        assert len(set(raw)) > 1
        tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        advantages, _ = compute_grpo_outcome_advantage(
            tensor,
            torch.ones_like(tensor),
            np.array(["same_actual_task"] * 4),
            norm_adv_by_std_in_grpo=True,
        )
        family = rows[0]["query_type"]
        strict_keys = {
            "backadj_minimal_sets": ["valid_adjustment_set"],
            "best_intervention": ["optimal_action"],
            "mediator_set": ["mediators_exact_match", "order_exact_match"],
        }.get(family)
        group_rows = []
        for row, advantage in zip(rows, advantages[:, 0].tolist(), strict=True):
            trace = traces[row["request_id"]]
            assert trace["tape_key"] == row["tape_key"] and trace["completed"] == row["completed"]
            strict = None
            if strict_keys:
                strict = bool(row["completed"] and all(row["task_metrics"][k] for k in strict_keys))
            group_rows.append(
                {
                    **row,
                    "answer": trace["answer"],
                    "official_scalar_advantage": advantage,
                    "strict_correct": strict,
                    "successful_interventions": trace["valid_experiments"].get("intervene", 0),
                    "protocol_errors": trace["protocol_errors"],
                }
            )
        generated += int(metrics[step]["train/num_gen_batches"])
        if "training/generated_batches_total" in metrics[step]:
            assert generated == metrics[step]["training/generated_batches_total"]
        record = {
            "step": step,
            "task_index": task["index"],
            "tape_key": task["tape_key"],
            "family": family,
            "strong_discordant": task.get("discordant"),
            "generated_batches_total": generated,
            "rows": group_rows,
        }
        records.append(record)
        families[family].extend(group_rows)
    sampler = iter(create_rl_sampler(config.data, list(range(500))))
    consumed = [int(next(sampler)) for _ in range(generated)]
    retained = [r["task_index"] for r in records]
    assert [i for i in consumed if i in set(retained)] == retained
    assert consumed[-1] == retained[-1]
    filtered = [i for i in consumed if i not in set(retained)]
    assert len(filtered) == generated - len(steps)
    by_index = {task["index"]: task for task in tasks.values()}
    summary = {}
    for family, rows in families.items():
        discrete = rows[0]["strict_correct"] is not None
        summary[family] = {
            "retained_groups": len(rows) // 4,
            "trajectories": len(rows),
            "completed": sum(r["completed"] for r in rows),
            "positive_advantage": sum(r["official_scalar_advantage"] > 0 for r in rows),
            "strict_correct": sum(r["strict_correct"] is True for r in rows) if discrete else None,
            "strict_correct_positive": sum(
                r["strict_correct"] is True and r["official_scalar_advantage"] > 0 for r in rows
            )
            if discrete
            else None,
            "strict_incorrect_positive": sum(
                r["strict_correct"] is False and r["official_scalar_advantage"] > 0 for r in rows
            )
            if discrete
            else None,
            "length_penalized": sum(r["official_overlong_reward"] < 0 for r in rows),
        }
    strong = [group for group in records if group["strong_discordant"]]
    successes = [
        {"step": group["step"], "task_index": group["task_index"], **row}
        for group in strong
        for row in group["rows"]
        if row["strict_correct"]
    ]
    assert not torch.cuda.is_initialized()
    report = {
        "script_sha256": sha(Path(__file__)),
        "official_advantage_source_sha256": sha(official),
        "input_sha256": fingerprints,
        "accepted_sha256": sha(args.accepted),
        "configuration_sha256": sha(args.configuration),
        "completed_updates": len(steps),
        "generated_groups": generated,
        "filtered_groups": [
            {"index": i, "family": by_index[i]["query_type"], "tape_key": by_index[i]["tape_key"]}
            for i in filtered
        ],
        "families": summary,
        "strong_groups": len(strong),
        "successful_strong_trajectories": successes,
        "groups": records,
        "official_metrics": [metrics[step] for step in steps],
        "scope": "All completed retained groups in the contiguous snapshots; "
        "verified source prefixes. "
        "CPU calls the pinned official scalar outcome advantage. This is not GPU loss/gradient "
        "replay or evidence that every positive-advantage trajectory becomes more probable "
        "after a shared update.",
    }
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(
        json.dumps(
            {
                "updates": len(steps),
                "generated": generated,
                "filtered": filtered,
                "families": summary,
                "strong_successes": len(successes),
            }
        )
    )


if __name__ == "__main__":
    main()
