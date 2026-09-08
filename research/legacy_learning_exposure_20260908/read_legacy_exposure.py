"""Summarize existing legacy checkpoint history, without running any model.

The archived checkpoint contains full-precision records for updates 1--850.
This report selects updates 1--800 and compares two descriptive online windows.
One logged update represents four trajectories from one task group, under the
historical generation_batch_size=4 and num_generations=4 configuration.
Groups are not asserted to be statistically independent worlds.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_SHA256 = "97cfbd493909bbccf743f8044202e76f509d27b6ca9327df99e07960b1fde692"
TASKS = (
    "ate",
    "individual_counterfactual_probability",
    "backadj_minimal_sets",
    "best_intervention",
    "mediator_set",
)
ERROR_SPECS = {
    "ate": ("ate", "effect/ate_mse", "rmse", True),
    "individual_counterfactual_probability": (
        "cf",
        "effect/cf_endpoint_mse",
        "endpoint_rmse",
        True,
    ),
    "best_intervention": (
        "decision",
        "effect/decision_normalized_regret",
        "normalized_regret",
        False,
    ),
}
TRAJECTORIES_PER_GROUP = 4


def family(row: dict) -> str:
    names = [
        key.split("/")[1]
        for key in row
        if key.startswith("task/") and key.endswith("/reward_raw")
    ]
    assert len(names) == 1 and names[0] in TASKS, names
    return names[0]


def terminal_error(rows: list[dict], task: str) -> dict | None:
    """Aggregate scorer diagnostics over valid terminal answers, not all rows."""
    if task not in ERROR_SPECS:
        return None
    prefix, metric, name, take_root = ERROR_SPECS[task]
    count_key = f"effect/{prefix}_count"
    coverage_key = f"effect/{prefix}_coverage"
    for row in rows:
        count = row[count_key]
        assert count == int(count) and 0 <= count <= TRAJECTORIES_PER_GROUP
        assert math.isclose(row[coverage_key], count / TRAJECTORIES_PER_GROUP)
        if count:
            assert math.isfinite(row[metric]) and row[metric] >= 0
    valid_count = int(sum(row[count_key] for row in rows))
    weighted_sum = sum(row[count_key] * row[metric] for row in rows if row[count_key])
    total_count = len(rows) * TRAJECTORIES_PER_GROUP
    mean = weighted_sum / valid_count if valid_count else None
    return {
        "name": name,
        "source_metric": metric,
        "value": math.sqrt(mean) if take_root and mean is not None else mean,
        "valid_terminal_count": valid_count,
        "total_trajectories": total_count,
        "valid_terminal_coverage": valid_count / total_count,
        "weighted_source_metric_sum": weighted_sum,
        "aggregation": (
            "sqrt(sum(group_valid_count * group_mean_squared_error) / sum(group_valid_count))"
            if take_root
            else "sum(group_valid_count * group_normalized_regret) / sum(group_valid_count)"
        ),
        "scope": "Error is conditional on scorer-recorded valid terminal answers.",
    }


def summarize(rows: list[dict], task: str | None = None) -> dict:
    zero = [row for row in rows if row["grad_norm"] == 0]
    statuses = Counter(
        "all_zero"
        if row["reward"] == 0
        else "all_one" if row["reward"] == 1 else "constant_partial"
        for row in zero
    )
    clipped = sum(row["completions/clipped_ratio"] * TRAJECTORIES_PER_GROUP for row in rows)
    assert math.isclose(clipped, round(clipped))
    report = {
        "updates": len(rows),
        "task_group_exposures": len(rows),
        "trajectory_exposures": len(rows) * TRAJECTORIES_PER_GROUP,
        "first_update": rows[0]["step"],
        "last_update": rows[-1]["step"],
        "zero_current_gradient_groups": len(zero),
        "zero_current_gradient_fraction": len(zero) / len(rows),
        "zero_reward_std_groups": sum(row["reward_std"] == 0 for row in rows),
        "nonzero_reward_std_but_zero_gradient_groups": sum(
            row["reward_std"] != 0 and row["grad_norm"] == 0 for row in rows
        ),
        "zero_gradient_by_logged_reward": {
            status: statuses[status] for status in ("all_zero", "all_one", "constant_partial")
        },
        "logged_reward_mean": statistics.mean(row["reward"] for row in rows),
        "logged_reward_std_mean": statistics.mean(row["reward_std"] for row in rows),
        "logged_mean_completion_tokens": statistics.mean(
            row["completions/mean_length"] for row in rows
        ),
        "logged_mean_tool_call_frequency": statistics.mean(
            row["tools/call_frequency"] for row in rows
        ),
        "length_clipped_trajectories": round(clipped),
        "length_clipped_fraction": clipped / (len(rows) * TRAJECTORIES_PER_GROUP),
    }
    if task is not None:
        report["terminal_error"] = terminal_error(rows, task)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "trainer-state-checkpoint850.json.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "summary.json")
    args = parser.parse_args()
    compressed = args.source.read_bytes()
    raw = gzip.decompress(compressed)
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == SOURCE_SHA256, (sha, SOURCE_SHA256)
    checkpoint = json.loads(raw)
    history = checkpoint["log_history"]
    assert checkpoint["global_step"] == 850
    assert [row["step"] for row in history] == list(range(1, 851))
    rows = history[:800]
    assert [row["step"] for row in rows] == list(range(1, 801))
    assert Counter(family(row) for row in rows) == {task: 160 for task in TASKS}
    for row in rows:
        assert all(
            math.isfinite(value) for value in row.values() if isinstance(value, (float, int))
        )
        assert (row["grad_norm"] == 0) == (row["reward_std"] == 0)
        assert (row["grad_norm"] == 0) == (row["frac_reward_zero_std"] == 1)
        # Separate FP32 logged reductions differ by at most the spacing at 1.0.
        assert abs(row["reward"] - row[f"task/{family(row)}/reward_raw"]) <= 2**-23
    report = {
        "source": {
            "archive": args.source.name,
            "uncompressed_sha256": sha,
            "uncompressed_bytes": len(raw),
            "archive_sha256": hashlib.sha256(compressed).hexdigest(),
            "archive_bytes": len(compressed),
            "original_local_path": (
                "audit/rl_server_20260906_1725/trainer_state.json"
            ),
            "checkpoint_global_step": checkpoint["global_step"],
            "history_updates": [1, 850],
            "history_records": len(history),
        },
        "reader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selected_updates": [1, 800],
        "trajectories_per_logged_group": TRAJECTORIES_PER_GROUP,
        "grouping_basis": {
            "source": "audit/rl_server_20260906_1725/analysis_sources/train_grpo_resource_smoke.py",
            "generation_batch_size": 4,
            "num_generations": 4,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "num_iterations": 1,
        },
        "overall": summarize(rows),
        "by_task": {
            task: summarize([row for row in rows if family(row) == task], task)
            for task in TASKS
        },
        "max_logged_reward_vs_family_raw_difference": max(
            abs(row["reward"] - row[f"task/{family(row)}/reward_raw"]) for row in rows
        ),
        "online_windows": {},
        "scope": [
            "Legacy trainer and checkpoints are disqualified as evidence of current "
            "official DAPO behavior.",
            "Task-group exposures are not an audited count of distinct or statistically "
            "independent worlds.",
            "Different online tasks and policies occur in the two windows; differences "
            "are descriptive, not causal learning gains.",
            "No model generation, checkpoint evaluation, training update, or new task "
            "generation was performed.",
            "Zero current gradient does not imply no Adam momentum or parameter movement.",
            "Nonzero reward differences do not establish that correct reusable reasoning "
            "was reinforced.",
            "Clipped ratio counts logged generation length clipping; it does not count "
            "every possible termination failure.",
            "Backdoor and mediator valid-terminal coverage and exact correctness were "
            "not logged and are not invented.",
        ],
    }
    for name, start, end in (("early_1_200", 1, 200), ("late_601_800", 601, 800)):
        selected = [row for row in rows if start <= row["step"] <= end]
        report["online_windows"][name] = {
            "updates": [start, end],
            "by_task": {
                task: summarize([row for row in selected if family(row) == task], task)
                for task in TASKS
            },
        }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"source_verified": True, "updates": [1, 800], "overall": report["overall"]}))


if __name__ == "__main__":
    main()
