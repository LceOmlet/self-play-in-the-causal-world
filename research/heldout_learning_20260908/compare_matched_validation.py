"""Compare complete official passes on the same original25 tasks and plot them."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    before, after = [json.loads(path.read_text()) for path in [args.before, args.after]]
    assert before["complete_official_pass"] and after["complete_official_pass"]
    assert before["step"] == 0 and after["step"] == 25
    assert before["dataset_sha256"] == after["dataset_sha256"]
    assert len(before["records"]) == len(after["records"]) == 25
    indexed = {r["tape_key"]: r for r in before["records"]}
    assert len(indexed) == 25 and set(indexed) == {r["tape_key"] for r in after["records"]}
    pairs = []
    for new in after["records"]:
        old = indexed[new["tape_key"]]
        for field in ["index", "query_type", "public_user_message_sha256", "input_sha256"]:
            assert old[field] == new[field]
        old_metrics = (old["trajectory"] or {}).get("task_metrics") or {}
        new_metrics = (new["trajectory"] or {}).get("task_metrics") or {}
        pairs.append(
            {
                "index": old["index"],
                "tape_key": old["tape_key"],
                "family": old["query_type"],
                "before_completed": old["completed"],
                "after_completed": new["completed"],
                "before_raw_quality": old["raw_quality"],
                "after_raw_quality": new["raw_quality"],
                "raw_quality_delta": new["raw_quality"] - old["raw_quality"],
                "before_shaped_reward": old["shaped_reward"],
                "after_shaped_reward": new["shaped_reward"],
                "before_answer": (old["trajectory"] or {}).get("answer"),
                "after_answer": (new["trajectory"] or {}).get("answer"),
                "exact_same_output": old["output_sha256"] == new["output_sha256"],
                "paired_task_metrics": {
                    key: {
                        "before": old_metrics[key],
                        "after": new_metrics[key],
                        "delta": new_metrics[key] - old_metrics[key],
                    }
                    for key in old_metrics.keys() & new_metrics.keys()
                }
                if old["completed"] and new["completed"]
                else {},
            }
        )
    families = {}
    for family in before["families"]:
        group = [p for p in pairs if p["family"] == family]
        assert len(group) == 5
        families[family] = {
            "tasks": 5,
            "before": before["families"][family],
            "after": after["families"][family],
            "paired_error_indices": [p["index"] for p in group if p["paired_task_metrics"]],
        }
    transitions = Counter(
        f"{int(p['before_completed'])}->{int(p['after_completed'])}" for p in pairs
    )
    result = {
        "script_sha256": sha(Path(__file__)),
        "before_report_sha256": sha(args.before),
        "after_report_sha256": sha(args.after),
        "dataset_sha256": before["dataset_sha256"],
        "mean_raw_quality_before": sum(p["before_raw_quality"] for p in pairs) / 25,
        "mean_raw_quality_after": sum(p["after_raw_quality"] for p in pairs) / 25,
        "completion_transitions": dict(transitions),
        "raw_quality_improved": sum(p["raw_quality_delta"] > 0 for p in pairs),
        "raw_quality_worsened": sum(p["raw_quality_delta"] < 0 for p in pairs),
        "raw_quality_unchanged": sum(p["raw_quality_delta"] == 0 for p in pairs),
        "identical_full_outputs": sum(p["exact_same_output"] for p in pairs),
        "families": families,
        "pairs": pairs,
        "scope": "One greedy rollout per model on each of the same25 original tasks. "
        "Task errors are paired only when both answers exist. This descriptive comparison "
        "does not establish population learning gains, training-caused degradation, or "
        "non-learnability. Equal inputs do not imply bitwise deterministic GPU execution.",
    }
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    order = [
        "ate",
        "individual_counterfactual_probability",
        "backadj_minimal_sets",
        "best_intervention",
        "mediator_set",
    ]
    labels = [
        "ATE",
        "Counterfactual\ninterval",
        "Backdoor\nset",
        "Best\nintervention",
        "Mediators\n+ order",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.35, 1.35, 1]})
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e6e9ec", linewidth=0.7)
    for ax, metric, title, ymax in [
        (axes[0], "mean_raw_quality", "Raw task quality", 1.0),
        (axes[1], "completed_answers", "Answers submitted / 5", 5.6),
        (axes[2], "strict_success_count", "Strict successes / 5", 5.6),
    ]:
        selected = list(range(5)) if metric != "strict_success_count" else [2, 3, 4]
        positions = np.arange(len(selected))
        for name, offset, color, label in [
            ("before", -0.18, "#929da5", "Original base"),
            ("after", 0.18, "#176b91", "Update 25"),
        ]:
            values = [families[order[j]][name][metric] for j in selected]
            bars = ax.bar(positions + offset, values, width=0.34, color=color, label=label)
            ax.bar_label(
                bars,
                labels=[f"{v:.3f}" if metric == "mean_raw_quality" else str(v) for v in values],
                padding=3,
                fontsize=8,
            )
        ax.set_xticks(positions, [labels[j] for j in selected], fontsize=8)
        ax.set_ylim(0, ymax)
        ax.set_title(title, fontsize=11)
        if metric != "mean_raw_quality":
            ax.set_yticks(range(6))
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("Matched Qwen3.5-9B validation: original base vs update 25", fontsize=14)
    fig.text(
        0.5,
        0.02,
        "Same 25 worlds; 5 per family; one greedy trajectory each. "
        "Missing answers count as failures, not zero task error.",
        ha="center",
        fontsize=9,
        color="#4b555c",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    args.figures.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "svg", "pdf"]:
        target = args.figures / f"base-vs-update25.{extension}"
        assert not target.exists()
        fig.savefig(target, dpi=180)
    plt.close(fig)
    print(json.dumps({k: v for k, v in result.items() if k not in ["families", "pairs"]}))


if __name__ == "__main__":
    main()
