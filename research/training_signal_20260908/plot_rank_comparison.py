"""Compare the frozen fixed programs; this is not a model-learning figure."""

import gzip
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    previous = root.parent / "active_decision_diagnostic_20260907/rescored-answers.json.gz"
    source = root / "observation-rank-records.jsonl"
    evidence = json.loads((root / "observation-rank-summary.json").read_text())
    assert sha(previous) == evidence["inputs_sha256"]["rescored-answers.json.gz"]
    assert sha(source) == evidence["records_sha256"]
    old = json.loads(gzip.decompress(previous.read_bytes()))["episodes"]
    old = {(row["seed_id"], row["replicate"]): row for row in old}
    records = [json.loads(line) for line in source.read_bytes().splitlines()]
    assert len(records) == len(old) == 100
    style_path = (
        root.parent / "rl_correctness_20260907/official_execution/original_plot_training_curves.py"
    )
    spec = importlib.util.spec_from_file_location("existing_figure_style", style_path)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.style()
    plt, np = style.plt, style.np
    results = {}
    for label, subset in [
        ("all", records),
        ("strong", [r for r in records if r["strong"]]),
        ("concordant", [r for r in records if not r["strong"]]),
    ]:
        method_rows = {"observed_best": [], "observed_second": [], "active_adjusted": []}
        for row in subset:
            prior = old[(row["seed_id"], row["replicate"])]
            for name, values in [
                ("observed_best", prior["evaluated"]["full_budget_passive"]),
                ("active_adjusted", prior["evaluated"]["active_adjusted"]),
            ]:
                method_rows[name].append((values["diagnostic"]["optimal_action"], values["reward"]))
            second = row["pure_observation_checkpoints"][-1]["second_rank"]
            method_rows["observed_second"].append(
                (second["metrics"]["optimal_action"], second["quality"])
            )
        result = {"episodes": len(subset), "worlds": len({r["seed_id"] for r in subset})}
        for name, values in method_rows.items():
            result[name] = {
                "correct": sum(v[0] for v in values),
                "accuracy": float(np.mean([v[0] for v in values])),
                "mean_quality": float(np.mean([v[1] for v in values])),
            }
        result["active_vs_second_paired_counts"] = {
            f"{int(a)}->{int(b)}": n
            for (a, b), n in sorted(
                Counter(
                    zip(
                        [v[0] for v in method_rows["active_adjusted"]],
                        [v[0] for v in method_rows["observed_second"]],
                        strict=True,
                    )
                ).items()
            )
        }
        results[label] = result
    report = {
        "script_sha256": sha(Path(__file__)),
        "records_sha256": sha(source),
        "original_program_scores_sha256": sha(previous),
        "results": results,
        "scope": "Previously analyzed 50 worlds/two tapes, not independent model training "
        "or a selection-adjusted significance test. Rank2 reuses "
        "the original pure-observation stream.",
    }
    with (root / "rank-comparison.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    fig.subplots_adjust(left=0.065, right=0.98, top=0.65, bottom=0.22, wspace=0.20)
    fig.suptitle("强反转诊断：需要更强的观察基线", x=0.065, y=0.98, ha="left", fontsize=22)
    fig.text(
        0.065,
        0.895,
        "50 个固定世界，每个 2 条随机流；全部沿用原任务和原观测预算。",
        color=style.MUTED,
    )
    labels = ["全体\n100 条", "强反转\n80 条", "观察／因果一致\n20 条"]
    names = ["all", "strong", "concordant"]
    for ax, metric, title in [
        (axes[0], "accuracy", "选中真正最优动作的比例"),
        (axes[1], "mean_quality", "平均原始终止质量"),
    ]:
        for offset, method, color, label in [
            (-0.26, "observed_best", "#95a2ac", "纯观察：选排名第一"),
            (0.0, "observed_second", "#d18a2d", "纯观察：选排名第二"),
            (0.26, "active_adjusted", "#197d79", "原干预＋调整程序"),
        ]:
            values = [results[name][method][metric] for name in names]
            bars = ax.bar(np.arange(3) + offset, values, width=0.24, color=color, label=label)
            ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=3, fontsize=9)
        ax.set_xticks(np.arange(3), labels)
        ax.set_ylim(0, 1.10)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_title(title, pad=15)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        ncol=3,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.065, 0.84),
    )
    fig.text(
        0.065, 0.085, "这些是固定程序的诊断结果，不是 Qwen3.5-9B 的训练成绩。", color=style.MUTED
    )
    fig.text(
        0.065,
        0.035,
        "排名第二规则没有做因果调整；该结果削弱原有判别性论证，不证明任务不可学。",
        color=style.MUTED,
    )
    artifacts = style.save_figure(fig, root / "figures", "observation-rank-baselines")
    manifest = {
        "script_sha256": sha(Path(__file__)),
        "style_sha256": sha(style_path),
        "report_sha256": sha(root / "rank-comparison.json"),
        "files": {p.name: sha(p) for p in artifacts},
    }
    with (root / "figures/rank-figure-manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
