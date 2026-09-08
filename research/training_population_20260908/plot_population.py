"""Plot all scored training samples alongside DAPO-retained samples."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generation_axis(ax, total, boundaries):
    target_stride = max(1, total / 6)
    magnitude = 10 ** math.floor(math.log10(target_stride))
    stride = next(
        value * magnitude for value in (1, 2, 5, 10) if value * magnitude >= target_stride
    )
    ticks = {
        1,
        total,
        *(tick for tick in range(stride, total, stride) if total - tick >= stride / 3),
    }
    ax.set_xlim(0.5, total + 0.5)
    ax.set_xticks(sorted(ticks))
    for coordinate, _ in boundaries:
        if 0.5 <= coordinate <= total + 0.5:
            ax.axvline(coordinate, color="#9aa9b8", lw=0.8, ls="--", alpha=0.7)


def phase_note(boundaries, total):
    labels = [
        f"G={coordinate:g}：{label}"
        for coordinate, label in boundaries
        if 0.5 <= coordinate <= total + 0.5
    ]
    return "\n竖线（生成组序号）：" + "；".join(labels) + "。" if labels else ""


def plot_references(data, style, output, boundaries=()):
    """Use identical completed trajectories on both sides of each comparison."""
    np, plt = style.np, style.plt
    families = [
        ("ate", "平均处理效应 · ATE", "TV 误差"),
        ("individual_counterfactual_probability", "反事实概率区间", "端点平均绝对误差"),
        ("best_intervention", "最佳干预选择", "归一化遗憾"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.4))
    fig.subplots_adjust(left=0.07, right=0.975, top=0.76, bottom=0.21, hspace=0.48, wspace=0.3)
    fig.suptitle(
        "模型与同题观察参考：得分和真实误差",
        x=0.07,
        y=0.975,
        ha="left",
        fontsize=22,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.917,
        f"前 {data['completed_updates']} 次更新 · 动态过滤前全部训练采样 · "
        "仅比较有最终答案的相同轨迹",
        color=style.MUTED,
    )
    handles = []
    for col, (family, title, error_label) in enumerate(families):
        groups = []
        for g in data["groups"]:
            if g["query_type"] == family:
                rows = [
                    t["reference_comparison"]
                    for t in g["trajectories"]
                    if t["reference_comparison"] is not None
                ]
                if rows:
                    groups.append((g["generation"], rows))
        count = sum(len(rows) for _, rows in groups)
        total = 4 * data["families"][family]["generated_groups"]
        axes[0, col].set_title(f"{title}\n有效比较 {count}/{total} 条", pad=10)
        denominator = np.cumsum([len(rows) for _, rows in groups])
        variants = [
            ("model", "模型", "#2878b5", "-"),
            ("observational", "总体观察参考", "#7d8590", "--"),
        ]
        if family == "best_intervention":
            variants.append(("observational_second", "观察排名第二（干预题）", "#c77b22", ":"))
        for row, metric in enumerate(("score", "error")):
            ax = axes[row, col]
            for key, label, color, linestyle in variants:
                values = (
                    np.cumsum([sum(r[f"{key}_{metric}"] for r in rows) for _, rows in groups])
                    / denominator
                )
                (line,) = ax.plot(
                    [g for g, _ in groups],
                    values,
                    color=color,
                    ls=linestyle,
                    lw=1.8,
                    marker="o",
                    ms=3,
                    label=label,
                )
                if row == 0 and col == 2:
                    handles.append(line)
            ax.set_ylabel("最终答案得分 ↑" if row == 0 else error_label + " ↓")
            if row == 0:
                ax.set_ylim(-0.03, 1.04)
            else:
                ax.set_ylim(bottom=0)
            generation_axis(ax, data["generated_groups"], boundaries)
            ax.set_xlabel("实际生成题组序号（累计均值）")
            ax.grid(axis="y")
            ax.tick_params(length=0)
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.062, 0.88), ncol=3, frameon=False
    )
    fig.text(
        0.07,
        0.105,
        "参考使用总体观察概率，不是基座模型成绩，也不是有限采样程序的实测成绩。"
        + phase_note(boundaries, data["generated_groups"]),
        color=style.MUTED,
    )
    fig.text(
        0.07,
        0.066,
        "缺少答案的轨迹没有数值误差，未填 0；其失败计入全采样得分与完成率。各题使用同一评分函数。",
        color=style.MUTED,
    )
    fig.text(
        0.07,
        0.027,
        "强反转使观察第一名参照偏弱，故同时列第二名。训练题不断变化，这些曲线不证明泛化或学习增益。",
        color=style.MUTED,
    )
    return style.save_figure(fig, output, "same-task-reference-score-and-error")


def plot_structure(data, style, output, boundaries=()):
    np, plt = style.np, style.plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.64, bottom=0.23, wspace=0.3)
    fig.suptitle("结构误差与严格正确", x=0.07, y=0.975, ha="left", fontsize=23, fontweight="bold")
    fig.text(
        0.07,
        0.87,
        f"前 {data['completed_updates']} 次更新 · 全部训练采样（含过滤组）· "
        "编辑距离与 F1 只统计已提交答案",
        color=style.MUTED,
    )
    for col, (family, fields, title) in enumerate(
        [
            ("backadj_minimal_sets", [("edit_distance", "编辑距离", "#188875")], "后门调整集"),
            (
                "mediator_set",
                [("mediator_f1", "中介集合 F1", "#c04f73"), ("order_f1", "路径边 F1", "#8b5bb3")],
                "中介与路径边",
            ),
        ]
    ):
        rows = [
            (g["generation"], [t["terminal_score"] for t in g["trajectories"] if t["completed"]])
            for g in data["groups"]
            if g["query_type"] == family
        ]
        rows = [(g, r) for g, r in rows if r]
        count = np.cumsum([len(r) for _, r in rows])
        for field, label, color in fields:
            values = np.cumsum([sum(t[field] for t in r) for _, r in rows]) / count
            axes[col].plot(
                [g for g, _ in rows], values, lw=1.7, marker="o", ms=3, label=label, color=color
            )
        axes[col].set_title(title)
        axes[col].set_ylabel("累计编辑距离 ↓" if col == 0 else "累计 F1 ↑")
        axes[col].set_ylim(bottom=0)
        if col == 1:
            axes[col].set_ylim(0, 1.04)
            axes[col].legend(frameon=False, fontsize=10)
        generation_axis(axes[col], data["generated_groups"], boundaries)
        axes[col].set_xlabel("实际生成题组序号")
    ax = axes[2]
    families = ["backadj_minimal_sets", "best_intervention", "mediator_set"]
    counts = [data["families"][f]["all_strict_correct"] for f in families]
    totals = [4 * data["families"][f]["generated_groups"] for f in families]
    bars = ax.bar(
        ["后门", "最佳干预", "中介与边"],
        [n / d for n, d in zip(counts, totals, strict=True)],
        color=["#188875", "#cf7c24", "#c04f73"],
        width=0.5,
    )
    ax.bar_label(bars, labels=[f"{n}/{d}" for n, d in zip(counts, totals, strict=True)], padding=5)
    ax.set_ylim(0, 1.04)
    ax.set_title("严格正确率（包括未提交答案）")
    ax.set_ylabel("严格正确比例 ↑")
    for ax in axes:
        ax.grid(axis="y")
        ax.tick_params(length=0)
    fig.text(
        0.07,
        0.09,
        "部分分不等于答对：后门要求合法调整集；干预要求零遗憾；中介要求集合与路径边同时精确匹配。"
        + phase_note(boundaries, data["generated_groups"]),
        color=style.MUTED,
    )
    return style.save_figure(fig, output, "structure-errors-and-strict-correctness")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase-boundary",
        action="append",
        default=[],
        metavar="GENERATION:LABEL",
        help="Add a generation-axis boundary, e.g. 86.5:持续新题; may be repeated.",
    )
    args = parser.parse_args()
    boundaries = []
    for value in args.phase_boundary:
        coordinate, separator, label = value.partition(":")
        try:
            coordinate = float(coordinate)
        except ValueError:
            parser.error("--phase-boundary requires GENERATION:LABEL")
        if not separator or not label or not math.isfinite(coordinate) or coordinate <= 0:
            parser.error(
                "--phase-boundary requires a positive finite GENERATION and nonempty LABEL"
            )
        boundaries.append((coordinate, label))
    assert len({coordinate for coordinate, _ in boundaries}) == len(boundaries)
    boundaries.sort()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    groups = data["groups"]
    style_path = Path(__file__).resolve().parents[1] / (
        "rl_correctness_20260907/official_execution/original_plot_training_curves.py"
    )
    spec = importlib.util.spec_from_file_location("existing_curve_style", style_path)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.style()
    plt, np = style.plt, style.np
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.subplots_adjust(left=0.065, right=0.97, top=0.77, bottom=0.19, hspace=0.48, wspace=0.25)
    fig.suptitle(
        "全部训练采样的累计得分", x=0.065, y=0.975, ha="left", fontsize=23, fontweight="bold"
    )
    fig.text(
        0.065,
        0.915,
        f"前 {data['completed_updates']} 次更新 · {len(groups)} 个已完成采样题组 · "
        f"全部 {data['all_training_trajectories']} 条 / "
        f"保留 {data['retained_training_trajectories']} 条",
        color=style.MUTED,
    )
    fig.text(
        0.065,
        0.855,
        "实线：全部训练采样；灰色虚线：筛选后保留样本。纵轴为累计均值，不是正确率。",
        color=style.MUTED,
    )
    for ax, (family, title, color) in zip(axes.flat, style.TASKS, strict=False):
        for keep, line_color, linestyle in [(False, color, "-"), (True, "#6f7d89", "--")]:
            rows = [g for g in groups if g["query_type"] == family and (not keep or g["retained"])]
            x = [g["generation"] for g in rows]
            means = np.cumsum([sum(g["qualities"]) for g in rows]) / (
                4 * np.arange(1, len(rows) + 1)
            )
            ax.plot(x, means, color=line_color, ls=linestyle, lw=1.5, marker="o", ms=3)
        ax.set_title(title, pad=12)
        ax.set_ylim(-0.03, 1.04)
        ax.set_ylabel("最终答案得分")
    ax = axes.flat[-1]
    for keep, color, linestyle in [(False, "#284c6b", "-"), (True, "#6f7d89", "--")]:
        rows = [g for g in groups if not keep or g["retained"]]
        values = np.cumsum([g["completed"] for g in rows]) / (4 * np.arange(1, len(rows) + 1))
        ax.plot([g["generation"] for g in rows], values, color=color, ls=linestyle, lw=1.5)
    ax.set_title("提交最终答案的比例（累计）", pad=12)
    ax.set_ylim(-0.03, 1.04)
    ax.set_ylabel("完成率")
    for ax in axes.flat:
        generation_axis(ax, data["generated_groups"], boundaries)
        ax.set_xlabel("实际生成题组序号（含过滤组）")
        ax.grid(axis="y")
        ax.tick_params(length=0)
    fig.text(
        0.065,
        0.075,
        "同题四个回答得分相同会被过滤，包括全对、全错和相同部分分；模型未提交答案时得分为 0。"
        + phase_note(boundaries, data["generated_groups"]),
        color=style.MUTED,
    )
    fig.text(
        0.065,
        0.032,
        "已排除验证、被取消的未完成题组及重复恢复片段。训练题不断变化，累计曲线不证明同题能力提高。",
        color=style.MUTED,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    files = style.save_figure(fig, args.output, "all-training-samples")
    files += plot_references(data, style, args.output, boundaries)
    files += plot_structure(data, style, args.output, boundaries)
    manifest = {
        "input": str(args.input),
        "input_sha256": sha(args.input),
        "script_sha256": sha(Path(__file__)),
        "style_sha256": sha(style_path),
        "phase_boundary_axis": "generation",
        "phase_boundaries": [
            {"coordinate": coordinate, "label": label} for coordinate, label in boundaries
        ],
        "files": {p.name: sha(p) for p in files},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
