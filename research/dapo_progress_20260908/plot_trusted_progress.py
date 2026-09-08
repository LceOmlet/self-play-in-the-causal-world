"""Adapt the existing figure style to joined, trusted official DAPO snapshots.

No smoothing or capability-gain claim: tasks differ between updates. All task
errors come from independently recorded environment metrics, never 1-reward.
"""

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-progress", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    style_path = Path(__file__).resolve().parents[1] / (
        "rl_correctness_20260907/official_execution/original_plot_training_curves.py"
    )
    spec = importlib.util.spec_from_file_location("existing_curve_style", style_path)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.style()
    plt, np = style.plt, style.np
    colors = {key: color for key, _, color in style.TASKS}
    labels = {key: label for key, label, _ in style.TASKS}
    metrics, rows = {}, []
    for path in args.inputs:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert not data["nonfinite_metrics"]
        for metric in data["official_metrics"]:
            if "actor/grad_norm" in metric:
                assert metric["step"] not in metrics
                metrics[metric["step"]] = metric
        rows.extend(data["retained_rollouts"])
    native = {
        record["completed_updates"]: record
        for record in [json.loads(p.read_text(encoding="utf-8")) for p in args.native_progress]
    }
    assert all(r["event_join_method"] != "unresolved" for r in rows)
    assert len({r["request_id"] for r in rows}) == len(rows)
    by_step = defaultdict(list)
    for row in rows:
        by_step[row["step"]].append(row)
    steps = sorted(by_step)
    assert steps == list(range(1, max(steps) + 1))
    missing_metrics = set(steps) - set(metrics)
    assert set(metrics) <= set(steps) and missing_metrics <= set(native)
    generated = 0
    for step in steps:
        batch = by_step[step]
        assert len(batch) == 4 and len({r["query_type"] for r in batch}) == 1
        if step in missing_metrics:
            # A saved native update can precede validation and final logging.
            # Preserve the reward/error samples; do not fabricate missing logs.
            assert native[step]["version"] == 1
            assert native[step]["generated_batches"] >= generated + 1
            generated = native[step]["generated_batches"]
            continue
        assert (
            abs(
                np.mean([r["shaped_reward_from_components"] for r in batch])
                - metrics[step]["critic/score/mean"]
            )
            < 1e-6
        )
        generated += int(metrics[step]["train/num_gen_batches"])
        if "training/generated_batches_total" in metrics[step]:
            assert generated == metrics[step]["training/generated_batches_total"]

    def axis(ax, title, ymax=1.04):
        ax.set_title(title, pad=12)
        ax.set_xlim(0.5, max(steps) + 0.5)
        stride = max(5, 5 * ((max(steps) + 49) // 50))
        regular_ticks = [
            s for s in range(stride, max(steps), stride) if max(steps) - s >= stride / 3
        ]
        ax.set_xticks(sorted({1, max(steps), *regular_ticks}))
        ax.set_xlabel("官方累计更新步数")
        ax.set_ylim(-0.03 * ymax, ymax)
        ax.grid(axis="y")
        ax.tick_params(length=0)
        for boundary in [5.5, 15.5]:
            ax.axvline(boundary, color="#9aa9b8", lw=0.8, ls="--", alpha=0.7)

    def error_points(family, key, transform=lambda x: x):
        result = []
        for row in rows:
            task = row["task_metrics"]
            if row["query_type"] == family and task is not None and key in task:
                result.append((row["step"], transform(task[key])))
        return result

    def draw(ax, points, color, label=None, marker="o"):
        groups = defaultdict(list)
        for step, value in points:
            groups[step].append(float(value))
        for step, values in groups.items():
            jitter = np.linspace(-0.08, 0.08, len(values))
            ax.scatter(step + jitter, values, s=25, alpha=0.32, color=color, marker=marker)
        xs = sorted(groups)
        ys = [float(np.mean(groups[x])) for x in xs]
        ax.plot(xs, ys, marker=marker, ms=5, lw=1.1, color=color, label=label)

    def strict_failures(family, keys):
        points = []
        for row in rows:
            if row["query_type"] != family:
                continue
            if not row["completed"]:
                points.append((row["step"], 1.0))
            else:
                task = row["task_metrics"]
                assert task is not None and all(key in task for key in keys)
                points.append((row["step"], 1 - float(all(task[key] for key in keys))))
        return points

    artifacts = []
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.5))
    fig.subplots_adjust(left=0.075, right=0.96, top=0.76, bottom=0.12, hspace=0.43)
    fig.suptitle(
        "官方 DAPO：奖励与更新耗时", x=0.075, y=0.985, ha="left", fontsize=22, fontweight="bold"
    )
    fig.text(
        0.075,
        0.915,
        f"{len(steps)} 次真实更新 · {len(rows)} 条保留轨迹 · {generated} 个生成题组（含过滤组）",
        color=style.MUTED,
    )
    axis(axes[0], "最终答案得分（未扣超长惩罚）：浅色为单条轨迹，实点为同题组均值")
    for family, _, color in style.TASKS:
        draw(
            axes[0],
            [(r["step"], r["raw_quality"]) for r in rows if r["query_type"] == family],
            color,
            labels[family],
        )
    axes[0].plot(
        steps,
        [metrics.get(s, {}).get("critic/score/mean", np.nan) for s in steps],
        "--",
        color="#566675",
        lw=1,
        alpha=0.8,
        label="训练奖励均值（含长度惩罚）",
    )
    minimum_reward = min(0.0, min(m["critic/score/mean"] for m in metrics.values()))
    axes[0].set_ylim(minimum_reward - 0.04, 1.04)
    axes[0].set_ylabel("最终答案得分")
    axes[0].legend(ncol=3, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, 1.39))
    maximum = max(m["timing_s/step"] / 60 for m in metrics.values())
    axis(
        axes[1], "训练步耗时（验证另计）：题目、轨迹长度及补采样次数均会改变工作量", maximum * 1.13
    )
    axes[1].plot(
        steps,
        [metrics.get(s, {}).get("timing_s/step", np.nan) / 60 for s in steps],
        "-o",
        color="#284c6b",
        ms=4,
        lw=1.2,
        label="训练步",
    )
    axes[1].plot(
        steps,
        [metrics.get(s, {}).get("timing_s/gen", np.nan) / 60 for s in steps],
        "-o",
        color="#df9b42",
        ms=3,
        lw=1.1,
        label="生成累计",
    )
    axes[1].set_ylabel("分钟")
    axes[1].legend(frameon=False, loc="upper right")
    validation_seconds = {
        s: metrics[s]["timing_s/testing"]
        for s in metrics
        if metrics[s].get("timing_s/testing", 0) > 0
    }
    if validation_seconds:
        axes[1].text(
            0.46,
            0.95,
            "另计固定验证："
            + "；".join(f"第 {s} 步 {t / 60:.1f} 分钟" for s, t in validation_seconds.items()),
            transform=axes[1].transAxes,
            va="top",
            color=style.MUTED,
            fontsize=9,
        )
    footnote = "虚线：第 6 步起启用编译；第 16 步起使用已验收的进度补丁。"
    if missing_metrics:
        footnote = (
            "第 " + "、".join(map(str, sorted(missing_metrics)))
            + " 步原生更新已保存；取消验证时尚未打印最终日志，相关曲线留空。"
        )
    fig.text(0.075, 0.06, footnote, color=style.MUTED)
    fig.text(
        0.075,
        0.027,
        "各步使用不同训练题。这是采样时表现，不能据此证明更新后的能力提高。",
        color=style.MUTED,
    )
    artifacts += style.save_figure(fig, args.output, "trusted-rewards-and-time")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.80, bottom=0.14, hspace=0.55, wspace=0.25)
    fig.suptitle(
        "环境任务误差与严格失败比例", x=0.06, y=0.97, ha="left", fontsize=23, fontweight="bold"
    )
    fig.text(0.06, 0.915, "误差直接取自已唯一匹配的原始环境事件；数值越低越好。", color=style.MUTED)
    definitions = [
        ("ate", "total_variation_error", "ATE：总变差误差"),
        (
            "individual_counterfactual_probability",
            "mean_absolute_endpoint_error",
            "反事实：区间端点平均绝对误差",
        ),
        ("best_intervention", "normalized_regret", "最佳干预：归一化遗憾"),
        ("backadj_minimal_sets", "edit_distance", "后门调整集：编辑距离"),
    ]
    coverage = {}
    for ax, (family, key, title) in zip(axes.flat, definitions, strict=False):
        points = error_points(family, key)
        ymax = max(1.04, max((v for _, v in points), default=0) * 1.16)
        axis(ax, title, ymax)
        draw(ax, points, colors[family])
        coverage[key] = len(points)
    axis(axes[1, 1], "中介与路径边：1 − F1")
    for key, label, color, marker in [
        ("mediator_f1", "中介集合", "#c04f73", "o"),
        ("order_f1", "路径边", "#74529a", "s"),
    ]:
        points = error_points("mediator_set", key, lambda x: 1 - float(x))
        draw(axes[1, 1], points, color, label, marker)
        coverage[key] = len(points)
    axes[1, 1].legend(frameon=False, fontsize=9)
    axis(axes[1, 2], "同题组严格失败比例")
    for family, key, label in [
        ("backadj_minimal_sets", "valid_adjustment_set", "后门集未答对"),
        ("best_intervention", "optimal_action", "非最优干预"),
    ]:
        draw(axes[1, 2], strict_failures(family, [key]), colors[family], label)
    mediator_failures = strict_failures(
        "mediator_set", ["mediators_exact_match", "order_exact_match"]
    )
    draw(axes[1, 2], mediator_failures, colors["mediator_set"], "中介／路径边非全对")
    axes[1, 2].legend(frameon=False, fontsize=8)
    fig.text(
        0.06,
        0.075,
        "误差只使用已记录指标，不填零；右下严格失败比例计入未完成轨迹。没有把 1 − 奖励当成误差。",
        color=style.MUTED,
    )
    fig.text(
        0.06,
        0.034,
        "每次更新仅 4 条同题轨迹，且题目随步数变化；这些点不构成固定题目的前后能力对照。",
        color=style.MUTED,
    )
    artifacts += style.save_figure(fig, args.output, "trusted-task-errors")
    manifest = {
        "inputs": {str(p): sha(p) for p in args.inputs},
        "script_sha256": sha(Path(__file__)),
        "adapted_style_sha256": sha(style_path),
        "updates": steps,
        "missing_official_metric_steps": sorted(missing_metrics),
        "native_progress_inputs": {str(p): sha(p) for p in args.native_progress},
        "retained_trajectories": len(rows),
        "incomplete_trajectories": sum(not row["completed"] for row in rows),
        "generated_groups": generated,
        "metric_coverage": coverage,
        "validation_seconds_separate_from_training_step": validation_seconds,
        "files": {p.name: sha(p) for p in artifacts},
        "scope": "Training rollout metrics on changing tasks; no causal capability-gain claim.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {k: manifest[k] for k in ["updates", "retained_trajectories", "generated_groups"]}
        )
    )


if __name__ == "__main__":
    main()
