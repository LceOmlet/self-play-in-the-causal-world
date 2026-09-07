"""Plot GRPO rewards and conditional environment errors from a frozen log.

Adapted from ../plot_quality_cost_20260906.py: local matplotlib runtime,
headless rendering, research figure styling, CSV export, and source hashes.
Does not connect to a server or modify a training run.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "plot_dependencies"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

TASKS = [
    ("ate", "平均处理效应 · ATE", "#2878b5"),
    ("individual_counterfactual_probability", "个体反事实概率", "#8b5bb3"),
    ("backadj_minimal_sets", "后门调整集", "#188875"),
    ("best_intervention", "最佳干预选择", "#cf7c24"),
    ("mediator_set", "中介变量集合", "#c04f73"),
]
ERRORS = [
    ("ate", "ate", "effect/ate_mse", "effect/ate_rmse", "ATE · 均方根误差", "#2878b5"),
    ("individual_counterfactual_probability", "cf", "effect/cf_endpoint_mse",
     "effect/cf_endpoint_rmse", "反事实区间端点 · 均方根误差", "#8b5bb3"),
    ("best_intervention", "decision", "effect/decision_normalized_regret",
     "effect/decision_normalized_regret", "最佳干预 · 归一化遗憾", "#cf7c24"),
]
INK = "#263446"
MUTED = "#667385"


def parse_log(path: Path, max_steps: int):
    records = {}
    progress_step = None
    parse_failures = []
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n").splitlines(), 1):
        progress = re.search(r"(\d+)/" + str(max_steps) + r"\s+\[", line)
        if progress:
            progress_step = int(progress.group(1))
        if not line.strip().startswith("{'loss':"):
            continue
        try:
            record = {k: float(v) for k, v in ast.literal_eval(line.strip()).items()}
            step = round(record["epoch"] * max_steps)
            if progress_step is not None and step != progress_step:
                raise ValueError(f"progress {progress_step} disagrees with epoch-derived step {step}")
            if not all(math.isfinite(v) for v in record.values()):
                raise ValueError("nonfinite metric")
            families = [t for t, _, _ in TASKS if f"task/{t}/reward_raw" in record]
            if len(families) != 1:
                raise ValueError(f"expected one task family, got {families}")
            record.update(step=step, task=families[0])
            if step in records and records[step] != record:
                raise ValueError("conflicting duplicate step")
            records[step] = record
        except (ValueError, SyntaxError, KeyError) as exc:
            parse_failures.append({"line": number, "error": str(exc)})
    if parse_failures:
        raise ValueError(parse_failures)
    rows = [records[k] for k in sorted(records)]
    if not rows:
        raise ValueError("No training metrics in log")
    steps = [r["step"] for r in rows]
    if steps != list(range(steps[0], steps[-1] + 1)):
        raise ValueError("Log has missing optimizer steps")
    return rows


def mean_curve(values, window):
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        out[i] = np.mean(arr[i-window+1:i+1])
    return out


def error_summary(rows, prefix, key):
    count_key = f"effect/{prefix}_count"
    coverage_key = f"effect/{prefix}_coverage"
    valid_count = sum(r.get(count_key, 0) for r in rows)
    total_count = 0
    for r in rows:
        count, coverage = r.get(count_key, 0), r.get(coverage_key, 0)
        if coverage:
            total_count += count / coverage
        else:
            # generation_batch_size = num_generations = 4, confirmed in source.
            total_count += 4
    weighted = sum(r[count_key] * r[key] for r in rows if r.get(count_key, 0) > 0)
    value = weighted / valid_count if valid_count else math.nan
    if prefix in ("ate", "cf"):
        value = math.sqrt(value)
    return value, valid_count / total_count, valid_count, total_count


def error_curve(rows, prefix, key, window):
    values = np.full(len(rows), np.nan)
    coverage = np.full(len(rows), np.nan)
    for i in range(window-1, len(rows)):
        values[i], coverage[i], _, _ = error_summary(rows[i-window+1:i+1], prefix, key)
    return values, coverage


def style():
    candidate = Path("C:/Windows/Fonts/msyh.ttc")
    if candidate.exists():
        font_manager.fontManager.addfont(str(candidate))
        family = font_manager.FontProperties(fname=str(candidate)).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": 10, "text.color": INK,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.facecolor": "#ffffff", "figure.facecolor": "#f7f9fc",
        "grid.color": "#e1e7ef", "grid.linewidth": .75,
        "axes.unicode_minus": False, "svg.fonttype": "path",
        "pdf.fonttype": 42, "savefig.facecolor": "#f7f9fc",
    })


def base_axis(ax, first, last, ymax=1.04):
    ax.set_xlim(first-6, last+6)
    ax.set_ylim(-.025, ymax)
    ax.set_xticks([v for v in range(250, last+1, 100) if v >= first-10])
    ax.grid(axis="y")
    ax.tick_params(axis="both", length=0, pad=7)
    ax.set_xlabel("全局训练步数", labelpad=8)
    ax.margins(x=0)


def save_figure(fig, output, name):
    artifacts = []
    for extension in ("png", "svg", "pdf"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=180)
        artifacts.append(path)
    plt.close(fig)
    return artifacts


def window_record(name, rows):
    return {
        "name": name, "start_step": rows[0]["step"], "end_step": rows[-1]["step"],
        "n_steps": len(rows), "mean_reward": float(np.mean([r["reward"] for r in rows])),
        "tasks": {t: {"n_steps": sum(r["task"] == t for r in rows),
                       "mean_reward": float(np.mean([r["reward"] for r in rows if r["task"] == t]))}
                  for t, _, _ in TASKS},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, default=HERE / "train.log")
    ap.add_argument("--output", type=Path, default=HERE / "figures")
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--window", type=int, default=100, help="Global step trailing window")
    ap.add_argument("--task-window", type=int, default=20, help="Same-family observation trailing window")
    args = ap.parse_args()
    if args.window < 1 or args.task_window < 1:
        ap.error("windows must be positive")
    rows = parse_log(args.log, args.max_steps)
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    first, last = rows[0]["step"], rows[-1]["step"]
    by_task = {t: [r for r in rows if r["task"] == t] for t, _, _ in TASKS}
    for task, prefix, key, _, _, _ in ERRORS:
        for row in by_task[task]:
            count = row.get(f"effect/{prefix}_count", 0)
            coverage = row.get(f"effect/{prefix}_coverage", 0)
            if not (0 <= count <= 4 and 0 <= coverage <= 1):
                raise ValueError("Unexpected count/coverage")
            if abs(count - 4 * coverage) > 1e-5:
                raise ValueError("Generation batch size changed")
            if count > 0 and key not in row:
                raise ValueError("Missing conditional error despite valid answers")
    artifacts = []
    summary = {
        "scope": "Online training only; current resumed run; no held-out evaluation.",
        "steps": [first, last], "records": len(rows),
        "smoothing": {"global_steps": args.window, "same_task_observations": args.task_window,
                      "alignment": "trailing, full window only"},
        "windows": [window_record("first100", rows[:100]),
                    window_record("previous100", rows[-200:-100]),
                    window_record("last100", rows[-100:])],
        "error_windows": {},
    }
    fig, axs = plt.subplots(3, 2, figsize=(14.6, 12.8))
    fig.suptitle("强化学习奖励曲线", x=.065, y=.974, ha="left", fontsize=23, fontweight="bold")
    fig.text(.065, .939, f"Qwen3.5-9B  ·  GRPO / terminal-quality-v10  ·  step {first}–{last}  ·  2026-09-06 快照",
             fontsize=11, color=MUTED)
    initial = summary["windows"][0]["mean_reward"]
    recent = summary["windows"][-1]["mean_reward"]
    previous = summary["windows"][1]["mean_reward"]
    fig.text(.065, .904, f"平均奖励   最初 100 步 {initial:.3f}    →    前一百步 {previous:.3f}    →    最近 100 步 {recent:.3f}",
             fontsize=13, color=INK)
    all_step = np.array([r["step"] for r in rows])
    ax = axs.flat[0]
    values = [r["reward"] for r in rows]
    ax.scatter(all_step, values, s=9, color="#60728b", alpha=.12, linewidths=0)
    ax.plot(all_step, mean_curve(values, args.window), color=INK, lw=2.6)
    ax.set_title(f"总体奖励  /  {args.window} 步滑动均值", pad=13, fontsize=13)
    ax.set_ylabel("平均训练奖励 ↑")
    base_axis(ax, first, last)
    for ax, (task, label, color) in zip(list(axs.flat)[1:], TASKS):
        rr = by_task[task]
        xx = [r["step"] for r in rr]
        yy = [r["reward"] for r in rr]
        smooth = mean_curve(yy, args.task_window)
        ax.scatter(xx, yy, color=color, s=15, alpha=.17, linewidths=0)
        ax.plot(xx, smooth, color=color, lw=2.6)
        ax.set_title(label, pad=13, fontsize=13)
        ax.text(.97, .96, f"最近 {args.task_window} 次  {smooth[-1]:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, color=color)
        ax.set_ylabel("该任务平均奖励 ↑")
        base_axis(ax, first, last)
    fig.text(.065, .043, f"浅色点：每步原始批次奖励。实线：总体最近 {args.window} 步；各任务最近 {args.task_window} 次出现（约 100 个全局步）。",
             fontsize=10, color=MUTED)
    fig.text(.065, .019, "仅画完整窗口；横轴均为全局步数。在线任务实例随训练变化，曲线不代表固定验证集表现。", fontsize=10, color=MUTED)
    fig.subplots_adjust(left=.065, right=.975, top=.842, bottom=.105, hspace=.49, wspace=.20)
    artifacts += save_figure(fig, args.output, "reward_curves")

    fig, axes = plt.subplots(2, 3, figsize=(16.3, 9.1))
    fig.suptitle("环境任务误差与有效答案覆盖率", x=.062, y=.968, ha="left", fontsize=22, fontweight="bold")
    fig.text(.062, .922, f"step {first}–{last}  ·  每类任务最近 {args.task_window} 次出现  ·  上排越低越好，下排越高越好",
             fontsize=11, color=MUTED)
    for j, (task, prefix, key, raw_key, title, color) in enumerate(ERRORS):
        rr = by_task[task]
        xx = [r["step"] for r in rr]
        raw = [r.get(raw_key, math.nan) for r in rr]
        vals, coverage = error_curve(rr, prefix, key, args.task_window)
        summary["error_windows"][task] = {}
        for label, chunk in [("first20", rr[:20]), ("previous20", rr[-40:-20]), ("last20", rr[-20:])]:
            val, cov, count, total = error_summary(chunk, prefix, key)
            summary["error_windows"][task][label] = {"error": val, "coverage": cov,
                "valid_answers": count, "total_answers": total, "start_step": chunk[0]["step"],
                "end_step": chunk[-1]["step"]}
        ax = axes[0, j]
        ax.scatter(xx, raw, s=15, color=color, alpha=.2, linewidths=0)
        ax.plot(xx, vals, color=color, lw=2.6)
        base_axis(ax, first, last, 1.05)
        ax.set_title(title, fontsize=12, pad=14)
        ax.set_ylabel("归一化遗憾 ↓" if prefix == "decision" else "RMSE ↓")
        ax.text(.97, .965, f"最新窗口  {vals[-1]:.3f}", transform=ax.transAxes,
                ha="right", va="top", color=color, fontsize=11)
        max_raw = max(v for v in raw if math.isfinite(v))
        if max_raw > 1.05:
            ax.set_ylim(-.025, max_raw * 1.08)
        ax = axes[1, j]
        rc = [r.get(f"effect/{prefix}_coverage", 0) for r in rr]
        ax.scatter(xx, rc, color=color, s=17, alpha=.2, linewidths=0)
        ax.plot(xx, coverage, color=color, lw=2.6)
        base_axis(ax, first, last, 1.07)
        ax.axhline(1, lw=1, color="#99a6b6", ls="--", alpha=.6)
        ax.set_title("有效终止答案覆盖率", fontsize=12, pad=14)
        ax.set_ylabel("有效答案数 / 全部生成数 ↑")
        ax.text(.97, .09, f"最新窗口  {coverage[-1]:.1%}", transform=ax.transAxes,
                ha="right", va="bottom", color=color, fontsize=11)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    fig.text(.062, .098, "误差只覆盖有效终止答案，未完成或无效答案不计入误差；下排同时保留它们造成的覆盖率损失。", fontsize=10, color=MUTED)
    fig.text(.062, .064, "RMSE = √[Σ(有效答案数 × 批次 MSE) / Σ有效答案数]；遗憾按有效答案数加权。浅色点为未平滑批次值。", fontsize=10, color=MUTED)
    fig.text(.062, .030, "后门调整集和中介集合未记录独立误差指标，仅在奖励图中展示；没有将 1 − 奖励当成实际任务误差。", fontsize=10, color=MUTED)
    fig.subplots_adjust(left=.062, right=.973, top=.84, bottom=.21, hspace=.51, wspace=.29)
    artifacts += save_figure(fig, args.output, "environment_errors")

    csvpath = args.output / "training_metrics.csv"
    keys = ["step", "task"] + sorted(set().union(*(r.keys() for r in rows)) - {"step", "task"})
    with csvpath.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    artifacts.append(csvpath)
    summarypath = args.output / "curve_summary.json"
    summarypath.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts.append(summarypath)
    manifest = {
        "adapted_from": str(HERE.parent / "plot_quality_cost_20260906.py"),
        "source_log": str(args.log.resolve()), "source_sha256": hashlib.sha256(args.log.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metric_source": "src/cpt_world/trl_environment.py:_log_terminal_effect_metrics",
        "source_precision": "Rounded values printed in train.log; no interpolation or imputed errors.",
        "runtime": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
        "artifacts": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts},
    }
    (args.output / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"steps": [first, last], "records": len(rows), "error_windows": summary["error_windows"],
                      "artifacts": [str(p) for p in artifacts]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

