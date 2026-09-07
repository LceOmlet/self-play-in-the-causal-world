"""Adapt the existing research plot style to actual timing and task errors."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = Path(__file__).resolve().parents[1] / 'rl_correctness_20260907/official_execution/original_plot_training_curves.py'
    spec = importlib.util.spec_from_file_location('existing_training_plots', original)
    existing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(existing)
    plt, np = existing.plt, existing.np
    existing.style()
    progress = json.loads((args.snapshot / 'progress.json').read_text())
    balance = json.loads((args.snapshot / 'generation-balance.json').read_text())
    steps = balance['steps']
    rows = [r for r in progress['retained_rollouts'] if r['step'] in {s['step'] for s in steps}]
    assert [s['step'] for s in steps] == [1, 2] and all(s['trajectories'] == 4 for s in steps)
    assert len(rows) == 8 and all(r['raw_quality'] == r['shaped_reward_from_components'] for r in rows)
    assert all(r['query_type'] == 'backadj_minimal_sets' for r in rows)
    args.output.mkdir(parents=True, exist_ok=True)
    table = [{'step': r['step'], 'trajectory_id': r['request_id'], 'quality': r['raw_quality'],
              'shaped_reward': r['shaped_reward_from_components'], **r['task_metrics']} for r in rows]
    with (args.output / 'trajectories.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    blue, orange, teal, grey = '#2878b5', '#cf7c24', '#188875', '#bdc7d4'
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.6))
    fig.suptitle('Qwen3.5-9B · 两次官方更新的耗时与任务误差', x=.07, y=.973,
                 ha='left', fontsize=21, fontweight='bold')
    fig.text(.07, .928, '两道不同的后门任务，每题四条轨迹；均来自各步更新前策略，不能据此判断学习收益。',
             fontsize=11, color=existing.MUTED)
    x = np.array([s['step'] for s in steps])
    ax = axes[0, 0]
    generation = np.array([s['generation_seconds'] / 60 for s in steps])
    totals = np.array([s['step_seconds'] / 60 for s in steps])
    ax.bar(x, generation, width=.5, color=blue, label='生成阶段')
    ax.bar(x, totals - generation, bottom=generation, width=.5, color=grey, label='更新、同步及其余阶段')
    for xi, total in zip(x, totals, strict=True):
        ax.text(xi, total + .8, f'{total:.2f}', ha='center')
    ax.set(title='第二步仍有 96.2% 时间用于生成', ylabel='分钟', xlabel='实际更新步', xticks=x,
           ylim=(0, max(totals) * 1.2))
    ax.legend(frameon=False, fontsize=9, loc='upper left')

    trajectory_x = np.arange(1, len(rows) + 1)
    colors = [teal if r['task_metrics']['valid_adjustment_set'] else orange for r in rows]
    labels = [f'{r["step"]}-{i % 4 + 1}' for i, r in enumerate(rows)]
    ax = axes[0, 1]
    ax.bar(trajectory_x, [r['raw_quality'] for r in rows], width=.6, color=colors)
    for xi, row in zip(trajectory_x, rows, strict=True):
        ax.text(xi, row['raw_quality'] + .025, f'{row["raw_quality"]:.3f}', ha='center', fontsize=8)
    ax.set(title='奖励较高仍可能是不合法答案', ylabel='环境质量 / 实际训练奖励',
           xlabel='更新步－轨迹', xticks=trajectory_x, xticklabels=labels, ylim=(0, 1.13))
    ax.plot([], [], color=teal, linewidth=8, label='合法调整集')
    ax.plot([], [], color=orange, linewidth=8, label='不合法调整集')
    ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=.95,
              fontsize=9, loc='lower left')

    ax = axes[1, 0]
    errors = [r['task_metrics']['edit_distance'] for r in rows]
    ax.bar(trajectory_x, errors, width=.6, color=colors)
    for xi, error in zip(trajectory_x, errors, strict=True):
        ax.text(xi, error + .08, str(error), ha='center')
    ax.set(title='第二步四个答案均未满足后门调整条件', ylabel='到合法调整集的编辑距离',
           xlabel='更新步－轨迹', xticks=trajectory_x, xticklabels=labels, yticks=[0, 1, 2, 3], ylim=(0, 3.6))

    ax = axes[1, 1]
    pending = [s['mean_pending_generation_requests'] for s in steps]
    ax.bar(x, pending, width=.5, color=blue)
    ax.axhline(4, color=grey, linestyle='--', linewidth=1.5)
    for xi, value in zip(x, pending, strict=True):
        ax.text(xi, value + .1, f'{value:.2f}', ha='center')
    ax.text(.98, .94, '允许最多四路', transform=ax.transAxes, ha='right', fontsize=9, color=existing.MUTED)
    ax.set(title='四条轨迹并不代表全程四路并行', ylabel='平均待完成生成请求数（含等待）',
           xlabel='实际更新步', xticks=x, ylim=(0, 4.5))
    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.grid(axis='y')
    second = next(s for s in steps if s['step'] == 2)
    fig.text(.07, .049, f'第二步至少 {second["single_pending_generation_seconds_lower_bound"] / 60:.2f} 分钟只有一条待完成生成请求；这是计时下界，不是 GPU 利用率。',
             fontsize=10, color=existing.MUTED)
    fig.text(.07, .023, '长度惩罚均为零。未改变任务、奖励、采样或官方 DAPO；编译与 LoRA 合并的 GPU 提速验收尚未完成。',
             fontsize=10, color=existing.MUTED)
    fig.subplots_adjust(left=.07, right=.97, top=.855, bottom=.135, hspace=.40, wspace=.25)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(args.output / f'generation-and-errors.{extension}', dpi=170)
    plt.close(fig)
    manifest = {
        'source_progress_sha256': hashlib.sha256((args.snapshot / 'progress.json').read_bytes()).hexdigest(),
        'source_balance_sha256': hashlib.sha256((args.snapshot / 'generation-balance.json').read_bytes()).hexdigest(),
        'adapted_from': str(original), 'updates': [s['step'] for s in steps], 'trajectories': len(rows),
        'scope': 'Actual update timing and task errors; different worlds, not a learning-effect comparison.',
        'artifacts': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())
                      if p.is_file() and p.name != 'manifest.json'},
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
