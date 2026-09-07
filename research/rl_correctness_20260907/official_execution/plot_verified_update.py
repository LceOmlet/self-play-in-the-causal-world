"""Adapt the existing training/error plot style to one verified actual update.

All four trajectories are generated before the update; this is not a learning
curve or an assessment of post-update capability. No discarded run is plotted.
"""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE/'original_plot_training_curves.py'
spec = importlib.util.spec_from_file_location('existing_training_plots', OLD)
existing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(existing)
plt, np = existing.plt, existing.np
existing.style()
source = HERE/'run-02-mask-v1-audit-only/execution-acceptance.json'
acceptance = json.loads(source.read_text(encoding='utf-8'))
assert acceptance['passed'] and acceptance['official_updates'] == 1
rows = []
for index, trajectory in enumerate(acceptance['trajectories'], 1):
    env = trajectory['environment']
    assert env['query_type'] == 'ate' and env['completed']
    score = env['terminal_score']
    penalty = -max(trajectory['response_tokens']-(30720-4096), 0)/4096
    rows.append({'trajectory': index, 'request_id': trajectory['request_id'],
                 'quality': env['raw_reward'], 'shaped_reward': env['raw_reward']+penalty,
                 'overlong_shaping': penalty, 'ate_tv_error': score['total_variation_error'],
                 'ate_rmse': math.sqrt(score['squared_error']),
                 'action_tokens': trajectory['action_tokens'], 'tool_tokens': trajectory['tool_tokens'],
                 'observations': env['observations_used'], 'experiments': env['queries_used']})
out = HERE/'figures'
out.mkdir(exist_ok=True)
with (out/'verified-update-trajectories.csv').open('w', encoding='utf-8-sig', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

x = np.arange(1, len(rows)+1)
fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.6))
fig.suptitle('官方 DAPO · 一次更新的执行验收', x=.07, y=.973, ha='left', fontsize=22, fontweight='bold')
fig.text(.07, .928, '原始 Qwen3.5-9B 基座  ·  同一道 ATE 任务的四条独立轨迹  ·  动作掩码已逐段核对',
         fontsize=11, color=existing.MUTED)
blue, orange, teal, grey = '#2878b5', '#cf7c24', '#188875', '#bdc7d4'
ax = axes[0, 0]
quality = [r['quality'] for r in rows]
ax.bar(x, quality, color=blue, width=.58, label='环境原始质量')
ax.scatter(x, [r['shaped_reward'] for r in rows], marker='D', facecolors='none',
           edgecolors=orange, linewidth=1.8, s=65, label='实际训练奖励', zorder=4)
for xi, q in zip(x, quality, strict=True):
    ax.text(xi, q+.025, f'{q:.3f}', ha='center', fontsize=10)
ax.set(ylim=(0, 1.05), title='环境质量与官方长度惩罚后的奖励', ylabel='奖励 ↑', xticks=x, xlabel='轨迹编号')
ax.legend(frameon=False, fontsize=9, loc='upper right')

ax = axes[0, 1]
ax.bar(x-.17, [r['ate_tv_error'] for r in rows], width=.32, color=orange, label='总变差误差')
ax.bar(x+.17, [r['ate_rmse'] for r in rows], width=.32, color=teal, label='分量均方根误差')
ax.set(title='对隐藏因果真值的实际误差', ylabel='误差 ↓', xticks=x, xlabel='轨迹编号')
ax.legend(frameon=False, fontsize=9)

ax = axes[1, 0]
actions = np.array([r['action_tokens'] for r in rows])
tools = np.array([r['tool_tokens'] for r in rows])
ax.bar(x, actions, width=.58, color=blue, label='动作 token：计入损失')
ax.bar(x, tools, bottom=actions, width=.58, color=grey, label='工具 token：仅作上下文')
ax.set(title='损失掩码保留模型动作、排除工具文本', ylabel='token 数', xticks=x, xlabel='轨迹编号')
ax.legend(frameon=False, fontsize=9)

ax = axes[1, 1]
observations = [r['observations'] for r in rows]
error = [r['ate_tv_error'] for r in rows]
ax.scatter(observations, error, color=orange, s=65)
for r in rows:
    ax.annotate(str(r['trajectory']), (r['observations'], r['ate_tv_error']), xytext=(7, 6), textcoords='offset points')
ax.set_xscale('symlog', linthresh=1)
ax.set(title='实际标量观测开销与误差', xlabel='计费标量观测数', ylabel='总变差误差 ↓')
fig.text(.07, .045, '一个训练步不足以判断收敛。上述轨迹均由更新前策略生成，不能用它们推断训练后的能力变化。',
         fontsize=10, color=existing.MUTED)
fig.subplots_adjust(left=.07, right=.97, top=.86, bottom=.13, hspace=.39, wspace=.25)
for extension in ('png', 'svg', 'pdf'):
    fig.savefig(out/f'verified-update.{extension}', dpi=170)
plt.close(fig)
manifest = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'adapted_from': str(OLD), 'training_steps': 1, 'trajectories': len(rows),
            'scope': 'Execution audit. Pre-update rollouts only. No convergence or post-update capability claim.',
            'artifacts': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}}
(out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(manifest, ensure_ascii=False, indent=2))
