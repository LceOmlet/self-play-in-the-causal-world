"""Render the measured GPU comparison using the existing Matplotlib runtime."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font', type=Path, default=Path('C:/Windows/Fonts/msyh.ttc'))
    args = parser.parse_args()
    report = json.loads(args.comparison.read_text(encoding='utf-8'))
    assert report['identical_inputs_weights_and_sampling']
    assert all(r['eager_cached_prompt_tokens_mean'] == r['compiled_cached_prompt_tokens_mean']
               and r['eager_cached_prompt_tokens_mean'] is not None for r in report['timing'])
    fontManager.addfont(str(args.font))
    font = FontProperties(fname=str(args.font)).get_name()
    plt.rcParams.update({'font.family': font, 'axes.unicode_minus': False,
                         'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'path'})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8), sharey=True)
    fig.subplots_adjust(top=.76, bottom=.26, left=.07, right=.98, wspace=.12)
    fig.suptitle('同一 Qwen3.5-9B：编译执行的实测收益', fontsize=18, y=.97)
    fig.text(.5, .90, '同一非零 LoRA · 相同输入 · 每条固定输出 256 token · 每种条件重复 2 次',
             ha='center', fontsize=10.5, color='#404b59')
    cases = ['p2048_b1', 'p2048_b4', 'p24576_b1', 'p24576_b4']
    labels = ['2K / 单路', '2K / 四路', '24K / 单路', '24K / 四路']
    bars = None
    for ax, phase, title in zip(axes, ['cold_prefix', 'reused_prefix'],
                               ['清空前缀缓存', '复用相同前缀'], strict=True):
        rows = [next(r for r in report['timing'] if r['case'] == case and r['phase'] == phase)
                for case in cases]
        left = ax.bar([i-.18 for i in range(4)], [r['eager_seconds_mean'] for r in rows],
                      width=.34, color='#426d9b', label='eager')
        right = ax.bar([i+.18 for i in range(4)], [r['compiled_seconds_mean'] for r in rows],
                       width=.34, color='#d77c34', label='编译 / CUDA Graph')
        bars = (left, right)
        ax.bar_label(left, fmt='%.1f', padding=3, fontsize=9)
        ax.bar_label(right, fmt='%.1f', padding=3, fontsize=9)
        for i, row in enumerate(rows):
            ax.text(i, 57, f"{row['eager_to_compiled_wall_ratio']:.2f}×", ha='center',
                    fontsize=11, fontweight='bold', color='#233d58')
        ax.set_title(title, fontsize=12, pad=12)
        ax.set_xticks(range(4), labels)
        ax.set_ylim(0, 61)
        ax.grid(axis='y', color='#dce2e8', linewidth=.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('每次生成调用耗时 / 秒')
    fig.legend(handles=[bars[0], bars[1]], labels=['eager', '编译 / CUDA Graph'],
               loc='upper center', bbox_to_anchor=(.5, .87), ncol=2, frameon=False)
    startup = report['startup_seconds']
    fig.text(.07, .14, f"图上方为实测耗时比；两边缓存命中 token 数相同。首次启动："
             f"eager {startup['eager']:.1f} 秒，编译 {startup['compiled']:.1f} 秒。", fontsize=9.5)
    fig.text(.07, .08, '耗时包含预填充与主机开销；此图验证推理收益，'
             '实际训练的同步、Token-TIS 与更新另行验收。',
             fontsize=9.5, color='#4b5563')
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [args.output/f'compilation-speed.{suffix}' for suffix in ('png', 'svg')]
    for path in paths:
        fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    manifest = {'comparison_sha256': hashlib.sha256(args.comparison.read_bytes()).hexdigest(),
                'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (args.output/'compilation-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
