"""Standalone scientific plots; missing native diagnostics remain missing."""
import argparse
from datetime import datetime, timezone, timedelta
from fractions import Fraction
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / '.deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import numpy as np

FAMILIES = ['ate', 'individual_counterfactual_probability', 'backadj_minimal_sets', 'best_intervention', 'mediator_set']
NAMES = ['ATE', 'Counterfactual', 'Backdoor adjustment', 'Decision', 'Mediator']
COLORS = ['#2878b5', '#f08a24', '#3b9460', '#8963b5', '#d45d71']
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 130, 'savefig.dpi': 150, 'axes.grid': True, 'grid.alpha': .18})
CST = timezone(timedelta(hours=8))


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def number(x):
    if isinstance(x, bool):
        return float(x)
    return float(Fraction(str(x))) if x is not None else math.nan


def aggregate(rows, field, transform='mean'):
    cases = [case for row in rows for case in row['cases']]
    values = [number(c['reward'] if field == 'reward' else (c['terminal_score'] or {}).get(field)) for c in cases]
    values = [v for v in values if math.isfinite(v)]
    mean = sum(values) / len(values) if values else math.nan
    if transform == 'sqrt' and values:
        mean = math.sqrt(mean)
    elif transform == 'one-minus' and values:
        mean = 1 - mean
    return {'value': mean if math.isfinite(mean) else None, 'valid_count': len(values), 'total_count': len(cases)}


def native_curve(ax, rows, field, label, color, transform='mean', support=None):
    x = [r['step'] for r in rows]
    raw = [aggregate([r], field, transform) for r in rows]
    y = [r['value'] if r['value'] is not None else math.nan for r in raw]
    if any(math.isfinite(v) for v in y):
        ax.plot(x, y, '.-', color=color, alpha=.7, lw=1, label=label)
    windows = {}
    for window, style in ((5, '-'), (20, '--')):
        points = []
        for i in range(len(rows)):
            if i < window - 1:
                points.append(None)
            else:
                a = aggregate(rows[i-window+1:i+1], field, transform)
                a.update(start_step=rows[i-window+1]['step'], end_step=rows[i]['step'], updates=window)
                points.append(a)
        windows[str(window)] = points
        vals = [p['value'] if p and p['value'] is not None else math.nan for p in points]
        if any(math.isfinite(v) for v in vals):
            ax.plot(x, vals, style, color=color, lw=2, label=f'{label}: {window} full updates')
    if support is not None:
        support.append({'field': field, 'label': label, 'transform': transform, 'steps': x,
                        'families': sorted({r['family'] for r in rows}),
                        'raw': raw, 'trailing_full_updates': windows})


def finish(fig, axes, title, out, name):
    fig.suptitle(title, fontsize=13)
    for ax in np.array(axes).flat:
        if ax.has_data():
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=7, loc='best')
        elif ax.axison:
            ax.text(.5, .5, 'No completed observations', ha='center', va='center', transform=ax.transAxes, color='gray')
        if not ax.get_xlabel():
            ax.set_xlabel('Completed PPO step')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=1))
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(out / name, bbox_inches='tight')
    plt.close(fig)


def main(out, history_path):
    capture = read(out / 'capture.json')
    rows = [json.loads(x) for x in (out / 'series.jsonl').read_text(encoding='utf-8').splitlines()]
    checked = datetime.fromisoformat(capture['checked_utc']).astimezone(CST).strftime('%Y-%m-%d %H:%M:%S CST')
    batch=capture['configuration']['load']['gradient_accumulation_steps']
    title = f'PPO | 32K / optimizer batch {batch} / micro 1 | checked {checked}'
    def family_rows(family):
        result=[]
        for r in rows:
            selected=[c for c in r['cases'] if c.get('family',r['family'])==family]
            if selected:result.append({**r,'cases':selected,'family':family})
        return result
    support = []
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for i, (family, label) in enumerate([(None, 'All families')] + list(zip(FAMILIES, NAMES))):
        ax = axes.flat[i]
        subset = rows if family is None else family_rows(family)
        native_curve(ax, subset, 'reward', 'Batch mean', COLORS[i % 5], support=support)
        for r in subset:
            ax.scatter([r['step']] * len(r['cases']), [c['reward'] for c in r['cases']], color=COLORS[i % 5], alpha=.25, s=14)
        ax.set(title=f'{label}: {len(subset)} updates', ylabel='Owner reward', ylim=(-.04, 1.04))
    finish(fig, axes, title + '\nRaw answers + batch mean; trends use complete 5/20-update windows only', out, 'rewards.png')

    specifications = [
        ('ate', 'ATE errors', [('squared_error', 'MSE', 'mean'), ('squared_error', 'RMSE', 'sqrt'), ('total_variation_error', 'TV', 'mean')]),
        (FAMILIES[1], 'Counterfactual endpoint errors', [('mean_squared_endpoint_error', 'MSE', 'mean'), ('mean_squared_endpoint_error', 'RMSE', 'sqrt'), ('mean_absolute_endpoint_error', 'MAE', 'mean')]),
        ('best_intervention', 'Decision errors', [('regret', 'Regret', 'mean'), ('normalized_regret', 'Normalized regret', 'mean')]),
        ('backadj_minimal_sets', 'Backdoor adjustment errors', [('edit_distance', 'Native edit distance', 'mean')]),
        ('mediator_set', 'Mediator errors', [('mediator_f1', '1 - node F1', 'one-minus'), ('order_f1', '1 - order F1', 'one-minus')])]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (family, label, fields) in zip(axes.flat, specifications):
        subset = family_rows(family)
        for i, (field, name, transform) in enumerate(fields):
            native_curve(ax, subset, field, name, COLORS[i], transform, support)
        ax.set(title=label, ylabel='Conditional native metric')
        ax.set_ylim(bottom=0)
    ax = axes.flat[5]
    for family, label, color in zip(FAMILIES, NAMES, COLORS):
        subset = family_rows(family)
        ax.plot([r['step'] for r in subset], [sum(c['submitted'] for c in r['cases']) / len(r['cases']) for r in subset], '.-', label=label, color=color)
    ax.set(title='Terminal coverage (unsubmitted errors stay missing)', ylabel='Submitted fraction', ylim=(-.04, 1.04))
    finish(fig, axes, title + '\nNative scorer fields; RMSE = sqrt(pooled MSE); no reward inversion or zero imputation', out, 'task-errors.png')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    subset = family_rows('mediator_set')
    for ax, prefix, label in zip(axes[:2], ('mediator', 'order'), ('Mediator nodes', 'Mediator order pairs')):
        for i, field in enumerate(('precision', 'recall', 'f1')):
            native_curve(ax, subset, f'{prefix}_{field}', field, COLORS[i], support=support)
        ax.set(title=label, ylabel='Native score on submitted answers', ylim=(-.04, 1.04))
    for family, field, label, color in [('backadj_minimal_sets', 'valid_adjustment_set', 'Valid backdoor set', COLORS[0]),
        ('mediator_set', 'mediators_exact_match', 'Exact mediator nodes', COLORS[1]),
        ('mediator_set', 'order_exact_match', 'Exact order pairs', COLORS[2])]:
        subset = family_rows(family)
        native_curve(axes[2], subset, field, label, color, support=support)
    axes[2].set(title='Native exact / valid flags', ylim=(-.04, 1.04))
    finish(fig, axes, title + '\nDirect precision / recall / F1 / validity fields; missing families are not estimated', out, 'discovery.png')

    x = [r['step'] for r in rows]
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    for ax, field, label in zip(axes.flat[:2], ('policy_loss_global', 'value_loss_global'), ('Policy loss (sum of globally normalized microbatches)', 'Value loss (sum of globally normalized microbatches)')):
        ax.plot(x, [r[field] for r in rows], '.-', color=COLORS[0]); ax.set(title=label, ylabel='Loss')
    axes[1, 0].plot(x, [r['metrics']['grad_norm'] for r in rows], '.-', label='Finite norm before clipping')
    axes[1, 0].axhline(1, color=COLORS[1], ls='--', label='Clip limit (exceeding it is normal)')
    axes[1, 0].set(title='Gradient norm', ylabel='L2 norm')
    for field, label, color in [('ppo_kl_token_weighted', 'Approx PPO KL', COLORS[0]), ('policy_clipfrac_token_weighted', 'Policy clip fraction', COLORS[1]), ('value_clipfrac_token_weighted', 'Value clip fraction', COLORS[2])]:
        axes[1, 1].plot(x, [r[field] for r in rows], '.-', label=label, color=color)
    axes[1, 1].set(title='Before-update PPO diagnostics', ylabel='Token-weighted metric')
    for field, label, color in [('explicit_numerics_error_outputs', 'Explicit Python error outputs', COLORS[0]), ('protocol_error_outputs', 'Protocol errors', COLORS[1]), ('tool_exceptions', 'Tool exceptions', COLORS[3]), ('state_lost_outputs', 'Sandbox state losses', COLORS[4])]:
        axes[2, 0].plot(x, [sum(c[field] for c in r['cases']) for r in rows], '.-', label=label, color=color)
    axes[2, 0].set(title='Observed tool diagnostics (counts)', ylabel='Per optimizer batch')
    names = list(rows[0]['checks']) if rows else []
    case_names = list(rows[0]['cases'][0]['checks']) if rows else []
    if rows:
        matrix = [[int(r['checks'][key]) for r in rows] for key in names]
        matrix += [[int(all(c['checks'][key] is True for c in r['cases'])) for r in rows] for key in case_names]
        axes[2, 1].imshow(matrix, vmin=0, vmax=1, cmap='RdYlGn', interpolation='nearest', aspect='auto')
        axes[2, 1].set_yticks(range(len(names + case_names)), names + case_names, fontsize=7)
        indexes = np.unique(np.linspace(0, len(rows)-1, min(6, len(rows))).astype(int))
        axes[2, 1].set_xticks(indexes, [x[i] for i in indexes]); axes[2, 1].grid(False)
    axes[2, 1].set(title='All-case checks: green=pass, red=fail', xlabel='Completed PPO step')
    finish(fig, axes, title + '\nSingle PPO epoch: ratios/KL/clipping are measured before the optimizer step', out, 'checks.png')

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for field, label, color in [('rollout_seconds', 'Rollout wall time', COLORS[0]), ('update_seconds', 'Old-policy / GAE / update', COLORS[1])]:
        axes[0, 0].plot(x, [r['metrics'][field] for r in rows], '.-', color=color, label=label)
    axes[0, 0].set(title='Measured phase times (first shapes include tuning)', ylabel='Seconds')
    axes[0, 1].plot(x, [r['generation_tokens_per_second'] for r in rows], '.-', label='Generation incl. prefill', color=COLORS[0])
    axes[0, 1].plot(x, [r['update_policy_tokens_per_second'] for r in rows], '.-', label='Update incl. old-policy/GAE', color=COLORS[1])
    axes[0, 1].set(title='Measured throughput', ylabel='Model tokens / second')
    for field, label, color in [('model_tokens', 'Model output', COLORS[0]), ('feedback_tokens', 'Tool feedback', COLORS[1]), ('total_context_tokens', 'Total context', COLORS[2])]:
        axes[1, 0].plot(x, [sum(c[field] for c in r['cases'])/len(r['cases']) for r in rows], '.-', label=label, color=color)
    axes[1, 0].axhline(32768, color='gray', ls='--', label='Fixed context limit'); axes[1, 0].set(title='Mean per-answer lengths', ylabel='Tokens')
    for field, label, color in [('at_context_limit', 'Exactly at context cap', COLORS[0]), ('submitted', 'Submitted', COLORS[2])]:
        axes[1, 1].plot(x, [sum(c[field] for c in r['cases'])/len(r['cases']) for r in rows], '.-', label=label, color=color)
    axes[1, 1].set(title='Observed terminal / context coverage', ylabel='Fraction', ylim=(-.04, 1.04))
    zeros = []
    for r in rows:
        for c in r['cases']:
            if c['budget_fraction'] > 0:
                axes[2, 0].scatter(r['step'], c['budget_fraction'], color=COLORS[0], s=18, alpha=.6)
            else:
                zeros.append(r['step'])
    axes[2, 0].set_yscale('log')
    if zeros:
        axes[2, 0].scatter(zeros, [-.08]*len(zeros), transform=axes[2, 0].get_xaxis_transform(), marker='x', clip_on=False, color=COLORS[0], label='True zero (outside log axis)')
    axes[2, 0].set(title='Native used / total observation budget', ylabel='Fraction (log; zeros separate)')
    for field, label, color in [('queries_used', 'Successful queries', COLORS[0]), ('numerics_calls', 'Numerics calls', COLORS[1]), ('tool_calls', 'All tool calls', COLORS[2])]:
        axes[2, 1].plot(x, [sum(c[field] for c in r['cases'])/len(r['cases']) for r in rows], '.-', label=label, color=color)
    axes[2, 1].set(title='Mean observed calls per answer', ylabel='Count')
    finish(fig, axes, title, out, 'efficiency-budget.png')

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for ax, field, label in [(axes[0, 0], 'total_context_tokens', 'Final learning context: prompt + model + tool feedback'),
        (axes[1, 0], 'model_tokens', 'Model-generated tokens retained for learning'),
        (axes[1, 1], 'feedback_tokens', 'Tool-feedback tokens retained in context')]:
        values = [[c[field] for c in r['cases']] for r in rows]
        if values:
            ax.fill_between(x, [min(v) for v in values], [max(v) for v in values], color=COLORS[0], alpha=.15, label='Min-max of batch answers')
            ax.plot(x, [sum(v)/len(v) for v in values], '.-', color=COLORS[0], label='Mean of batch answers')
            for step, val in zip(x, values, strict=True):
                ax.scatter([step]*len(val), val, color=COLORS[0], alpha=.35, s=16)
        ax.set(title=label, ylabel='Tokens', ylim=(0, 34000))
    axes[0, 0].axhline(32768, color=COLORS[1], ls='--', label='Fixed 32768-token context cap')
    bottom = np.zeros(len(rows))
    for field, label, color in [('prompt_tokens', 'Initial prompt', COLORS[2]),
        ('model_tokens', 'Model output', COLORS[0]), ('feedback_tokens', 'Tool feedback', COLORS[1])]:
        values = np.array([sum(c[field] for c in r['cases'])/len(r['cases']) for r in rows])
        axes[0, 1].bar(x, values, bottom=bottom, color=color, label=label, width=.65)
        bottom += values
    axes[0, 1].axhline(32768, color='gray', ls='--')
    axes[0, 1].set(title='Mean context composition (exact token counts)', ylabel='Tokens', ylim=(0, 34000))
    finish(fig, axes, title + '\nCompleted trajectories only; total context includes initial prompt and all retained tool feedback', out, 'context-length.png')

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    for ax, values, label, unit in [(axes[0, 0], [r['metrics']['learning_rate'] for r in rows], 'Actual learning rate', 'Learning rate'),
        (axes[0, 1], [r['combined_loss_global'] for r in rows], 'Policy loss + 0.1 * value loss', 'Global loss'),
        (axes[1, 0], [r['weight_sync']['seconds'] if r['weight_sync'] else math.nan for r in rows], 'Updated actor weight synchronization', 'Seconds'),
        (axes[1, 1], [r['metrics']['active_policy_tokens'] for r in rows], 'Actual active policy-token denominator', 'Tokens / optimizer batch')]:
        ax.plot(x, values, '.-', color=COLORS[0]); ax.set(title=label, ylabel=unit)
    finish(fig, axes, title, out, 'optimizer-state.png')

    history = [json.loads(x) for x in history_path.read_text(encoding='utf-8').splitlines()] if history_path.exists() else []
    history = [r for r in history if r.get('result') == 'ok' and r['capture']['checked_utc'] != capture['checked_utc']]
    history.append({'capture': capture})
    captures = [r['capture'] for r in history]
    times = [datetime.fromisoformat(c['checked_utc']).astimezone(CST) for c in captures]
    def ram(c):
        value = c['service'].get('MemoryCurrent')
        return int(value) / 2**30 if value and value.isdigit() else math.nan
    def with_gaps(values):
        xs, ys = [], []
        for i, (stamp, value) in enumerate(zip(times, values, strict=True)):
            if i:
                a, b = captures[i-1], captures[i]
                if (a['boot_id'] != b['boot_id'] or a['service']['InvocationID'] != b['service']['InvocationID']
                        or b['unix'] - a['unix'] > 5400):
                    xs.append(stamp); ys.append(math.nan)
            xs.append(stamp); ys.append(value)
        return xs, ys
    cpu = [math.nan]
    for a, b in zip(captures, captures[1:]):
        counter_a = a['service'].get('CPUUsageNSec', '')
        counter_b = b['service'].get('CPUUsageNSec', '')
        dt = b['unix'] - a['unix']
        same = a['boot_id'] == b['boot_id'] and a['service']['InvocationID'] == b['service']['InvocationID']
        cpu.append((int(counter_b)-int(counter_a))/1e9/dt if same and dt > 0 and counter_a.isdigit() and counter_b.isdigit() else math.nan)
    def gpu_metric(key, factor=1.):
        return [float(c['gpu'][key]) * factor if c.get('gpu', {}).get(key) is not None else math.nan for c in captures]
    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    axes[0, 0].plot(*with_gaps(gpu_metric('utilization_percent_mean')), '.-', label='5-sample check-window mean')
    axes[0, 0].fill_between(times, gpu_metric('utilization_percent_min'), gpu_metric('utilization_percent_max'), alpha=.15, label='Observed min-max')
    axes[0, 0].set(title='Actual NVML GPU utilization (not hourly average)', ylabel='Percent', ylim=(-2, 102))
    axes[0, 1].plot(*with_gaps(gpu_metric('physical_memory_used_bytes', 1/2**30)), '.-', label='Used: latest real NVML sample')
    axes[0, 1].plot(*with_gaps(gpu_metric('physical_memory_total_bytes', 1/2**30)), '.--', label='NVML total', color=COLORS[1])
    axes[0, 1].set(title='Actual physical GPU memory (NVML)', ylabel='GiB', ylim=(0, None))
    axes[1, 0].plot(*with_gaps(gpu_metric('power_watts')), '.-')
    axes[1, 0].set(title='Actual GPU power (latest sample at each check)', ylabel='Watts', ylim=(0, None))
    axes[1, 1].plot(*with_gaps([ram(c) for c in captures]), '.-'); axes[1, 1].set(title='Training cgroup RAM incl. sandboxes', ylabel='GiB')
    axes[2, 0].plot(*with_gaps(cpu), '.-'); axes[2, 0].set(title='CPU interval mean (same boot + invocation only)', ylabel='CPU cores')
    axes[2, 1].plot(*with_gaps([c['status']['global_step'] - max((x['step'] for x in c['checkpoints'] if x['complete_files']), default=0) for c in captures]), '.-')
    axes[2, 1].set(title='Steps since last complete-file checkpoint', ylabel='Steps (not proof of reloadability)')
    for ax, key in [(axes[0, 0], 'utilization_percent_mean'), (axes[0, 1], 'physical_memory_used_bytes'), (axes[1, 0], 'power_watts')]:
        if not any(math.isfinite(v) for v in gpu_metric(key)):
            ax.text(.5, .5, 'No valid NVML samples at these checks', ha='center', va='center', transform=ax.transAxes, color='gray')
    for ax in axes.flat:
        ax.set_xlabel('Check time (Asia/Shanghai)')
        if ax.axison:
            lower = times[0] - timedelta(minutes=30)
            upper = times[-1] + timedelta(minutes=30)
            ax.set_xlim(lower, upper)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M', tz=CST))
        if ax.has_data():
            ax.tick_params(axis='x', rotation=15)
    finish(fig, axes, title + '\nObserved check snapshots; gaps and restarts are not interpolated', out, 'resources.png')
    (out / 'plot-series.json').write_text(json.dumps(support, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    latest_checkpoint = max((x['step'] for x in capture['checkpoints'] if x['complete_files']), default=0)
    summary = {'checked_utc': capture['checked_utc'], 'completed_updates': capture.get('captured_global_step', len(rows)), 'updates_in_current_run':len(rows),
        'optimizer_batch_size':batch,'resume_checkpoint':capture['configuration'].get('resume_checkpoint'),
        'live_status_global_step': capture['status']['global_step'],
        'active_step': capture['status'].get('active_step'), 'stage': capture['status']['stage'],
        'active_group':capture['status'].get('active_group'),
        'rollout_groups':capture['status'].get('rollout_groups'),
        'completed_cases_in_active_batch':capture['status'].get('completed_cases'),
        'service_active': capture['service']['ActiveState'] == 'active', 'latest_checkpoint_files_step': latest_checkpoint,
        'source_matches_launch': capture['source_matches_launch'], 'load_unchanged': capture['learning_load_unchanged'],
        'all_step_checks_passed': all(all(r['checks'].values()) and all(all(c['checks'].values()) for c in r['cases']) for r in rows) if rows else None,
        'submitted': sum(c['submitted'] for r in rows for c in r['cases']), 'answers': sum(len(r['cases']) for r in rows),
        'failure': capture['failure'], 'native_error_policy': 'conditional on accepted terminals; no reward inversion',
        'gpu_available': capture.get('gpu', {}).get('available', False),
        'gpu_utilization_check_window_mean': capture.get('gpu', {}).get('utilization_percent_mean'),
        'gpu_utilization_samples': capture.get('gpu', {}).get('valid_utilization_samples', 0),
        'families': {f: {'updates_with_family':len(family_rows(f)),'groups':sum(c.get('family',r['family'])==f for r in rows for c in r['cases'])//4,
            'submitted': sum(c['submitted'] for r in family_rows(f) for c in r['cases'])} for f in FAMILIES}}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--history', type=Path, required=True)
    args = parser.parse_args()
    main(args.capture_dir, args.history)
