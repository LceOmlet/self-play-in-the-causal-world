"""Offline plots from native metrics, decoded feedback and timestamped telemetry."""
from datetime import datetime, timezone
import json
import math


def timestamp(text):
    return datetime.fromisoformat(text.replace('Z', '+00:00')).astimezone(timezone.utc)


def mean(values):
    return sum(values)/len(values) if values else None


def draw(out, rows, plt, title, checked):
    times = [timestamp(row['utc']) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for key, caption in [('mean_length', 'mean'), ('min_length', 'min'), ('max_length', 'max')]:
        axes[0, 0].plot(times, [r.get('completions/'+key, math.nan) for r in rows], '.-', label=caption)
    axes[0, 0].set(title='Native generated length (tool output excluded)', ylabel='Tokens / answer')
    axes[0, 0].legend()
    for key, label in [('completions/clipped_ratio', 'clipped fraction'), ('tools/failure_frequency', 'native tool failure frequency')]:
        axes[0, 1].plot(times, [r.get(key, math.nan) for r in rows], '.-', label=label)
    axes[0, 1].set(title='Native clipping / tool failures')
    axes[0, 1].legend()
    axes[1, 0].plot(times, [r.get('tools/call_frequency', math.nan) for r in rows], '.-')
    axes[1, 0].set(title='Native tool calls', ylabel='Calls / answer')
    axes[1, 1].plot(times, [r.get('step_time', math.nan) for r in rows], '.-')
    axes[1, 1].set(title='Native step time (cold compilation included)', ylabel='Seconds')
    finish(fig, axes, plt, out/'rollout-length.png', title)

    budget = json.loads((out/'budgets.json').read_text())
    records = budget['records']
    by_step = {row['step']: [r for r in records if r['step'] == row['step']] for row in rows}
    points = []
    for row in rows:
        answers = by_step[row['step']]
        complete = [r for r in answers if r['status'] == 'complete_logged']
        points.append({'step': row['step'], 'utc': row['utc'],
                       'coverage': len(complete)/4,
                       'fraction': sum(r['observed_consumed'] for r in complete)/sum(r['initial'] for r in complete) if complete else None,
                       'consumed': mean([r['observed_consumed'] for r in complete]),
                       'remaining': mean([r['observed_remaining'] for r in complete]),
                       'queries': mean([r['accepted_queries'] for r in complete]),
                       'rejections': mean([r['rejected_act_calls'] for r in complete]),
                       'numerics': mean([r['numerics_calls'] for r in complete])})
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, fields, caption in [
        (axes[0, 0], [('fraction', 'used / allocated')], 'Budget usage (allocation-weighted)'),
        (axes[0, 1], [('consumed', 'consumed'), ('remaining', 'remaining')], 'Scalar observation budget / answer'),
        (axes[1, 0], [('queries', 'accepted queries'), ('rejections', 'rejected act'), ('numerics', 'numerics')], 'Calls / complete logged answer'),
        (axes[1, 1], [('coverage', 'complete log coverage')], 'Budget accounting coverage / four answers')]:
        for key, label in fields:
            ax.plot(times, [p[key] if p[key] is not None else math.nan for p in points], '.-', label=label)
        ax.set_title(caption)
        ax.legend()
    axes[0, 0].set_ylim(0, 1.05)
    axes[1, 1].set_ylim(0, 1.05)
    finish(fig, axes, plt, out/'budget-usage.png', title,
           'Decoded-log reconstruction, not an authenticated owner ledger. Ambiguous / partial logs excluded; coverage shown.')

    resources = [json.loads(line) for line in (out/'resources.jsonl').read_text().splitlines() if line.strip()] if (out/'resources.jsonl').exists() else []
    rt = [timestamp(r['utc']) for r in resources]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    gpu = [r.get('gpus', [{}])[0] if r.get('gpus') else {} for r in resources]
    axes[0, 0].plot(rt, [g.get('util_percent') for g in gpu], '.-', label='whole GPU 0')
    axes[0, 0].set(title='GPU utilization snapshots', ylabel='%')
    axes[0, 0].set_ylim(0, 105)
    axes[0, 1].plot(rt, [g.get('used_mib') for g in gpu], '.-')
    axes[0, 1].set(title='GPU allocated memory snapshots', ylabel='MiB')
    axes[0, 2].plot(rt, [r.get('cpu_cores') for r in resources], '.-')
    axes[0, 2].set(title='Training + sandbox CPU (interval average)', ylabel='CPU cores used')
    for key, label in [('ram_bytes', 'training cgroup'), ('host_ram_used_bytes', 'whole host')]:
        axes[1, 0].plot(rt, [r.get(key, math.nan)/2**30 for r in resources], '.-', label=label)
    axes[1, 0].set(title='RAM snapshots (cgroup includes sandboxes)', ylabel='GiB')
    axes[1, 0].legend()
    axes[1, 1].plot(rt, [g.get('power_w') for g in gpu], '.-')
    axes[1, 1].set(title='GPU power snapshots', ylabel='W')
    axes[1, 2].plot(rt, [r.get('disk_free_bytes', math.nan)/2**30 for r in resources], '.-')
    axes[1, 2].set(title='Run filesystem free space', ylabel='GiB')
    axes[1, 2].ticklabel_format(axis='y', style='plain', useOffset=False)
    if not resources:
        for ax in axes.flat:
            ax.text(.5, .5, 'No telemetry recorded yet', ha='center', transform=ax.transAxes)
    finish(fig, axes, plt, out/'resources.png', title,
           '60-second read-only sampling; no backfill. GPU/RAM/power are snapshots, not whole-hour means. CPU includes answer sandboxes.')
    last_hour = [r for r in rows if 0 <= (timestamp(checked)-timestamp(r['utc'])).total_seconds() <= 3600]
    statuses = {name: sum(r['status'] == name for r in records) for name in ['complete_logged', 'partial_logged', 'unavailable']}
    return {'last_60_minutes_completed_steps': len(last_hour), 'rollout_latest': {k: v for k, v in rows[-1].items() if k.startswith(('completions/', 'tools/'))} if rows else {},
            'budget_status_counts': statuses, 'budget_expected_answers': len(rows)*4,
            'budget_missing_answers': len(rows)*4-len(records), 'budget_error': budget.get('error'),
            'budget_by_step': points, 'resource_samples': len(resources),
            'resource_first_utc': resources[0]['utc'] if resources else None,
            'resource_last': resources[-1] if resources else None}


def finish(fig, axes, plt, path, title, footnote=None):
    import matplotlib.dates as dates

    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.set_xlabel('Completion / sample time (UTC)')
        ax.xaxis.set_major_formatter(dates.DateFormatter('%m-%d\n%H:%M', tz=timezone.utc))
        if ax.has_data():
            ax.xaxis.set_major_locator(dates.AutoDateLocator(minticks=3, maxticks=6))
    fig.suptitle(title)
    if footnote:
        fig.text(.5, .015, footnote, ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(path, dpi=150)
    plt.close(fig)
