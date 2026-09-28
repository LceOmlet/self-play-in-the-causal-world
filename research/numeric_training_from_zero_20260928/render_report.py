"""Offline plots for ONLY the fresh numeric-tools training branch."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-dir',required=True,type=Path)
    parser.add_argument('--original-plotter', type=Path)
    args = parser.parse_args()
    out = args.capture_dir.resolve()
    manifest = json.loads((out/'manifest.json').read_text())
    for name, expected in manifest.items():
        assert Path(name).name == name, name
        assert hashlib.sha256((out/name).read_bytes()).hexdigest() == expected, name
    original_path = args.original_plotter
    if original_path is None:
        original_path = REPO.parent/'audit/rl_server_20260906_1725/plot_training_curves.py'
        if not original_path.exists():
            original_path = REPO/'research/rl_correctness_20260907/official_execution/original_plot_training_curves.py'
    spec = importlib.util.spec_from_file_location('original_cpt_plot',
                 original_path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    plt = original.plt
    original.style()
    capture = json.loads((out/'capture.json').read_text())
    status = json.loads((out/'status.json').read_text())
    rows = [json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
    rows = [row for row in rows if 'loss' in row]
    assert [row['step'] for row in rows] == list(range(1,len(rows)+1))
    assert len(rows) <= status['completed_optimizer_steps']
    if (out/'saved-trainer-state.json').exists():
        state = json.loads((out/'saved-trainer-state.json').read_text())
        assert state['global_step'] == capture['saved_step']
        history = [r for r in state['log_history'] if 'loss' in r]
        assert [r['step'] for r in history] == list(range(1,state['global_step']+1))
        for saved in history:
            observed = rows[saved['step']-1]
            for key,value in saved.items():
                assert observed[key] == value, (saved['step'],key)
    groups = {}
    labels = {'ate':'ATE','individual_counterfactual_probability':'Counterfactual',
              'backadj_minimal_sets':'Backdoor sets','best_intervention':'Best intervention',
              'mediator_set':'Mediator set'}
    title = f"From zero + NumPy/SciPy | {status['completed_optimizer_steps']}/10000 | {capture['checked_utc']}"
    fig, axes = plt.subplots(2,3,figsize=(14,8))
    for ax, (family,label) in zip(list(axes.flat)[:5],labels.items()):
        key = f'task/{family}/reward_raw'
        group = [r for r in rows if key in r]
        groups[family] = group
        x,y = [r['step'] for r in group],[r[key] for r in group]
        ax.plot(x,y,'.-',alpha=.5,label='one group / 4 answers')
        if len(y)>=20:
            ax.plot(x,original.mean_curve(y,20),label='20-group mean')
        if not group:
            ax.text(.5,.5,'No completed group yet',ha='center',transform=ax.transAxes)
        ax.set(title=label,xlabel='Optimizer step',ylabel='Raw terminal reward',ylim=(0,1))
        ax.grid(alpha=.25)
    axes.flat[5].axis('off')
    axes.flat[5].text(0,.95,
        f"Fresh base + new LoRA + empty Adam\nSaved checkpoint: {capture['saved_step']}\n"
        f"Numeric sessions created: {capture['numeric_sessions_created']}\n\n"
        "Only this branch; no old-history splice.\nOnline descriptions, not controlled evaluation.\n"
        "An optimizer step at lr=0 is not a weight change.\n\n"
        "Missing diagnostics are not zero.\nFour answers are one task group.",va='top',linespacing=1.6)
    fig.suptitle(title)
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(out/'rewards.png',dpi=150)
    plt.close(fig)
    errors = [('ate','ate','effect/ate_mse','ATE RMSE'),
              ('individual_counterfactual_probability','cf','effect/cf_endpoint_mse','CF endpoint RMSE'),
              ('best_intervention','decision','effect/decision_normalized_regret','Normalized regret')]
    fig,axes = plt.subplots(2,3,figsize=(14,8))
    summaries = {}
    for column,(family,prefix,key,label) in enumerate(errors):
        eligible = [r for r in groups[family] if f'effect/{prefix}_count' in r]
        values,coverages = [],[]
        for i in range(len(eligible)):
            # Reuse original pooled-MSE -> sqrt / count-weighted regret logic.
            value,coverage,_,_ = original.error_summary(eligible[max(0,i-19):i+1],prefix,key)
            values.append(value);coverages.append(coverage)
        x = [r['step'] for r in eligible]
        axes[0,column].plot(x,values,'.-')
        axes[1,column].plot(x,coverages,'.-',color='#188875')
        for i,caption in [(0,label),(1,'Valid terminal coverage')]:
            axes[i,column].set(title=labels[family]+' / '+caption,xlabel='Optimizer step')
            axes[i,column].grid(alpha=.25)
            if not eligible:
                axes[i,column].text(.5,.5,'Not recorded yet',ha='center',transform=axes[i,column].transAxes)
        axes[1,column].set_ylim(0,1.03)
        if eligible:
            val,cov,n,total = original.error_summary(eligible,prefix,key)
            summaries[family] = {'error':val if math.isfinite(val) else None,
                                'coverage':cov,'valid_terminals':n,'answer_count':total}
        else:
            summaries[family] = None
    fig.suptitle(title)
    fig.text(.5,.015,'Up to 20 same-family groups, pooled by valid terminals; missing values stay missing.',ha='center')
    fig.tight_layout(rect=(0,.04,1,.94))
    fig.savefig(out/'task-errors.png',dpi=150)
    plt.close(fig)
    summary = {'capture':capture,'status':status,'logged_steps':len(rows),
        'family_task_groups':{f:len(g) for f,g in groups.items()},'native_errors':summaries,
        'missing_native_diagnostics':['Backdoor validity/edit, separate mediator node/edge F1 and strict correctness are not recorded by this original owner.'],
        'scope':'Current branch descriptive online curves only; no old history and no controlled tool-effect claim.'}
    spec = importlib.util.spec_from_file_location('plot_usage', HERE/'plot_usage.py')
    usage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(usage)
    summary['usage'] = usage.draw(out, rows, plt, title, capture['checked_utc'])
    previous = []
    for path in out.parent.glob('*/summary.json'):
        if path.parent == out:
            continue
        try:
            candidate = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(candidate, dict) or not isinstance(candidate.get('capture'), dict):
            continue
        if candidate['capture'].get('run') == capture['run'] and candidate['capture'].get('checked_utc', '') < capture['checked_utc'] and 'status' in candidate:
            previous.append(candidate)
    if previous:
        last = max(previous, key=lambda s: s['capture']['checked_utc'])
        summary['since_previous_check'] = {
            'previous_utc': last['capture']['checked_utc'],
            'elapsed_minutes': (usage.timestamp(capture['checked_utc']) - usage.timestamp(last['capture']['checked_utc'])).total_seconds()/60,
            'new_steps': status['completed_optimizer_steps']-last['status']['completed_optimizer_steps']}
    summary['health'] = {
        'source_hashes_match': all(capture['source_hash_matches'].values()),
        'process_source_matches': capture['process'].get('source_matches', False),
        'native_history_contiguous': True,
        'service_active': 'ActiveState=active' in capture['systemd'],
        'telemetry_age_seconds': (usage.timestamp(capture['checked_utc']) - usage.timestamp(summary['usage']['resource_last']['utc'])).total_seconds() if summary['usage']['resource_last'] else None}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'logged_steps':len(rows),'completed':status['completed_optimizer_steps'],
                      'saved':capture['saved_step'],'output':str(out)}))

if __name__=='__main__': main()
