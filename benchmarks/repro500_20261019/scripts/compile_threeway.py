"""Unseal all three 100-case phases together, then export the three-way table."""
import argparse
import csv
import hashlib
import io
from pathlib import Path
import sys

import compile_results as paired


def all_three_gate(agent_runtime, base_runtime):
    original = paired.read(agent_runtime/'progress.json')
    baseline = paired.read(base_runtime/'progress.json')
    for phase in (original['agent'],original['qwen'],baseline['qwen']):
        assert phase['completed'] == phase['total'] == 100, 'All THREE phases must finish before unsealing'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent-runtime', type=Path, required=True)
    parser.add_argument('--trained-transcripts', type=Path, required=True)
    parser.add_argument('--base-runtime', type=Path, required=True)
    parser.add_argument('--base-transcripts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    all_three_gate(args.agent_runtime, args.base_runtime)
    # Gate above precedes all score/transcript reads, including paired exports.
    original_argv = sys.argv
    try:
        for model, runtime, transcripts in (
            ('checkpoint2350',args.agent_runtime,args.trained_transcripts),
            ('base',args.base_runtime,args.base_transcripts)):
            sys.argv = ['compile_results.py','--runtime',str(args.agent_runtime),
                '--qwen-runtime',str(runtime),'--model-label',model,
                '--model-transcripts',str(transcripts),'--output',str(args.output/('pair-'+model))]
            paired.main()
    finally:
        sys.argv = original_argv
    trained = paired.read(args.output/'pair-checkpoint2350/paired-results.json')
    base = paired.read(args.output/'pair-base/paired-results.json')
    assert set(trained) == set(base) and len(trained)==100
    rows = {}
    for task_id in trained:
        assert trained[task_id]['agent'] == base[task_id]['agent']
        rows[task_id] = {'agent':trained[task_id]['agent'],
            'base':base[task_id]['qwen'],'checkpoint2350':trained[task_id]['qwen'],
            'base_model_usage':base[task_id]['model_usage'],
            'checkpoint2350_model_usage':trained[task_id]['model_usage'],
            'base_model_termination':base[task_id]['model_termination_reason'],
            'checkpoint2350_model_termination':trained[task_id]['model_termination_reason']}
    evaluation = paired.read(paired.SUITE/'data/eval100.json')
    groups = {**evaluation['task_ids_by_family'], 'Overall':evaluation['task_ids']}
    report = {'dataset_sha256':paired.read(paired.SUITE/'data/manifest.json')['tasks_merkle_input_sha256'],
        'paired_cases':100,'phase_counts':{'agent':100,'base':100,'checkpoint2350':100},
        'all_model_prompts_tools_capsules_match_agent':True,'groups':{},
        'metric':'Native terminal-quality-v10 reward [0,1], higher is better; unfinished=0',
        'context_note':'Five isolated agent conversations of 20 tasks; each 9B run starts fresh per task. Numerical state is fresh per task for all three. Internal reasoning-token budgets are not matched.'}
    lines = ['# Three-way comparison on the same 100 frozen cases','',report['metric'],'',
        '| Family | n | Isolated agents | Original 9B | Checkpoint-2350 | Trained − base |',
        '|---|---:|---:|---:|---:|---:|']
    for group, ids in groups.items():
        metrics = {model:paired.summarize([rows[t][model] for t in ids])
            for model in ('agent','base','checkpoint2350')}
        delta = paired.paired_summary([(rows[t]['checkpoint2350'],rows[t]['base']) for t in ids])
        report['groups'][group] = {'n':len(ids),'models':metrics,
            'trained_minus_base':delta['mean_agent_minus_qwen'],
            'trained_minus_base_paired_bootstrap_95pct':delta['paired_bootstrap_95pct'],
            'trained_wins':delta['agent_wins'],'ties':delta['ties'],'base_wins':delta['qwen_wins']}
        lines.append(f"| {group} | {len(ids)} | {metrics['agent']['mean_reward']:.4f} | {metrics['base']['mean_reward']:.4f} | {metrics['checkpoint2350']['mean_reward']:.4f} | {delta['mean_agent_minus_qwen']:+.4f} |")
    lines += ['',report['context_note'],'',
        'The five-family subset was fixed before solving. Both 9B runs began after the original 100 agent cases terminated. No terminal case was resampled.',
        '', 'Each pair directory contains exact diagnostics, per-case CSV/JSON and raw owner/model transcripts. Confidence intervals are descriptive paired bootstrap intervals with 10,000 draws and seed 20261019; each family contains only 20 cases.']
    paired.write(args.output/'threeway-results.json', rows)
    paired.write(args.output/'comparison.json', report)
    (args.output/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    table = io.StringIO(newline='')
    writer = csv.writer(table,lineterminator='\n')
    writer.writerow(['task_id','family','agent_reward','base_reward','checkpoint2350_reward',
        'trained_minus_base','agent_minus_base','agent_minus_trained'])
    for t in evaluation['task_ids']:
        a,b,c = [rows[t][model]['reward'] for model in ('agent','base','checkpoint2350')]
        writer.writerow([t,rows[t]['agent']['family'],a,b,c,c-b,a-b,a-c])
    (args.output/'threeway-results.csv').write_text(table.getvalue(),encoding='utf-8')
    proof = paired.read(args.base_runtime/'prior-agent-proof.json')
    paired.write(args.output/'base-agent-completion-proof.json',proof)
    paired.write(args.output/'all-result-checksums.json',
        {str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest()
         for p in args.output.rglob('*') if p.is_file() and p.name!='all-result-checksums.json'})
    print(report['groups']['Overall'])


if __name__ == '__main__':
    main()
