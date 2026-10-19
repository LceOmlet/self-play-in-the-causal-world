"""Unseal/export the paired experiment only after both 100-case phases finish."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import statistics
import tarfile

SUITE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def summarize(records):
    return {'n':len(records), 'submitted':sum(r['submitted'] for r in records),
        'mean_reward':statistics.mean(r['reward'] for r in records),
        'mean_observations':statistics.mean(r['observations_used'] for r in records),
        'mean_budget_fraction':statistics.mean(r['observations_used']/r['budget'] for r in records),
        'mean_tool_calls':statistics.mean(r['tool_calls'] for r in records),
        'termination_counts':dict(Counter(r['reason'] for r in records))}


def paired_summary(pairs):
    deltas = [a['reward']-q['reward'] for a,q in pairs]
    rng = random.Random(20261019)
    draws = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(10000))
    return {'n':len(pairs), 'agent':summarize([a for a,q in pairs]),
        'qwen':summarize([q for a,q in pairs]), 'mean_agent_minus_qwen':statistics.mean(deltas),
        'paired_bootstrap_95pct': [draws[249], draws[9749]],
        'bootstrap_draws':10000, 'bootstrap_seed':20261019,
        'agent_wins':sum(d>1e-12 for d in deltas),
        'ties':sum(abs(d)<=1e-12 for d in deltas),
        'qwen_wins':sum(d < -1e-12 for d in deltas)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--model-transcripts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    progress = read(args.runtime/'progress.json')
    # This check MUST precede opening any private score/answer files.
    for kind in ('agent','qwen'):
        assert progress[kind]['total'] == progress[kind]['completed'] == 100, 'Both phases must finish first'
    manifest = read(SUITE/'data/manifest.json')
    evaluation = read(SUITE/'data/eval100.json')
    ids = evaluation['task_ids']
    assert len(ids) == len(set(ids)) == 100
    controls = read(args.runtime/'private-control.json')
    secrets = [r['token'].encode() for r in controls['roles']]
    pairs = []
    by_id = {}
    archive_sources = []
    agent_end_times = []
    qwen_start_times = []
    for task_id in ids:
        records = []
        prompts = []
        for kind in ('agent','qwen'):
            folder = args.runtime/kind/task_id
            result = read(folder/'private-result.json')
            assert result['task_id'] == task_id and result['role'] == kind
            assert 0 <= result['reward'] <= 1
            records.append(result)
            prompts.append(read(folder/'prompt.json'))
            events = [json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
            if kind == 'agent':
                agent_end_times.append(events[-1]['utc'])
            else:
                qwen_start_times.append(events[0]['utc'])
            for name in ('private-result.json','prompt.json','events.jsonl'):
                archive_sources.append((folder/name, f'owner/{kind}/{task_id}/{name}'))
        assert prompts[0] == prompts[1], 'Paired public prompt/tools/capsule differ'
        transcript = args.model_transcripts/(hashlib.sha256(task_id.encode()).hexdigest()+'.json')
        model = read(transcript)
        assert model['task_id'] == task_id and model['model'] == 'checkpoint2350'
        assert model['messages'][:len(prompts[0]['messages'])] == prompts[0]['messages']
        assert model['tools'] == prompts[0]['tools']
        archive_sources.append((transcript, 'model/'+transcript.name))
        pairs.append(tuple(records))
        by_id[task_id] = {'agent':records[0], 'qwen':records[1],
            'model_termination_reason':model['termination_reason'], 'model_usage':model['usage'],
            'model_requests':model['model_requests']}
    assert max(agent_end_times) <= min(qwen_start_times), 'Phase order violated'
    for kind in ('agent','qwen'):
        assert {p.name for p in (args.runtime/kind).iterdir() if p.is_dir()} == set(ids)
    assert len(list(args.model_transcripts.glob('*.json'))) == 100
    report = {'dataset_sha256':manifest['tasks_merkle_input_sha256'],
        'selection_sha256':manifest['eval100_sha256'], 'paired_prompts_and_tools_identical':True,
        'all_agent_submissions_precede_qwen_actions':True,
        'agent_last_terminal_utc':max(agent_end_times),'qwen_first_action_utc':min(qwen_start_times),
        'metric':'unchanged terminal-quality-v10 native reward, [0,1]; unfinished=0',
        'context_note':'Five isolated agent conversations of 20 cases; Qwen fresh conversation per case. Fresh numerical sessions for both; reasoning-token budgets are not matched.',
        'overall':paired_summary(pairs), 'by_family':{}}
    for family, members in evaluation['task_ids_by_family'].items():
        report['by_family'][family] = paired_summary([(by_id[t]['agent'],by_id[t]['qwen']) for t in members])
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output/'paired-results.json', by_id)
    write(args.output/'comparison.json', report)
    table = io.StringIO(newline='')
    fields = ['task_id','family','agent_reward','qwen_reward','agent_minus_qwen',
        'agent_submitted','qwen_submitted','agent_observations','qwen_observations','model_termination_reason']
    writer = csv.DictWriter(table, fieldnames=fields)
    writer.writeheader()
    for task_id in ids:
        a,q = by_id[task_id]['agent'],by_id[task_id]['qwen']
        writer.writerow({'task_id':task_id,'family':a['family'],
            'agent_reward':a['reward'],'qwen_reward':q['reward'],
            'agent_minus_qwen':a['reward']-q['reward'],'agent_submitted':a['submitted'],
            'qwen_submitted':q['submitted'],'agent_observations':a['observations_used'],
            'qwen_observations':q['observations_used'],
            'model_termination_reason':by_id[task_id]['model_termination_reason']})
    (args.output/'paired-results.csv').write_text(table.getvalue(), encoding='utf-8')
    lines = ['# Agent versus checkpoint-2350, same 100 tasks', '',
        'Native terminal-quality-v10 mean reward (higher is better, 0–1). Unfinished cases remain zero.', '',
        '| Family | n | Isolated agent | 9B | Difference | Agent / tie / 9B wins |',
        '|---|---:|---:|---:|---:|---:|']
    for family, row in list(report['by_family'].items())+[('Overall',report['overall'])]:
        lines.append(f"| {family} | {row['n']} | {row['agent']['mean_reward']:.4f} | {row['qwen']['mean_reward']:.4f} | {row['mean_agent_minus_qwen']:+.4f} | {row['agent_wins']} / {row['ties']} / {row['qwen_wins']} |")
    lines += ['',report['context_note'],'',
        'All paired prompts, tool schemas and capsule hashes match. All 100 agent cases terminated before Qwen actions began.',
        '', 'Per-case native diagnostics, submissions and resource use are in paired-results.json/csv. Raw owner events and model responses are in transcripts.tar.gz.',
        '', 'Bootstrap intervals in comparison.json are descriptive paired case-resampling intervals (10,000 draws, fixed seed); each family has only 20 cases.']
    (args.output/'COMPARISON.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    with tarfile.open(args.output/'transcripts.tar.gz', 'w:gz') as archive:
        for path, name in archive_sources:
            blob = path.read_bytes()
            assert not any(secret in blob for secret in secrets), 'Runtime credential found in export'
            info = tarfile.TarInfo(name)
            info.size = len(blob);info.mtime=0;info.mode=0o644
            archive.addfile(info, io.BytesIO(blob))
    checksums = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()
        if p.is_file() and p.name != 'result-checksums.json'}
    write(args.output/'result-checksums.json', checksums)
    print(json.dumps({'exported_pairs':100,'dataset_sha256':report['dataset_sha256'],
        'comparison':report['overall']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
