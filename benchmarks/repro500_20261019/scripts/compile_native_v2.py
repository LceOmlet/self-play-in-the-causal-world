"""Export the completed per-case native run, then compare archived evidence locally.

Never runs models or tools. Remote export gates all score reads on 100 terminal
cases; local comparison verifies the downloaded archive against both 9B archives.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import compile_results as paired
from compile_threeway import native_diagnostics
from verify_model_export import members

SUITE = Path(__file__).resolve().parents[1]
read, write = paired.read, paired.write


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def export(root, output):
    runtime = root/'isolated-agent-v2-runtime'
    phase = read(runtime/'progress.json')['agent']
    assert phase['total'] == phase['completed'] == 100
    ids = read(SUITE/'data/eval100.json')['task_ids']
    assert {p.name for p in (runtime/'agent').iterdir() if p.is_dir()} == set(ids)
    assert all((runtime/'agent'/t/'private-result.json').is_file() for t in ids)
    controls = read(runtime/'private-control.json')
    roles = controls['roles']
    assert len(roles) == 100 and len({r['token'] for r in roles}) == 100
    assert {r['task_ids'][0] for r in roles} == set(ids)
    assert all(r['kind'] == 'agent' and len(r['task_ids']) == 1 for r in roles)
    assert controls['protocol_label'] == 'native-subagent-per-case-v2'
    assert controls['native_generation_token_budget_matched'] is False
    assert controls['native_generation_usage_available'] is False
    secrets = [r['token'].encode() for r in roles]
    for name in ('recovered-base-runtime', 'timeout-checkpoint-runtime'):
        secrets.extend(r['token'].encode() for r in read(root/name/'private-control.json')['roles'])
    rows, sources = {}, []
    for task in ids:
        folder = runtime/'agent'/task
        prompt = read(folder/'prompt.json')
        result = read(folder/'private-result.json')
        assert result['task_id'] == task and result['role'] == 'agent'
        assert 0 <= result['reward'] <= 1 and result['submitted']
        assert result['reason'] == 'terminal_answer'
        for model in ('recovered-base-runtime', 'timeout-checkpoint-runtime'):
            other = root/model/'qwen'/task
            assert prompt == read(other/'prompt.json')
            reference = read(other/'private-result.json')
            assert result['budget'] == reference['budget']
            assert result['initial_public_qwen_tokens'] == reference['initial_public_qwen_tokens']
        events = [json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
        assert events[-1]['event'] == 'response' and events[-1]['response']['ended'] is True
        requests = [e['request'] for e in events if e['event'] == 'request']
        assert all(r['task_id'] == task for r in requests)
        rows[task] = result
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                sources.append((path, 'owner/agent/'+str(path.relative_to(runtime/'agent')).replace('\\','/')))
    source_names = ('server.py', 'serve_isolated_agents.py', 'capsule.py',
                    'native_agent_v2_protocol.json', 'model_profile.json',
                    'data/manifest.json', 'data/eval100.json', 'scripts/compile_native_v2.py')
    sources += [(SUITE/n, 'source/'+n) for n in source_names]
    sources.append((runtime/'progress.json', 'owner/progress.json'))
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with tarfile.open(output/'native-owner-transcripts.tar.gz', 'w:gz') as archive:
        for path, name in sources:
            blob = path.read_bytes()
            assert not any(secret in blob for secret in secrets), 'Credential in export'
            hashes[name] = sha(blob)
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(blob), 0, 0o644
            archive.addfile(info, io.BytesIO(blob))
    write(output/'native-results.json', rows)
    write(output/'owner-audit.json', {
        'cases':100, 'submitted':100, 'all_terminal_answer':True,
        'single_task_roles':100, 'unique_role_tokens':100,
        'all_case_prompts_tools_capsules_budgets_initial_public_tokens_match_both_9b':True,
        'all_terminal_responses_verified':True, 'all_requests_remain_in_assigned_case':True,
        'export_contains_role_credentials':False, 'archive_file_count':len(hashes),
        'source_sha256':{n:hashes['source/'+n] for n in source_names},
        'protocol':read(SUITE/'native_agent_v2_protocol.json')})
    write(output/'archive-members-sha256.json', hashes)
    write(output/'export-sha256.json', {p.name:sha(p.read_bytes()) for p in output.iterdir()
        if p.is_file() and p.name != 'export-sha256.json'})
    print(json.dumps({'native_cases':100,'submitted':100,'archive_files':len(hashes)}))


def compare(output, models):
    for name, digest in read(output/'export-sha256.json').items():
        assert sha((output/name).read_bytes()) == digest
    native = members(output/'native-owner-transcripts.tar.gz')
    hashes = read(output/'archive-members-sha256.json')
    assert set(native) == set(hashes)
    assert all(sha(blob) == hashes[name] for name, blob in native.items())
    evaluation = read(SUITE/'data/eval100.json')
    ids = evaluation['task_ids']
    records = read(output/'native-results.json')
    old = read(models/'final-results/threeway-results.json')
    assert set(records) == set(old) == set(ids)
    references = {m:members(models/f'final-results/pair-{m}/transcripts.tar.gz')
                  for m in ('base','checkpoint2350')}
    rows = {}
    for t in ids:
        prefix = f'owner/agent/{t}/'
        a = json.loads(native[prefix+'private-result.json'])
        assert a == records[t]
        prompt = json.loads(native[prefix+'prompt.json'])
        for m, archive in references.items():
            qprefix = f'owner/qwen/{t}/'
            assert prompt == json.loads(archive[qprefix+'prompt.json'])
            q = json.loads(archive[qprefix+'private-result.json'])
            assert q == old[t][m]
            assert a['budget'] == q['budget']
            assert a['initial_public_qwen_tokens'] == q['initial_public_qwen_tokens']
        rows[t] = {**old[t], 'agent':a}
    manifest = read(SUITE/'data/manifest.json')
    assert sha(native['source/data/manifest.json']) == sha((SUITE/'data/manifest.json').read_bytes())
    for name in ('server.py','serve_isolated_agents.py','capsule.py','model_profile.json',
                 'native_agent_v2_protocol.json','scripts/compile_native_v2.py'):
        assert sha(native['source/'+name]) == sha((SUITE/name).read_bytes()), name
    groups = {**evaluation['task_ids_by_family'], 'Overall':ids}
    report = {'dataset_sha256':manifest['tasks_merkle_input_sha256'], 'paired_cases':100,
        'protocol':read(SUITE/'native_agent_v2_protocol.json'),
        'metric':'Native terminal-quality-v10 reward [0,1], not accuracy',
        'legacy_agent_results_included':False, 'groups':{}}
    lines = ['# Per-case native subagents versus original and trained 9B', '',
        'All three groups completed the same fixed 100 cases. Native agents submitted 100 final answers; original 9B submitted 20 and checkpoint-2350 submitted 24.', '',
        '**Reward is not accuracy. Native hidden reasoning/generation compute and full system context are not matched to 9B.**',
        'Each native case uses a fresh fork_turns=none conversation with a single-task capability. Task prompts, tool schemas, capsules, observation budgets, initial public token counts and owner source match both 9B runs. Native solver instructions additionally constrain transport and prohibit external computation. The old five-conversation agent run is excluded.', '',
        '| Family | n | Native per-case agent | Original 9B | Checkpoint-2350 |',
        '|---|---:|---:|---:|---:|']
    for group, tasks in groups.items():
        metrics = {m:paired.summarize([rows[t][m] for t in tasks])
                   for m in ('agent','base','checkpoint2350')}
        comparisons = {}
        for left,right in (('agent','base'),('agent','checkpoint2350'),('checkpoint2350','base')):
            d = paired.paired_summary([(rows[t][left],rows[t][right]) for t in tasks])
            comparisons[left+'_minus_'+right] = {k:v for k,v in d.items() if k not in ('agent','qwen')}
        report['groups'][group] = {'n':len(tasks),'models':metrics,'comparisons':comparisons,
            'native_diagnostics':{m:native_diagnostics([rows[t][m] for t in tasks])
                                  for m in metrics}}
        lines.append(f"| {group} | {len(tasks)} | {metrics['agent']['mean_reward']:.4f} | {metrics['base']['mean_reward']:.4f} | {metrics['checkpoint2350']['mean_reward']:.4f} |")
    lines += ['', '| Overall comparison | Mean difference | Paired bootstrap 95% interval | Left / tie / right wins |',
              '|---|---:|---|---:|']
    for label,d in report['groups']['Overall']['comparisons'].items():
        lo,hi = d['paired_bootstrap_95pct']
        lines.append(f"| {label} | {d['mean_agent_minus_qwen']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {d['agent_wins']} / {d['ties']} / {d['qwen_wins']} |")
    lines += ['', '| Overall resource use | Native agent | Original 9B | Checkpoint-2350 |',
        '|---|---:|---:|---:|']
    for field in ('submitted','mean_tool_calls','mean_observations','mean_budget_fraction'):
        values = [str(round(report['groups']['Overall']['models'][m][field],4))
                  for m in ('agent','base','checkpoint2350')]
        lines.append('| '+ ' | '.join([field]+values)+' |')
    lines += ['', 'Tool calls are owner act/numerics calls, excluding prompt and transport retries. Observation budgets vary by family; mean budget fraction is reported separately. Hidden model reasoning/generation tokens are not comparable.', '',
        'All native cases ended with terminal_answer. Legal 9B failures remain zero reward; infrastructure failures were recovered without resampling and were not treated as performance zeros. The earlier model audit and archives are retained under ../model-evaluation/.', '',
        'Native diagnostics, exact answers/errors, per-case rewards and paired comparisons are in JSON/CSV. Raw owner requests, responses, request caches, terminal records and source identities are archived. Bootstrap intervals use 10,000 paired case resamples with seed 20261019; they do not capture solver/run-to-run variability. Each family has only 20 cases.', '',
        'The two 9B groups finished before the replacement native run finished. Scores were sealed from native solvers; the parent first unsealed this native run only after all 100 owner terminal files existed. Overall Codex instructions and hidden native generation/context limits cannot be audited as identical to the 9B model profile.']
    write(output/'threeway-results.json', rows)
    write(output/'comparison.json', report)
    (output/'COMPARISON.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    table = io.StringIO(newline='')
    writer = csv.writer(table,lineterminator='\n')
    writer.writerow(['task_id','family','native_reward','base_reward','checkpoint2350_reward',
        'native_submitted','base_submitted','checkpoint2350_submitted',
        'native_tool_calls','base_tool_calls','checkpoint2350_tool_calls'])
    for t in ids:
        writer.writerow([t,rows[t]['agent']['family']]+[rows[t][m][f]
            for f in ('reward','submitted','tool_calls') for m in ('agent','base','checkpoint2350')])
    (output/'threeway-results.csv').write_text(table.getvalue(),encoding='utf-8')
    write(output/'download-verification.json',{'passed':True,'cases':100,
        'archive_files':len(native),'all_archive_hashes_match':True,
        'all_prompts_tools_capsules_budgets_initial_public_tokens_match_both_models':True,
        'owner_capsule_server_and_protocol_source_match_local':True,
        'both_model_archived_terminal_records_match_previously_verified_results':True})
    print(json.dumps(report['groups']['Overall'],ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export-root',type=Path)
    p.add_argument('--models',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.export_root:
        export(args.export_root,args.output)
    else:
        compare(args.output,args.models)
