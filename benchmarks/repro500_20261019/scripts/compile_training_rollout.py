"""Unseal and archive the two completed training-rollout evaluations."""
import argparse
import io
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_training_rollout import SUITE, digest, read, write
from compile_results import paired_summary, summarize
from compile_threeway import native_diagnostics


def compile_run(root, output):
    # Metadata-only gates precede every score read.
    for model in ('base','checkpoint2350'):
        phase = read(root/model/'progress.json')
        assert phase['stage'] == 'complete' and phase['total'] == phase['completed'] == 100
        assert not (root/model/'failure.json').exists()
    evaluation = read(SUITE/'data/eval100.json')
    ids = evaluation['task_ids']
    native = read(root.parent/'native-v2-results/native-results.json')
    rows, sources = {}, []
    for model in ('base','checkpoint2350'):
        assert {p.name for p in (root/model/'cases').iterdir() if p.is_dir()} == set(ids)
        for path in (root/model).rglob('*'):
            if path.is_file() and 'numeric-sessions' not in path.parts:
                sources.append((path,f'{model}/'+path.relative_to(root/model).as_posix()))
    assert read(root/'base/contract.json') == read(root/'checkpoint2350/contract.json')
    runner_hash = digest((SUITE/'run_training_rollout.py').read_bytes())
    for model in ('base','checkpoint2350'):
        assert read(root/model/'runner-source.json')['run_training_rollout.py_sha256'] == runner_hash
    contract = read(root/'base/contract.json')
    for task in ids:
        prompts = [read(root/model/'cases'/task/'prompt.json') for model in ('base','checkpoint2350')]
        assert prompts[0] == prompts[1]
        assert prompts[0] == read(root.parent/'isolated-agent-v2-runtime/agent'/task/'prompt.json')
        pair = {}
        for model in ('base','checkpoint2350'):
            row = read(root/model/'cases'/task/'private-result.json')
            assert row['task_id'] == task and row['model'] == model and row['valid_terminated']
            assert 0 <= row['reward'] <= 1
            assert row['budget'] == native[task]['budget']
            rollout = read(root/model/'cases'/task/'rollout.json')
            assert len(rollout['completion_ids']) <= contract['training_settings']['max_completion_length']
            assert len(rollout['completion_ids']) == len(rollout['tool_mask']) == len(rollout['logprobs'])
            events = [json.loads(line) for line in (root/model/'cases'/task/'events.jsonl').read_text().splitlines()]
            calls = [e for e in events if e['kind']=='generation_request']
            responses = [e for e in events if e['kind']=='generation_response']
            assert len(calls) == len(responses) == row['generation_calls']
            assert all(e['max_tokens'] == 31497 for e in calls)
            pair[model] = row
        rows[task] = pair
    output.mkdir(parents=True,exist_ok=False)
    write(output/'contract.json',contract)
    write(output/'paired-results.json',rows)
    groups = {**evaluation['task_ids_by_family'],'Overall':ids}
    report = {'version':contract['version'],'paired_cases':100,
        'metric':'Native terminal-quality-v10 reward, not accuracy',
        'extra_8k_generation_cap':False,'generation_methods_reused_unchanged':True,
        'task_interface':contract['task_interface'],
        'training_task_interface_identical':False,'groups':{}}
    lines = ['# Original versus checkpoint-2350: training rollout generation', '',
        'Same fixed 100 tasks and final on-demand-data-v1.1 environment. Actual training TRL generation/tokenization/parsing/tool-loop methods are reused unchanged. No extra 8192-token per-turn cap.', '',
        'Training settings: 32768 context, 31497 total rollout completion budget (including tool feedback), temperature 1, top_p 1, top_k 0, repetition penalty 1, thinking disabled. One rollout per evaluation task. This replaces the earlier custom 8K/temperature-0 evaluation as the requested model comparison.', '',
        '**Generation logic matches training; the later frozen task interface intentionally differs from training\'s old automatic-observation-feedback adapter. Reward is not accuracy.**', '',
        '| Family | n | Original 9B | Checkpoint-2350 | Difference | Submitted original / trained |',
        '|---|---:|---:|---:|---:|---:|']
    for group,tasks in groups.items():
        d = paired_summary([(rows[t]['checkpoint2350'],rows[t]['base']) for t in tasks])
        report['groups'][group] = {'n':len(tasks),'comparison':d,
            'diagnostics':{model:native_diagnostics([rows[t][model] for t in tasks])
                           for model in ('base','checkpoint2350')}}
        lines.append(f"| {group} | {len(tasks)} | {d['qwen']['mean_reward']:.4f} | {d['agent']['mean_reward']:.4f} | {d['mean_agent_minus_qwen']:+.4f} | {d['qwen']['submitted']} / {d['agent']['submitted']} |")
    d = report['groups']['Overall']['comparison']
    lines += ['',f"Trained / tie / original wins: {d['agent_wins']} / {d['ties']} / {d['qwen_wins']}. Paired bootstrap 95% interval: {d['paired_bootstrap_95pct']}.", '',
        'The old 8K results are retained separately and must not be pooled with this run. Native agents did not have matched hidden compute; their previous scores are not part of this new two-model test.', '',
        'Raw token IDs, sampled-token logprobs, generated text, parsed completions, tool masks, tool calls/results, original acquired observations, capsule identities and source fingerprints are retained in the archive. No optimizer or training weights were changed.']
    (output/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    write(output/'comparison.json',report)
    hashes = {}
    for name in ('run_training_rollout.py','scripts/compile_training_rollout.py',
                 'scripts/tests_training_rollout.py','scripts/run_training_rollout_pipeline.py'):
        sources.append((SUITE/name,'source/'+name))
    for path in (root/'merged-checkpoint2350').glob('merge-receipt.json'):
        sources.append((path,'source/merge-receipt.json'))
    if 'engine_execution' in contract:
        receipt = root/'tolerance-report.json'
        assert digest(receipt.read_bytes()) == contract['engine_execution']['tolerance_report_sha256']
        assert read(receipt)['passed']
        sources.append((receipt,'source/tolerance-report.json'))
        for name in ('scripts/run_graph_rollout_pipeline.py','scripts/vllm_speed_tolerance.py',
                     'scripts/tests_vllm_tolerance_gate.py'):
            sources.append((SUITE/name,'source/'+name))
    with tarfile.open(output/'raw-rollouts.tar.gz','w:gz') as archive:
        for path,name in sources:
            blob = path.read_bytes()
            # This offline runner has no role tokens or HTTP credentials.
            assert 'private-control' not in name
            hashes[name] = digest(blob)
            info = tarfile.TarInfo(name)
            info.size,info.mtime,info.mode = len(blob),0,0o644
            archive.addfile(info,io.BytesIO(blob))
    write(output/'archive-members-sha256.json',hashes)
    write(output/'verification.json',{'passed':True,'paired_cases':100,
        'all_requested_generation_caps_31497':True,'extra_per_turn_8k_limit':False,
        'all_prompts_tools_capsules_paired':True,'all_budgets_match_fixed_cases':True,
        'all_completions_follow_training_total_budget':True,
        'raw_archive_files':len(hashes),'infrastructure_failures_scored_as_zero':0})
    write(output/'all-result-checksums.json',{p.name:digest(p.read_bytes()) for p in output.iterdir()
        if p.is_file() and p.name!='all-result-checksums.json'})
    print(json.dumps({'paired_cases':100,'archive_files':len(hashes)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    compile_run(args.root,args.output)
