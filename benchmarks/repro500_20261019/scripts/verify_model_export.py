"""Verify downloaded model results and exact continuation from archived bytes.

No model requests, tool execution, or reads of the replacement native-agent run.
Legacy agent records in these archives remain exploratory only.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

SUITE = Path(__file__).resolve().parents[1]
EXPECTED = {
    'recovery-source-transcripts.tar.gz': 'cf4b6aa74ced7ec0a59693c30b61919dda35d99b11632ae61723b42430a4f232',
    'recovery-completion-audit.json': '312d158f41234315c19e65cca7748b04a48201152013cd5adeab54660b4018de',
    'final-results/all-result-checksums.json': '707b5e2d1e3772084dc7744064c4e66ed4d3230b0918bd9a4d488f16c8907d01',
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def members(archive):
    result = {}
    with tarfile.open(archive) as source:
        for item in source:
            path = PurePosixPath(item.name)
            assert not path.is_absolute() and '..' not in path.parts
            assert not item.issym() and not item.islnk()
            if item.isfile():
                assert item.name not in result
                result[item.name] = source.extractfile(item).read()
    return result


def completions(record):
    return [e for e in record['events'] if e['kind'] == 'completion']


def strip_uuid(request):
    return {k: v for k, v in request.items() if k != 'request_id'}


def verify(root):
    for name, sha in EXPECTED.items():
        assert digest((root/name).read_bytes()) == sha, name
    output = root/'final-results'
    checksums = read(output/'all-result-checksums.json')
    for name, sha in checksums.items():
        path = PurePosixPath(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert digest((output/name).read_bytes()) == sha, name
    manifest = read(SUITE/'data/manifest.json')
    selection = read(SUITE/'data/eval100.json')
    ids = selection['task_ids']
    entries = {x['task_id']: x for x in manifest['tasks']}
    assert len(ids) == len(set(ids)) == 100
    comparison = read(output/'comparison.json')
    assert comparison['dataset_sha256'] == manifest['tasks_merkle_input_sha256']
    assert comparison['agent_comparison_validity'] == 'exploratory_only_not_fair_controlled_comparison'
    rows = read(output/'threeway-results.json')
    assert set(rows) == set(ids)
    audit = read(root/'recovery-completion-audit.json')
    assert audit['model_cases'] == 200 and audit['continued_http_timeout_cases'] == 64
    assert audit['read_timeout_seconds'] == 3600
    sources = members(root/'recovery-source-transcripts.tar.gz')
    profile = read(SUITE/'model_profile.json')
    archived = {model: members(output/f'pair-{model}/transcripts.tar.gz')
        for model in ('base', 'checkpoint2350')}
    continued = 0
    counts = {}
    for model, source_name in (('base', 'base-transcripts'), ('checkpoint2350', 'qwen-transcripts')):
        archive = archived[model]
        evidence = next(x for x in audit['models'] if x['model'] == model)
        proof = {x['task_id']: x for x in evidence['cases_evidence']}
        assert set(proof) == set(ids) and evidence['cases'] == 100
        summary = json.loads(archive['model/run_summary.json'])
        assert summary['model'] == model and len(summary['tasks']) == 100
        assert {x['task_id'] for x in summary['tasks']} == set(ids)
        submitted = 0
        for task_id in ids:
            filename = digest(task_id.encode()) + '.json'
            raw = archive['model/'+filename]
            record = json.loads(raw)
            assert record['task_id'] == task_id and record['model'] == model
            assert 'ended_unix' in record and record['model_requests'] > 0
            assert not record.get('error_type') and not record.get('finish_error_type')
            assert record['termination_reason'] not in ('model_api_failure', 'runner_or_owner_failure')
            assert record['profile'] == profile
            prompt_raw = archive[f'owner/qwen/{task_id}/prompt.json']
            prompt = json.loads(prompt_raw)
            assert prompt['capsule_sha256'] == entries[task_id]['sha256']
            assert record['tools'] == prompt['tools']
            assert record['messages'][:len(prompt['messages'])] == prompt['messages']
            assert prompt_raw == archive[f'owner/agent/{task_id}/prompt.json']
            assert prompt_raw == archived['base'][f'owner/qwen/{task_id}/prompt.json']
            owner = json.loads(archive[f'owner/qwen/{task_id}/private-result.json'])
            assert owner == rows[task_id][model]
            assert owner['budget'] == rows[task_id]['agent']['budget']
            submitted += bool(owner['submitted'])
            original_raw = sources[source_name+'/'+filename]
            assert digest(original_raw) == proof[task_id]['original_first_sha256']
            before = completions(json.loads(original_raw))
            after = completions(record)
            assert len(before) == 1 and after
            assert after[0]['request'] == before[0]['request']
            assert after[0]['response'] == before[0]['response']
            for turn in after:
                request = turn['request']
                assert request['model'] == profile['models'][model]
                assert request['temperature'] == profile['temperature']
                assert request['seed'] == profile['seed']
                assert request['chat_template_kwargs'] == {'enable_thinking': False}
                assert 0 < request['max_tokens'] <= profile['max_tokens_per_turn']
                assert turn['prompt_tokens_tokenizer'] + request['max_tokens'] <= profile['max_model_len']
            if model == 'checkpoint2350':
                prior_raw = sources['recovered-qwen-transcripts/'+filename]
                assert digest(prior_raw) == proof[task_id]['continuation_source_sha256']
                prior = json.loads(prior_raw)
                if prior['termination_reason'] not in ('model_api_failure', 'runner_or_owner_failure'):
                    assert raw == prior_raw
                    continue
                continued += 1
                turns = completions(prior)
                assert turns[-1]['error_type'] == 'TimeoutError' and 'response' not in turns[-1]
                saved = turns[:-1]
                for old, new in zip(saved, after):
                    assert old['request'] == new['request'] and old['response'] == new['response']
                assert len(after) > len(saved) and after[len(saved)]['request'] == turns[-1]['request']
                old_tools = [e for e in prior['events'] if e['kind'] == 'tool']
                new_tools = [e for e in record['events'] if e['kind'] == 'tool']
                assert len(new_tools) >= len(old_tools)
                for old, new in zip(old_tools, new_tools):
                    assert strip_uuid(old['request']) == strip_uuid(new['request'])
                    assert old['response'] == new['response']
                assert record['continuation_source']['returned_answers_resampled'] == 0
        counts[model] = {'valid_terminated': 100, 'submitted': submitted}
    assert continued == 64
    return {'status': 'verified', 'dataset_sha256': comparison['dataset_sha256'],
        'paired_cases': 100, 'models': counts, 'continued_http_timeout_cases': continued,
        'all_first_replies_and_returned_completions_preserved': True,
        'all_replayed_tool_requests_and_feedback_identical': True,
        'public_prompts_tools_capsules_budgets_identical': True,
        'model_profile_and_request_limits_verified': True,
        'downloaded_file_checksums_verified': len(checksums) + len(EXPECTED),
        'legacy_agent_comparison': 'exploratory only; replacement native run is separate'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=SUITE/'results/model-evaluation')
    args = parser.parse_args()
    report = verify(args.root)
    (args.root/'download-verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    comparison = read(args.root/'final-results/comparison.json')
    names = {'ate': 'ATE', 'backadj_minimal_sets': 'Backdoor adjustment sets',
        'best_intervention': 'Best intervention',
        'individual_counterfactual_probability': 'Individual counterfactual',
        'mediator_set': 'Mediator set', 'Overall': 'Overall'}
    lines = ['# Original versus trained 9B: same 100 frozen tasks', '',
        'Both 100-case runs have valid terminal records. Final answers were submitted in '
        '20 original-model cases and 24 trained-model cases. Legal unfinished cases remain zero reward; '
        'transport/executor failures are excluded and were recovered without resampling returned answers.', '',
        '| Family | n | Original 9B reward | Checkpoint-2350 reward | Difference | Final answers: original / trained |',
        '|---|---:|---:|---:|---:|---:|']
    for family, group in comparison['groups'].items():
        base, trained = group['models']['base'], group['models']['checkpoint2350']
        lines.append(f"| {names[family]} | {group['n']} | {base['mean_reward']:.4f} | "
            f"{trained['mean_reward']:.4f} | {group['trained_minus_base']:+.4f} | "
            f"{base['submitted']}/{group['n']} / {trained['submitted']}/{group['n']} |")
    overall = comparison['groups']['Overall']
    interval = overall['trained_minus_base_paired_bootstrap_95pct']
    lines += ['', f"Paired cases: trained wins {overall['trained_wins']}, ties {overall['ties']}, "
        f"original wins {overall['base_wins']}. Mean reward difference: {overall['trained_minus_base']:+.4f}; "
        f"descriptive paired-bootstrap 95% interval [{interval[0]:+.4f}, {interval[1]:+.4f}]. "
        'The interval includes zero; this run does not establish a reliable training improvement.', '',
        'Reward is the unchanged native terminal-quality-v10 value in [0,1], not accuracy. '
        'Both models use the same runner, task owner, prompts, tools, observation budgets, sandbox, '
        'thinking=false, temperature=0, seed=20261019, 32768-token context and 8192-token per-turn generation cap.', '',
        'Download/provenance verification: `download-verification.json`; exact first replies, all returned '
        'completion prefixes and replayed tool feedback were checked from archived bytes for all 200 cases. '
        'The 64 checkpoint transport timeouts were continued with 3600-second waits.', '',
        '**The old five-agent results inside `final-results/` are exploratory only.** '
        'They retained 20-case histories and did not match native generation/context limits. '
        'The replacement native subagent group uses a fresh conversation per case, is still running, '
        'and is not included or combined with this archived group. Its hidden native reasoning/generation '
        'budget remains unobservable and must be disclosed in the eventual comparison.']
    (args.root/'MODEL_COMPARISON.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
