"""Continue censored HTTP requests without resampling any returned completion.

Successful terminal cases are copied unchanged. A timeout case replays every
returned completion and tool receipt against a fresh owner; any discrepancy
stops the case before further inference. Only the request with no response is
reissued, with the identical payload, seed and token limits. No scores are read.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
import shutil

import run_9b


INFRA_REASONS = ('runner_or_owner_failure', 'model_api_failure')


def classification(record):
    assert 'ended_unix' in record, 'Source run must be terminal before continuation'
    assert not record.get('error_type') and not record.get('finish_error_type')
    if record['termination_reason'] not in INFRA_REASONS:
        return 'complete'
    completions = [e for e in record['events'] if e['kind'] == 'completion']
    assert record['termination_reason'] == 'model_api_failure'
    assert completions and completions[-1].get('error_type') == 'TimeoutError'
    assert 'response' not in completions[-1]
    assert all('response' in e and not e.get('error_type') for e in completions[:-1])
    assert all('response' in e for e in record['events'] if e['kind'] == 'tool')
    return 'timeout'


def source_records(source, task_ids):
    summary = json.loads((source/'run_summary.json').read_text())
    assert len(summary['tasks']) == 100
    assert {t['task_id'] for t in summary['tasks']} == set(task_ids)
    records = {}
    for task_id in task_ids:
        path = source/(hashlib.sha256(task_id.encode()).hexdigest()+'.json')
        record = json.loads(path.read_text())
        assert record['task_id'] == task_id and record['model'] == summary['model']
        records[task_id] = (path, record, classification(record))
    return summary['model'], records


def continue_task(task_id, source_path, source, *, tokenizer, profile, owner_url,
                  model_api, output_dir, timeout=3600, owner=run_9b.owner_request,
                  completion=run_9b.post_json):
    assert classification(source) == 'timeout'
    assert source['task_id'] == task_id and source['profile'] == profile
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    turns = [e for e in source['events'] if e['kind'] == 'completion']
    saved = turns[:-1]
    prior_tools = [e for e in source['events'] if e['kind'] == 'tool']
    counters = {'completions': 0, 'tools': 0, 'retried_censored_request': False}

    def replay_completion(url, payload, wait):
        index = counters['completions']
        if index < len(saved):
            assert payload == saved[index]['request'], 'Saved completion payload changed'
            counters['completions'] += 1
            return copy.deepcopy(saved[index]['response'])
        if not counters['retried_censored_request']:
            assert counters['tools'] == len(prior_tools), 'Prior tools not fully replayed'
            assert payload == turns[-1]['request'], 'Censored request payload changed'
            counters['retried_censored_request'] = True
        return completion(url, payload, wait)

    def replay_owner(url, payload, wait):
        if payload['op'] == 'prompt':
            result = owner(url, payload, wait)
            assert result['messages'] == source['messages'][:len(result['messages'])]
            assert result['tools'] == source['tools']
            return result
        if payload['op'] in ('act', 'numerics') and counters['tools'] < len(prior_tools):
            previous = prior_tools[counters['tools']]
            strip_uuid = lambda value: {k:v for k,v in value.items() if k != 'request_id'}
            assert strip_uuid(payload) == strip_uuid(previous['request']), 'Replay tool changed'
            result = owner(url, payload, wait)
            assert result == previous['response'], 'Replay tool feedback differs; refusing inference'
            counters['tools'] += 1
            return result
        assert counters['tools'] == len(prior_tools), 'Replay ended before all saved tools'
        return owner(url, payload, wait)

    result = run_9b.run_task(task_id, tokenizer=tokenizer, profile=profile,
        model=source['model'], owner_url=owner_url, model_api=model_api,
        output_dir=output_dir, timeout=timeout, owner=replay_owner, completion=replay_completion)
    destination = output_dir/source_path.name
    record = json.loads(destination.read_text())
    returned = [e for e in record['events'] if e['kind'] == 'completion']
    for index in range(counters['completions']):
        assert returned[index]['request'] == saved[index]['request']
        assert returned[index]['response'] == saved[index]['response']
        returned[index]['reused_recorded_completion'] = {'source_sha256':source_sha, 'index':index}
    record['continuation_source'] = {'filename':source_path.name, 'sha256':source_sha,
        'replayed_completions':counters['completions'], 'replayed_tools':counters['tools'],
        'censored_request_reissued_identically':counters['retried_censored_request'],
        'read_timeout_seconds':timeout, 'returned_answers_resampled':0}
    record['api_requests_this_run'] -= counters['completions']
    record['reused_responses'] += counters['completions']
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(destination)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--owner-url',required=True)
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=3600)
    args = parser.parse_args()
    assert args.timeout >= 3600
    task_ids = json.loads(args.index.read_text())['task_ids']
    assert len(task_ids) == len(set(task_ids)) == 100
    model, records = source_records(args.source,task_ids)
    profile = json.loads(Path(run_9b.__file__).with_name('model_profile.json').read_text())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(profile['base_model_path'])
    args.output_dir.mkdir(parents=True,exist_ok=False)
    summaries = []
    futures = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for task_id in task_ids:
            path, record, status = records[task_id]
            assert record['profile'] == profile
            if status == 'complete':
                shutil.copyfile(path,args.output_dir/path.name)
                summaries.append({'task_id':task_id,'termination_reason':record['termination_reason'],
                    'transcript':path.name,'model_requests':record['model_requests'],
                    'copied_terminal_trace_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            else:
                futures.append(executor.submit(continue_task,task_id,path,record,
                    tokenizer=tokenizer,profile=profile,owner_url=args.owner_url,
                    model_api=profile['model_api'],output_dir=args.output_dir,timeout=args.timeout))
        for future in as_completed(futures):
            summary = future.result();summaries.append(summary)
            print(json.dumps(summary),flush=True)
    (args.output_dir/'run_summary.json').write_text(json.dumps({'model':model,
        'tasks':summaries,'continuation_source':str(args.source),
        'completed_cases_copied':sum(status=='complete' for _,_,status in records.values()),
        'timeout_cases_continued':len(futures)},indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
