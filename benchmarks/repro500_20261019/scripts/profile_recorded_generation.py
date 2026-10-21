"""Profile recorded public generation events; never read task scores or labels."""
import argparse
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_training_rollout import read, write, digest


def profile(runtime,output):
    events, sources, retained = [], {}, 0
    paths = sorted((runtime/'cases').glob('ate-*/events.jsonl'))
    for path in paths:
        # Only valid fully recorded rollouts; do not infer task reward from termination.
        rollout_path = path.with_name('rollout.json')
        if not rollout_path.exists():
            continue
        retained += sum(read(rollout_path)['tool_mask'])
        sources[str(path)] = digest(path.read_bytes())
        for line in path.read_text().splitlines():
            e = json.loads(line)
            if e['kind'] in ('generation_request','generation_response'):
                e['task_id'] = path.parent.name
                events.append(e)
    pending, assigned, calls = {}, {}, []
    for e in sorted(events,key=lambda x:x['unix']):
        key = (e['task_id'],e['turn'])
        if e['kind'] == 'generation_request':
            assert key not in pending and key not in assigned
            pending[key] = e
        else:
            if key not in assigned:
                assert key in pending
                call = {'requests':list(pending.values()),'responses':[]}
                calls.append(call)
                for k in pending:
                    assigned[k] = call
                pending.clear()
            assigned[key]['responses'].append(e)
    assert not pending
    rows = []
    for call in calls:
        req,res = call['requests'],call['responses']
        assert {(r['task_id'],r['turn']) for r in req} == {(r['task_id'],r['turn']) for r in res}
        tokens = [len(r['outputs'][0]['token_ids']) for r in res]
        seconds = min(r['unix'] for r in res)-min(r['unix'] for r in req)
        rows.append({'requests':len(req),'prompt_lengths':[len(r['prompt_token_ids']) for r in req],
            'output_lengths':tokens,'generation_wall_seconds':seconds,
            'decode_step_proxy':max(tokens),'generated_model_tokens':sum(tokens),
            'effective_decode_width':sum(tokens)/max(tokens)})
    total = sum(r['generated_model_tokens'] for r in rows)
    seconds = sum(r['generation_wall_seconds'] for r in rows)
    write(output,{'case_count':len(sources),'generation_calls':len(rows),
        'generated_model_tokens_before_training_truncation':total,
        'retained_model_tokens_after_training_truncation':retained,
        'discarded_fraction':1-retained/total,
        'generation_wall_seconds':seconds,'raw_generated_tokens_per_second':total/seconds,
        'retained_tokens_per_generation_second':retained/seconds,
        'weighted_effective_decode_width':total/sum(r['decode_step_proxy'] for r in rows),
        'median_prompt_length':statistics.median(n for r in rows for n in r['prompt_lengths']),
        'max_prompt_length':max(n for r in rows for n in r['prompt_lengths']),
        'call_details':rows,'source_sha256':sources,
        'limits':['Effective width is output lengths / longest output, not measured per-step GPU occupancy.',
            'Generation wall includes prefill and batch completion; this does not isolate a causal speed contribution.',
            'Discarded tokens reflect original trainer truncation; no change to training generation logic is proposed.']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    profile(args.runtime,args.output)
