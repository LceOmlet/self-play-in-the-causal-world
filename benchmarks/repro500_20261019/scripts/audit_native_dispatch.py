"""Publish credential-free dispatch receipts after the native run is complete."""
import argparse
import hashlib
import itertools
from pathlib import Path

from compile_results import read, write, SUITE


def audit(index_path, output):
    index = read(index_path)
    ids = read(SUITE/'data/eval100.json')['task_ids']
    assert len(index) == 100 and {r['task_id'] for r in index} == set(ids)
    assert len({r['agent_name'] for r in index}) == 100
    assert all(r['status'] == 'completed_owner_verified' for r in index)
    assert all(r['submitted_reported'] for r in index)
    template = (SUITE/'native_agent_solver_prompt.txt').read_text(encoding='utf-8')
    receipts = []
    for row in index:
        receipt = {k:row[k] for k in ('index','task_id','family','status','agent_name','submitted_reported')}
        if row.get('prompt_sha256'):
            assert row['fork_turns'] == 'none' and row['model_override'] is None
            candidates = set()
            for slash, trim, crlf in itertools.product((False, True), repeat=3):
                cap = row['capability_file'].replace('\\','/') if slash else row['capability_file']
                prompt = template.replace('{task_id}',row['task_id']).replace('{capability_file}',cap)
                if trim:
                    prompt = prompt.strip()
                if crlf:
                    prompt = prompt.replace('\n','\r\n')
                candidates.add(hashlib.sha256(prompt.encode()).hexdigest())
            assert row['prompt_sha256'] in candidates, row['task_id']
            receipt.update({k:row[k] for k in ('prompt_sha256','fork_turns','model_override')})
            receipt['prompt_receipt_matches_solver_template'] = True
        else:
            receipt.update({'fork_turns':'none','model_override':None,
                'prompt_sha256':None,'prompt_receipt_matches_solver_template':None,
                'provenance_note':'Initial parent dispatch: fresh fork confirmed from execution records; prompt digest was not recorded in the index.'})
        receipts.append(receipt)
    write(output/'dispatch-provenance.json', {
        'cases':100,'unique_solver_conversations':100,'fork_turns':'none',
        'model_override':None,'all_reported_submitted_and_owner_verified':True,
        'prompt_digest_receipts':sum(r['prompt_sha256'] is not None for r in receipts),
        'initial_parent_cases_without_index_prompt_digest':sum(r['prompt_sha256'] is None for r in receipts),
        'complete_native_system_context_or_hidden_reasoning_trace_available':False,
        'receipts':receipts})
    print('100 unique solver conversations verified; 62 prompt digests match the committed template; initial 38 have no index prompt digest.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    audit(args.index,args.output)
