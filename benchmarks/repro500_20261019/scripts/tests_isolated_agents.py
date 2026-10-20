"""Synthetic single-case capability checks; no real tasks or grades."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve_isolated_agents import single_task_roles
from isolated_agent_client import prepare_request


def main():
    ids = [f'synthetic-{i}' for i in range(100)]
    evaluation = {'task_ids': ids, 'task_ids_by_family': {
        f'family-{i}': ids[i*20:(i+1)*20] for i in range(5)}}
    tokens = iter(f'token-{i}' for i in range(100))
    roles = single_task_roles(evaluation, lambda: next(tokens))
    assert len(roles) == 100
    assert all(len(role['task_ids']) == 1 for role in roles.values())
    assert {role['task_ids'][0] for role in roles.values()} == set(ids)
    cap = {'task_id': ids[0], 'token': 'synthetic'}
    request = prepare_request(cap, {'op': 'act', 'command': 'synthetic', 'request_id': 'fixed-id-123'})
    assert request['task_id'] == ids[0] and request['request_id'] == 'fixed-id-123'
    short = {'op': 'act', 'command': 'synthetic', 'request_id': 'a001'}
    first = prepare_request(cap, short)
    assert first == prepare_request(cap, short)
    assert 8 <= len(first['request_id']) <= 128
    for invalid in ({'op': 'prompt', 'task_id': ids[1]}, {'op': 'next'}, {'op': 'status'}):
        try:
            prepare_request(cap, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Cross-case or global operation accepted')
    print('100 single-case roles; cross-case/global access rejected; UUID preserved: PASS')


if __name__ == '__main__':
    main()
