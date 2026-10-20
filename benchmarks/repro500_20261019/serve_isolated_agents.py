"""Fresh native-subagent exam: one immutable task capability per conversation.

Uses the original frozen task owner and numerical sandbox without changing their
budgets or scoring. Native Codex hidden reasoning/generation usage is unavailable;
this protocol does not claim matched native generation compute with Qwen.
"""
import argparse
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import secrets


def single_task_roles(evaluation, token_factory=lambda: secrets.token_urlsafe(30)):
    ids = evaluation['task_ids']
    assert len(ids) == len(set(ids)) == 100
    families = {t: f for f, tasks in evaluation['task_ids_by_family'].items() for t in tasks}
    assert set(families) == set(ids)
    roles = {}
    for task_id in ids:
        token = token_factory()
        assert token not in roles
        roles[token] = {'kind': 'agent', 'family': families[task_id], 'task_ids': [task_id]}
    return roles


def main():
    import server

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=server.ROOT/'data')
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18735)
    parser.add_argument('--tokenizer', default='/home/chen/models/Qwen/Qwen3.5-9B')
    args = parser.parse_args()
    exam = server.Exam(args)
    evaluation = json.loads((args.data/'eval100.json').read_text())
    # No server accepts requests until the original family capabilities have
    # been replaced. Every solver receives only its single-case capability.
    exam.roles = single_task_roles(evaluation)
    controls = json.loads((exam.runtime/'private-control.json').read_text())
    controls['roles'] = [{'token': token, **role} for token, role in exam.roles.items()]
    controls['agent_context'] = 'One fresh fork_turns=none native subagent per case'
    controls['native_generation_token_budget_matched'] = False
    controls['native_generation_usage_available'] = False
    controls['protocol_label'] = 'native-subagent-per-case-v2'
    controls.pop('qwen_phase_gate', None)
    server.write(exam.runtime/'private-control.json', controls)
    exam.progress()
    http = ThreadingHTTPServer(('127.0.0.1', args.port), server.Handler)
    http.exam = exam
    print(json.dumps({'ready': True, 'port': args.port, 'single_case_roles': len(exam.roles),
        'dataset_sha256': exam.manifest['tasks_merkle_input_sha256']}), flush=True)
    try:
        http.serve_forever()
    finally:
        exam.factory.close()


if __name__ == '__main__':
    main()
