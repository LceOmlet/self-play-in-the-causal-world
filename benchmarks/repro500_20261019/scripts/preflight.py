"""Synthetic-only sealed-service integration check; never opens benchmark cases."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server


class SyntheticEnvironment:
    def reset(self, **kwargs):
        self.episode = SimpleNamespace(completed=False, remaining_budget=10,
            queries_used=0, observations_used=0, terminal_score=None,
            terminal_reward=0, budget=SimpleNamespace(max_observations=10))

    def act(self, command):
        if command.get('answer') is not None:
            self.episode.completed = True
            return 'Synthetic submission accepted.'
        self.episode.remaining_budget -= 1
        self.episode.queries_used += 1
        self.episode.observations_used += 1
        return json.dumps({'type': 'batch_result', 'batch': {'n': 1,
            'joint_histogram': {'columns': ['synthetic'], 'rows': [[1, 1]]}},
            'remaining_budget': self.episode.remaining_budget})

    def get_reward(self):
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    data = args.root / 'synthetic-data'
    data.mkdir()
    grouped = {f'synthetic_{i}': [f'synthetic_{i}'] for i in range(5)}
    evaluation = {'task_ids_by_family': grouped, 'task_ids': list(grouped)}
    entries = []
    for task_id in grouped:
        capsule = {'row': {'prompt': [{'role': 'user', 'content':
            'ON-DEMAND DATA PROTOCOL v1.1: Synthetic preflight only.'}]}}
        blob = server.canonical(capsule)
        compressed = gzip.compress(blob, mtime=0)
        name = task_id + '.json.gz'
        (data / name).write_bytes(compressed)
        entries.append({'task_id': task_id, 'family': task_id, 'file': name,
            'sha256': hashlib.sha256(blob).hexdigest(),
            'compressed_sha256': hashlib.sha256(compressed).hexdigest()})
    server.write(data / 'eval100.json', evaluation)
    server.write(data / 'manifest.json', {'tasks': entries, 'owner': {'files': {}},
        'eval100_sha256': hashlib.sha256(server.canonical(evaluation)).hexdigest(),
        'tasks_merkle_input_sha256': 'synthetic-only'})
    server.FrozenEnvironment = SyntheticEnvironment
    exam = server.Exam(SimpleNamespace(runtime=args.root/'runtime', data=data,
        tokenizer='/home/chen/models/Qwen/Qwen3.5-9B', port=0))
    roles = [(token, role) for token, role in exam.roles.items()]
    qwen = next(t for t, r in roles if r['kind'] == 'qwen')
    try:
        exam.dispatch(qwen, {'op': 'next'})
        raise AssertionError('Qwen gate did not reject premature start')
    except ValueError:
        pass
    for i, (token, role) in enumerate((t, r) for t, r in roles if r['kind']=='agent'):
        prompt = exam.dispatch(token, {'op':'next'})
        task_id = prompt['task_id']
        req = {'op':'act','task_id':task_id,'request_id':'synthetic-experiment',
            'command':{'sample':1}}
        first = exam.dispatch(token, req)
        assert first == exam.dispatch(token, req)
        case = exam.cases[('agent', task_id)]
        assert case['env'].episode.remaining_budget == 9
        if i == 0:
            numeric = exam.dispatch(token, {'op':'numerics','task_id':task_id,
                'request_id':'synthetic-numerics', 'code':
                "import numpy as np\nprint(np.linalg.solve([[2.,0.],[0.,4.]],[6.,8.]))\nprint(len(public['observations']))"})
            assert '[3. 2.]' in numeric['output'] and '\n1' in numeric['output'], numeric
        end = exam.dispatch(token, {'op':'act','task_id':task_id,
            'request_id':'synthetic-terminal','command':{'answer':0}})
        assert end['ended']
        assert exam.dispatch(token, {'op':'next'})['done']
    assert exam.dispatch(qwen, {'op':'next'})['task_id'] == list(grouped)[0]
    exam.factory.close()
    server.write(args.root/'report.json', {'synthetic_only':True,
        'idempotent_experiment':True,'real_nsjail_numpy':True,
        'public_observation_bridge':True,'agent_completion_gate':True,
        'qwen_phase_gate':True,'progress':exam.progress()})
    print('Synthetic sealed-service preflight PASS', flush=True)


if __name__ == '__main__':
    main()
