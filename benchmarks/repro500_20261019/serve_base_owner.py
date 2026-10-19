"""Independent base-model owner, gated by the original completed agent phase.

Does not restart or alter the first exam. Imports completion metadata only;
all base-model episodes and numerical sessions start empty from frozen capsules.
"""
import argparse
import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path

import server


def completed_agent_proof(source, data):
    progress = json.loads((source/'progress.json').read_text())
    assert progress['agent']['completed'] == progress['agent']['total'] == 100
    manifest = json.loads((data/'manifest.json').read_text())
    evaluation = json.loads((data/'eval100.json').read_text())
    controls = json.loads((source/'private-control.json').read_text())
    assert controls['dataset_sha256'] == manifest['tasks_merkle_input_sha256']
    assigned = [t for role in controls['roles'] if role['kind']=='agent' for t in role['task_ids']]
    assert len(assigned) == len(set(assigned)) == 100
    assert set(assigned) == set(evaluation['task_ids'])
    # Hash opaque completion records; neither read nor expose scores here.
    terminal_files = {t:hashlib.sha256((source/'agent'/t/'private-result.json').read_bytes()).hexdigest()
        for t in assigned}
    return {'dataset_sha256':controls['dataset_sha256'], 'completed_agent_tasks':100,
        'source_progress_sha256':hashlib.sha256((source/'progress.json').read_bytes()).hexdigest(),
        'agent_terminal_record_sha256':terminal_files,
        'imported_scores':False, 'base_episodes_reused':False}


class BaselineExam(server.Exam):
    def __init__(self, args):
        proof = completed_agent_proof(args.agent_runtime.resolve(), args.data.resolve())
        super().__init__(args)
        for role in self.roles.values():
            if role['kind']=='agent':
                for task_id in role['task_ids']:
                    self.cases[('agent',task_id)] = {'ended':True, 'imported_completion':True}
        control_path = self.runtime/'private-control.json'
        controls = json.loads(control_path.read_text())
        controls['model_label'] = 'base'
        controls['prior_agent_completion_proof'] = 'prior-agent-proof.json'
        server.write(control_path, controls)
        server.write(self.runtime/'prior-agent-proof.json', proof)
        self.progress()

    def get_case(self, role, task_id):
        if role['kind']=='agent':
            raise ValueError('Original agent phase already completed; baseline owner has no agent episodes')
        return super().get_case(role, task_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=server.ROOT/'data')
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--agent-runtime', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18730)
    parser.add_argument('--tokenizer', default='/home/chen/models/Qwen/Qwen3.5-9B')
    args = parser.parse_args()
    exam = BaselineExam(args)
    http = ThreadingHTTPServer(('127.0.0.1',args.port), server.Handler)
    http.exam = exam
    print(json.dumps({'ready':True,'port':args.port,'model':'base',
        'dataset_sha256':exam.manifest['tasks_merkle_input_sha256'],
        'original_agent_completion_verified':True}), flush=True)
    try:
        http.serve_forever()
    finally:
        exam.factory.close()


if __name__ == '__main__':
    main()
