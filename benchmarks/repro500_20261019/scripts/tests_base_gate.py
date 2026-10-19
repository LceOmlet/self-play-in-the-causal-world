"""Synthetic completion-proof checks without opening a real capsule."""
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from serve_base_owner import completed_agent_proof


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source,data = root/'source',root/'data'
        ids = [f'synthetic-{i:03d}' for i in range(100)]
        write(data/'manifest.json',{'tasks_merkle_input_sha256':'synthetic-digest'})
        write(data/'eval100.json',{'task_ids':ids})
        write(source/'private-control.json',{'dataset_sha256':'synthetic-digest',
            'roles':[{'kind':'agent','task_ids':ids}]})
        write(source/'progress.json',{'agent':{'completed':99,'total':100}})
        try:
            completed_agent_proof(source,data)
            raise RuntimeError('Incomplete agent phase accepted')
        except AssertionError:
            pass
        write(source/'progress.json',{'agent':{'completed':100,'total':100}})
        for task_id in ids:
            folder = source/'agent'/task_id
            folder.mkdir(parents=True)
            (folder/'private-result.json').write_bytes(b'Opaque synthetic completion; not a JSON score')
        proof = completed_agent_proof(source,data)
        assert proof['completed_agent_tasks']==100 and proof['imported_scores'] is False
        assert len(proof['agent_terminal_record_sha256'])==100
        assert proof['base_episodes_reused'] is False
        print('baseline gate rejects incomplete phase and fingerprints 100 opaque completions: PASS')


if __name__=='__main__':
    main()
