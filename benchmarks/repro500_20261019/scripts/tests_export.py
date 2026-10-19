"""Synthetic end-to-end checks for the unsealing gate and paired exporter."""
import hashlib
import json
from pathlib import Path
import sys
import shutil
import tarfile
import tempfile
from unittest.mock import patch

import compile_results as export
import compile_threeway as threeway


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        runtime = root/'runtime'
        transcripts = root/'model'
        output = root/'output'
        argv = ['compile_results.py', '--runtime', str(runtime), '--model-transcripts',str(transcripts), '--output',str(output)]
        write(runtime/'progress.json', {k:{'total':100,'completed':99} for k in ('agent','qwen')})
        with patch.object(sys, 'argv', argv), patch.object(export, 'SUITE', root/'suite'):
            try:
                export.main()
                raise RuntimeError('Premature score gate accepted')
            except AssertionError as error:
                assert 'Both phases must finish first' in str(error)
            assert not output.exists()
        print('unfinished gate rejects before any private file exists: PASS')
        write(runtime/'progress.json', {k:{'total':100,'completed':100} for k in ('agent','qwen')})
        write(runtime/'private-control.json', {'roles':[{'token':'synthetic-secret-not-in-results'}]})
        grouped = {f'family_{i}':[f'family_{i}-{j:03d}' for j in range(20)] for i in range(5)}
        ids = [task for members in grouped.values() for task in members]
        write(root/'suite/data/manifest.json', {'tasks_merkle_input_sha256':'synthetic-dataset', 'eval100_sha256':'synthetic-selection'})
        write(root/'suite/data/eval100.json', {'task_ids':ids,'task_ids_by_family':grouped})
        for task_id in ids:
            prompt = {'messages':[{'role':'user','content':'Synthetic case '+task_id}],
                'tools':[], 'capsule_sha256':hashlib.sha256(task_id.encode()).hexdigest()}
            for kind in ('agent','qwen'):
                folder = runtime/kind/task_id
                write(folder/'prompt.json', prompt)
                write(folder/'private-result.json', {'task_id':task_id, 'family':task_id.split('-')[0],
                    'role':kind, 'reward':0.75 if kind=='agent' else 0.25, 'submitted':True,
                    'observations_used':10 if kind=='agent' else 20, 'budget':100,
                    'tool_calls':3,'reason':'terminal_answer'})
                (folder/'events.jsonl').write_text(json.dumps({'utc':'2026-01-01T00:00:00Z' if kind=='agent' else '2026-01-02T00:00:00Z'})+'\n')
            write(transcripts/(hashlib.sha256(task_id.encode()).hexdigest()+'.json'),
                {'task_id':task_id,'model':'checkpoint2350','messages':prompt['messages'],
                'tools':[], 'termination_reason':'owner_ended','usage':{},'model_requests':1})
        write(transcripts/'run_summary.json', {'model':'checkpoint2350','tasks':[{'task_id':t} for t in ids]})
        with patch.object(sys, 'argv', argv), patch.object(export, 'SUITE', root/'suite'):
            export.main()
        report = json.loads((output/'comparison.json').read_text())
        assert report['overall']['n'] == 100
        assert report['overall']['mean_agent_minus_qwen'] == 0.5
        assert report['overall']['paired_bootstrap_95pct'] == [0.5,0.5]
        assert report['overall']['agent_wins'] == 100
        assert report['overall']['agent']['mean_budget_fraction'] == 0.1
        assert len((output/'paired-results.csv').read_text().splitlines()) == 101
        with tarfile.open(output/'transcripts.tar.gz') as archive:
            assert len(archive.getnames()) == 701
            assert 'model/run_summary.json' in archive.getnames()
            assert not any('private-control' in name for name in archive.getnames())
        print('100 paired cases, run summary, metrics, phase order and safe archive: PASS')
        base_runtime = root/'base-runtime'
        base_model = root/'base-model'
        shutil.copytree(runtime, base_runtime)
        shutil.copytree(transcripts, base_model)
        for task_id in ids:
            path = base_runtime/'qwen'/task_id/'private-result.json'
            value = json.loads(path.read_text())
            value['reward'] = 0.125
            write(path,value)
            path = base_model/(hashlib.sha256(task_id.encode()).hexdigest()+'.json')
            value = json.loads(path.read_text())
            value['model'] = 'base'
            write(path,value)
        write(base_runtime/'prior-agent-proof.json',{'synthetic_only':True})
        write(base_runtime/'progress.json', {k:{'total':100,'completed':99} for k in ('agent','qwen')})
        three_output = root/'three-output'
        three_argv = ['compile_threeway.py','--agent-runtime',str(runtime),
            '--trained-transcripts',str(transcripts),'--base-runtime',str(base_runtime),
            '--base-transcripts',str(base_model),'--output',str(three_output)]
        with patch.object(sys,'argv',three_argv), patch.object(export,'SUITE',root/'suite'):
            try:
                threeway.main()
                raise RuntimeError('Incomplete baseline was unsealed')
            except AssertionError as error:
                assert 'All THREE phases' in str(error)
            assert not three_output.exists()
            write(base_runtime/'progress.json', {k:{'total':100,'completed':100} for k in ('agent','qwen')})
            threeway.main()
        report = json.loads((three_output/'comparison.json').read_text())
        overall = report['groups']['Overall']
        assert overall['models']['agent']['mean_reward']==0.75
        assert overall['models']['base']['mean_reward']==0.125
        assert overall['models']['checkpoint2350']['mean_reward']==0.25
        assert overall['trained_minus_base']==0.125
        assert len((three_output/'threeway-results.csv').read_text().splitlines())==101
        print('three-way gate and identical 100-case agent/base/trained table: PASS')


if __name__ == '__main__':
    main()
