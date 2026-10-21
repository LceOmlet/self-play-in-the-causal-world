"""One-shot base + checkpoint evaluation with the actual training rollout code.

Never resumes or changes training, never reuses old 8K outputs, and refuses any
restart of an already-started phase. Model/executor failures block the pipeline.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_training_rollout import SUITE, read, write, digest


def main(root, contract):
    root.mkdir(parents=True,exist_ok=True)
    status_path = root/'pipeline-status.json'
    assert not status_path.exists(), 'Pipeline already started; no automatic restart/resampling'
    assert not (root/'base').exists() and not (root/'checkpoint2350').exists()
    # Place the supervisor in controller before delegated sandbox subgroups exist.
    sys.path.insert(0,str(SUITE/'frozen/protocol'))
    from secure_backend import delegated_cgroup
    delegated_cgroup()

    def status(stage, **extra):
        write(status_path,{'stage':stage,'updated_unix':time.time(),**extra})

    def run(arguments, log_path):
        with log_path.open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u',*map(str,arguments)],check=True,stdout=log,stderr=subprocess.STDOUT)

    try:
        status('preflight')
        run([SUITE/'scripts/tests_training_rollout.py'],root/'synthetic-tests.log')
        write(root/'preflight.json',{'synthetic_tests_passed':5,
            'tests_sha256':digest((SUITE/'scripts/tests_training_rollout.py').read_bytes()),
            'runner_sha256':digest((SUITE/'run_training_rollout.py').read_bytes()),
            'contract_sha256':digest(contract.read_bytes()),'production_model_calls':0})
        status('running_base',base_completed=0,checkpoint_completed=0)
        run([SUITE/'run_training_rollout.py','--model','base','--contract',contract,
             '--runtime',root/'base'],root/'base.log')
        assert read(root/'base/progress.json')['completed'] == 100
        status('waiting_for_checkpoint_merge',base_completed=100,checkpoint_completed=0)
        merged = root/'merged-checkpoint2350'
        while not (merged/'merge-receipt.json').exists():
            raw = subprocess.check_output(['systemctl','--user','show',
                'repro500-v3-merge-20261021.service','-p','ActiveState','-p','ExecMainStatus'],text=True)
            state = dict(line.split('=',1) for line in raw.splitlines())
            if state['ActiveState'] in ('inactive','failed'):
                raise RuntimeError('Checkpoint merge ended without its receipt')
            time.sleep(20)
        status('running_checkpoint',base_completed=100,checkpoint_completed=0)
        run([SUITE/'run_training_rollout.py','--model','checkpoint2350','--contract',contract,
             '--runtime',root/'checkpoint2350','--merged-checkpoint',merged],root/'checkpoint2350.log')
        assert read(root/'checkpoint2350/progress.json')['completed'] == 100
        status('exporting',base_completed=100,checkpoint_completed=100)
        run([SUITE/'scripts/compile_training_rollout.py','--root',root,
             '--output',root/'results'],root/'export.log')
        assert read(root/'results/verification.json')['passed']
        status('complete',base_completed=100,checkpoint_completed=100,
               output=str(root/'results'),extra_per_turn_8k_limit=False)
    except BaseException as error:
        status('failed',error_type=type(error).__name__,message=str(error),
               automatic_restart_or_resample=False)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True)
    args = parser.parse_args()
    main(args.root,args.contract)
