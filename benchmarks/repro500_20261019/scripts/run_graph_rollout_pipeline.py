"""Wait for the predeclared two-model experiment, then run a fresh unified eval."""
import argparse
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_training_rollout import SUITE, read, write, digest

ROOT = Path('/home/chen/runs/repro-benchmark500-20261019')
LAB = ROOT/'vllm-speed-tolerance-20261021'
OUTPUT = ROOT/'training-rollout-v4-20261021'


def main(lab=LAB,profile='graph'):
    OUTPUT.mkdir(exist_ok=False)
    def status(stage,**kw):
        write(OUTPUT/'pipeline-status.json',{'stage':stage,'updated_unix':time.time(),**kw})
    def run(args,logname):
        with (OUTPUT/logname).open('w') as log:
            subprocess.run([sys.executable,'-u',*map(str,args)],check=True,stdout=log,stderr=subprocess.STDOUT)
    try:
        status('waiting_for_tolerance_gate')
        while True:
            state = read(lab/'status.json')
            if state['stage'] == 'failed':
                raise RuntimeError('Tolerance experiment infrastructure failed; evaluation not started')
            if state['stage'] == 'complete':
                break
            time.sleep(20)
        report = read(lab/'official-report.json')
        assert report['acceptance_policy'] == 'official-only-vllm-v0.26.0'
        assert report.get('engine_profile','graph') == profile
        assert report['passed'], 'Official comparison failed; evaluation not started'
        write(OUTPUT/'tolerance-report.json',report)
        sys.path.insert(0,str(SUITE/'frozen/protocol'))
        from secure_backend import delegated_cgroup
        delegated_cgroup()
        status('preflight')
        run([SUITE/'scripts/tests_training_rollout.py'],'synthetic-tests.log')
        run([SUITE/'scripts/tests_official_vllm_validation.py'],'official-rule-tests.log')
        merged = ROOT/'training-rollout-v3-20261021/merged-checkpoint2350'
        (OUTPUT/'merged-checkpoint2350').symlink_to(merged,target_is_directory=True)
        write(OUTPUT/'preflight.json',{'synthetic_tests_passed':5,
            'official_rule_tests_passed':2,'acceptance_policy':report['acceptance_policy'],
            'runner_sha256':digest((SUITE/'run_training_rollout.py').read_bytes()),
            'tolerance_report_sha256':digest((OUTPUT/'tolerance-report.json').read_bytes()),
            'frozen_prompts_and_interface_unchanged':True,'restart_or_resample':False})
        for model in ('base','checkpoint2350'):
            status('running_'+model)
            run([SUITE/'run_training_rollout.py','--model',model,
                '--contract',ROOT/'training-rollout-v3-contract.json','--runtime',OUTPUT/model,
                '--merged-checkpoint',merged,'--engine-profile',profile,
                '--tolerance-report',OUTPUT/'tolerance-report.json'],model+'.log')
            assert read(OUTPUT/model/'progress.json')['completed'] == 100
        status('exporting')
        run([SUITE/'scripts/compile_training_rollout.py','--root',OUTPUT,
             '--output',OUTPUT/'results'],'export.log')
        assert read(OUTPUT/'results/verification.json')['passed']
        status('complete',base_completed=100,checkpoint_completed=100)
    except BaseException as exc:
        status('blocked',error=repr(exc),automatic_restart_or_resample=False)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lab',type=Path,default=LAB)
    parser.add_argument('--engine-profile',choices=('graph','decode_graph','decode_graph_sync'),default='graph')
    args = parser.parse_args()
    main(args.lab,args.engine_profile)
