"""Test the official offline deterministic-scheduling setting with decode graphs.

No tolerance changes: use the same upstream Graph assertion and same fixtures.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import vllm_speed_tolerance as original
import vllm_decode_graphs_tolerance as decode
from official_vllm_validation import validate, official_rules, PINNED
from run_training_rollout import read, write, digest

ROOT = original.ROOT
LAB = ROOT/'vllm-sync-decode-graphs-tolerance-20261021'
FIRST = original.LAB
ENV = {'VLLM_ENABLE_V1_MULTIPROCESSING':'0'}


def worker(model,repeat,control=False):
    assert all(os.environ.get(k) == v for k,v in ENV.items())
    if control:
        original.LAB = LAB
        original.worker(model,'eager',repeat)
        path = LAB/f'{model}-eager-{repeat}.json'
    else:
        decode.LAB = LAB
        decode.worker(model,repeat)
        path = LAB/f'{model}-graph-{repeat}.json'
    row = read(path)
    row['environment'] = ENV
    row['profile'] = 'eager_sync_control' if control else 'decode_graph_sync'
    write(path,row)


def main():
    LAB.mkdir(exist_ok=False)
    official_rules()
    write(LAB/'protocol.json',{'acceptance_policy':'official-only-vllm-v0.26.0',
        'engine_profile':'decode_graph_sync','compilation_config':decode.CONFIG,
        'environment':ENV,'official_sources_sha256':PINNED,
        'gate':'Unchanged upstream Graph greedy text lowercased equality',
        'scheduling_source':'https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/usage/reproducibility.md',
        'source_sha256':digest(Path(__file__).read_bytes()),
        'fixtures_sha256':digest((FIRST/'fixtures.json').read_bytes()),
        'baseline':'Original unchanged eager controls plus a checkpoint eager control using deterministic scheduling',
        'all_attempts_retained':True,'project_epsilon':None})
    for name in ('fixtures.json','official-utils.py'):
        shutil.copyfile(FIRST/name,LAB/name)
    hashes = {}
    for model in ('base','checkpoint2350'):
        for repeat in range(2):
            name = f'{model}-eager-{repeat}.json'
            shutil.copyfile(FIRST/name,LAB/name)
            hashes[name] = digest((LAB/name).read_bytes())
    write(LAB/'baseline-sha256.json',hashes)
    environment = {**os.environ,**ENV}
    def run(model,repeat,control=False):
        label = f'{model}-'+('eager-sync-control' if control else 'graph')+f'-{repeat}'
        write(LAB/'status.json',{'stage':'running','active':label})
        with (LAB/(label+'.log')).open('w') as log:
            subprocess.run([sys.executable,'-u',__file__,'--worker',model,str(repeat),
                *(['--control'] if control else [])],check=True,env=environment,
                stdout=log,stderr=subprocess.STDOUT)
    try:
        # One new checkpoint eager process isolates the scheduling change. Keep
        # the original two controls as the deployment comparison, never replace
        # them with a convenient successful run.
        run('checkpoint2350',2,control=True)
        control = read(LAB/'checkpoint2350-eager-2.json')
        reference = read(LAB/'checkpoint2350-eager-0.json')
        equality = [a[1].lower() == b[1].lower()
                    for a,b in zip(control['greedy'],reference['greedy'],strict=True)]
        write(LAB/'scheduling-control.json',{'all_greedy_outputs_equal':all(equality),
            'per_fixture_equal':equality,'original_eager_controls_unchanged':True})
        assert all(equality), 'Deterministic scheduling changed the eager baseline'
        for model in ('base','checkpoint2350'):
            for repeat in range(2):
                run(model,repeat)
        report = validate(LAB,'decode_graph_sync')
        write(LAB/'status.json',{'stage':'complete','passed':report['passed']})
    except BaseException as exc:
        write(LAB/'status.json',{'stage':'failed','error':repr(exc),'automatic_retry':False})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--worker',nargs=2)
    p.add_argument('--control',action='store_true')
    args = p.parse_args()
    if args.worker:
        worker(args.worker[0],int(args.worker[1]),args.control)
    else:
        main()
