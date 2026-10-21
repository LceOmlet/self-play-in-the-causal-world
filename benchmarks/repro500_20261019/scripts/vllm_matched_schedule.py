"""Isolate graph arithmetic using eager and graph engines with identical scheduling.

The original asynchronous reference is checked separately by the upstream
hybrid batching rule. Graph comparisons retain the exact upstream text rule.
"""
import argparse
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
FIRST = original.LAB
LAB = ROOT/'vllm-fixed-schedule-graphs-20261021'
ENV = {'VLLM_ENABLE_V1_MULTIPROCESSING':'0'}


def worker(model,mode,repeat,numeric_only=False):
    assert all(os.environ.get(k) == v for k,v in ENV.items())
    shapes = () if numeric_only else ('mixed',)
    if mode == 'eager':
        original.LAB = LAB
        original.worker(model,mode,repeat,shapes=shapes)
    else:
        decode.LAB = LAB
        decode.worker(model,repeat,shapes=shapes)
    path = LAB/f'{model}-{mode}-{repeat}.json'
    row = read(path)
    row.update(environment=ENV,profile='eager_sync' if mode == 'eager' else 'decode_graph_sync',
               numeric_only=numeric_only)
    write(path,row)


def main():
    LAB.mkdir(exist_ok=False)
    official_rules()
    write(LAB/'protocol.json',{'acceptance_policy':'official-only-vllm-v0.26.0',
        'engine_profile':'decode_graph_sync','compilation_config':decode.CONFIG,'environment':ENV,
        'official_sources_sha256':PINNED,'source_sha256':digest(Path(__file__).read_bytes()),
        'numerical_gate':'Eager and graph with identical fixed scheduling: official Graph text equality.',
        'scheduling_bridge':'Original asynchronous eager versus fixed-schedule eager: official hybrid test_batching check_logprobs_close.',
        'project_epsilon':None,'same_fixtures_all_comparisons':True,
        'independent_eager_and_graph_processes_per_model':2,
        'timing':'One warmed eager engine and two warmed graph engines per model; mixed4, 512 tokens, two timed repetitions.',
        'earlier_attempts':'Retained, never overwritten or used as replacement successful samples.'})
    for name in ('fixtures.json','official-utils.py'):
        shutil.copyfile(FIRST/name,LAB/name)
    references = LAB/'asynchronous-reference'
    references.mkdir()
    provenance = {}
    for model in ('base','checkpoint2350'):
        for repeat in range(2):
            name = f'{model}-eager-{repeat}.json'
            shutil.copyfile(FIRST/name,references/name)
            provenance['asynchronous-reference/'+name] = {'source':str(FIRST/name),'sha256':digest((FIRST/name).read_bytes())}
    # Retain the first actual fixed-schedule checkpoint control, even though it
    # differed from the old asynchronous reference. No success-based selection.
    previous = ROOT/'vllm-sync-decode-graphs-tolerance-20261021/checkpoint2350-eager-2.json'
    assert read(previous)['environment'] == ENV
    shutil.copyfile(previous,LAB/'checkpoint2350-eager-0.json')
    provenance['checkpoint2350-eager-0.json'] = {'source':str(previous),'sha256':digest(previous.read_bytes())}
    write(LAB/'reference-provenance.json',provenance)
    environment = {**os.environ,**ENV}
    def run(model,mode,repeat,numeric_only=False):
        label = f'{model}-{mode}-{repeat}'
        write(LAB/'status.json',{'stage':'running','active':label})
        with (LAB/(label+'.log')).open('w') as log:
            subprocess.run([sys.executable,'-u',__file__,'--worker',model,mode,str(repeat),
                *(['--numeric-only'] if numeric_only else [])],check=True,env=environment,
                stdout=log,stderr=subprocess.STDOUT)
    try:
        run('checkpoint2350','eager',1,True)
        a,b = [read(LAB/f'checkpoint2350-eager-{i}.json') for i in range(2)]
        same = [x[1].lower() == y[1].lower() for x,y in zip(a['greedy'],b['greedy'],strict=True)]
        write(LAB/'fixed-schedule-repeat-control.json',{'all_equal':all(same),'per_fixture':same})
        assert all(same), 'Fixed-schedule eager control is not repeatable'
        for repeat in range(2):
            run('base','eager',repeat,repeat == 1)
        for model in ('base','checkpoint2350'):
            for repeat in range(2):
                run(model,'graph',repeat)
        report = validate(LAB,'decode_graph_sync')
        write(LAB/'status.json',{'stage':'complete','passed':report['passed']})
    except BaseException as exc:
        write(LAB/'status.json',{'stage':'failed','error':repr(exc),'automatic_retry':False})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--worker',nargs=3)
    p.add_argument('--numeric-only',action='store_true')
    args = p.parse_args()
    if args.worker:
        worker(args.worker[0],args.worker[1],int(args.worker[2]),args.numeric_only)
    else:
        main()
