"""CUDA decode graphs with eager operators, checked by official Graph test logic.

Reuses the exact eager controls and fixed inputs. No project numeric tolerance.
Starts GPU work only if the compiled candidate fails the official compile rule.
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
import vllm_speed_tolerance as original
from official_vllm_validation import validate, official_rules, PINNED
from run_training_rollout import BASE, write, read, digest

FIRST = original.LAB
LAB = FIRST.parent/'vllm-decode-graphs-tolerance-20261021'
CONFIG = {'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY'}


def worker(model,repeat):
    # Same measurements and ordering as original.worker, with explicit eager
    # operators + decode graphs replacing default Inductor compilation.
    import math
    from vllm import LLM, SamplingParams
    fixtures = read(LAB/'fixtures.json')
    started = time.monotonic()
    llm = LLM(model=BASE if model == 'base' else str(original.OLD/'merged-checkpoint2350'),
        tokenizer=BASE,dtype='bfloat16',max_model_len=32768,max_num_seqs=4,
        gpu_memory_utilization=.75,max_num_batched_tokens=4096,seed=0,
        enable_prefix_caching=True,model_impl='vllm',enforce_eager=False,
        compilation_config=CONFIG,logprobs_mode='processed_logprobs',
        limit_mm_per_prompt={'image':0,'video':0})
    startup = time.monotonic()-started
    def generate(prefixes,params):
        return llm.generate([{'prompt_token_ids':p} for p in prefixes],params,use_tqdm=False)
    def serialize(o):
        c = o.outputs[0]
        return [list(c.token_ids),c.text,[{str(k):v.logprob for k,v in step.items()}
                                       for step in c.logprobs]]
    greedy = []
    for i in range(0,len(fixtures),4):
        greedy.extend(serialize(o) for o in generate([f['prefix'] for f in fixtures[i:i+4]],
            SamplingParams(temperature=0,max_tokens=128,ignore_eos=True,logprobs=5)))
    probes = []
    for f in fixtures:
        for offset in (0,16,128):
            prefix = f['prefix']+f['continuation'][:offset]
            target = f['continuation'][offset]
            o = generate([prefix],SamplingParams(temperature=1,max_tokens=1,
                logprobs=1,logprob_token_ids=[target]))[0]
            value = o.outputs[0].logprobs[0][target].logprob
            assert math.isfinite(value)
            probes.append({'fixture':f['id'],'offset':offset,'target':target,
                           'prefix_sha256':digest(__import__('json').dumps(prefix).encode()),'logprob':value})
    timings = []
    for shape in ('mixed','clone'):
        prefixes = [f['prefix'] for f in fixtures[:4]]
        if shape == 'clone':
            prefixes = [prefixes[0]]*4
        params = SamplingParams(temperature=1,top_p=1,top_k=0,seed=42,
            max_tokens=512,min_tokens=512,ignore_eos=True,logprobs=0)
        generate(prefixes,params)
        for r in range(2):
            t = time.monotonic()
            outputs = generate(prefixes,params)
            elapsed = time.monotonic()-t
            tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            assert tokens == 2048
            timings.append({'shape':shape,'repeat':r,'seconds':elapsed,
                            'tokens':tokens,'tokens_per_second':tokens/elapsed})
    write(LAB/f'{model}-graph-{repeat}.json',{'model':model,'mode':'graph','repeat':repeat,
        'profile':'decode_graph','compilation_config_requested':CONFIG,
        'fixtures_sha256':digest((LAB/'fixtures.json').read_bytes()),
        'startup_seconds':startup,'greedy':greedy,'probes':probes,'timings':timings,
        'compilation_config':str(llm.llm_engine.vllm_config.compilation_config)})


def main():
    LAB.mkdir(exist_ok=False)
    official_rules()
    write(LAB/'protocol.json',{'engine_profile':'decode_graph','compilation_config':CONFIG,
        'acceptance_policy':'official-only-vllm-v0.26.0',
        'gate':'Official TestFullCUDAGraph greedy response text equality after lowercasing',
        'official_sources_sha256':PINNED,'project_epsilon':None,
        'fixtures_sha256':digest((FIRST/'fixtures.json').read_bytes()),
        'baseline':'Reuse exact two independent eager controls per model from first experiment',
        'source_sha256':digest(Path(__file__).read_bytes()),
        'hypothesis':'Keep eager operator arithmetic; remove per-token CPU kernel launch overhead using decode graphs.'})
    write(LAB/'status.json',{'stage':'waiting_for_first_experiment'})
    try:
        while True:
            state = read(FIRST/'status.json')
            if state['stage'] == 'failed':
                raise RuntimeError('Original experiment infrastructure failed')
            if state['stage'] == 'complete':
                break
            time.sleep(20)
        if validate(FIRST,'graph')['passed']:
            write(LAB/'status.json',{'stage':'not_needed','reason':'Compiled candidate passed official compile rule'})
            return
        assert digest((original.SUITE/'scripts/vllm_speed_tolerance.py').read_bytes()) == read(FIRST/'sources.json')['script_sha256']
        for name in ('fixtures.json','official-utils.py'):
            shutil.copyfile(FIRST/name,LAB/name)
        baselines = {}
        for model in ('base','checkpoint2350'):
            for repeat in range(2):
                name = f'{model}-eager-{repeat}.json'
                shutil.copyfile(FIRST/name,LAB/name)
                baselines[name] = digest((LAB/name).read_bytes())
        write(LAB/'baseline-sha256.json',baselines)
        for model in ('base','checkpoint2350'):
            for repeat in range(2):
                name = f'{model}-graph-{repeat}'
                write(LAB/'status.json',{'stage':'running','active':name})
                with (LAB/f'{name}.log').open('w') as log:
                    subprocess.run([sys.executable,'-u',__file__,'--worker',model,str(repeat)],
                                   check=True,stdout=log,stderr=subprocess.STDOUT)
        report = validate(LAB,'decode_graph')
        write(LAB/'status.json',{'stage':'complete','passed':report['passed']})
    except BaseException as exc:
        write(LAB/'status.json',{'stage':'failed','error':repr(exc),'automatic_retry':False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker',nargs=2)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker[0],int(args.worker[1]))
    else:
        main()
