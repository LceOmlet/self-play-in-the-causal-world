"""Fixed-prefix eager/graph experiment; no task owner, grading or tool execution."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

SUITE = Path(__file__).resolve().parents[1]
ROOT = Path('/home/chen/runs/repro-benchmark500-20261019')
OLD = ROOT/'training-rollout-v3-20261021'
LAB = ROOT/'vllm-speed-tolerance-20261021'
sys.path.insert(0, str(SUITE))
from run_training_rollout import BASE, write, read, digest


def prepare():
    LAB.mkdir(exist_ok=False)
    # Numerical acceptance comes only from the applicable pinned upstream rule.
    write(LAB/'protocol.json', {
        'version':1,'variable':'enforce_eager only; all other engine settings identical',
        'models':['base','checkpoint2350'],'independent_process_repeats':2,
        'fixed_prefix_offsets':[0,16,128],'greedy_tokens':128,
        'throughput_tokens':512,'throughput_repeats_per_engine':2,
        'gate':'Pinned official compile generate_close comparison; see official_vllm_validation.py',
        'fixed_context_logprob_differences':'Descriptive only; no project epsilon or repeat-noise acceptance margin.',
        'limitations':['No universal epsilon is asserted.',
            'Selected-token processed logprobs do not bound full-vocabulary raw logits.',
            'Fixtures reuse only recorded public token contexts; no labels or grades read.',
            'Diagnostic 128/512-token limits do not apply to the 31497-token evaluation.']})
    candidates = []
    for path in sorted((OLD/'base/cases').glob('*/events.jsonl')):
        for line in path.read_text().splitlines():
            e = json.loads(line)
            if e['kind'] != 'generation_response':
                continue
            continuation = e['outputs'][0]['token_ids']
            if len(continuation) < 129:
                continue
            candidates.append({'source':str(path),'source_sha256':digest(path.read_bytes()),
                'turn':e['turn'],'prefix':e['prompt_token_ids'], 'continuation':continuation})
    # Four short actual contexts, alternating available families, plus long contexts.
    short = []
    for family in ('ate-','individual_counterfactual_probability-'):
        options = [c for c in candidates if Path(c['source']).parent.name.startswith(family)
                   and c['turn'] == 1]
        short.extend(options[:2])
    assert len(short) == 4
    fixtures = [{**c,'continuation':c['continuation'][:129]} for c in short]
    for length in (8000,16000,24000,30000):
        c = next(c for c in candidates if len(c['prefix']) <= length
                 and len(c['prefix'])+len(c['continuation']) >= length+129)
        joined = c['prefix']+c['continuation']
        fixtures.append({**c,'prefix':joined[:length],'continuation':joined[length:length+129]})
    for i,f in enumerate(fixtures):
        f['id'] = i
        f['prefix_sha256'] = digest(json.dumps(f['prefix']).encode())
    write(LAB/'fixtures.json',fixtures)
    source = Path('/home/chen/runs/official-grpo-rollout-repair-20260928/upstream-inference/utils.py')
    (LAB/'official-utils.py').write_bytes(source.read_bytes())
    write(LAB/'sources.json',{'script_sha256':digest(Path(__file__).read_bytes()),
        'fixture_sha256':digest((LAB/'fixtures.json').read_bytes()),
        'official_utils_sha256':digest(source.read_bytes()),
        'official_source':'https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/tests/models/utils.py'})


def worker(model,mode,repeat,*,shapes=('mixed','clone')):
    from vllm import LLM, SamplingParams
    fixtures = read(LAB/'fixtures.json')
    started = time.monotonic()
    llm = LLM(model=BASE if model == 'base' else str(OLD/'merged-checkpoint2350'),
        tokenizer=BASE,dtype='bfloat16',max_model_len=32768,max_num_seqs=4,
        gpu_memory_utilization=.75,max_num_batched_tokens=4096,seed=0,
        enable_prefix_caching=True,model_impl='vllm',enforce_eager=mode == 'eager',
        logprobs_mode='processed_logprobs',limit_mm_per_prompt={'image':0,'video':0})
    startup = time.monotonic()-started
    def generate(prefixes, params):
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
                           'prefix_sha256':digest(json.dumps(prefix).encode()),'logprob':value})
    timings = []
    for shape in shapes:
        prefixes = [f['prefix'] for f in fixtures[:4]]
        if shape == 'clone':
            prefixes = [prefixes[0]]*4
        params = SamplingParams(temperature=1,top_p=1,top_k=0,seed=42,
            max_tokens=512,min_tokens=512,ignore_eos=True,logprobs=0)
        generate(prefixes,params)  # warm-up excluded
        for r in range(2):
            t = time.monotonic()
            outputs = generate(prefixes,params)
            elapsed = time.monotonic()-t
            tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            assert tokens == 2048
            timings.append({'shape':shape,'repeat':r,'seconds':elapsed,
                            'tokens':tokens,'tokens_per_second':tokens/elapsed})
    write(LAB/f'{model}-{mode}-{repeat}.json',{'model':model,'mode':mode,'repeat':repeat,
        'fixtures_sha256':digest((LAB/'fixtures.json').read_bytes()),
        'startup_seconds':startup,'greedy':greedy,'probes':probes,'timings':timings,
        'compilation_config':str(llm.llm_engine.vllm_config.compilation_config)})


def compare():
    from official_vllm_validation import validate
    return validate(LAB,'graph')


def orchestrate():
    prepare()
    try:
        for model in ('base','checkpoint2350'):
            for repeat in range(2):
                for mode in ('eager','graph'):
                    name = f'{model}-{mode}-{repeat}'
                    write(LAB/'status.json',{'stage':'running','active':name})
                    with (LAB/f'{name}.log').open('w') as log:
                        subprocess.run([sys.executable,'-u',__file__,'--worker',model,mode,str(repeat)],
                                       check=True,stdout=log,stderr=subprocess.STDOUT)
        compare()
        write(LAB/'status.json',{'stage':'complete','passed':read(LAB/'official-report.json')['passed']})
    except BaseException as exc:
        write(LAB/'status.json',{'stage':'failed','error':repr(exc),'automatic_retry':False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker',nargs=3)
    parser.add_argument('--compare',action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker[0],args.worker[1],int(args.worker[2]))
    elif args.compare:
        compare()
    else:
        orchestrate()
