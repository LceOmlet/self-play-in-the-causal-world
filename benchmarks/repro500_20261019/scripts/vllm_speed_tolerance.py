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
    # Declare the acceptance rule before any candidate engine runs.
    write(LAB/'protocol.json', {
        'version':1,'variable':'enforce_eager only; all other engine settings identical',
        'models':['base','checkpoint2350'],'independent_process_repeats':2,
        'fixed_prefix_offsets':[0,16,128],'greedy_tokens':128,
        'throughput_tokens':512,'throughput_repeats_per_engine':2,
        'gate':{'official_vllm_reciprocal_top5_at_first_divergence':True,
                'fixed_context_selected_logprob_mean_p99_max':
                'each cross-mode statistic <= max(eager repeat, graph repeat) + 2**-7',
                'minimum_warmed_mixed_batch_speedup_each_model':1.05},
        'limitations':['Project numeric epsilon is not an official vLLM tolerance.',
            'Rank comparator stops at first divergence; no full trajectory equivalence claim.',
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


def worker(model,mode,repeat):
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
    for shape in ('mixed','clone'):
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
    import numpy as np
    tree = ast.parse((LAB/'official-utils.py').read_text())
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'check_logprobs_close')
    ns = {}
    exec('from __future__ import annotations\nimport warnings\n'+ast.unparse(function),ns)
    def rank(a,b):
        def convert(items):
            return [(x[0],x[1],[{int(k):v for k,v in s.items()} for s in x[2]]) for x in items]
        try:
            ns['check_logprobs_close'](outputs_0_lst=convert(a['greedy']),
                outputs_1_lst=convert(b['greedy']),name_0=a['mode'],name_1=b['mode'],warn_on_mismatch=False)
            return True
        except AssertionError:
            return False
    def delta(a,b):
        values = []
        for x,y in zip(a['probes'],b['probes'],strict=True):
            assert {k:v for k,v in x.items() if k != 'logprob'} == {k:v for k,v in y.items() if k != 'logprob'}
            values.append(abs(x['logprob']-y['logprob']))
        return {'mean':float(np.mean(values)),'p99':float(np.percentile(values,99)),'max':max(values)}
    models = {}
    for model in ('base','checkpoint2350'):
        e = [read(LAB/f'{model}-eager-{i}.json') for i in range(2)]
        g = [read(LAB/f'{model}-graph-{i}.json') for i in range(2)]
        controls = [delta(e[0],e[1]),delta(g[0],g[1])]
        thresholds = {k:max(c[k] for c in controls)+2**-7 for k in controls[0]}
        cross = [delta(a,b) for a in e for b in g]
        numeric = all(c[k] <= thresholds[k] for c in cross for k in thresholds)
        ranks = [rank(a,b) for a in e for b in g]
        speeds = {mode:{shape:statistics.median(t['tokens_per_second'] for r in runs
            for t in r['timings'] if t['shape'] == shape) for shape in ('mixed','clone')}
            for mode,runs in [('eager',e),('graph',g)]}
        speedup = speeds['graph']['mixed']/speeds['eager']['mixed']
        models[model] = {'controls':controls,'thresholds':thresholds,'cross_mode':cross,
            'official_rank_passes':ranks,'fixed_prefix_numeric_passed':numeric,
            'speeds':speeds,'mixed_speedup':speedup,
            'passed':numeric and all(ranks) and speedup >= 1.05}
    write(LAB/'report.json',{'models':models,'passed':all(m['passed'] for m in models.values())})


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
        write(LAB/'status.json',{'stage':'complete','passed':read(LAB/'report.json')['passed']})
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
