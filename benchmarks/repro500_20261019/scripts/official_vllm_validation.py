"""Replay official v0.26 compilation/Graph comparison logic on measured outputs.

No project epsilon, repeat-noise allowance, or kernel-level tensor tolerance.
This ports the official comparison to our actual model weights and prefixes;
it does not claim that upstream pytest parametrizes this particular model.
"""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_training_rollout import BASE, digest, read, write

FIRST = Path('/home/chen/runs/repro-benchmark500-20261019/vllm-speed-tolerance-20261021')
SOURCES = FIRST/'official-applicability'
PINNED = {
    'tests__utils.py':'cf6d6006d9084e2f3025c3a48c64fd07a51aee1be17c97bd2e4e9cabfdb7b602',
    'tests__compile__fullgraph__test_basic_correctness.py':'1ed4461ab5b88b80ce45072a43c48b4391959ce49d1e24b2217169b0250865a5',
    'tests__compile__fullgraph__test_full_cudagraph.py':'cb48fec6e04f054c1a772c73030c7a16ed3a719dbabebc6cbffe89c5665f38a6',
}


def official_rules():
    for name,expected in PINNED.items():
        assert digest((SOURCES/name).read_bytes()) == expected,name
    tree = ast.parse((SOURCES/'tests__utils.py').read_text())
    helper = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == '_test_completion_close')
    ns = {}
    exec('from __future__ import annotations\n'+ast.unparse(helper),ns)
    tree = ast.parse((SOURCES/'tests__compile__fullgraph__test_full_cudagraph.py').read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'TestFullCUDAGraph')
    test = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == 'test_full_cudagraph')
    assertion = next(n for n in ast.walk(test) if isinstance(n,ast.Assert))
    graph_rule = compile(ast.Expression(assertion.test),'<pinned official graph assertion>','eval')
    return ns['_test_completion_close'],graph_rule


def rounded_result(helper,top5):
    # Replay only the response consumed by the unchanged upstream helper. Actual
    # data come from the measured first token, never fabricated probabilities.
    def create(**kw):
        assert kw['max_tokens'] == 1 and kw['logprobs'] == 5 and kw['temperature'] == 0
        return NS(choices=[NS(logprobs=NS(top_logprobs=[top5]))])
    return helper(NS(completions=NS(create=create)),model=BASE,prompt='recorded fixed prefix')


def validate(root,profile):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    helper,graph_rule = official_rules()
    def top5(choice):
        # Match the default completion API's decoded-token dictionary keys.
        return {tokenizer.decode([int(k)]):v for k,v in choice[2][0].items()}
    models = {}
    for model in ('base','checkpoint2350'):
        e = [read(root/f'{model}-eager-{i}.json') for i in range(2)]
        g = [read(root/f'{model}-graph-{i}.json') for i in range(2)]
        assert len({r['fixtures_sha256'] for r in e+g}) == 1
        pairs = []
        for er,a in enumerate(e):
            for gr,b in enumerate(g):
                rows = []
                for i,(x,y) in enumerate(zip(a['greedy'],b['greedy'],strict=True)):
                    assert len(x[0]) == len(y[0]) == 128
                    lhs,rhs = rounded_result(helper,top5(x)),rounded_result(helper,top5(y))
                    graph_pass = eval(graph_rule,{}, {
                        'piecewise_res':NS(outputs=[NS(text=x[1])]),
                        'full_res':NS(outputs=[NS(text=y[1])])})
                    rows.append({'fixture':i,'compile_rounded_top5_passed':lhs == rhs,
                        'compile_reference':lhs,'compile_candidate':rhs,
                        'graph_greedy_text_passed':bool(graph_pass),
                        'exact_greedy_token_ids':x[0] == y[0]})
                pairs.append({'eager_repeat':er,'candidate_repeat':gr,'fixtures':rows})
        # Compiler arithmetic may change greedy text, as its upstream test
        # explicitly notes. For a graph-only change keep the compiler unchanged
        # and use the upstream Graph test's text equality instead.
        key = 'compile_rounded_top5_passed' if profile == 'graph' else 'graph_greedy_text_passed'
        passed = all(row[key] for pair in pairs for row in pair['fixtures'])
        speed = {mode:statistics.median(t['tokens_per_second'] for r in runs
                    for t in r['timings'] if t['shape'] == 'mixed')
                 for mode,runs in [('eager',e),('candidate',g)]}
        models[model] = {'passed':passed,'official_rule_applied':key,
            'pairs':pairs,'mixed_tokens_per_second':speed,
            'mixed_speedup':speed['candidate']/speed['eager']}
    result = {'acceptance_policy':'official-only-vllm-v0.26.0',
        'engine_profile':profile,'passed':all(m['passed'] for m in models.values()),
        'compilation_config':{'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY'}
                             if profile == 'decode_graph' else 'vLLM default',
        'official_sources_sha256':PINNED,'models':models,
        'scope':{'compile_rule':'Unchanged _test_completion_close: top5 logprobs rounded to 2 decimals, whole dictionary equal.',
            'graph_rule':'Unchanged TestFullCUDAGraph assertion: greedy response text lowercased, then exactly equal.',
            'adaptation':'Actual 9B weights, 8 recorded token prefixes, batch size 4, first-token compile comparison and 128-token graph comparison; upstream fixtures/models replaced.',
            'no_universal_epsilon':True,'repeat_noise_is_not_an_acceptance_allowance':True,
            'not_claimed':['Upstream pytest includes Qwen3.5-9B in these parametrizations',
                'Whole-vocabulary raw-logit tolerance','Equivalence of arbitrary sampled full task trajectories']}}
    write(root/'official-report.json',result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--profile',choices=('graph','decode_graph'),required=True)
    args = p.parse_args()
    validate(args.root,args.profile)
