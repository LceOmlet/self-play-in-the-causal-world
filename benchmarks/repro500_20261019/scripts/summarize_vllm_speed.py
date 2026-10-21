"""Add descriptive diagnostics without changing the predeclared acceptance gate."""
import argparse
import statistics
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_training_rollout import read, write


def summarize(root):
    import numpy as np
    def probe_delta(a,b):
        values = []
        for x,y in zip(a['probes'],b['probes'],strict=True):
            assert {k:v for k,v in x.items() if k != 'logprob'} == {k:v for k,v in y.items() if k != 'logprob'}
            values.append(abs(x['logprob']-y['logprob']))
        return {'mean':float(np.mean(values)),'p99':float(np.percentile(values,99)),
                'max':max(values),'count':len(values)}
    result = {'official_report':read(root/'official-report.json'),'models':{},
        'scope':{'fixed_context_logprob_probes':'24 single-request contexts per engine',
                 'greedy_rank_probes':'8 prefixes in batches of four; top5 at first divergence',
                 'throughput':'four equal-length 512-token outputs; warmed; aggregate model tokens/s',
                 'not_estimated':['Full task wall-time speedup','Full-vocabulary numeric bounds',
                                  'Numerical equivalence of every future task trajectory']},
        'historical_observation':{
            'v3_first20_ate':{'minutes_per4':40.72,'model_tokens_per_case':15850.8,
                'end_to_end_model_tokens_per_second':25.95,'all_tool_seconds':70.9},
            'training20_ate_steps2251_2346':{'minutes_per4':10.72,
                'model_tokens_per_case':10233.5,'end_to_end_model_tokens_per_second':63.63},
            'caveat':'Historical groups differ in questions, weights, output lengths and batch diversity; ratio is not a controlled causal estimate.'}}
    for model in ('base','checkpoint2350'):
        runs = {(mode,r):read(root/f'{model}-{mode}-{r}.json')
                for mode in ('eager','graph') for r in range(2)}
        speeds = {}
        shapes = sorted(set(t['shape'] for r in (runs['eager',0],runs['eager',1]) for t in r['timings'])
            & set(t['shape'] for r in (runs['graph',0],runs['graph',1]) for t in r['timings']))
        for mode in ('eager','graph'):
            speeds[mode] = {'cold_start_seconds':[runs[mode,r]['startup_seconds'] for r in range(2)]}
            for shape in shapes:
                values = [t['tokens_per_second'] for r in range(2)
                          for t in runs[mode,r]['timings'] if t['shape'] == shape]
                speeds[mode][shape] = {'median':statistics.median(values),
                    'min':min(values),'max':max(values),'measurements':values}
        pairs = []
        for er in range(2):
            for gr in range(2):
                a,b = runs['eager',er],runs['graph',gr]
                prefixes = []
                for fixture,(x,y) in enumerate(zip(a['greedy'],b['greedy'],strict=True)):
                    divergence = next((i for i,(u,v) in enumerate(zip(x[0],y[0])) if u != v),None)
                    # The first output is conditioned on identical recorded prefixes.
                    common = set(x[2][0]) & set(y[2][0])
                    deltas = [abs(x[2][0][k]-y[2][0][k]) for k in common]
                    prefixes.append({'fixture':fixture,'first_divergence':divergence,
                        'exact_128_tokens':x[0] == y[0],
                        'batch4_first_step_common_top5_tokens':len(common),
                        'batch4_first_step_common_logprob_max_delta':max(deltas) if deltas else None})
                pairs.append({'eager_repeat':er,'graph_repeat':gr,'prefixes':prefixes})
        result['models'][model] = {'timing':speeds,'greedy_pairs':pairs,
            'fixed_prefix_logprob_differences':{'role':'Descriptive; no acceptance threshold',
                'eager_repeat':probe_delta(runs['eager',0],runs['eager',1]),
                'candidate_repeat':probe_delta(runs['graph',0],runs['graph',1]),
                'cross_mode':[probe_delta(runs['eager',er],runs['graph',gr])
                              for er in range(2) for gr in range(2)]},
            'clone_over_mixed':{m:speeds[m]['clone']['median']/speeds[m]['mixed']['median']
                                for m in ('eager','graph') if 'clone' in speeds[m]}}
    write(root/'analysis.json',result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True,type=Path)
    summarize(parser.parse_args().root)
