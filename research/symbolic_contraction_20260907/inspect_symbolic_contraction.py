import json
import math
from pathlib import Path
import pickle
import sys

from cpt_world import counterfactual_solver as s
from cpt_world import query_truth as q

ROOT = Path('/home/chen/runs/symbolic-contraction-20260907')
index = int(sys.argv[1])
world, seed = pickle.loads((Path('/home/chen/runs/kernel-root-cause-20260907') / f'task-{index}.pkl').read_bytes())
owner = None
original = s._SparseResponseModel._build_twin_probability


class Captured(BaseException):
    pass


def build(self):
    global owner
    owner = self
    return original(self)


def capture(world, model, factors, tokens, local_factor, **kwargs):
    involving = [factor for factor in factors if any(t in factor.scope for t in tokens)] + [local_factor]
    union = tuple(sorted({t for factor in involving for t in factor.scope} - set(tokens)))
    report = {
        'index': index, 'seed_id': seed['seed_id'], 'domains': world.domains,
        'parents': world.parents, 'edges': world.edges,
        'treatment': owner.treatment, 'outcome': owner.outcome,
        'outcome_events': owner.outcome_events,
        'terminal_event_endpoint': owner.terminal_event_endpoint,
        'baseline_value': owner.baseline_value, 'treatment_value': owner.treatment_value,
        'affected': owner.affected, 'shared': owner.shared,
        'mechanism_affected': owner.mechanism_affected,
        'context_counts': {n:len(c) for n,c in owner.contexts.items()},
        'context_components': {
            n: [{'contexts': len(c.indices), 'edges': len(c.edges), 'forest': c.forest}
                for c in components]
            for n, components in owner.context_components.items()
        },
        'tokens': tokens, 'message_scope': union,
        'message_cells': math.prod(world.domains[t[0]] for t in union),
        'downstream_scopes': [f.scope for f in factors], 'local_scope':local_factor.scope,
        'variables_before_contraction': model.getNVars(),
    }
    (ROOT / f'case-{index}-first-contraction.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise Captured()


s._SparseResponseModel._build_twin_probability = build
s._eliminate_factor_tokens = capture
try:
    q.compute_query_truth(world, seed, counterfactual_endpoint_time_limit_seconds=5.0)
except Captured:
    pass
