"""Inspect the frozen structural class without executing counterfactual search."""
import json
import pickle
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent / 'rl_algorithm_fix_20260906/project'
DATA = ROOT.parent / 'kernel_root_cause_20260907'
pkg = types.ModuleType('cpt_world')
pkg.__path__ = [str(PROJECT / 'src/cpt_world')]
sys.modules['cpt_world'] = pkg
from cpt_world import query_truth as q

rows = []
for i in range(74):
    world, seed = pickle.loads((DATA / f'task-{i}.pkl').read_bytes())
    query = seed['query']
    x = q._resolve_seed_node(world, seed, query['treatment'])
    y = q._resolve_seed_node(world, seed, query['outcome'])
    affected = q._descendants(world, x) & (q._ancestors(world, y) | {y})
    if len(affected) != 2 or x in world.parents[y]:
        continue
    m = next(iter(affected - {y}))
    rows.append({'index':i, 'treatment':x, 'mediator':m, 'outcome':y,
                 'mediator_parents':world.parents[m], 'outcome_parents':world.parents[y],
                 'mediator_states':world.domains[m], 'outcome_states':world.domains[y],
                 'unconditioned_mediator':world.parents[m] == (x,)})
(ROOT / 'indirect-chain-structure.json').write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
