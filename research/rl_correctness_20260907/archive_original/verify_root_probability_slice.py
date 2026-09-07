"""Independent enumeration checks for the root-CPT slice theorem."""
import dataclasses
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
import sys
import types

from root_probability_slice import affine, certified_slice, relation

ROOT = Path(__file__).resolve().parent
project = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent/'rl_algorithm_fix_20260906/project'
package = types.ModuleType('cpt_world')
package.__path__ = [str(project/'src/cpt_world')]
sys.modules['cpt_world'] = package
from cpt_world import world_space as w
from cpt_world.query_truth import interventional_probability, worldspec_projected_interventional_distribution


def enumerate_probability(world, interventions, event):
    total = F(0)
    for state in product(*(range(domain) for domain in world.domains)):
        if any(state[n] != x for n, x in interventions.items()):
            continue
        if any(state[n] != x for n, x in event.items()):
            continue
        weight = F(1)
        for node in range(len(world.variables)):
            if node in interventions:
                continue
            row = 0
            for parent in world.parents[node]:
                row = row*world.domains[parent] + state[parent]
            weight *= world.cpt[node][row][state[node]]
        total += weight
    return total


def root_pair_world(world, z, pair, rho):
    row = list(world.cpt[z][0])
    total = row[pair[0]] + row[pair[1]]
    row[pair[0]], row[pair[1]] = total*(1-rho), total*rho
    return dataclasses.replace(world, cpt={**world.cpt, z: (tuple(row),)})


def endpoints(world, z, x, y, pair):
    worlds = [root_pair_world(world, z, pair, r) for r in (0, 1)]
    causal = [[interventional_probability(concrete, {x: a}, y, 1) for concrete in worlds] for a in range(world.domains[x])]
    joints, masses = [], []
    for concrete in worlds:
        law = dict(worldspec_projected_interventional_distribution(concrete, {}, (x, y)))
        joints.append([law[a, 1] for a in range(world.domains[x])])
        masses.append([sum(law[a, b] for b in range(world.domains[y])) for a in range(world.domains[x])])
    return causal, list(map(list, zip(*joints))), list(map(list, zip(*masses)))


def fixture(actions, variant, root_states):
    # A second root W and arbitrary X/W interactions in Y defeat the symmetric
    # three-node shortcut. All mechanisms are strictly positive rationals.
    domains = (root_states, 2, actions, 2)
    cpt = {0: ((F(2, 5), F(3, 5)),), 1: ((F(3, 7), F(4, 7)),)}
    if root_states == 3:
        cpt[0] = ((F(2, 7), F(3, 7), F(2, 7)),)
    x_rows = []
    for z, v in product(range(root_states), range(2)):
        weights = [1 + ((a + 2)*(z + 3) + 5*v + variant*(a + 1)) % 19 for a in range(actions)]
        x_rows.append(tuple(F(p, sum(weights)) for p in weights))
    cpt[2] = tuple(x_rows)
    cpt[3] = tuple((1-F(1+(variant*7+z*13+v*3+a*5+z*a*11) % 31, 33),
                    F(1+(variant*7+z*13+v*3+a*5+z*a*11) % 31, 33))
                   for z, v, a in product(range(root_states), range(2), range(actions)))
    return w.WorldSpec(family='root-slice-proof', topology='Z,W -> X -> Y with Z,W -> Y',
                       variables=('Z', 'W', 'X', 'Y'), domains=domains,
                       state_names=tuple(tuple(str(i) for i in range(d)) for d in domains),
                       edges=((0, 2), (1, 2), (0, 3), (1, 3), (2, 3)),
                       parents={0: (), 1: (), 2: (0, 1), 3: (0, 1, 2)}, cpt=cpt)


def serial(value):
    if isinstance(value, F):
        return {'rational': str(value), 'float': float(value)}
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


records = []
comparisons = 0
threshold = F(w.BEST_INTERVENTION_STRONG_REVERSAL_MIN_GAP)
for actions, variant, root_states in product((2, 3, 4), range(1, 13), (2, 3)):
    world = fixture(actions, variant, root_states)
    pair = (0, root_states-1)
    causal, joints, masses = endpoints(world, 0, 2, 3, pair)
    lines = [list(map(affine, item)) for item in (causal, joints, masses)]
    for objective in ('maximize', 'minimize'):
        certificate = certified_slice(causal, joints, masses, threshold=threshold, objective=objective)
        assert certificate['uncertain_mass'] <= certificate['polynomials'] * 2 * certificate['isolator_width']
        for rho in [F(i, 29) for i in range(1, 29)]:
            concrete = root_pair_world(world, 0, pair, rho)
            mu = [enumerate_probability(concrete, {2: a}, {3: 1}) for a in range(actions)]
            nu = [enumerate_probability(concrete, {}, {2: a, 3: 1}) /
                  enumerate_probability(concrete, {}, {2: a}) for a in range(actions)]
            choose = max if objective == 'maximize' else min
            optimum = choose(mu)
            observed = {a for a in range(actions) if nu[a] == choose(nu)}
            truth = not any(mu[a] == optimum for a in observed) and abs(optimum-choose(mu[a] for a in observed)) >= threshold
            assert relation(*lines, rho, objective, threshold) == truth
            accepted = any(lo < rho < hi for lo, hi in certificate['accepted'])
            rejected = any(lo < rho < hi for lo, hi in certificate['rejected'])
            uncertain = any(lo <= rho <= hi for lo, hi in certificate['root_boxes'])
            assert accepted or rejected or uncertain
            if not uncertain:
                assert accepted == truth
            comparisons += 1
        records.append({'actions': actions, 'variant': variant, 'root_states': root_states, 'objective': objective, 'certificate': serial(certificate)})
result = {'passed': True, 'worlds': 72, 'slices': len(records), 'independent_enumeration_comparisons': comparisons,
          'actual_production_threshold': float(threshold),
          'nonempty_slices': sum(r['certificate']['mass_lower']['float'] > 0 for r in records),
          'max_uncertain_mass': max(r['certificate']['uncertain_mass']['float'] for r in records),
          'records': records, 'integrated_into_sampler': False}
(ROOT/'root-slice-acceptance.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))
