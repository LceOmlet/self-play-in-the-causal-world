"""Find and certify the optimal legal experiment for the existing weak-edge witness.

This strengthens the earlier generic KL upper bound using the true scalar cost.
It is a worst-case two-world theorem, not the frequency of hard generated tasks.
"""
from fractions import Fraction as F
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = Path('/home/chen/projects/self-play-in-the-causal-world-rewardv10-a92ab8e-20260831')
sys.path.insert(0, str(PROJECT/'src'))
from cpt_world import OutcomeTape, WorldSpecEpisode
from cpt_world.identification import INTERACTION_SURFACE_VERSION
from cpt_world.query_truth import worldspec_interventional_distribution
from cpt_world.world_space import assemble_seed
import mpmath as mp

source = Path('/home/chen/runs/environment-validation-20260907/prove_finite_budget_limit.py')
spec = importlib.util.spec_from_file_location('frozen_witness', source)
witness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(witness)
delta, budget = F(1, 10000), 8*2**14
worlds = [witness.world(F(0)), witness.world(delta)]
actions = [{}] + [{node: state} for node in range(8) if node not in (0, 3) for state in range(2)]
single_checks, full_likelihood_checks = 0, 0
for action in actions:
    laws = [dict(worldspec_interventional_distribution(world, action)) for world in worlds]
    for node in range(8):
        marginals = [tuple(sum(p for state, p in law.items() if state[node] == value) for value in range(2)) for law in laws]
        assert marginals[0] == marginals[1]
        single_checks += 1
    if 2 in action:  # do(Z) deletes precisely the only differing CPT.
        assert laws[0] == laws[1]
    else:
        for state, base in laws[0].items():
            if base == 0:
                assert laws[1][state] == 0
                continue
            expected = 1 + (2*delta if state[0] == state[2] else -2*delta)
            assert laws[1][state] / base == expected
            full_likelihood_checks += 1
        pairs = [{(x, z): sum(p for state, p in law.items() if (state[0], state[2]) == (x, z))
                  for x in range(2) for z in range(2)} for law in laws]
        assert all(p == F(1, 4) for p in pairs[0].values())
        assert sum(p for (x, z), p in pairs[1].items() if x == z) == F(1, 2)+delta

# Check the proposed maximally informative two-scalar measurement through the
# real episode protocol, including the readonly anchor constraint and fee.
episode_records = []
for world in worlds:
    seed = assemble_seed(world, 'mechanism_hidden', 'mediator_set', 'discovery',
                         seed_id='FINITE-BUDGET-SAME-PUBLIC-PROMPT', anchors={'treatment': 0, 'outcome': 3},
                         manipulability={name: i not in (0, 3) for i, name in enumerate(world.variables)},
                         readable={name: True for name in world.variables}, observation_bandwidth=8,
                         observation_budget_exponent=14, observation_budget=budget,
                         interaction_surface_version=INTERACTION_SURFACE_VERSION)
    episode = WorldSpecEpisode(world, seed, OutcomeTape('optimal-experiment-proof'))
    labels = seed['visible_schema']['variable_labels']
    command = {'type': 'intervene', 'target': labels['M'], 'value': 'state_0',
               'measure': [labels['X'], labels['Z']], 'batch_size': 1}
    step = episode.step(json.dumps(command))
    assert step.kind == 'batch'
    assert episode.observations_used == 2 and episode.queries_used == 1
    episode_records.append({'legal': True, 'scalar_cost': episode.observations_used})

n = budget//2
ratio_hi, ratio_lo = 1+2*delta, 1-2*delta
threshold_real = -n*math.log1p(-2*float(delta))/(math.log1p(2*float(delta))-math.log1p(-2*float(delta)))
threshold = math.ceil(threshold_real)
# Certify the discrete likelihood-ratio threshold by integer arithmetic.
denominator = pow(ratio_hi.denominator, n)
assert ratio_hi.denominator == ratio_lo.denominator
assert pow(ratio_hi.numerator, threshold)*pow(ratio_lo.numerator, n-threshold) >= denominator
assert pow(ratio_hi.numerator, threshold-1)*pow(ratio_lo.numerator, n-threshold+1) < denominator
mp.mp.dps = 60


def binomial_tail(start, p, direction):
    probability = mp.exp(mp.loggamma(n+1)-mp.loggamma(start+1)-mp.loggamma(n-start+1)
                         + start*mp.log(p)+(n-start)*mp.log1p(-p))
    total = probability
    if direction == 'up':
        for k in range(start, n):
            probability *= mp.mpf(n-k)/(k+1)*p/(1-p)
            total += probability
    else:
        for k in range(start, 0, -1):
            probability *= mp.mpf(k)/(n-k+1)*(1-p)/p
            total += probability
    return total


error0 = binomial_tail(threshold, mp.mpf(1)/2, 'up')
error1 = binomial_tail(threshold-1, mp.mpf(1)/2+mp.mpf(delta.numerator)/delta.denominator, 'down')
kl_per_pair = -.5*math.log1p(-4*float(delta)**2)
result = {'passed': True, 'scalar_budget': budget, 'independent_informative_pairs': n,
          'delta': str(delta), 'legal_actions_checked': len(actions),
          'identical_single_scalar_marginals': single_checks,
          'full_likelihood_ratio_checks': full_likelihood_checks,
          'real_protocol_checks': episode_records,
          'optimal_measurement': 'do(M=0), measure (X,Z); each informative unit costs exactly 2 scalars',
          'sufficient_statistic': 'C = 1[X=Z], Bernoulli(1/2) versus Bernoulli(1/2+delta)',
          'integer_certified_likelihood_threshold': threshold,
          'type_i_error': float(error0), 'type_ii_error': float(error1),
          'optimal_equal_prior_error_numeric': float((error0+error1)/2),
          'binomial_tail_precision_digits': 60,
          'kl_per_scalar_upper_bound': kl_per_pair/2,
          'pinsker_error_lower_bound': (1-math.sqrt(n*kl_per_pair/2))/2,
          'scope': 'Exact Bayes decision formula for the existing two-world witness; binomial tails evaluated numerically. No sampler prevalence claim.'}
(ROOT/'optimal-legal-experiment-acceptance.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
