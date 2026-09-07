"""Exact-rational two-world witness for the finite-budget structural task limit."""

import json
import math
from fractions import Fraction as F
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))

from cpt_world import OutcomeTape, WorldSpecEpisode
from cpt_world.identification import INTERACTION_SURFACE_VERSION
from cpt_world.query_truth import compute_query_truth, worldspec_interventional_distribution
from cpt_world.world_space import WorldSpec, assemble_seed


def world(delta):
    names = ('X', 'M', 'Z', 'Y', 'U4', 'U5', 'U6', 'U7')
    edges = [(0, 1), (1, 3), (2, 3)]
    if delta:
        edges.append((0, 2))
    edges = tuple(sorted(edges))
    parents = {i: tuple(p for p, child in edges if child == i) for i in range(8)}
    cpt = {i: ((F(1, 2), F(1, 2)),) for i in range(8)}
    cpt[1] = ((F(3, 4), F(1, 4)), (F(1, 4), F(3, 4)))
    if delta:
        cpt[2] = ((F(1, 2) + delta, F(1, 2) - delta),
                  (F(1, 2) - delta, F(1, 2) + delta))
    cpt[3] = ((F(4, 5), F(1, 5)), (F(1, 2), F(1, 2)),
              (F(1, 2), F(1, 2)), (F(1, 5), F(4, 5)))
    return WorldSpec(family='finite-budget-witness', topology='weak-added-path',
                     variables=names, domains=(2,) * 8,
                     state_names=(('zero', 'one'),) * 8,
                     edges=edges, parents=parents, cpt=cpt)


def main():
    delta = F(1, 10000)
    budget = 8 * 2**14
    worlds = [world(F(0)), world(delta)]
    episodes = []
    for w in worlds:
        seed = assemble_seed(w, 'mechanism_hidden', 'mediator_set', 'discovery',
            seed_id='FINITE-BUDGET-SAME-PUBLIC-PROMPT', anchors={'treatment': 0, 'outcome': 3},
            manipulability={n: i not in {0, 3} for i, n in enumerate(w.variables)},
            readable={n: True for n in w.variables}, observation_bandwidth=8,
            observation_budget_exponent=14, observation_budget=budget,
            interaction_surface_version=INTERACTION_SURFACE_VERSION)
        episodes.append(WorldSpecEpisode(w, seed, OutcomeTape('witness')))
    assert episodes[0].initial_messages() == episodes[1].initial_messages()
    truth = [compute_query_truth(w, e.seed) for w, e in zip(worlds, episodes)]
    assert truth[0]['mediators'] == ('M',)
    assert truth[1]['mediators'] == ('M', 'Z')
    laws = []
    for action in [{}] + [{i: state} for i in range(8) if i not in {0, 3} for state in range(2)]:
        base, alternative = [dict(worldspec_interventional_distribution(w, action)) for w in worlds]
        assert set(base) == set(alternative)
        tv = sum((abs(alternative[x] - base[x]) for x in base), F(0)) / 2
        assert all(alternative[x] == 0 for x in base if base[x] == 0)
        chi2 = sum(((alternative[x] - base[x])**2 / base[x] for x in base if base[x] > 0), F(0))
        assert tv <= delta
        assert chi2 <= 4 * delta**2
        laws.append({'action': action, 'tv_exact': str(tv), 'chi2_exact': str(chi2)})
    # KL <= chi2 per unit; adaptive chain rule and data processing give
    # KL(transcripts) <= 4*B*delta^2 since each IID unit costs >=1 scalar.
    # Pinsker then gives TV <= delta*sqrt(2B). Equal-prior testing error
    # is at least (1-TV)/2, for ANY adaptive randomized legal policy.
    transcript_kl_bound = 4 * budget * delta**2
    transcript_tv_bound = math.sqrt(float(transcript_kl_bound) / 2)
    report = {'delta_exact': str(delta), 'nodes': 8, 'scalar_budget': budget,
              'public_prompts_identical': True, 'both_pass_production_contract': True,
              'truths': truth, 'all_legal_one_unit_laws': laws,
              'transcript_kl_upper_bound_exact': str(transcript_kl_bound),
              'transcript_tv_upper_bound': transcript_tv_bound,
              'equal_prior_Z_membership_error_lower_bound': (1 - transcript_tv_bound) / 2,
              'scope': 'Worst-case witness in the declared positive/minimal causal class; does not estimate its frequency in the sampler.',
              'math_reference': 'https://stanford.edu/class/stats311/lecture-notes.pdf'}
    output = Path(__file__).resolve().parent/'finite-budget-proof.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
