from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace
from fractions import Fraction as F
from itertools import product

from cpt_world.identification import (
    INTERACTION_SURFACE_VERSION,
    validate_population_identification,
)
from cpt_world.query_truth import worldspec_projected_interventional_distribution as projected
from cpt_world.rendering import render_seed_task_prompt
from cpt_world.world import OutcomeTape
from cpt_world.world_runtime import WorldSpecEpisode
from cpt_world.world_space import (
    WorldGrammar,
    WorldSpec,
    assemble_sampled_anchor_tasks,
    assemble_seed,
)


def reversal_world(mediating_z=False):
    edges = [(0, 1) if mediating_z else (1, 0), (0, 2), (0, 3), (1, 3), (2, 3)]
    parents = {i: tuple(sorted(p for p, child in edges if child == i)) for i in range(4)}
    cpt = {i: ((F(1, 2), F(1, 2)),) for i in range(4)}
    cpt[1 if mediating_z else 0] = ((F(3, 4), F(1, 4)), (F(1, 4), F(3, 4)))
    cpt[2] = ((F(7, 10), F(3, 10)), (F(3, 10), F(7, 10)))
    rows = []
    for x, z, m in product((0, 1), repeat=3):
        sx, sz, sm = 2 * x - 1, 2 * z - 1, 2 * m - 1
        p = F(1, 2) + F(1, 10) * sz - F(3, 100) * sx + F(1, 100) * sm
        p += F(3, 1000) * sx * sz + F(2, 1000) * sz * sm
        p += F(1, 1000) * sx * sm + F(7, 10000) * sx * sz * sm
        rows.append((1 - p, p))
    cpt[3] = tuple(rows)
    return WorldSpec(
        family="identification-proof",
        topology="covered-edge-reversal",
        variables=("X", "Z", "M", "Y"),
        domains=(2,) * 4,
        state_names=(("zero", "one"),) * 4,
        edges=tuple(sorted(edges)),
        parents=parents,
        cpt=cpt,
    )


def recover_factors_from_distributions(domains, source, outcome, natural, interventions):
    """Proof check only: receives distributions, never the hidden graph or CPTs."""
    assignments = list(product(*(range(d) for d in domains)))
    nonanchors = [i for i in range(len(domains)) if i not in {source, outcome}]
    factors = {i: {} for i in range(len(domains))}
    for v in assignments:
        for i in nonanchors:
            measured = v[:i] + v[i + 1 :]
            factors[i][v] = natural[v] / interventions[i, v[i]][measured]
    remainder = {
        v: natural[v] / _fraction_product(factors[i][v] for i in nonanchors) for v in assignments
    }
    for v in assignments:
        fx = sum(
            (remainder[v[:outcome] + (y,) + v[outcome + 1 :]] for y in range(domains[outcome])),
            F(0),
        )
        factors[source][v] = fx
        factors[outcome][v] = remainder[v] / fx
    return factors


def _fraction_product(values):
    result = F(1)
    for value in values:
        result *= value
    return result


class PopulationIdentificationTests(unittest.TestCase):
    def validate(self, world, manipulability=None, bandwidth=4):
        validate_population_identification(
            world,
            0,
            3,
            manipulability or {"X": False, "Z": True, "M": True, "Y": False},
            dict.fromkeys(world.variables, True),
            bandwidth,
        )

    def check_recovery(self, world):
        self.validate(world)
        natural = dict(projected(world, {}, (0, 1, 2, 3)))
        experiments = {
            (i, state): dict(projected(world, {i: state}, tuple(j for j in range(4) if j != i)))
            for i in (1, 2)
            for state in range(2)
        }
        recovered = recover_factors_from_distributions(world.domains, 0, 3, natural, experiments)
        for i in range(4):
            for assignment, value in recovered[i].items():
                row = 0
                for parent in world.parents[i]:
                    row = row * world.domains[parent] + assignment[parent]
                self.assertEqual(value, world.cpt[i][row][assignment[i]])
            recovered_parents = {
                j
                for j in range(4)
                if j != i
                and any(
                    recovered[i][v] != recovered[i][v[:j] + (1 - v[j],) + v[j + 1 :]]
                    for v in recovered[i]
                )
            }
            self.assertEqual(recovered_parents, set(world.parents[i]))

    def test_old_equivalence_is_broken_by_legal_z_intervention(self):
        a, b = reversal_world(), reversal_world(True)
        self.assertEqual(dict(projected(a, {}, (0, 1, 2, 3))), dict(projected(b, {}, (0, 1, 2, 3))))
        for state in (0, 1):
            self.assertEqual(
                dict(projected(a, {2: state}, (0, 1, 3))),
                dict(projected(b, {2: state}, (0, 1, 3))),
            )
            pa, pb = dict(projected(a, {1: state}, (0,))), dict(projected(b, {1: state}, (0,)))
            self.assertEqual(sum(abs(pa[v] - pb[v]) for v in pa) / 2, F(1, 4))
        self.check_recovery(a)
        self.check_recovery(b)

    def test_factor_proof_also_recovers_an_observationally_unfaithful_world(self):
        world = reversal_world(True)
        edges = ((0, 2), (0, 3), (2, 3))
        cpt = {0: ((F(1, 2), F(1, 2)),), 1: ((F(1, 2), F(1, 2)),)}
        cpt[2] = ((F(3, 4), F(1, 4)), (F(1, 4), F(3, 4)))
        cpt[3] = tuple(
            (1 - p, p)
            for x, m in product((0, 1), repeat=2)
            for p in [F(1, 2) + F(1, 5) * (2 * m - 1) - F(1, 10) * (2 * x - 1)]
        )
        world = replace(
            world,
            edges=edges,
            parents={i: tuple(p for p, child in edges if child == i) for i in range(4)},
            cpt=cpt,
        )
        # X and Y are independent observationally despite X -> Y and X -> M -> Y.
        self.assertEqual(set(dict(projected(world, {}, (0, 3))).values()), {F(1, 4)})
        self.check_recovery(world)

    def test_validator_rejects_missing_interventions_and_joint_measurements(self):
        with self.assertRaisesRegex(ValueError, "every non-anchor"):
            self.validate(reversal_world(), {"X": False, "Z": False, "M": True, "Y": False})
        with self.assertRaisesRegex(ValueError, "full joint"):
            self.validate(reversal_world(), bandwidth=1)

    def test_validator_rejects_zero_probabilities_and_inactive_declared_edges(self):
        world = reversal_world()
        for rows, message in [
            (((F(1), F(0)),), "strictly positive"),
            (((F(1, 2), F(1, 2)),) * 2, "minimality"),
        ]:
            cpt = dict(world.cpt)
            cpt[2] = rows
            with self.assertRaisesRegex(ValueError, message):
                self.validate(replace(world, cpt=cpt))

    def test_sampled_tasks_keep_recorded_old_budgets_and_disclose_identification_prior(self):
        # Golden values recorded from the running old owner before modification.
        fixtures = [
            (0, "ate", 0, 65536),
            (1, "individual_counterfactual_probability", 0, 16384),
            (2, "backadj_minimal_sets", 0, 24576),
            (6, "best_intervention", 2, 32768),
            (4, "mediator_set", 0, 36864),
        ]
        for sample, family, anchor, budget in fixtures:
            with self.subTest(family=family):
                ((world, seed),) = assemble_sampled_anchor_tasks(
                    WorldGrammar(), sample, family, anchor
                )
                episode = WorldSpecEpisode(world, seed, OutcomeTape("identification-audit"))
                self.assertEqual(episode.budget.max_observations, budget)
                self.assertEqual(seed["observation_budget"], budget)
                self.assertEqual(episode.measure_max, len(world.variables))
                self.assertEqual(sum(seed["manipulability"].values()), len(world.variables) - 2)
                self.assertEqual(seed["interaction_surface_version"], INTERACTION_SURFACE_VERSION)
                prompt = render_seed_task_prompt(seed)
                self.assertIn(f"observation budget is {budget} scalar values", prompt)
                self.assertIn("is an ancestor of", prompt)
                self.assertIn("no hidden confounders", prompt)
                labels = seed["visible_schema"]["variable_labels"]
                command = {"type": "observe", "measure": list(labels.values()), "batch_size": 1}
                episode.step(json.dumps(command))
                self.assertEqual(episode.remaining_budget, budget - len(world.variables))

    def test_runtime_revalidates_actual_permissions_instead_of_trusting_version_label(self):
        world = reversal_world()
        seed = assemble_seed(
            world,
            "mechanism_hidden",
            "ate",
            "target_query",
            seed_id="proof-runtime",
            anchors={"treatment": 0, "outcome": 3},
            observation_bandwidth=4,
            observation_budget=2048,
            interaction_surface_version=INTERACTION_SURFACE_VERSION,
        )
        corrupted = copy.deepcopy(seed)
        corrupted["manipulability"]["Z"] = False
        with self.assertRaisesRegex(ValueError, "every non-anchor"):
            WorldSpecEpisode(world, corrupted, OutcomeTape("bad-permissions"))


if __name__ == "__main__":
    unittest.main()
