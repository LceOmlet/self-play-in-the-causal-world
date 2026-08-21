from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from fractions import Fraction
from itertools import product
from pathlib import Path

from cpt_world import (
    OWNER_STATUS_IMPLEMENTED,
    WorldGrammar,
    WorldSpec,
    ate_effect,
    backdoor_adjustment_sets,
    collider_bias_effect,
    compute_query_truth,
    counterfactual_transition_bounds,
    load_bnlearn_world,
    load_candidate_seed_manifest,
    load_cladder_world,
    mediator_set_truth,
    query_truth_owner_status,
    sample_world,
    sample_worldspec_assignment,
    worldspec_interventional_distribution,
    worldspec_projected_interventional_distribution,
)


def _reference_projected_distribution(
    world: WorldSpec,
    interventions: dict[int, int],
    measure: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], Fraction], ...]:
    totals = {
        assignment: Fraction(0)
        for assignment in product(*(range(world.domains[node]) for node in measure))
    }
    for values, probability in worldspec_interventional_distribution(world, interventions):
        selected = tuple(values[node] for node in measure)
        totals[selected] += probability
    return tuple(totals.items())


def _binary_chain_world() -> WorldSpec:
    return WorldSpec(
        family="test_dag",
        topology="X-to-Y",
        variables=("X", "Y"),
        domains=(2, 2),
        state_names=(("x0", "x1"), ("y0", "y1")),
        edges=((0, 1),),
        parents={0: (), 1: (0,)},
        cpt={
            0: ((Fraction(1, 2), Fraction(1, 2)),),
            1: (
                (Fraction(3, 4), Fraction(1, 4)),
                (Fraction(1, 4), Fraction(3, 4)),
            ),
        },
    )


class QueryTruthOwnerTests(unittest.TestCase):
    def test_variable_elimination_matches_full_joint_reference_exactly(self) -> None:
        grammar = WorldGrammar(node_counts=(2, 3, 4), max_domain_size=4)
        for seed in range(24):
            world = sample_world(grammar, seed)
            measures = tuple((node,) for node in range(len(world.variables)))
            if len(world.variables) >= 2:
                measures += ((0, len(world.variables) - 1),)
            interventions = ({}, {0: 0}, {0: world.domains[0] - 1})
            for fixed in interventions:
                for measure in measures:
                    with self.subTest(seed=seed, fixed=fixed, measure=measure):
                        self.assertEqual(
                            worldspec_projected_interventional_distribution(
                                world,
                                fixed,
                                measure,
                            ),
                            _reference_projected_distribution(world, fixed, measure),
                        )

    def test_ancestral_sampler_matches_exact_hard_do_law_on_uniform_grid(self) -> None:
        world = _binary_chain_world()
        grid = tuple(Fraction(odd, 8) for odd in (1, 3, 5, 7))
        for interventions in ({}, {0: 1}):
            observed = {
                assignment: 0
                for assignment, _ in worldspec_interventional_distribution(
                    world,
                    interventions,
                )
            }
            for uniforms in product(grid, repeat=len(world.variables)):
                assignment = sample_worldspec_assignment(world, interventions, uniforms)
                observed[assignment] += 1
            reference = dict(worldspec_interventional_distribution(world, interventions))
            with self.subTest(interventions=interventions):
                self.assertEqual(
                    {
                        assignment: Fraction(count, len(grid) ** len(world.variables))
                        for assignment, count in observed.items()
                    },
                    reference,
                )

    def test_ancestral_sampler_matches_full_joint_reference_across_sampled_worlds(self) -> None:
        grammar = WorldGrammar(node_counts=(2, 3, 4), max_domain_size=4)
        for seed in range(12):
            world = sample_world(grammar, seed)
            intervention_cases = ({}, {0: world.domains[0] - 1})
            for interventions in intervention_cases:
                for assignment, reference_mass in worldspec_interventional_distribution(
                    world,
                    interventions,
                ):
                    if reference_mass == 0:
                        continue
                    uniforms = [Fraction(0)] * len(world.variables)
                    reconstructed_mass = Fraction(1)
                    for node in range(len(world.variables)):
                        if node in interventions:
                            continue
                        row_index = 0
                        for parent in world.parents.get(node, ()):
                            row_index = row_index * world.domains[parent] + assignment[parent]
                        row = world.cpt[node][row_index]
                        state = assignment[node]
                        lower = sum(row[:state], start=Fraction(0))
                        upper = lower + row[state]
                        uniforms[node] = (lower + upper) / 2
                        reconstructed_mass *= row[state]
                    with self.subTest(
                        seed=seed, interventions=interventions, assignment=assignment
                    ):
                        self.assertEqual(reconstructed_mass, reference_mass)
                        self.assertEqual(
                            sample_worldspec_assignment(
                                world,
                                interventions,
                                tuple(uniforms),
                            ),
                            assignment,
                        )

    def test_registry_distinguishes_registered_from_implemented(self) -> None:
        for query_type in (
            "ate",
            "counterfactual_transition_bounds",
            "backadj_minimal_sets",
            "best_intervention",
            "mediator_set",
        ):
            self.assertEqual(query_truth_owner_status(query_type), OWNER_STATUS_IMPLEMENTED)

    def test_ate_matches_cladder_groundtruth(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        models = json.loads(
            (repo / "data/worlds/cladder/meta-models-subset.json").read_text(encoding="utf-8")
        )
        for model in models:
            world = load_cladder_world(int(model["model_id"]))
            self.assertAlmostEqual(
                float(ate_effect(world, "X", "Y")),
                float(model["groundtruth"]["ATE(Y | X)"]),
                places=12,
            )

    def test_ate_on_sampled_edge_is_exact(self) -> None:
        grammar = WorldGrammar(node_counts=(2,), max_domain_size=2)
        for seed in range(200):
            world = sample_world(grammar, seed)
            if world.edges == ((0, 1),):
                expected = world.cpt[1][1][1] - world.cpt[1][0][1]
                self.assertEqual(ate_effect(world, "V0", "V1"), expected)
                return
        self.fail("no sampled binary edge found")

    def test_ate_on_sampled_multi_valued_edge_is_exact(self) -> None:
        grammar = WorldGrammar(node_counts=(2,), max_domain_size=5)
        for seed in range(400):
            world = sample_world(grammar, seed)
            if world.edges == ((0, 1),) and max(world.domains) > 2:
                expected = world.cpt[1][1][1] - world.cpt[1][0][1]
                self.assertEqual(ate_effect(world, "V0", "V1"), expected)
                return
        self.fail("no sampled multi-valued edge found")

    def test_counterfactual_transition_bounds_are_sharp_without_selecting_an_scm(self) -> None:
        world = WorldSpec(
            family="test_dag",
            topology="X-to-Y",
            variables=("X", "Y"),
            domains=(2, 2),
            state_names=(("x0", "x1"), ("y0", "y1")),
            edges=((0, 1),),
            parents={0: (), 1: (0,)},
            cpt={
                0: ((Fraction(1, 2), Fraction(1, 2)),),
                1: (
                    (Fraction(4, 5), Fraction(1, 5)),
                    (Fraction(1, 5), Fraction(4, 5)),
                ),
            },
        )
        self.assertEqual(
            counterfactual_transition_bounds(world, "X", "Y"),
            (Fraction(3, 5), Fraction(4, 5)),
        )

    def test_best_intervention_matches_pinned_cancer_seed(self) -> None:
        seeds = {seed.seed_id: seed for seed in load_candidate_seed_manifest()}
        seed = seeds["SEED-BN-CANCER-BESTINT"]
        world = load_bnlearn_world("data/worlds/bnlearn/cancer.bif")
        truth = compute_query_truth(world, asdict(seed))
        self.assertEqual(truth["target"], "Cancer")
        self.assertEqual(truth["value"], 1)
        self.assertEqual(truth["probability"], Fraction(3, 10))

    def test_best_intervention_matches_pinned_survey_seed(self) -> None:
        seeds = {seed.seed_id: seed for seed in load_candidate_seed_manifest()}
        seed = seeds["SEED-BN-SURVEY-BESTINT"]
        world = load_bnlearn_world("data/worlds/bnlearn/survey.bif")
        truth = compute_query_truth(world, asdict(seed))
        self.assertEqual(truth["target"], "R")
        self.assertEqual(truth["value"], 1)
        self.assertEqual(truth["probability"], Fraction(1831319, 3125000))

    def test_backdoor_adjustment_sets_match_known_dag_motifs(self) -> None:
        expected = {
            810: (("V1",),),
            60: (("V1",), ("V3",)),
            330: ((),),
            0: ((),),
        }
        for model_id, sets in expected.items():
            world = load_cladder_world(model_id)
            self.assertEqual(backdoor_adjustment_sets(world, "X", "Y"), sets)

    def test_collider_conditioned_contrast_remains_available_for_ate_diagnostics(self) -> None:
        world = load_cladder_world(330)
        effect = collider_bias_effect(world, "X", "Y", "V3")
        self.assertLess(effect, 0)
        self.assertNotEqual(effect, 0)

    def test_mediator_set_matches_pinned_asia_seed(self) -> None:
        seeds = {seed.seed_id: seed for seed in load_candidate_seed_manifest()}
        seed = seeds["SEED-BN-ASIA-MEDIATOR"]
        world = load_bnlearn_world("data/worlds/bnlearn/asia.bif")
        self.assertEqual(
            mediator_set_truth(world, "smoke", "dysp"),
            (
                ("lung", "bronc", "either"),
                (
                    ("smoke", "lung"),
                    ("smoke", "bronc"),
                    ("lung", "either"),
                    ("bronc", "dysp"),
                    ("either", "dysp"),
                ),
            ),
        )
        truth = compute_query_truth(world, asdict(seed))
        self.assertEqual(truth["mediators"], ("lung", "bronc", "either"))
        self.assertEqual(
            truth["order"],
            (
                ("smoke", "lung"),
                ("smoke", "bronc"),
                ("lung", "either"),
                ("bronc", "dysp"),
                ("either", "dysp"),
            ),
        )

    def test_mediator_set_matches_cancer_pathway(self) -> None:
        world = load_bnlearn_world("data/worlds/bnlearn/cancer.bif")
        self.assertEqual(
            mediator_set_truth(world, "Pollution", "Dyspnoea"),
            (("Cancer",), (("Pollution", "Cancer"), ("Cancer", "Dyspnoea"))),
        )


if __name__ == "__main__":
    unittest.main()
