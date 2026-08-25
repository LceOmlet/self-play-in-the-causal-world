from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
from unittest.mock import patch

from cpt_world import (
    OWNER_STATUS_IMPLEMENTED,
    WorldGrammar,
    WorldSpec,
    ate_effect,
    backdoor_adjustment_sets,
    collider_bias_effect,
    compute_query_truth,
    counterfactual_transition_bounds,
    individual_counterfactual_frechet_outer_bounds,
    individual_counterfactual_probability_bounds,
    interventional_frechet_transition_outer_bounds,
    load_bnlearn_world,
    load_candidate_seed_manifest,
    load_cladder_world,
    mediator_set_truth,
    query_truth_owner_status,
    reference_counterfactual_transition_bounds,
    reference_individual_counterfactual_probability_bounds,
    sample_world,
    sample_worldspec_assignment,
    sparse_counterfactual_transition_bounds,
    sparse_individual_counterfactual_probability_bounds,
    validate_individual_counterfactual_probability,
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


def _binary_world_from_edges(node_count: int, edges: tuple[tuple[int, int], ...]) -> WorldSpec:
    parents = {
        node: tuple(parent for parent, child in edges if child == node)
        for node in range(node_count)
    }
    return WorldSpec(
        family="test_dag",
        topology="ordered-dag",
        variables=tuple(f"V{node}" for node in range(node_count)),
        domains=(2,) * node_count,
        state_names=tuple(("state_0", "state_1") for _ in range(node_count)),
        edges=edges,
        parents=parents,
        cpt={
            node: tuple((Fraction(1, 2), Fraction(1, 2)) for _ in range(2 ** len(parents[node])))
            for node in range(node_count)
        },
    )


def _reference_backdoor_adjustment_sets(
    world: WorldSpec, treatment: int, outcome: int
) -> tuple[tuple[str, ...], ...]:
    def descendants(node: int) -> frozenset[int]:
        seen: set[int] = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for parent, child in world.edges:
                if parent == current and child not in seen:
                    seen.add(child)
                    stack.append(child)
        return frozenset(seen)

    neighbors = {
        node: {
            child if parent == node else parent
            for parent, child in world.edges
            if parent == node or child == node
        }
        for node in range(len(world.variables))
    }
    paths: list[tuple[int, ...]] = []

    def visit(current: int, path: list[int], seen: set[int]) -> None:
        if current == outcome:
            paths.append(tuple(path))
            return
        for neighbor in sorted(neighbors[current]):
            if neighbor in seen:
                continue
            visit(neighbor, [*path, neighbor], {*seen, neighbor})

    for parent in world.parents[treatment]:
        visit(parent, [treatment, parent], {treatment, parent})

    edge_set = set(world.edges)
    descendant_sets = {node: descendants(node) for node in range(len(world.variables))}

    def path_is_open(path: tuple[int, ...], condition: frozenset[int]) -> bool:
        for index in range(1, len(path) - 1):
            previous, node, following = path[index - 1 : index + 2]
            collider = (previous, node) in edge_set and (following, node) in edge_set
            if collider:
                if node not in condition and not (descendant_sets[node] & condition):
                    return False
            elif node in condition:
                return False
        return True

    treatment_descendants = descendants(treatment)
    allowed = tuple(
        node
        for node in range(len(world.variables))
        if node not in {treatment, outcome} and node not in treatment_descendants
    )
    valid: list[frozenset[int]] = []
    for size in range(len(allowed) + 1):
        for subset in combinations(allowed, size):
            condition = frozenset(subset)
            if all(not path_is_open(path, condition) for path in paths):
                valid.append(condition)
    minimal = [candidate for candidate in valid if not any(other < candidate for other in valid)]
    minimal.sort(key=lambda candidate: (len(candidate), tuple(sorted(candidate))))
    return tuple(
        tuple(world.variables[node] for node in sorted(candidate)) for candidate in minimal
    )


class QueryTruthOwnerTests(unittest.TestCase):
    def test_variable_elimination_matches_full_joint_reference_with_float_tolerance(self) -> None:
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
                        accelerated = worldspec_projected_interventional_distribution(
                            world,
                            fixed,
                            measure,
                        )
                        reference = _reference_projected_distribution(world, fixed, measure)
                        self.assertEqual(
                            tuple(assignment for assignment, _ in accelerated),
                            tuple(assignment for assignment, _ in reference),
                        )
                        for (_, actual), (_, expected) in zip(accelerated, reference, strict=True):
                            tolerance = 1e-9 + 1e-9 * max(abs(float(actual)), abs(float(expected)))
                            self.assertLessEqual(abs(float(actual) - float(expected)), tolerance)

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
            "individual_counterfactual_probability",
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

    def test_ate_uses_the_requested_multivalued_states(self) -> None:
        world = WorldSpec(
            family="test_dag",
            topology="three-state-x-to-y",
            variables=("X", "Y"),
            domains=(3, 3),
            state_names=(("x0", "x1", "x2"), ("y0", "y1", "y2")),
            edges=((0, 1),),
            parents={0: (), 1: (0,)},
            cpt={
                0: ((Fraction(1, 3),) * 3,),
                1: (
                    (Fraction(7, 10), Fraction(2, 10), Fraction(1, 10)),
                    (Fraction(2, 10), Fraction(5, 10), Fraction(3, 10)),
                    (Fraction(1, 10), Fraction(3, 10), Fraction(6, 10)),
                ),
            },
        )
        seed = {
            "query": {
                "type": "ate",
                "treatment": "X",
                "outcome": "Y",
                "baseline_value": "state_2",
                "treatment_value": "state_0",
                "outcome_state": "state_1",
            }
        }
        self.assertEqual(compute_query_truth(world, seed)["effect"], Fraction(-1, 10))

    def test_best_intervention_uses_the_requested_multivalued_outcome_state(self) -> None:
        world = WorldSpec(
            family="test_dag",
            topology="three-state-d-to-y",
            variables=("D", "Y"),
            domains=(3, 3),
            state_names=(("d0", "d1", "d2"), ("y0", "y1", "y2")),
            edges=((0, 1),),
            parents={0: (), 1: (0,)},
            cpt={
                0: ((Fraction(1, 3),) * 3,),
                1: (
                    (Fraction(7, 10), Fraction(2, 10), Fraction(1, 10)),
                    (Fraction(2, 10), Fraction(5, 10), Fraction(3, 10)),
                    (Fraction(1, 10), Fraction(3, 10), Fraction(6, 10)),
                ),
            },
        )
        seed = {
            "query": {
                "type": "best_intervention",
                "decision_target": "D",
                "outcome": "Y",
                "objective": "maximize",
                "outcome_state": "state_2",
            }
        }
        truth = compute_query_truth(world, seed)
        self.assertEqual(truth["value"], 2)
        self.assertEqual(truth["probability"], Fraction(3, 5))

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

    def test_individual_counterfactual_direct_bounds_condition_on_factual_outcome(
        self,
    ) -> None:
        world = _binary_chain_world()
        expected = (Fraction(2, 3), Fraction(1))
        arguments = {
            "factual_value": 0,
            "counterfactual_value": 1,
            "factual_outcome_state": 0,
            "target_outcome_state": 1,
        }
        self.assertEqual(
            individual_counterfactual_probability_bounds(world, "X", "Y", **arguments),
            expected,
        )
        self.assertEqual(
            reference_individual_counterfactual_probability_bounds(world, "X", "Y", **arguments),
            expected,
        )
        self.assertEqual(
            individual_counterfactual_frechet_outer_bounds(world, "X", "Y", **arguments),
            expected,
        )

    def test_individual_counterfactual_sparse_owner_matches_explicit_reference(
        self,
    ) -> None:
        world = WorldSpec(
            family="test_dag",
            topology="X-to-M-to-Y",
            variables=("X", "M", "Y"),
            domains=(2, 2, 2),
            state_names=(("0", "1"),) * 3,
            edges=((0, 1), (1, 2)),
            parents={0: (), 1: (0,), 2: (1,)},
            cpt={
                0: ((Fraction(1, 2), Fraction(1, 2)),),
                1: (
                    (Fraction(4, 5), Fraction(1, 5)),
                    (Fraction(1, 5), Fraction(4, 5)),
                ),
                2: (
                    (Fraction(9, 10), Fraction(1, 10)),
                    (Fraction(1, 10), Fraction(9, 10)),
                ),
            },
        )
        for factual_value, counterfactual_value in ((0, 1), (1, 0)):
            for factual_state, target_state in product(range(2), repeat=2):
                arguments = {
                    "factual_value": factual_value,
                    "counterfactual_value": counterfactual_value,
                    "factual_outcome_state": factual_state,
                    "target_outcome_state": target_state,
                }
                with self.subTest(**arguments):
                    expected = reference_individual_counterfactual_probability_bounds(
                        world, "X", "Y", **arguments
                    )
                    sparse = sparse_individual_counterfactual_probability_bounds(
                        world, 0, 2, **arguments
                    )
                    produced = individual_counterfactual_probability_bounds(
                        world, "X", "Y", **arguments
                    )
                    self.assertAlmostEqual(sparse.lower, float(expected[0]), places=8)
                    self.assertAlmostEqual(sparse.upper, float(expected[1]), places=8)
                    self.assertAlmostEqual(float(produced[0]), float(expected[0]), places=8)
                    self.assertAlmostEqual(float(produced[1]), float(expected[1]), places=8)
                    outer = individual_counterfactual_frechet_outer_bounds(
                        world, "X", "Y", **arguments
                    )
                    self.assertLessEqual(float(outer[0]), float(produced[0]) + 1e-9)
                    self.assertGreaterEqual(float(outer[1]), float(produced[1]) - 1e-9)

    def test_individual_counterfactual_no_path_is_same_person_identity(self) -> None:
        world = _binary_world_from_edges(3, ((0, 1),))
        common = {
            "factual_value": 0,
            "counterfactual_value": 1,
            "factual_outcome_state": 1,
        }
        self.assertEqual(
            individual_counterfactual_probability_bounds(
                world, 0, 2, target_outcome_state=1, **common
            ),
            (Fraction(1), Fraction(1)),
        )
        self.assertEqual(
            individual_counterfactual_probability_bounds(
                world, 0, 2, target_outcome_state=0, **common
            ),
            (Fraction(0), Fraction(0)),
        )

    def test_individual_scalar_verifier_is_exact_or_fail_closed(self) -> None:
        world = _binary_chain_world()
        arguments = {
            "factual_value": 0,
            "counterfactual_value": 1,
            "factual_outcome_state": 0,
            "target_outcome_state": 1,
        }
        exact = validate_individual_counterfactual_probability(world, "X", "Y", 0.8, **arguments)
        self.assertEqual(exact["status"], "exact")
        self.assertTrue(exact["compatible"])

        with patch(
            "cpt_world.query_truth.individual_counterfactual_probability_bounds",
            side_effect=RuntimeError("time limit reached"),
        ):
            rejected = validate_individual_counterfactual_probability(
                world, "X", "Y", 0.0, **arguments
            )
            unresolved = validate_individual_counterfactual_probability(
                world, "X", "Y", 0.8, **arguments
            )
        self.assertEqual(rejected["status"], "rejected_by_frechet_outer")
        self.assertFalse(rejected["compatible"])
        self.assertEqual(unresolved["status"], "unresolved_timeout")
        self.assertIsNone(unresolved["compatible"])

    def test_markovian_counterfactual_bounds_can_be_tighter_than_marginal_frechet(
        self,
    ) -> None:
        world = WorldSpec(
            family="test_dag",
            topology="X-to-M-to-Y",
            variables=("X", "M", "Y"),
            domains=(2, 2, 2),
            state_names=(("0", "1"),) * 3,
            edges=((0, 1), (1, 2)),
            parents={0: (), 1: (0,), 2: (1,)},
            cpt={
                0: ((Fraction(1, 2), Fraction(1, 2)),),
                1: ((Fraction(9, 10), Fraction(1, 10)),) * 2,
                2: ((Fraction(9, 10), Fraction(1, 10)),) * 2,
            },
        )
        self.assertEqual(
            interventional_frechet_transition_outer_bounds(world, "X", "Y"),
            (Fraction(0), Fraction(1, 10)),
        )
        expected = (Fraction(0), Fraction(1, 50))
        self.assertEqual(reference_counterfactual_transition_bounds(world, "X", "Y"), expected)
        produced = counterfactual_transition_bounds(world, "X", "Y")
        self.assertAlmostEqual(float(produced[0]), float(expected[0]), places=8)
        self.assertAlmostEqual(float(produced[1]), float(expected[1]), places=8)

    def test_direct_counterfactual_fast_path_matches_reference_with_shared_parent(
        self,
    ) -> None:
        world = WorldSpec(
            family="test_dag",
            topology="X-and-Z-to-Y",
            variables=("X", "Z", "Y"),
            domains=(2, 2, 2),
            state_names=(("0", "1"),) * 3,
            edges=((0, 2), (1, 2)),
            parents={0: (), 1: (), 2: (0, 1)},
            cpt={
                0: ((Fraction(1, 2), Fraction(1, 2)),),
                1: ((Fraction(3, 4), Fraction(1, 4)),),
                2: (
                    (Fraction(9, 10), Fraction(1, 10)),
                    (Fraction(4, 5), Fraction(1, 5)),
                    (Fraction(3, 5), Fraction(2, 5)),
                    (Fraction(1, 5), Fraction(4, 5)),
                ),
            },
        )
        self.assertEqual(
            counterfactual_transition_bounds(world, "X", "Y"),
            reference_counterfactual_transition_bounds(world, "X", "Y"),
        )

    def test_on_demand_counterfactual_solver_matches_reference_on_a_cyclic_context_graph(
        self,
    ) -> None:
        world = WorldSpec(
            family="test_dag",
            topology="direct-and-mediated-triangle",
            variables=("X", "M", "Y"),
            domains=(2, 2, 2),
            state_names=(("0", "1"),) * 3,
            edges=((0, 1), (0, 2), (1, 2)),
            parents={0: (), 1: (0,), 2: (0, 1)},
            cpt={
                0: ((Fraction(1, 2), Fraction(1, 2)),),
                1: (
                    (Fraction(4, 5), Fraction(1, 5)),
                    (Fraction(1, 5), Fraction(4, 5)),
                ),
                2: (
                    (Fraction(9, 10), Fraction(1, 10)),
                    (Fraction(3, 5), Fraction(2, 5)),
                    (Fraction(2, 5), Fraction(3, 5)),
                    (Fraction(1, 10), Fraction(9, 10)),
                ),
            },
        )
        cases = (
            (0, 1, 0, (Fraction(0), Fraction(4, 25))),
            (0, 1, 1, (Fraction(17, 25), Fraction(21, 25))),
            (1, 0, 0, (Fraction(17, 25), Fraction(21, 25))),
            (1, 0, 1, (Fraction(0), Fraction(4, 25))),
        )
        for baseline, treatment, outcome_state, expected in cases:
            with self.subTest(
                baseline=baseline,
                treatment=treatment,
                outcome_state=outcome_state,
            ):
                reference = reference_counterfactual_transition_bounds(
                    world,
                    "X",
                    "Y",
                    treatment_value=treatment,
                    baseline_value=baseline,
                    outcome_state=outcome_state,
                )
                with (
                    patch(
                        "cpt_world.counterfactual_solver._one_mediator_joint_bounds",
                        return_value=None,
                    ),
                    patch(
                        "cpt_world.counterfactual_solver._direct_treatment_terminal_bounds",
                        return_value=None,
                    ),
                    patch(
                        "cpt_world.counterfactual_solver."
                        "_partially_attainable_terminal_bounds",
                        return_value=None,
                    ),
                ):
                    result = sparse_counterfactual_transition_bounds(
                        world,
                        0,
                        2,
                        treatment_value=treatment,
                        baseline_value=baseline,
                        outcome_state=outcome_state,
                    )
                self.assertEqual(reference, expected)
                self.assertAlmostEqual(result.lower, float(reference[0]), places=8)
                self.assertAlmostEqual(result.upper, float(reference[1]), places=8)
                self.assertGreater(result.pair_kernel_entries, 0)
                self.assertGreater(result.dynamic_response_blocks, 0)
                produced = counterfactual_transition_bounds(
                    world,
                    "X",
                    "Y",
                    treatment_value=treatment,
                    baseline_value=baseline,
                    outcome_state=outcome_state,
                )
                self.assertAlmostEqual(float(produced[0]), float(reference[0]), places=8)
                self.assertAlmostEqual(float(produced[1]), float(reference[1]), places=8)

    def test_counterfactual_direct_multivalued_states_exhaustively_match_reference(
        self,
    ) -> None:
        world = WorldSpec(
            family="test_dag",
            topology="X3-to-Y3",
            variables=("X", "Y"),
            domains=(3, 3),
            state_names=(("0", "1", "2"), ("0", "1", "2")),
            edges=((0, 1),),
            parents={0: (), 1: (0,)},
            cpt={
                0: ((Fraction(1, 3),) * 3,),
                1: (
                    (Fraction(7, 10), Fraction(1, 5), Fraction(1, 10)),
                    (Fraction(1, 2), Fraction(3, 10), Fraction(1, 5)),
                    (Fraction(3, 10), Fraction(2, 5), Fraction(3, 10)),
                ),
            },
        )
        for baseline in range(3):
            for treatment in range(3):
                if baseline == treatment:
                    continue
                for outcome_state in range(3):
                    with self.subTest(
                        baseline=baseline,
                        treatment=treatment,
                        outcome_state=outcome_state,
                    ):
                        expected = reference_counterfactual_transition_bounds(
                            world,
                            "X",
                            "Y",
                            treatment_value=treatment,
                            baseline_value=baseline,
                            outcome_state=outcome_state,
                        )
                        self.assertEqual(
                            counterfactual_transition_bounds(
                                world,
                                "X",
                                "Y",
                                treatment_value=treatment,
                                baseline_value=baseline,
                                outcome_state=outcome_state,
                            ),
                            expected,
                        )

    def test_counterfactual_multivalued_chain_matches_reference(self) -> None:
        world = WorldSpec(
            family="test_dag",
            topology="X3-to-M2-to-Y3",
            variables=("X", "M", "Y"),
            domains=(3, 2, 3),
            state_names=(("0", "1", "2"), ("0", "1"), ("0", "1", "2")),
            edges=((0, 1), (1, 2)),
            parents={0: (), 1: (0,), 2: (1,)},
            cpt={
                0: ((Fraction(1, 3),) * 3,),
                1: (
                    (Fraction(4, 5), Fraction(1, 5)),
                    (Fraction(1, 2), Fraction(1, 2)),
                    (Fraction(1, 5), Fraction(4, 5)),
                ),
                2: (
                    (Fraction(7, 10), Fraction(1, 5), Fraction(1, 10)),
                    (Fraction(1, 10), Fraction(3, 10), Fraction(3, 5)),
                ),
            },
        )
        cases = ((0, 2, 2), (2, 0, 0), (0, 1, 1))
        for baseline, treatment, outcome_state in cases:
            with self.subTest(
                baseline=baseline,
                treatment=treatment,
                outcome_state=outcome_state,
            ):
                expected = reference_counterfactual_transition_bounds(
                    world,
                    "X",
                    "Y",
                    treatment_value=treatment,
                    baseline_value=baseline,
                    outcome_state=outcome_state,
                )
                produced = counterfactual_transition_bounds(
                    world,
                    "X",
                    "Y",
                    treatment_value=treatment,
                    baseline_value=baseline,
                    outcome_state=outcome_state,
                )
                self.assertAlmostEqual(float(produced[0]), float(expected[0]), places=8)
                self.assertAlmostEqual(float(produced[1]), float(expected[1]), places=8)

    def test_counterfactual_deterministic_and_no_path_boundaries(self) -> None:
        deterministic = WorldSpec(
            family="test_dag",
            topology="X2-to-M3-to-Y2-deterministic",
            variables=("X", "M", "Y"),
            domains=(2, 3, 2),
            state_names=(("0", "1"), ("0", "1", "2"), ("0", "1")),
            edges=((0, 1), (1, 2)),
            parents={0: (), 1: (0,), 2: (1,)},
            cpt={
                0: ((Fraction(1, 2), Fraction(1, 2)),),
                1: (
                    (Fraction(1), Fraction(0), Fraction(0)),
                    (Fraction(0), Fraction(0), Fraction(1)),
                ),
                2: (
                    (Fraction(1), Fraction(0)),
                    (Fraction(1, 2), Fraction(1, 2)),
                    (Fraction(0), Fraction(1)),
                ),
            },
        )
        self.assertEqual(
            reference_counterfactual_transition_bounds(deterministic, "X", "Y"),
            (Fraction(1), Fraction(1)),
        )
        produced = counterfactual_transition_bounds(deterministic, "X", "Y")
        self.assertAlmostEqual(float(produced[0]), 1.0, places=9)
        self.assertAlmostEqual(float(produced[1]), 1.0, places=9)
        reverse = counterfactual_transition_bounds(
            deterministic,
            "X",
            "Y",
            treatment_value=0,
            baseline_value=1,
            outcome_state=1,
        )
        self.assertAlmostEqual(float(reverse[0]), 0.0, places=9)
        self.assertAlmostEqual(float(reverse[1]), 0.0, places=9)

        no_path = _binary_world_from_edges(3, ((0, 1),))
        self.assertEqual(
            counterfactual_transition_bounds(no_path, 0, 2),
            (Fraction(0), Fraction(0)),
        )

    def test_equal_marginals_do_not_collapse_cross_world_uncertainty(self) -> None:
        world = WorldSpec(
            family="test_dag",
            topology="X-to-Y-equal-marginals",
            variables=("X", "Y"),
            domains=(2, 2),
            state_names=(("0", "1"),) * 2,
            edges=((0, 1),),
            parents={0: (), 1: (0,)},
            cpt={
                0: ((Fraction(1, 2), Fraction(1, 2)),),
                1: (
                    (Fraction(7, 10), Fraction(3, 10)),
                    (Fraction(7, 10), Fraction(3, 10)),
                ),
            },
        )
        self.assertEqual(
            counterfactual_transition_bounds(world, "X", "Y"),
            (Fraction(0), Fraction(3, 10)),
        )

    def test_counterfactual_shared_observed_cause_matches_reference(self) -> None:
        world = WorldSpec(
            family="test_dag",
            topology="Z-to-X-Z-to-Y-X-to-Y",
            variables=("Z", "X", "Y"),
            domains=(2, 2, 2),
            state_names=(("0", "1"),) * 3,
            edges=((0, 1), (0, 2), (1, 2)),
            parents={0: (), 1: (0,), 2: (0, 1)},
            cpt={
                0: ((Fraction(3, 5), Fraction(2, 5)),),
                1: (
                    (Fraction(4, 5), Fraction(1, 5)),
                    (Fraction(1, 5), Fraction(4, 5)),
                ),
                2: (
                    (Fraction(9, 10), Fraction(1, 10)),
                    (Fraction(3, 5), Fraction(2, 5)),
                    (Fraction(7, 10), Fraction(3, 10)),
                    (Fraction(1, 5), Fraction(4, 5)),
                ),
            },
        )
        expected = reference_counterfactual_transition_bounds(world, "X", "Y")
        self.assertEqual(expected, (Fraction(19, 50), Fraction(13, 25)))
        self.assertEqual(counterfactual_transition_bounds(world, "X", "Y"), expected)

    def test_counterfactual_near_deterministic_probabilities_match_reference(self) -> None:
        epsilon = Fraction(1, 10**6)
        world = WorldSpec(
            family="test_dag",
            topology="X-to-M-to-Y-near-deterministic",
            variables=("X", "M", "Y"),
            domains=(2, 2, 2),
            state_names=(("0", "1"),) * 3,
            edges=((0, 1), (1, 2)),
            parents={0: (), 1: (0,), 2: (1,)},
            cpt={
                0: ((Fraction(1, 2), Fraction(1, 2)),),
                1: ((1 - epsilon, epsilon), (epsilon, 1 - epsilon)),
                2: ((1 - epsilon, epsilon), (epsilon, 1 - epsilon)),
            },
        )
        expected = reference_counterfactual_transition_bounds(world, "X", "Y")
        result = sparse_counterfactual_transition_bounds(
            world,
            0,
            2,
            treatment_value=1,
            baseline_value=0,
            outcome_state=1,
        )
        self.assertAlmostEqual(result.lower, float(expected[0]), places=8)
        self.assertAlmostEqual(result.upper, float(expected[1]), places=8)

    def test_counterfactual_reconvergent_diamond_matches_reference(self) -> None:
        world = WorldSpec(
            family="test_dag",
            topology="X-to-A-and-B-to-Y",
            variables=("X", "A", "B", "Y"),
            domains=(2, 2, 2, 2),
            state_names=(("0", "1"),) * 4,
            edges=((0, 1), (0, 2), (1, 3), (2, 3)),
            parents={0: (), 1: (0,), 2: (0,), 3: (1, 2)},
            cpt={
                0: ((Fraction(1, 2), Fraction(1, 2)),),
                1: (
                    (Fraction(4, 5), Fraction(1, 5)),
                    (Fraction(1, 5), Fraction(4, 5)),
                ),
                2: (
                    (Fraction(7, 10), Fraction(3, 10)),
                    (Fraction(3, 10), Fraction(7, 10)),
                ),
                3: (
                    (Fraction(9, 10), Fraction(1, 10)),
                    (Fraction(3, 5), Fraction(2, 5)),
                    (Fraction(2, 5), Fraction(3, 5)),
                    (Fraction(1, 10), Fraction(9, 10)),
                ),
            },
        )
        expected = reference_counterfactual_transition_bounds(world, "X", "Y")
        result = sparse_counterfactual_transition_bounds(
            world,
            0,
            3,
            treatment_value=1,
            baseline_value=0,
            outcome_state=1,
        )
        self.assertEqual(expected, (Fraction(21, 50), Fraction(71, 100)))
        self.assertAlmostEqual(result.lower, float(expected[0]), places=8)
        self.assertAlmostEqual(result.upper, float(expected[1]), places=8)

    def test_counterfactual_query_rejects_state_boundary_errors(self) -> None:
        world = _binary_chain_world()
        with self.assertRaisesRegex(ValueError, "must differ"):
            counterfactual_transition_bounds(
                world,
                "X",
                "Y",
                treatment_value=0,
                baseline_value=0,
            )
        for field, arguments in (
            ("treatment value", {"treatment_value": 2}),
            ("baseline value", {"baseline_value": -1}),
            ("outcome state", {"outcome_state": 2}),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    counterfactual_transition_bounds(world, "X", "Y", **arguments)

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

    def test_backdoor_moral_graph_algorithm_matches_path_enumeration(self) -> None:
        for node_count in range(2, 6):
            edge_slots = tuple(combinations(range(node_count), 2))
            for edge_mask in range(1 << len(edge_slots)):
                edges = tuple(
                    edge for index, edge in enumerate(edge_slots) if edge_mask & (1 << index)
                )
                world = _binary_world_from_edges(node_count, edges)
                for treatment in range(node_count):
                    for outcome in range(node_count):
                        if treatment == outcome:
                            continue
                        with self.subTest(
                            node_count=node_count,
                            edge_mask=edge_mask,
                            treatment=treatment,
                            outcome=outcome,
                        ):
                            self.assertEqual(
                                backdoor_adjustment_sets(world, treatment, outcome),
                                _reference_backdoor_adjustment_sets(world, treatment, outcome),
                            )

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
