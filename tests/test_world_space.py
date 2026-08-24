from __future__ import annotations

import hashlib
import json
import random
import re
import unittest
from collections.abc import Mapping
from dataclasses import asdict, replace
from fractions import Fraction
from itertools import product
from pathlib import Path

from cpt_world import (
    QUERY_TYPES,
    Budget,
    WorldGrammar,
    WorldSpec,
    assemble_sampled_anchor_tasks,
    assemble_seed,
    check_seed_legality,
    compute_query_truth,
    iter_sampled_seeds,
    iter_upstream_worlds,
    iter_world_space,
    legal_query_anchors,
    legal_world,
    load_bnlearn_world,
    load_candidate_seed_manifest,
    load_cladder_world,
    profile_task_targets,
    render_seed_task_prompt,
    sample_task_world,
    sample_world,
    supports_query,
    supports_task,
    task_answerability,
    task_difficulty_profile,
)


def _indirect_ate_world(
    *,
    mediator_given_treatment: tuple[Fraction, Fraction],
    outcome_given_mediator: tuple[Fraction, Fraction] = (Fraction(1, 5), Fraction(4, 5)),
) -> WorldSpec:
    return WorldSpec(
        family="sampled_dag",
        topology="test-chain-x-m-y",
        variables=("X", "M", "Y"),
        domains=(2, 2, 2),
        state_names=(("0", "1"),) * 3,
        edges=((0, 1), (1, 2)),
        parents={0: (), 1: (0,), 2: (1,)},
        cpt={
            0: ((Fraction(1, 2), Fraction(1, 2)),),
            1: tuple((1 - probability, probability) for probability in mediator_given_treatment),
            2: tuple((1 - probability, probability) for probability in outcome_given_mediator),
        },
    )


def _indirect_ate_seed(world: WorldSpec, seed_id: str) -> Mapping[str, object]:
    return assemble_seed(
        world,
        "mechanism_hidden",
        "ate",
        "target_query",
        seed_id=seed_id,
        anchors={"treatment": 0, "outcome": 2},
        manipulability={"X": False, "M": True, "Y": False},
        readable={"X": True, "M": True, "Y": True},
    )


def _counterfactual_km_world(
    *,
    mediator_given_treatment: tuple[Fraction, Fraction],
    outcome_given_mediator: tuple[Fraction, Fraction],
) -> WorldSpec:
    return WorldSpec(
        family="sampled_dag",
        topology="test-chain-x-m-y-plus-z",
        variables=("X", "M", "Y", "Z"),
        domains=(2, 2, 2, 2),
        state_names=(("0", "1"),) * 4,
        edges=((0, 1), (1, 2)),
        parents={0: (), 1: (0,), 2: (1,), 3: ()},
        cpt={
            0: ((Fraction(1, 2), Fraction(1, 2)),),
            1: tuple((1 - probability, probability) for probability in mediator_given_treatment),
            2: tuple((1 - probability, probability) for probability in outcome_given_mediator),
            3: ((Fraction(1, 2), Fraction(1, 2)),),
        },
    )


def _counterfactual_km_seed(
    world: WorldSpec,
    seed_id: str,
    *,
    manipulable: frozenset[str],
    observation_bandwidth: int,
) -> Mapping[str, object]:
    return assemble_seed(
        world,
        "mechanism_hidden",
        "counterfactual_transition_bounds",
        "target_query",
        seed_id=seed_id,
        anchors={"treatment": 0, "outcome": 2},
        manipulability={name: name in manipulable for name in world.variables},
        readable={name: True for name in world.variables},
        observation_bandwidth=observation_bandwidth,
    )


def _decision_chain_world(*, positive_first_edge: bool) -> WorldSpec:
    mediator_probabilities = (
        (Fraction(1, 4), Fraction(3, 4))
        if positive_first_edge
        else (Fraction(3, 4), Fraction(1, 4))
    )
    return WorldSpec(
        family="sampled_dag",
        topology="test-chain-d-m-y",
        variables=("D", "M", "Y"),
        domains=(2, 2, 2),
        state_names=(("0", "1"),) * 3,
        edges=((0, 1), (1, 2)),
        parents={0: (), 1: (0,), 2: (1,)},
        cpt={
            0: ((Fraction(1, 2), Fraction(1, 2)),),
            1: tuple((1 - probability, probability) for probability in mediator_probabilities),
            2: (
                (Fraction(3, 4), Fraction(1, 4)),
                (Fraction(1, 4), Fraction(3, 4)),
            ),
        },
    )


def _decision_chain_seed(world: WorldSpec, seed_id: str) -> Mapping[str, object]:
    return assemble_seed(
        world,
        "mechanism_hidden",
        "best_intervention",
        "decision",
        seed_id=seed_id,
        anchors={"decision_target": 0, "outcome": 2, "objective": "maximize"},
        manipulability={"D": False, "M": True, "Y": False},
        readable={"D": True, "M": True, "Y": True},
    )


class WorldSpaceSamplerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grammar = WorldGrammar(node_counts=(3, 4))

    def test_default_node_count_support_is_three_through_fifteen(self) -> None:
        self.assertEqual(WorldGrammar().node_counts, tuple(range(3, 16)))

    def test_upstream_worlds_load_as_legal_worldspecs(self) -> None:
        upstream = iter_upstream_worlds()
        self.assertGreaterEqual(len(upstream), 8)
        for world in upstream:
            self.assertTrue(legal_world(world))
        self.assertIn(
            3,
            {
                domain
                for world in upstream
                if world.topology == "survey"
                for domain in world.domains
            },
        )

    def test_pinned_seed_worlds_are_inside_the_world_space(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        for seed in load_candidate_seed_manifest():
            source = seed.world_source
            if source.get("type") == "cladder_meta_model":
                world = load_cladder_world(int(source["model_id"]))
            elif source.get("type") == "bnlearn_bif":
                world = load_bnlearn_world(repo / str(source["file"]))
            else:
                self.fail(f"unexpected world source type in {seed.seed_id}")
            self.assertTrue(legal_world(world), seed.seed_id)
            self.assertEqual(world.variables, seed.internal_variable_names())

    def test_cladder_loader_reproduces_upstream_observational_truth(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        models = json.loads(
            (repo / "data/worlds/cladder/meta-models-subset.json").read_text(encoding="utf-8")
        )
        for model in models:
            world = load_cladder_world(int(model["model_id"]))
            x_index = world.variables.index("X")
            y_index = world.variables.index("Y")
            joint: dict[tuple[int, ...], Fraction] = {}
            for values in product(range(2), repeat=len(world.variables)):
                probability = Fraction(1)
                for child in range(len(world.variables)):
                    row_index = 0
                    for parent in world.parents[child]:
                        row_index = row_index * 2 + values[parent]
                    probability *= world.cpt[child][row_index][values[child]]
                joint[values] = probability
            numerator = sum(
                probability
                for values, probability in joint.items()
                if values[x_index] == 1 and values[y_index] == 1
            )
            denominator = sum(
                probability for values, probability in joint.items() if values[x_index] == 1
            )
            self.assertAlmostEqual(
                float(numerator / denominator),
                float(model["groundtruth"]["P(Y=1 | X=1)"]),
                places=12,
            )

    def test_world_space_enumeration_starts_with_upstream_worlds(self) -> None:
        upstream = iter_upstream_worlds()
        sampled_count = 3
        enumerated = iter_world_space(
            self.grammar,
            include_upstream=True,
            start_seed=0,
            count=sampled_count,
        )
        self.assertEqual(len(enumerated), len(upstream) + sampled_count)
        self.assertEqual(enumerated[: len(upstream)], upstream)
        self.assertEqual(
            enumerated[len(upstream) :],
            tuple(sample_world(self.grammar, seed) for seed in range(sampled_count)),
        )
        self.assertTrue(all(legal_world(world) for world in enumerated))

    def test_sampled_worlds_are_legal_and_deterministic(self) -> None:
        for seed in range(30):
            world = sample_world(self.grammar, seed)
            self.assertTrue(legal_world(world))
            self.assertEqual(world, sample_world(self.grammar, seed))

    def test_three_to_fifteen_node_sampling_is_reproducible(self) -> None:
        grammar = WorldGrammar(node_counts=tuple(range(3, 16)), max_domain_size=2)
        first = tuple(sample_world(grammar, seed) for seed in range(30))
        second = tuple(sample_world(grammar, seed) for seed in range(30))

        self.assertEqual(first, second)
        self.assertTrue(all(3 <= len(world.variables) <= 15 for world in first))
        self.assertTrue(all(legal_world(world) for world in first))

    def test_task_world_holds_the_first_sampled_node_count_fixed(self) -> None:
        grammar = WorldGrammar(node_counts=tuple(range(3, 16)), max_domain_size=2)
        for query_type in QUERY_TYPES:
            for sample_index in range(30):
                expected_count = random.Random(sample_index).choice(grammar.node_counts)
                world = sample_task_world(grammar, sample_index, query_type)
                self.assertEqual(len(world.variables), expected_count)
                self.assertTrue(supports_query(world, query_type))

    def test_world_first_sampler_selects_one_role_and_uses_shared_composition(self) -> None:
        grammar = WorldGrammar(node_counts=(3,), max_domain_size=2)
        for sample_index in range(100):
            expanded = iter_sampled_seeds(
                grammar,
                query_types=("ate",),
                start_seed=sample_index,
                count=1,
            )
            self.assertEqual(len(expanded), 1)
            match = re.search(r"-a(\d+)$", str(expanded[0]["seed_id"]))
            self.assertIsNotNone(match)
            anchor_index = int(match.group(1))
            composed = assemble_sampled_anchor_tasks(
                grammar,
                sample_index,
                "ate",
                anchor_index,
            )
            self.assertEqual(tuple(seed for _, seed in composed), expanded)
            self.assertTrue(all(legal_world(world) for world, _ in composed))
            return
        self.fail("no world-first sampled ATE task found")

    def test_opaque_label_collision_retries_with_a_deterministic_nonce(self) -> None:
        sample_index = 1923676317
        query_type = "counterfactual_transition_bounds"
        anchor_index = 3
        seed_id = f"SAMPLED-{sample_index}-{query_type}-target_query-a{anchor_index}"
        label_pool = "DEFGHIJKLMNOPQRSTUVW"
        first_attempts = tuple(
            "".join(
                label_pool[byte % len(label_pool)]
                for byte in hashlib.sha256(
                    f"cpt-world-space-labels-v1\0{seed_id}\0{index}\0{index}".encode()
                ).digest()[:3]
            )
            for index in range(6)
        )
        self.assertLess(len(set(first_attempts)), len(first_attempts))

        grammar = WorldGrammar(node_counts=(6,), max_domain_size=2)
        first = assemble_sampled_anchor_tasks(grammar, sample_index, query_type, anchor_index)
        second = assemble_sampled_anchor_tasks(grammar, sample_index, query_type, anchor_index)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        for _, seed in first:
            labels = tuple(seed["visible_schema"]["variable_labels"].values())
            self.assertEqual(len(labels), 6)
            self.assertEqual(len(set(labels)), 6)

    def test_declared_distribution_covers_declared_dags(self) -> None:
        for node_count, sample_size, expected_count in ((2, 200, 3), (3, 5000, 25)):
            grammar = WorldGrammar(node_counts=(node_count,), max_domain_size=2)
            canonical = set()
            for seed in range(sample_size):
                world = sample_world(grammar, seed)
                canonical.add(tuple(sorted(world.edges)))
            self.assertEqual(len(canonical), expected_count)

    def test_grammar_samples_multi_valued_domains_legally(self) -> None:
        grammar = WorldGrammar(node_counts=(3,), max_domain_size=5)
        for seed in range(1000):
            world = sample_world(grammar, seed)
            if world.domains == (5, 5, 5) and world.edges:
                self.assertTrue(legal_world(world))
                for rows in world.cpt.values():
                    for row in rows:
                        self.assertEqual(sum(row), 1)
                return
        self.fail("no sampled multi-valued world with edges found")

    def test_task_target_profiling_reports_distribution_without_thresholds(self) -> None:
        for seed in range(50):
            world = sample_world(self.grammar, seed)
            anchors = legal_query_anchors(world, "ate")
            if anchors:
                profile = profile_task_targets(self.grammar, seed, "ate", anchors[0])
                self.assertGreater(profile["sample_count"], 0)
                self.assertEqual(
                    profile["sample_count"],
                    profile["zero_count"] + profile["negative_count"] + profile["positive_count"],
                )
                self.assertLessEqual(profile["target_min"], profile["target_max"])
                return
        self.fail("no profileable ATE world found")

    def test_task_answerability_includes_passive_observation_laws(self) -> None:
        first = _indirect_ate_world(mediator_given_treatment=(Fraction(1, 4), Fraction(3, 4)))
        indistinguishable = _indirect_ate_world(
            mediator_given_treatment=(Fraction(2, 5), Fraction(3, 5))
        )
        distinguishable = _indirect_ate_world(
            mediator_given_treatment=(Fraction(1, 4), Fraction(3, 4)),
            outcome_given_mediator=(Fraction(1, 10), Fraction(9, 10)),
        )
        first_seed = _indirect_ate_seed(first, "ATE-FIRST")
        indistinguishable_seed = _indirect_ate_seed(indistinguishable, "ATE-INDIST")
        distinguishable_seed = _indirect_ate_seed(distinguishable, "ATE-DIST")

        passively_distinguishable = task_answerability(
            ((first, first_seed), (indistinguishable, indistinguishable_seed))
        )
        self.assertEqual(
            passively_distinguishable,
            {"ATE-FIRST": "answerable", "ATE-INDIST": "answerable"},
        )
        answerable = task_answerability(
            ((first, first_seed), (distinguishable, distinguishable_seed))
        )
        self.assertEqual(
            answerable,
            {"ATE-FIRST": "answerable", "ATE-DIST": "answerable"},
        )

    def test_decision_answerability_uses_passive_evidence_before_shared_action(self) -> None:
        positive = _decision_chain_world(positive_first_edge=True)
        negative = _decision_chain_world(positive_first_edge=False)
        positive_seed = _decision_chain_seed(positive, "DECISION-POSITIVE")
        negative_seed = _decision_chain_seed(negative, "DECISION-NEGATIVE")

        passively_distinguishable = task_answerability(
            ((positive, positive_seed), (negative, negative_seed))
        )
        self.assertEqual(
            passively_distinguishable,
            {
                "DECISION-POSITIVE": "answerable",
                "DECISION-NEGATIVE": "answerable",
            },
        )

        repeated_positive_seed = dict(positive_seed)
        repeated_positive_seed["seed_id"] = "DECISION-POSITIVE-REPEATED"
        answerable = task_answerability(
            ((positive, positive_seed), (positive, repeated_positive_seed))
        )
        self.assertEqual(
            answerable,
            {
                "DECISION-POSITIVE": "answerable",
                "DECISION-POSITIVE-REPEATED": "answerable",
            },
        )

    def test_counterfactual_answerability_precedes_k_and_m_and_truth_is_invariant(self) -> None:
        first = _counterfactual_km_world(
            mediator_given_treatment=(Fraction(1, 4), Fraction(3, 4)),
            outcome_given_mediator=(Fraction(1, 5), Fraction(4, 5)),
        )
        second = _counterfactual_km_world(
            mediator_given_treatment=(Fraction(2, 5), Fraction(3, 5)),
            outcome_given_mediator=(Fraction(1, 10), Fraction(9, 10)),
        )

        low_information = tuple(
            (
                world,
                _counterfactual_km_seed(
                    world,
                    f"CF-LOW-{index}",
                    manipulable=frozenset({"Z"}),
                    observation_bandwidth=1,
                ),
            )
            for index, world in enumerate((first, second))
        )
        self.assertEqual(
            task_answerability(low_information),
            {"CF-LOW-0": "answerable", "CF-LOW-1": "answerable"},
        )
        self.assertNotEqual(
            compute_query_truth(first, low_information[0][1]),
            compute_query_truth(second, low_information[1][1]),
        )

        wider_observation = tuple(
            (
                world,
                _counterfactual_km_seed(
                    world,
                    f"CF-M2-{index}",
                    manipulable=frozenset({"Z"}),
                    observation_bandwidth=2,
                ),
            )
            for index, world in enumerate((first, second))
        )
        self.assertEqual(
            task_answerability(wider_observation),
            {"CF-M2-0": "answerable", "CF-M2-1": "answerable"},
        )

        wider_intervention = tuple(
            (
                world,
                _counterfactual_km_seed(
                    world,
                    f"CF-K2-{index}",
                    manipulable=frozenset({"M", "Z"}),
                    observation_bandwidth=1,
                ),
            )
            for index, world in enumerate((first, second))
        )
        self.assertEqual(
            task_answerability(wider_intervention),
            {"CF-K2-0": "answerable", "CF-K2-1": "answerable"},
        )
        for candidate_group in (low_information, wider_observation, wider_intervention):
            for index, (world, seed) in enumerate(candidate_group):
                expected = compute_query_truth(world, low_information[index][1])
                self.assertEqual(compute_query_truth(world, seed), expected)

    def test_difficulty_profile_is_separate_from_family_answerability(self) -> None:
        for seed in range(200):
            world = sample_world(self.grammar, seed)
            anchors = legal_query_anchors(world, "ate")
            if not anchors:
                continue
            anchors = anchors[0]
            profile = task_difficulty_profile(self.grammar, seed, "ate", anchors, sample_count=50)
            self.assertNotIn("answerability", profile)
            self.assertIn("structure", profile)
            self.assertIn("target_profile", profile)
            self.assertGreaterEqual(
                profile["structure"]["node_count"],
                2,
            )
            return
        self.fail("no profileable ATE task found")

    def test_sample_task_world_never_resamples_on_numerical_answers(self) -> None:
        for query_type in (
            "ate",
            "counterfactual_transition_bounds",
            "backadj_minimal_sets",
            "best_intervention",
            "mediator_set",
        ):
            for seed in range(200):
                structural = sample_world(self.grammar, seed)
                anchors_list = legal_query_anchors(structural, query_type)
                if not anchors_list:
                    continue
                anchors = anchors_list[0]
                world = sample_task_world(self.grammar, seed, query_type, anchors)
                self.assertTrue(legal_world(world))
                self.assertEqual(world, structural)
                break
            else:
                self.fail(f"no structurally legal sampled {query_type} world found")

    def test_structural_discovery_uses_the_main_sampler_without_numeric_resampling(
        self,
    ) -> None:
        seeds = iter_sampled_seeds(
            self.grammar,
            count=30,
            query_types=("backadj_minimal_sets", "mediator_set"),
        )
        self.assertTrue(seeds)
        self.assertEqual(
            {seed["query"]["type"] for seed in seeds},
            {"backadj_minimal_sets", "mediator_set"},
        )
        for seed in seeds:
            self.assertEqual(check_seed_legality(seed), [])
            self.assertNotIn("answerability", seed)
            self.assertEqual(seed["task_head"]["head"], "discovery")
        with self.assertRaisesRegex(ValueError, "collider_bias"):
            iter_sampled_seeds(
                self.grammar,
                count=1,
                query_types=("collider_bias",),
            )

    def test_rendered_seed_hides_internal_world(self) -> None:
        grammar = WorldGrammar(node_counts=(3,), max_domain_size=2)
        for seed in range(200):
            world = sample_world(grammar, seed)
            anchors = legal_query_anchors(world, "ate")
            if not anchors:
                continue
            seed_obj = assemble_seed(
                world,
                "mechanism_hidden",
                "ate",
                "target_query",
                anchors=anchors[0],
                seed_id="RENDER-TEST",
            )
            prompt = render_seed_task_prompt(
                seed_obj,
                budget=Budget(max_observations=64),
            )
            for name in world.variables:
                self.assertIsNone(
                    re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", prompt)
                )
            self.assertIn("DOLENS HIDDEN-MECHANISM TASK", prompt)
            self.assertIn("Estimate the average treatment effect", prompt)
            self.assertNotIn("probability", prompt.lower())
            return
        self.fail("no renderable ATE world found")

    def test_sampled_ate_uses_main_pipeline_with_readonly_endpoints(self) -> None:
        seeds = iter_sampled_seeds(
            self.grammar,
            count=30,
            query_types=("ate",),
        )
        self.assertTrue(seeds)
        widths: set[int] = set()
        bandwidths: set[int] = set()
        for seed in seeds:
            self.assertNotIn("answerability", seed)
            labels = seed["visible_schema"]["variable_labels"]
            internal_by_label = {visible: internal for internal, visible in labels.items()}
            treatment = internal_by_label[seed["query"]["treatment"]]
            outcome = internal_by_label[seed["query"]["outcome"]]
            self.assertFalse(seed["manipulability"][treatment])
            self.assertFalse(seed["manipulability"][outcome])
            self.assertTrue(
                any(
                    allowed and name not in {treatment, outcome}
                    for name, allowed in seed["manipulability"].items()
                )
            )
            width = sum(seed["manipulability"].values())
            widths.add(width)
            self.assertGreaterEqual(width, 1)
            self.assertLessEqual(width, len(seed["world_source"]["variables"]) - 2)
            bandwidth = seed["observation_bandwidth"]
            bandwidths.add(bandwidth)
            self.assertGreaterEqual(bandwidth, 1)
            self.assertLessEqual(bandwidth, len(seed["world_source"]["variables"]))
        self.assertGreater(len(widths), 1)
        self.assertGreater(len(bandwidths), 1)

        repeated = iter_sampled_seeds(
            self.grammar,
            count=30,
            query_types=("ate",),
        )
        self.assertEqual(
            [
                (seed["seed_id"], seed["manipulability"], seed["observation_bandwidth"])
                for seed in seeds
            ],
            [
                (seed["seed_id"], seed["manipulability"], seed["observation_bandwidth"])
                for seed in repeated
            ],
        )

    def test_sampled_state_anchors_are_legal_and_not_numerically_filtered(self) -> None:
        grammar = WorldGrammar(node_counts=(5,), max_domain_size=5)
        observed_pairs: set[tuple[int, int]] = set()
        observed_outcome_states: set[int] = set()
        for sample_index in range(60):
            seed = iter_sampled_seeds(
                grammar,
                start_seed=sample_index,
                count=1,
                query_types=("ate",),
            )[0]
            world = sample_task_world(grammar, sample_index, "ate")
            labels = seed["visible_schema"]["variable_labels"]
            internal_by_label = {visible: internal for internal, visible in labels.items()}
            treatment = world.variables.index(internal_by_label[seed["query"]["treatment"]])
            outcome = world.variables.index(internal_by_label[seed["query"]["outcome"]])
            baseline = int(str(seed["query"]["baseline_value"]).removeprefix("state_"))
            comparison = int(str(seed["query"]["treatment_value"]).removeprefix("state_"))
            outcome_state = int(str(seed["query"]["outcome_state"]).removeprefix("state_"))
            self.assertNotEqual(baseline, comparison)
            self.assertIn(baseline, range(world.domains[treatment]))
            self.assertIn(comparison, range(world.domains[treatment]))
            self.assertIn(outcome_state, range(world.domains[outcome]))
            observed_pairs.add((baseline, comparison))
            observed_outcome_states.add(outcome_state)
            self.assertEqual(compute_query_truth(world, seed)["type"], "ate")
        self.assertGreater(len(observed_pairs), 1)
        self.assertGreater(len(observed_outcome_states), 1)

    def test_all_five_families_sample_k_and_m_within_the_declared_support(self) -> None:
        grammar = WorldGrammar(node_counts=(6,), max_domain_size=3)
        for query_type in QUERY_TYPES:
            seeds = iter_sampled_seeds(grammar, count=20, query_types=(query_type,))
            expected_count = 40 if query_type == "counterfactual_transition_bounds" else 20
            self.assertEqual(len(seeds), expected_count)
            for seed in seeds:
                labels = seed["visible_schema"]["variable_labels"]
                internal_by_label = {visible: internal for internal, visible in labels.items()}
                anchor_names = QUERY_TYPES[query_type]["anchors"]
                anchor_variables = {
                    internal_by_label[seed["query"][anchor_name]] for anchor_name in anchor_names
                }
                self.assertTrue(all(not seed["manipulability"][name] for name in anchor_variables))
                width = sum(seed["manipulability"].values())
                self.assertGreaterEqual(width, 1)
                self.assertLessEqual(width, 4)
                self.assertIn(seed["observation_bandwidth"], range(1, 7))

    def test_counterfactual_bounds_use_the_same_main_pipeline_and_k_m_surface(self) -> None:
        seeds = iter_sampled_seeds(
            self.grammar,
            count=30,
            query_types=("counterfactual_transition_bounds",),
        )
        self.assertTrue(seeds)
        widths: set[int] = set()
        bandwidths: set[int] = set()
        paired: dict[str, dict[str, Mapping[str, object]]] = {}
        for seed in seeds:
            self.assertEqual(seed["query"]["type"], "counterfactual_transition_bounds")
            self.assertNotIn("answerability", seed)
            self.assertNotIn("answerability_scope", seed)
            labels = seed["visible_schema"]["variable_labels"]
            internal_by_label = {visible: internal for internal, visible in labels.items()}
            treatment = internal_by_label[seed["query"]["treatment"]]
            outcome = internal_by_label[seed["query"]["outcome"]]
            self.assertFalse(seed["manipulability"][treatment])
            self.assertFalse(seed["manipulability"][outcome])
            widths.add(sum(seed["manipulability"].values()))
            bandwidths.add(int(seed["observation_bandwidth"]))
            answer_mode = str(seed["query"]["answer_mode"])
            base_seed_id = str(seed["seed_id"]).rsplit("-mode-", 1)[0]
            paired.setdefault(base_seed_id, {})[answer_mode] = seed
            prompt = render_seed_task_prompt(seed)
            self.assertIn("no single hidden SCM", prompt)
            if answer_mode == "sharp_interval":
                self.assertIn("sharp legal interval", prompt)
                self.assertIn('"lower": <number in [0,1]>', prompt)
            else:
                self.assertEqual(answer_mode, "compatible_value")
                self.assertIn("one value for q that is compatible", prompt)
                self.assertIn("does not claim that q is point identified", prompt)
                self.assertIn('"value": <number in [0,1]>', prompt)
        self.assertTrue(paired)
        for modes in paired.values():
            self.assertEqual(set(modes), {"sharp_interval", "compatible_value"})
            interval = modes["sharp_interval"]
            compatible = modes["compatible_value"]
            for field in (
                "world_source",
                "manipulability",
                "readable",
                "observation_bandwidth",
            ):
                self.assertEqual(interval[field], compatible[field])
            self.assertEqual(
                interval["visible_schema"]["variable_labels"],
                compatible["visible_schema"]["variable_labels"],
            )
        self.assertGreater(len(widths), 1)
        self.assertGreater(len(bandwidths), 1)

    def test_sampled_decision_separates_experiments_from_deployment(self) -> None:
        seeds = iter_sampled_seeds(
            self.grammar,
            count=30,
            query_types=("best_intervention",),
        )
        self.assertTrue(seeds)
        for seed in seeds:
            self.assertNotIn("answerability", seed)
            labels = seed["visible_schema"]["variable_labels"]
            internal_by_label = {visible: internal for internal, visible in labels.items()}
            decision_target = internal_by_label[seed["query"]["decision_target"]]
            outcome = internal_by_label[seed["query"]["outcome"]]
            self.assertFalse(seed["manipulability"][decision_target])
            self.assertFalse(seed["manipulability"][outcome])
            self.assertTrue(any(seed["manipulability"].values()))
            self.assertIn("observation_bandwidth", seed)
            prompt = render_seed_task_prompt(seed)
            self.assertIn("Final deployment decision", prompt)
            self.assertIn("Legal experimental do targets", prompt)
            experimental_line = next(
                line for line in prompt.splitlines() if line.startswith("Legal experimental")
            )
            self.assertNotIn(labels[decision_target], experimental_line)

    def test_pinned_seeds_render_without_internal_leaks(self) -> None:
        for seed in load_candidate_seed_manifest():
            prompt = render_seed_task_prompt(
                asdict(seed),
                budget=Budget(max_observations=64),
            )
            for name in seed.internal_variable_names():
                self.assertIsNone(
                    re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", prompt),
                    seed.seed_id,
                )
            for token in ("graph", "probability", "model_id", "bnlearn", "cladder", "Pollution"):
                self.assertNotIn(token, prompt)

    def test_render_is_invariant_to_hidden_cpt_changes(self) -> None:
        grammar = WorldGrammar(node_counts=(3,), max_domain_size=2)
        for seed in range(200):
            world = sample_world(grammar, seed)
            anchors = legal_query_anchors(world, "ate")
            if not anchors:
                continue
            changed = replace(
                world,
                cpt={
                    **world.cpt,
                    0: ((Fraction(1, 3), Fraction(2, 3)),),
                },
            )
            first = assemble_seed(
                world,
                "mechanism_hidden",
                "ate",
                "target_query",
                anchors=anchors[0],
                seed_id="HIDE-INVARIANT",
            )
            second = assemble_seed(
                changed,
                "mechanism_hidden",
                "ate",
                "target_query",
                anchors=anchors[0],
                seed_id="HIDE-INVARIANT",
            )
            self.assertNotEqual(first["world_source"]["cpt"], second["world_source"]["cpt"])
            budget = Budget(max_observations=64)
            self.assertEqual(
                render_seed_task_prompt(first, budget=budget),
                render_seed_task_prompt(second, budget=budget),
            )
            return
        self.fail("no invariant-render world found")

    def test_legal_anchor_assignments_never_use_disconnected_pairs(self) -> None:
        for seed in range(30):
            world = sample_world(self.grammar, seed)
            for query_type in ("ate", "backadj_minimal_sets", "mediator_set"):
                for anchors in legal_query_anchors(world, query_type):
                    source = int(anchors["treatment"])
                    target = int(anchors["outcome"])
                    self.assertIsNotNone(world.shortest_path_nodes(source, target))
                    if query_type == "mediator_set":
                        self.assertTrue(world.has_indirect_path(source, target))

    def test_sampled_seed_family_is_diverse_and_legality_checked(self) -> None:
        seeds = iter_sampled_seeds(
            self.grammar,
            start_seed=0,
            count=30,
            query_types=tuple(QUERY_TYPES),
        )
        self.assertGreater(len(seeds), 6)
        for seed in seeds:
            self.assertEqual(check_seed_legality(seed), [])
            self.assertGreaterEqual(sum(seed["manipulability"].values()), 1)
            self.assertGreaterEqual(seed["observation_bandwidth"], 1)
            self.assertLessEqual(
                seed["observation_bandwidth"],
                len(seed["world_source"]["variables"]),
            )
        self.assertGreater(len({seed["seed_id"].split("-a", 1)[0] for seed in seeds}), 6)
        sampled_types = {seed["query"]["type"] for seed in seeds}
        self.assertEqual(sampled_types, set(QUERY_TYPES))
        self.assertGreater(
            len({tuple(tuple(edge) for edge in seed["world_source"]["edges"]) for seed in seeds}),
            1,
        )

    def test_sampler_produces_more_tasks_than_pinned_seeds(self) -> None:
        seeds = []
        for seed in range(30):
            world = sample_world(self.grammar, seed)
            for query_type in QUERY_TYPES:
                if not supports_query(world, query_type):
                    continue
                for task_head in ("target_query", "discovery", "decision"):
                    if not supports_task(query_type, task_head):
                        continue
                    try:
                        seed_obj = assemble_seed(
                            world,
                            "mechanism_hidden",
                            query_type,
                            task_head,
                            seed_id=f"SAMPLED-{seed}-{query_type}-{task_head}",
                        )
                    except ValueError:
                        continue
                    self.assertEqual(check_seed_legality(seed_obj), [])
                    seeds.append(seed_obj)
        self.assertGreater(len(seeds), 6)
        self.assertGreater(
            len({tuple(tuple(edge) for edge in seed["world_source"]["edges"]) for seed in seeds}),
            1,
        )


if __name__ == "__main__":
    unittest.main()
