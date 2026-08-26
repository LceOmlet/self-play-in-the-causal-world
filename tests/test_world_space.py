from __future__ import annotations

import hashlib
import json
import math
import random
import re
import unittest
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, replace
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
from unittest.mock import patch

from cpt_world import (
    QUERY_TYPES,
    TASK_FAMILY_QUERY_TYPES,
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
from cpt_world.world_space import (
    _ET_V2_STRENGTH_CEILING,
    _build_world,
    _combine_effect_blocks,
    _contextual_parent_pair_score_scale,
    _exponential_tilt_rows,
    _minimum_backdoor_adjustment_size,
    _parent_interaction_projection,
    _project_joint_effect,
    _project_parent_subset_effect,
    _sample_et_v2_strength,
    _sample_parent_subset_balanced_effect,
    _sampled_backdoor_complexity,
    _sampled_role_assignments,
    _SampledStructure,
    _single_parent_pairwise_score_scale,
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
        "individual_counterfactual_probability",
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

    def test_default_node_count_support_is_eight_through_sixteen(self) -> None:
        self.assertEqual(WorldGrammar().node_counts, tuple(range(8, 17)))

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

    def test_eight_to_sixteen_node_sampling_is_reproducible(self) -> None:
        grammar = WorldGrammar(node_counts=tuple(range(8, 17)), max_domain_size=2)
        first = tuple(sample_world(grammar, seed) for seed in range(30))
        second = tuple(sample_world(grammar, seed) for seed in range(30))

        self.assertEqual(first, second)
        self.assertTrue(all(8 <= len(world.variables) <= 16 for world in first))
        self.assertTrue(all(legal_world(world) for world in first))

    def test_task_world_holds_the_first_sampled_node_count_fixed(self) -> None:
        grammar = WorldGrammar(node_counts=tuple(range(8, 17)), max_domain_size=2)
        for query_type in QUERY_TYPES:
            for sample_index in range(30):
                expected_count = random.Random(sample_index).choice(grammar.node_counts)
                world = sample_task_world(grammar, sample_index, query_type)
                self.assertEqual(len(world.variables), expected_count)
                self.assertTrue(supports_query(world, query_type))

    def test_formal_ate_and_backdoor_samplers_stratify_minimum_adjustment_size(self) -> None:
        grammar = WorldGrammar(max_domain_size=2)
        for query_type in ("ate", "backadj_minimal_sets"):
            for sample_index in range(80):
                world = sample_task_world(grammar, sample_index, query_type)
                target = _sampled_backdoor_complexity(sample_index, query_type)
                roles = _sampled_role_assignments(
                    len(world.variables),
                    world.edges,
                    query_type,
                    sample_index,
                )
                self.assertTrue(roles)
                self.assertTrue(
                    all(
                        _minimum_backdoor_adjustment_size(
                            len(world.variables),
                            world.edges,
                            role["treatment"],
                            role["outcome"],
                        )
                        == target
                        for role in roles
                    )
                )

    def test_backdoor_complexity_axis_is_uniform_and_seed_fixed(self) -> None:
        for query_type in ("ate", "backadj_minimal_sets"):
            first = tuple(
                _sampled_backdoor_complexity(sample_index, query_type)
                for sample_index in range(4000)
            )
            second = tuple(
                _sampled_backdoor_complexity(sample_index, query_type)
                for sample_index in range(4000)
            )
            self.assertEqual(first, second)
            counts = Counter(first)
            self.assertEqual(set(counts), {0, 1, 2, 3})
            self.assertTrue(all(900 <= counts[value] <= 1100 for value in range(4)))

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
        sample_index = 1000243
        query_type = "individual_counterfactual_probability"
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
        self.assertEqual(len(first), 1)
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
                        self.assertAlmostEqual(math.fsum(row), 1.0, places=12)
                return
        self.fail("no sampled multi-valued world with edges found")

    def test_parent_subset_blocks_are_orthogonal_and_pure(self) -> None:
        parent_domains = (2, 3, 2)
        child_domain = 4
        rng = random.Random(20260825)
        parent_subsets = tuple(
            positions
            for subset_size in range(1, len(parent_domains) + 1)
            for positions in combinations(range(len(parent_domains)), subset_size)
        )
        blocks = tuple(
            _project_parent_subset_effect(parent_domains, positions, child_domain, rng)
            for positions in parent_subsets
        )

        def squared_norm(table: tuple[tuple[float, ...], ...]) -> float:
            return math.fsum(value * value for row in table for value in row)

        for positions, block in zip(parent_subsets, blocks, strict=True):
            self.assertAlmostEqual(squared_norm(block), 1.0, places=12)
            for other_positions in parent_subsets:
                projection = _parent_interaction_projection(
                    block,
                    parent_domains,
                    other_positions,
                )
                self.assertAlmostEqual(
                    squared_norm(projection),
                    1.0 if other_positions == positions else 0.0,
                    places=11,
                )

        for left in range(len(blocks)):
            for right in range(left + 1, len(blocks)):
                inner = math.fsum(
                    a * b
                    for left_row, right_row in zip(blocks[left], blocks[right], strict=True)
                    for a, b in zip(left_row, right_row, strict=True)
                )
                self.assertAlmostEqual(inner, 0.0, places=12)

        shares = (0.04, 0.08, 0.12, 0.16, 0.20, 0.18, 0.22)
        combined = _combine_effect_blocks(blocks, shares)
        self.assertAlmostEqual(squared_norm(combined), 1.0, places=12)
        for positions, expected_share in zip(parent_subsets, shares, strict=True):
            projection = _parent_interaction_projection(
                combined,
                parent_domains,
                positions,
            )
            self.assertAlmostEqual(squared_norm(projection), expected_share, places=11)

    def test_parent_subset_energy_is_random_with_equal_expectation(self) -> None:
        parent_domains = (2, 2, 2)
        parent_subsets = tuple(
            positions
            for subset_size in range(1, len(parent_domains) + 1)
            for positions in combinations(range(len(parent_domains)), subset_size)
        )
        totals = [0.0] * len(parent_subsets)
        first_shares: tuple[float, ...] | None = None
        sample_count = 1000
        rng = random.Random(271828)
        for _ in range(sample_count):
            effect = _sample_parent_subset_balanced_effect(parent_domains, 3, rng)
            shares = tuple(
                math.fsum(
                    value * value
                    for row in _parent_interaction_projection(
                        effect,
                        parent_domains,
                        positions,
                    )
                    for value in row
                )
                for positions in parent_subsets
            )
            if first_shares is None:
                first_shares = shares
            for index, share in enumerate(shares):
                totals[index] += share

        self.assertIsNotNone(first_shares)
        self.assertGreater(max(first_shares) - min(first_shares), 0.05)
        expected = 1.0 / len(parent_subsets)
        for total in totals:
            self.assertAlmostEqual(total / sample_count, expected, delta=0.015)

    def test_one_parent_subset_sampling_preserves_the_original_direction_law_and_rng(self) -> None:
        separated_rng = random.Random(314159)
        original_rng = random.Random(314159)
        separated = _sample_parent_subset_balanced_effect((5,), 4, separated_rng)
        original = _project_joint_effect(5, 4, original_rng)

        for separated_row, original_row in zip(separated, original, strict=True):
            for separated_value, original_value in zip(separated_row, original_row, strict=True):
                self.assertAlmostEqual(separated_value, original_value, places=14)
        self.assertEqual(separated_rng.random(), original_rng.random())

    def test_et_v2_is_legal_and_does_not_use_an_additive_zero_wall(self) -> None:
        base = (0.4995, 0.4995, 0.001)
        direction = ((1.0, 0.0, -1.0), (-1.0, 0.0, 1.0))
        rows = _exponential_tilt_rows(base, direction, 1.0)

        for row in rows:
            self.assertTrue(all(0.0 < value < 1.0 for value in row))
            self.assertAlmostEqual(math.fsum(row), 1.0, places=12)
        self.assertGreater(abs(rows[0][0] - rows[1][0]), 0.4)

    def test_et_v2_is_scale_and_row_replication_invariant(self) -> None:
        base = (0.6, 0.3, 0.1)
        direction = ((1.0, -0.5, -0.5), (-1.0, 0.5, 0.5))
        original = _exponential_tilt_rows(base, direction, 0.8)
        scaled = _exponential_tilt_rows(
            base,
            tuple(tuple(7.0 * value for value in row) for row in direction),
            0.8,
        )
        replicated = _exponential_tilt_rows(base, (*direction, *direction), 0.8)

        self.assertEqual(original, scaled)
        self.assertEqual(replicated, (*original, *original))
        zero_strength = _exponential_tilt_rows(base, direction, 0.0)
        for row in zero_strength:
            for actual, expected in zip(row, base, strict=True):
                self.assertAlmostEqual(actual, expected, places=15)

    def test_et_v2_strength_has_unit_expected_squared_score_energy(self) -> None:
        self.assertAlmostEqual(_ET_V2_STRENGTH_CEILING**2 / 3.0, 1.0, places=15)
        rng = random.Random(161803)
        strengths = tuple(_sample_et_v2_strength(rng) for _ in range(10000))
        self.assertTrue(all(0.0 <= value < _ET_V2_STRENGTH_CEILING for value in strengths))
        self.assertAlmostEqual(
            math.fsum(value * value for value in strengths) / len(strengths),
            1.0,
            delta=0.02,
        )

        base = (0.6, 0.3, 0.1)
        direction = ((1.0, -0.5, -0.5), (-1.0, 0.5, 0.5))
        rows = _exponential_tilt_rows(
            base,
            direction,
            _ET_V2_STRENGTH_CEILING,
        )
        self.assertTrue(all(value > 0.0 for row in rows for value in row))
        with self.assertRaises(ValueError):
            _exponential_tilt_rows(
                base,
                direction,
                math.nextafter(_ET_V2_STRENGTH_CEILING, math.inf),
            )

    def test_single_parent_pairwise_score_scale_preserves_binary_contrast(self) -> None:
        rng = random.Random(141421)
        for parent_domain_size in range(2, 6):
            child_domain_size = 4
            direction = _sample_parent_subset_balanced_effect(
                (parent_domain_size,),
                child_domain_size,
                rng,
            )
            squared_rms = math.fsum(value * value for row in direction for value in row) / (
                parent_domain_size * child_domain_size
            )
            score_scale = _single_parent_pairwise_score_scale(parent_domain_size)
            normalized = tuple(
                tuple(score_scale * value / math.sqrt(squared_rms) for value in row)
                for row in direction
            )
            mean_pairwise_squared_contrast = math.fsum(
                (normalized[left][state] - normalized[right][state]) ** 2
                for left in range(parent_domain_size)
                for right in range(left + 1, parent_domain_size)
                for state in range(child_domain_size)
            ) / (math.comb(parent_domain_size, 2) * child_domain_size)
            self.assertAlmostEqual(mean_pairwise_squared_contrast, 4.0, places=11)

        self.assertEqual(_single_parent_pairwise_score_scale(2), 1.0)
        with self.assertRaises(ValueError):
            _single_parent_pairwise_score_scale(1)

    def test_contextual_parent_pair_scale_matches_single_parent_correction(self) -> None:
        rng = random.Random(173205)
        for parent_domain in range(2, 6):
            for child_domain in range(2, 6):
                direction = _sample_parent_subset_balanced_effect(
                    (parent_domain,),
                    child_domain,
                    rng,
                )
                self.assertAlmostEqual(
                    _contextual_parent_pair_score_scale(direction, (parent_domain,)),
                    _single_parent_pairwise_score_scale(parent_domain),
                    places=12,
                )

    def test_contextual_parent_pair_scale_fixes_multi_parent_contrast_at_four(self) -> None:
        rng = random.Random(223607)
        for parent_domains in ((2, 2), (2, 3), (5, 2, 4)):
            child_domain = 4
            direction = _sample_parent_subset_balanced_effect(
                parent_domains,
                child_domain,
                rng,
            )
            squared_rms = math.fsum(
                value * value for row in direction for value in row
            ) / (math.prod(parent_domains) * child_domain)
            scale = _contextual_parent_pair_score_scale(direction, parent_domains)
            normalized = tuple(
                tuple(scale * value / math.sqrt(squared_rms) for value in row)
                for row in direction
            )
            parent_means: list[float] = []
            for position, parent_domain in enumerate(parent_domains):
                suffix_count = math.prod(parent_domains[position + 1 :])
                prefix_count = math.prod(parent_domains[:position])
                contrasts = [
                    (
                        normalized[(prefix * parent_domain + left) * suffix_count + suffix][state]
                        - normalized[
                            (prefix * parent_domain + right) * suffix_count + suffix
                        ][state]
                    )
                    ** 2
                    for prefix in range(prefix_count)
                    for suffix in range(suffix_count)
                    for left in range(parent_domain)
                    for right in range(left + 1, parent_domain)
                    for state in range(child_domain)
                ]
                parent_means.append(math.fsum(contrasts) / len(contrasts))
            self.assertAlmostEqual(
                math.fsum(parent_means) / len(parent_means),
                4.0,
                places=11,
            )

        with self.assertRaises(ValueError):
            _contextual_parent_pair_score_scale(((1.0, -1.0),), ())

    def test_contextual_multi_parent_correction_does_not_touch_single_parent_chain(self) -> None:
        structure = _SampledStructure(
            node_count=5,
            domains=(2, 5, 3, 4, 2),
            variables=("V0", "V1", "V2", "V3", "V4"),
            edges=((0, 1), (1, 2), (2, 3), (3, 4)),
            parents=((), (0,), (1,), (2,), (3,)),
            topology="single-parent-chain",
        )
        with patch(
            "cpt_world.world_space._contextual_parent_pair_score_scale",
            side_effect=AssertionError("multi-parent normalization must not run"),
        ):
            world = _build_world(structure, random.Random(244949))
        self.assertTrue(legal_world(world))

    def test_et_v2_score_scale_changes_only_the_declared_radial_coordinate(self) -> None:
        base = (0.6, 0.3, 0.1)
        direction = ((1.0, -0.5, -0.5), (-1.0, 0.5, 0.5))
        scale = _single_parent_pairwise_score_scale(5)
        scaled = _exponential_tilt_rows(
            base,
            direction,
            0.8,
            score_scale=scale,
        )
        equivalent = _exponential_tilt_rows(base, direction, 0.8 * scale)
        self.assertEqual(scaled, equivalent)
        with self.assertRaises(ValueError):
            _exponential_tilt_rows(base, direction, 0.8, score_scale=0.0)

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

    def test_counterfactual_target_profile_never_substitutes_frechet_outer_bounds(self) -> None:
        for seed in range(50):
            world = sample_world(self.grammar, seed)
            anchors = legal_query_anchors(world, "individual_counterfactual_probability")
            if anchors:
                with self.assertRaisesRegex(
                    NotImplementedError,
                    "cannot substitute Frechet outer bounds",
                ):
                    profile_task_targets(
                        self.grammar,
                        seed,
                        "individual_counterfactual_probability",
                        anchors[0],
                    )
                return
        self.fail("no profileable individual-counterfactual world found")

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
            "individual_counterfactual_probability",
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
            prompt = render_seed_task_prompt(seed)
            self.assertIn(
                f"effect = P({seed['query']['outcome']}={seed['query']['outcome_state']} | "
                f"do({seed['query']['treatment']}={seed['query']['treatment_value']})) - "
                f"P({seed['query']['outcome']}={seed['query']['outcome_state']} | "
                f"do({seed['query']['treatment']}={seed['query']['baseline_value']}))",
                prompt,
            )
        self.assertGreater(len(observed_pairs), 1)
        self.assertGreater(len(observed_outcome_states), 1)

    def test_all_five_families_sample_k_and_m_within_the_declared_support(self) -> None:
        grammar = WorldGrammar(node_counts=(6,), max_domain_size=3)
        for query_type in QUERY_TYPES:
            seeds = iter_sampled_seeds(grammar, count=20, query_types=(query_type,))
            self.assertEqual(len(seeds), 20)
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

    def test_individual_counterfactual_probability_uses_main_pipeline_and_k_m_surface(
        self,
    ) -> None:
        seeds = iter_sampled_seeds(
            self.grammar,
            count=30,
            query_types=("individual_counterfactual_probability",),
        )
        self.assertTrue(seeds)
        widths: set[int] = set()
        bandwidths: set[int] = set()
        for seed in seeds:
            self.assertEqual(seed["query"]["type"], "individual_counterfactual_probability")
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
            self.assertNotIn("answer_mode", seed["query"])
            self.assertIn("factual_value", seed["query"])
            self.assertIn("counterfactual_value", seed["query"])
            self.assertIn("factual_outcome_state", seed["query"])
            prompt = render_seed_task_prompt(seed)
            self.assertIn("One individual was assigned", prompt)
            self.assertIn("for this same individual", prompt)
            self.assertIn("q may lie in a compatible interval", prompt)
            self.assertIn("one value in that interval", prompt)
            self.assertNotIn('"lower": <number in [0,1]>', prompt)
            self.assertIn('"value": <number in [0,1]>', prompt)
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
            verb = "minimizes" if seed["query"]["objective"] == "minimize" else "maximizes"
            self.assertIn(
                f"{verb} P({seed['query']['outcome']}={seed['query']['outcome_state']})",
                prompt,
            )
            self.assertIn("Legal experimental do targets", prompt)
            experimental_line = next(
                line for line in prompt.splitlines() if line.startswith("Legal experimental")
            )
            self.assertNotIn(labels[decision_target], experimental_line)

    def test_discovery_prompts_define_empty_adjustment_and_mediator_endpoints(self) -> None:
        backdoor, mediator = iter_sampled_seeds(
            self.grammar,
            count=1,
            query_types=("backadj_minimal_sets", "mediator_set"),
        )
        backdoor_prompt = render_seed_task_prompt(backdoor)
        self.assertIn('"adjustment_sets":[[]]', backdoor_prompt)

        mediator_prompt = render_seed_task_prompt(mediator)
        self.assertIn(
            f"excluding {mediator['query']['treatment']} and {mediator['query']['outcome']}",
            mediator_prompt,
        )
        self.assertIn("including edges incident to the two endpoints", mediator_prompt)

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

    def test_uniform_training_mixture_is_exactly_balanced(self) -> None:
        count = 3
        seeds = iter_sampled_seeds(
            self.grammar,
            start_seed=0,
            count=count,
            query_types=TASK_FAMILY_QUERY_TYPES,
        )
        family_counts = {
            query_type: sum(seed["query"]["type"] == query_type for seed in seeds)
            for query_type in TASK_FAMILY_QUERY_TYPES
        }
        self.assertEqual(
            TASK_FAMILY_QUERY_TYPES,
            (
                "ate",
                "individual_counterfactual_probability",
                "backadj_minimal_sets",
                "best_intervention",
                "mediator_set",
            ),
        )
        self.assertEqual(family_counts, dict.fromkeys(TASK_FAMILY_QUERY_TYPES, count))
        self.assertEqual(len(seeds), count * len(TASK_FAMILY_QUERY_TYPES))

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
