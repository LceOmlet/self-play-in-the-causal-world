from __future__ import annotations

import json
import unittest
from dataclasses import replace
from fractions import Fraction

from cpt_world import (
    TERMINAL_QUALITY_REWARD_VERSION,
    WorldGrammar,
    WorldSpec,
    assemble_seed,
    backdoor_adjustment_sets,
    legal_query_anchors,
    parse_terminal_answer,
    sample_task_world,
    sample_world,
    score_terminal_answer,
    terminal_quality_reward,
)

_HIDING_MODES = (
    "evidence_by_intervention_only",
    "manipulability_via_action_legality",
    "no_full_joint",
)


def _decision_world():
    world = WorldSpec(
        family="test_dag",
        topology="A-to-Y-from-B",
        variables=("A", "B", "Y"),
        domains=(2, 2, 2),
        state_names=(("a0", "a1"), ("b0", "b1"), ("y0", "y1")),
        edges=((0, 2), (1, 2)),
        parents={0: (), 1: (), 2: (0, 1)},
        cpt={
            0: ((Fraction(1, 2), Fraction(1, 2)),),
            1: ((Fraction(1, 2), Fraction(1, 2)),),
            2: (
                (Fraction(9, 10), Fraction(1, 10)),
                (Fraction(4, 5), Fraction(1, 5)),
                (Fraction(1, 5), Fraction(4, 5)),
                (Fraction(1, 10), Fraction(9, 10)),
            ),
        },
    )
    seed = assemble_seed(
        world,
        _HIDING_MODES,
        "best_intervention",
        "decision",
        anchors={"decision_target": 0, "outcome": 2, "objective": "minimize"},
        seed_id="SCORE-EXACT-DECISION",
    )
    return seed, world


def _backdoor_behavior_task():
    """Z confounds X/Y while M carries the directed X-to-Y effect."""

    world = WorldSpec(
        family="test_dag",
        topology="Z-to-X-and-Y-X-to-M-to-Y",
        variables=("Z", "X", "M", "Y"),
        domains=(2, 2, 2, 2),
        state_names=(("0", "1"),) * 4,
        edges=((0, 1), (0, 3), (1, 2), (2, 3)),
        parents={0: (), 1: (0,), 2: (1,), 3: (0, 2)},
        cpt={
            0: ((Fraction(1, 2), Fraction(1, 2)),),
            1: ((Fraction(4, 5), Fraction(1, 5)), (Fraction(1, 5), Fraction(4, 5))),
            2: ((Fraction(9, 10), Fraction(1, 10)), (Fraction(1, 10), Fraction(9, 10))),
            3: (
                (Fraction(9, 10), Fraction(1, 10)),
                (Fraction(3, 10), Fraction(7, 10)),
                (Fraction(7, 10), Fraction(3, 10)),
                (Fraction(1, 10), Fraction(9, 10)),
            ),
        },
    )
    seed = assemble_seed(
        world,
        _HIDING_MODES,
        "backadj_minimal_sets",
        "discovery",
        anchors={"treatment": 1, "outcome": 3},
        seed_id="SCORE-BACKDOOR-BEHAVIOR",
    )
    return seed, world


def _collider_behavior_task():
    """X and U are independent until conditioning on their collider C."""

    world = WorldSpec(
        family="test_dag",
        topology="X-to-C-from-U-X-and-U-to-Y",
        variables=("X", "U", "C", "Y"),
        domains=(2, 2, 2, 2),
        state_names=(("0", "1"),) * 4,
        edges=((0, 2), (1, 2), (0, 3), (1, 3)),
        parents={0: (), 1: (), 2: (0, 1), 3: (0, 1)},
        cpt={
            0: ((Fraction(1, 2), Fraction(1, 2)),),
            1: ((Fraction(1, 2), Fraction(1, 2)),),
            2: (
                (Fraction(9, 10), Fraction(1, 10)),
                (Fraction(1, 5), Fraction(4, 5)),
                (Fraction(1, 5), Fraction(4, 5)),
                (Fraction(1, 10), Fraction(9, 10)),
            ),
            3: (
                (Fraction(9, 10), Fraction(1, 10)),
                (Fraction(1, 5), Fraction(4, 5)),
                (Fraction(4, 5), Fraction(1, 5)),
                (Fraction(1, 10), Fraction(9, 10)),
            ),
        },
    )
    seed = assemble_seed(
        world,
        _HIDING_MODES,
        "backadj_minimal_sets",
        "discovery",
        anchors={"treatment": 0, "outcome": 3},
        seed_id="SCORE-COLLIDER-BEHAVIOR",
    )
    return seed, world


def _strong_weak_backdoor_task():
    """A and B are respectively strong and weak observed confounders."""

    def binary_row(probability_one: Fraction) -> tuple[Fraction, Fraction]:
        return Fraction(1) - probability_one, probability_one

    world = WorldSpec(
        family="test_dag",
        topology="A-and-B-to-X-and-Y-X-to-Y",
        variables=("A", "B", "X", "Y"),
        domains=(2, 2, 2, 2),
        state_names=(("0", "1"),) * 4,
        edges=((0, 2), (1, 2), (0, 3), (1, 3), (2, 3)),
        parents={0: (), 1: (), 2: (0, 1), 3: (0, 1, 2)},
        cpt={
            0: ((Fraction(1, 2), Fraction(1, 2)),),
            1: ((Fraction(1, 2), Fraction(1, 2)),),
            2: tuple(
                binary_row(probability)
                for probability in (
                    Fraction(1, 10),
                    Fraction(3, 20),
                    Fraction(17, 20),
                    Fraction(9, 10),
                )
            ),
            3: tuple(
                binary_row(probability)
                for probability in (
                    Fraction(1, 10),
                    Fraction(3, 10),
                    Fraction(3, 25),
                    Fraction(8, 25),
                    Fraction(7, 10),
                    Fraction(9, 10),
                    Fraction(18, 25),
                    Fraction(23, 25),
                )
            ),
        },
    )
    seed = assemble_seed(
        world,
        _HIDING_MODES,
        "backadj_minimal_sets",
        "discovery",
        anchors={"treatment": 2, "outcome": 3},
        seed_id="SCORE-STRONG-WEAK-BACKDOOR",
    )
    return seed, world


def _multiple_adjustment_task():
    """Either A or its parent B blocks the sole backdoor path."""

    world = WorldSpec(
        family="test_dag",
        topology="X-from-A-from-B-to-Y-and-X-to-Y",
        variables=("B", "A", "X", "Y"),
        domains=(2, 2, 2, 2),
        state_names=(("0", "1"),) * 4,
        edges=((0, 1), (1, 2), (0, 3), (2, 3)),
        parents={0: (), 1: (0,), 2: (1,), 3: (0, 2)},
        cpt={
            0: ((Fraction(1, 2), Fraction(1, 2)),),
            1: ((Fraction(4, 5), Fraction(1, 5)), (Fraction(1, 5), Fraction(4, 5))),
            2: ((Fraction(4, 5), Fraction(1, 5)), (Fraction(1, 5), Fraction(4, 5))),
            3: (
                (Fraction(9, 10), Fraction(1, 10)),
                (Fraction(1, 2), Fraction(1, 2)),
                (Fraction(1, 2), Fraction(1, 2)),
                (Fraction(1, 10), Fraction(9, 10)),
            ),
        },
    )
    seed = assemble_seed(
        world,
        _HIDING_MODES,
        "backadj_minimal_sets",
        "discovery",
        anchors={"treatment": 2, "outcome": 3},
        seed_id="SCORE-MULTIPLE-ADJUSTMENT",
    )
    return seed, world


def _find_task(query_type: str):
    grammar = WorldGrammar(node_counts=(2, 3, 4))
    for seed in range(300):
        structural = sample_world(grammar, seed)
        anchors_list = legal_query_anchors(structural, query_type)
        if not anchors_list:
            continue
        anchors = anchors_list[0]
        try:
            world = sample_task_world(grammar, seed, query_type, anchors)
        except ValueError:
            continue
        task_head = (
            "target_query"
            if query_type in {"ate", "individual_counterfactual_probability"}
            else "decision"
        )
        seed_obj = assemble_seed(
            world,
            "mechanism_hidden",
            query_type,
            task_head,
            anchors=anchors,
            seed_id=f"SCORE-{seed}-{query_type}",
        )
        return seed_obj, world
    raise AssertionError(f"no stable {query_type} task found")


class TaskScoringTests(unittest.TestCase):
    def test_target_query_parser_and_raw_score(self) -> None:
        seed_obj, world = _find_task("ate")
        truth = seed_obj["query"]
        self.assertEqual(truth["type"], "ate")
        outcome_label = str(seed_obj["query"]["outcome"])
        outcome_internal = next(
            internal
            for internal, label in seed_obj["visible_schema"]["variable_labels"].items()
            if label == outcome_label
        )
        domain = world.domains[world.variables.index(outcome_internal)]
        prediction = (0.25, -0.25) + (0.0,) * (domain - 2)
        raw = json.dumps(
            {
                "type": "answer",
                "effect": {
                    f"state_{state}": component for state, component in enumerate(prediction)
                },
            }
        )
        parsed = parse_terminal_answer(raw, seed_obj, world)
        self.assertEqual(parsed["kind"], "target_query")
        self.assertEqual(parsed["effect"], prediction)
        score = score_terminal_answer(raw, seed_obj, world)
        self.assertEqual(score["prediction"], tuple(Fraction(item) for item in prediction))
        self.assertEqual(score["l1_error"], sum(score["component_errors"]))
        self.assertEqual(score["total_variation_error"], score["l1_error"] / 2)
        self.assertIn("observational_shortcut_error", score)
        self.assertEqual(
            score["squared_error"],
            sum(error**2 for error in score["component_errors"]) / domain,
        )
        self.assertEqual(score["reward_scalarization"], TERMINAL_QUALITY_REWARD_VERSION)

    def test_target_query_rejects_malformed_answer(self) -> None:
        seed_obj, world = _find_task("ate")
        with self.assertRaises(ValueError):
            parse_terminal_answer('{"type":"answer"}', seed_obj, world)
        with self.assertRaises(ValueError):
            parse_terminal_answer('{"type":"answer","effect":"not-a-number"}', seed_obj, world)
        with self.assertRaises(ValueError):
            parse_terminal_answer(
                '{"type":"answer","effect":{"state_0":0.2,"state_1":0.2}}',
                seed_obj,
                world,
            )

    def test_counterfactual_terminal_contract_requires_an_ordered_roi(self) -> None:
        seed_obj, world = _find_task("individual_counterfactual_probability")
        for answer in (
            {"type": "answer", "value": 0.5},
            {"type": "answer", "lower": 0.8, "upper": 0.2},
            {"type": "answer", "lower": -0.1, "upper": 0.2},
            {"type": "answer", "lower": 0.1, "upper": 1.1},
        ):
            with self.assertRaises(ValueError):
                parse_terminal_answer(json.dumps(answer), seed_obj, world)

    def test_individual_counterfactual_probability_scores_both_roi_endpoints_continuously(
        self,
    ) -> None:
        from cpt_world import compute_query_truth

        roi_seed, world = _find_task("individual_counterfactual_probability")
        truth = compute_query_truth(world, roi_seed)
        self.assertIn(truth["certification"], {"exact", "epsilon_sharp"})
        self.assertLessEqual(truth["endpoint_error"], truth["endpoint_tolerance"])
        exact_roi = score_terminal_answer(
            json.dumps(
                {
                    "type": "answer",
                    "lower": float(truth["lower"]),
                    "upper": float(truth["upper"]),
                }
            ),
            roi_seed,
            world,
        )
        self.assertEqual(exact_roi["kind"], "counterfactual_roi")
        self.assertLess(exact_roi["mean_absolute_endpoint_error"], Fraction(1, 10**12))

        epsilon_truth = {
            "type": "individual_counterfactual_probability",
            "lower": Fraction(1, 5),
            "upper": Fraction(4, 5),
            "certification": "epsilon_sharp",
            "endpoint_error": 0.0015,
        }
        epsilon_score = score_terminal_answer(
            json.dumps({"type": "answer", "lower": 0.201, "upper": 0.799}),
            roi_seed,
            world,
            terminal_truth=epsilon_truth,
        )
        self.assertEqual(epsilon_score["certification"], "epsilon_sharp")
        self.assertEqual(epsilon_score["endpoint_error"], 0.0015)
        self.assertEqual(epsilon_score["truth"]["certification"], "epsilon_sharp")
        self.assertEqual(epsilon_score["mean_absolute_endpoint_error"], 0)

        shifted = score_terminal_answer(
            json.dumps({"type": "answer", "lower": 0.1, "upper": 0.9}),
            roi_seed,
            world,
            terminal_truth=epsilon_truth,
        )
        self.assertGreater(shifted["lower_endpoint_error"], 0)
        self.assertGreater(shifted["upper_endpoint_error"], 0)
        self.assertEqual(
            shifted["mean_absolute_endpoint_error"],
            (shifted["lower_endpoint_error"] + shifted["upper_endpoint_error"]) / 2,
        )

    def test_decision_optimal_answer_has_zero_regret(self) -> None:
        seed_obj, world = _find_task("best_intervention")
        from cpt_world import compute_query_truth

        truth = compute_query_truth(world, seed_obj)
        raw = json.dumps(
            {
                "type": "answer",
                "value": f"state_{truth['value']}",
            }
        )
        score = score_terminal_answer(raw, seed_obj, world)
        self.assertEqual(score["regret"], 0)
        self.assertEqual(score["normalized_regret"], 0)
        self.assertTrue(score["optimal_action"])
        self.assertGreater(score["probability_span"], 0)
        self.assertIn("observational_shortcut_error", score)

    def test_decision_suboptimal_answer_has_positive_regret(self) -> None:
        seed_obj, world = _find_task("best_intervention")
        from cpt_world import compute_query_truth

        truth = compute_query_truth(world, seed_obj)
        target_index = world.variables.index(str(truth["target"]))
        suboptimal_value = next(
            state for state in range(world.domains[target_index]) if state != truth["value"]
        )
        raw = json.dumps(
            {
                "type": "answer",
                "value": f"state_{suboptimal_value}",
            }
        )
        score = score_terminal_answer(raw, seed_obj, world)
        self.assertGreater(score["regret"], 0)
        self.assertGreater(score["normalized_regret"], 0)
        self.assertLessEqual(score["normalized_regret"], 1)
        self.assertFalse(score["optimal_action"])

    def test_decision_suboptimal_probability_and_regret_use_query_outcome(self) -> None:
        seed_obj, world = _decision_world()
        raw = json.dumps(
            {
                "type": "answer",
                "value": "state_1",
            }
        )
        score = score_terminal_answer(raw, seed_obj, world)
        self.assertEqual(score["optimal"], {"target": "A", "value": 0})
        self.assertEqual(score["optimal_probability"], Fraction(3, 20))
        self.assertEqual(score["chosen_probability"], Fraction(17, 20))
        self.assertEqual(score["regret"], Fraction(7, 10))
        self.assertEqual(
            score["candidate_probabilities"],
            (Fraction(3, 20), Fraction(17, 20)),
        )
        self.assertEqual(score["minimum_probability"], Fraction(3, 20))
        self.assertEqual(score["maximum_probability"], Fraction(17, 20))
        self.assertEqual(score["probability_span"], Fraction(7, 10))
        self.assertEqual(score["normalized_regret"], 1)

    def test_decision_observational_shortcut_error_is_raw_causal_action_regret(self) -> None:
        world = WorldSpec(
            family="test_dag",
            topology="B-confounds-A-to-Y",
            variables=("B", "A", "Y"),
            domains=(2, 2, 2),
            state_names=(("b0", "b1"), ("a0", "a1"), ("y0", "y1")),
            edges=((0, 1), (0, 2), (1, 2)),
            parents={0: (), 1: (0,), 2: (0, 1)},
            cpt={
                0: ((Fraction(1, 2), Fraction(1, 2)),),
                1: (
                    (Fraction(9, 10), Fraction(1, 10)),
                    (Fraction(1, 10), Fraction(9, 10)),
                ),
                2: (
                    (Fraction(3, 5), Fraction(2, 5)),
                    (Fraction(4, 5), Fraction(1, 5)),
                    (Fraction(1, 10), Fraction(9, 10)),
                    (Fraction(3, 10), Fraction(7, 10)),
                ),
            },
        )
        seed_obj = assemble_seed(
            world,
            _HIDING_MODES,
            "best_intervention",
            "decision",
            anchors={"decision_target": 1, "outcome": 2, "objective": "minimize"},
            seed_id="SCORE-CONFOUNDED-DECISION",
        )

        score = score_terminal_answer(
            json.dumps({"type": "answer", "value": "state_0"}),
            seed_obj,
            world,
        )

        self.assertEqual(score["candidate_probabilities"], (Fraction(13, 20), Fraction(9, 20)))
        self.assertEqual(score["observational_shortcut"], (Fraction(9, 20), Fraction(13, 20)))
        self.assertEqual(score["observational_choice"], 0)
        self.assertEqual(score["observational_shortcut_error"], Fraction(1, 5))
        self.assertEqual(score["observational_shortcut_normalized_regret"], 1)
        self.assertEqual(score["regret"], Fraction(1, 5))

    def test_decision_zero_span_accepts_every_tied_state_without_division(self) -> None:
        _, world = _decision_world()
        tied_rows = tuple((Fraction(1, 2), Fraction(1, 2)) for _ in world.cpt[2])
        tied_world = replace(world, cpt={**world.cpt, 2: tied_rows})
        seed_obj = assemble_seed(
            tied_world,
            _HIDING_MODES,
            "best_intervention",
            "decision",
            anchors={"decision_target": 0, "outcome": 2, "objective": "minimize"},
            seed_id="SCORE-TIED-DECISION",
        )
        score = score_terminal_answer(
            json.dumps(
                {
                    "type": "answer",
                    "value": "state_1",
                }
            ),
            seed_obj,
            tied_world,
        )

        self.assertEqual(score["probability_span"], 0)
        self.assertEqual(score["regret"], 0)
        self.assertEqual(score["normalized_regret"], 0)
        self.assertEqual(score["observational_shortcut_normalized_regret"], 0)
        self.assertTrue(score["optimal_action"])

    def test_decision_parser_rejects_hidden_names_and_noncanonical_states(self) -> None:
        seed_obj, world = _decision_world()
        invalid_answers = (0, "state_00", "state_2", "A")
        for value in invalid_answers:
            with self.assertRaises(ValueError):
                parse_terminal_answer(
                    json.dumps(
                        {
                            "type": "answer",
                            "value": value,
                        }
                    ),
                    seed_obj,
                    world,
                )


def _find_discovery_task(query_type: str):
    grammar = WorldGrammar(node_counts=(2, 3, 4))
    for seed in range(300):
        world = sample_world(grammar, seed)
        anchors_list = legal_query_anchors(world, query_type)
        if not anchors_list:
            continue
        seed_obj = assemble_seed(
            world,
            "mechanism_hidden",
            query_type,
            "discovery",
            anchors=anchors_list[0],
            seed_id=f"DISCOVERY-{seed}-{query_type}",
        )
        return seed_obj, world
    raise AssertionError(f"no {query_type} discovery task found")


class DiscoveryScoringTests(unittest.TestCase):
    @staticmethod
    def _score_adjustment_set(seed_obj, world, names: tuple[str, ...]):
        label_map = seed_obj["visible_schema"]["variable_labels"]
        raw = json.dumps(
            {
                "type": "answer",
                "adjustment_set": [label_map[name] for name in names],
            }
        )
        return score_terminal_answer(raw, seed_obj, world)

    def test_backadj_one_valid_adjustment_set_has_zero_effect_error(self) -> None:
        seed_obj, world = _find_discovery_task("backadj_minimal_sets")
        label_map = seed_obj["visible_schema"]["variable_labels"]
        treatment = world.variables.index(
            next(
                internal
                for internal, visible in label_map.items()
                if visible == seed_obj["query"]["treatment"]
            )
        )
        outcome = world.variables.index(
            next(
                internal
                for internal, visible in label_map.items()
                if visible == seed_obj["query"]["outcome"]
            )
        )
        adjustment_set = backdoor_adjustment_sets(world, treatment, outcome)[0]
        raw = json.dumps(
            {
                "type": "answer",
                "adjustment_set": [label_map[name] for name in adjustment_set],
            }
        )
        score = score_terminal_answer(raw, seed_obj, world)
        self.assertEqual(score["adjustment_error"], 0)
        self.assertEqual(terminal_quality_reward(score), 1)

    def test_backadj_scores_the_complete_submitted_set(self) -> None:
        seed_obj, world = _backdoor_behavior_task()
        valid = self._score_adjustment_set(seed_obj, world, ("Z",))
        unadjusted = self._score_adjustment_set(seed_obj, world, ())
        overadjusted = self._score_adjustment_set(seed_obj, world, ("Z", "M"))

        self.assertEqual(valid["adjustment_error"], 0)
        self.assertEqual(terminal_quality_reward(valid), 1)
        self.assertEqual(
            unadjusted["adjustment_error"], unadjusted["unadjusted_error"]
        )
        self.assertEqual(terminal_quality_reward(unadjusted), Fraction(1, 2))
        self.assertGreater(overadjusted["adjustment_error"], 0)
        self.assertLess(terminal_quality_reward(overadjusted), 1)

    def test_backadj_penalizes_opening_a_collider(self) -> None:
        seed_obj, world = _collider_behavior_task()
        valid = self._score_adjustment_set(seed_obj, world, ())
        collider = self._score_adjustment_set(seed_obj, world, ("C",))

        self.assertEqual(valid["unadjusted_error"], 0)
        self.assertEqual(valid["adjustment_error"], 0)
        self.assertEqual(terminal_quality_reward(valid), 1)
        self.assertGreater(collider["adjustment_error"], 0)
        self.assertEqual(terminal_quality_reward(collider), 0)

    def test_backadj_prioritizes_the_stronger_missing_confounder(self) -> None:
        seed_obj, world = _strong_weak_backdoor_task()
        complete = self._score_adjustment_set(seed_obj, world, ("A", "B"))
        keeps_strong = self._score_adjustment_set(seed_obj, world, ("A",))
        keeps_weak = self._score_adjustment_set(seed_obj, world, ("B",))

        self.assertEqual(complete["adjustment_error"], 0)
        self.assertLess(keeps_strong["adjustment_error"], keeps_weak["adjustment_error"])
        self.assertGreater(
            terminal_quality_reward(keeps_strong), terminal_quality_reward(keeps_weak)
        )

    def test_backadj_parser_rejects_duplicate_and_endpoint_labels(self) -> None:
        seed_obj, world = _backdoor_behavior_task()
        labels = seed_obj["visible_schema"]["variable_labels"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_terminal_answer(
                json.dumps(
                    {
                        "type": "answer",
                        "adjustment_set": [labels["Z"], labels["Z"]],
                    }
                ),
                seed_obj,
                world,
            )
        with self.assertRaisesRegex(ValueError, "exclude treatment and outcome"):
            parse_terminal_answer(
                json.dumps(
                    {"type": "answer", "adjustment_set": [labels["X"]]}
                ),
                seed_obj,
                world,
            )

    def test_backadj_does_not_require_every_minimal_adjustment_set(self) -> None:
        seed_obj, world = _multiple_adjustment_task()
        adjustment_sets = backdoor_adjustment_sets(world, "X", "Y")
        self.assertEqual(set(adjustment_sets), {("A",), ("B",)})
        label_map = seed_obj["visible_schema"]["variable_labels"]
        raw = json.dumps(
            {
                "type": "answer",
                "adjustment_set": [
                    label_map[name] for name in adjustment_sets[0]
                ],
            }
        )
        score = score_terminal_answer(raw, seed_obj, world)
        self.assertEqual(score["adjustment_error"], 0)
        self.assertEqual(terminal_quality_reward(score), 1)

    def test_mediator_correct_answer_has_full_f1(self) -> None:
        seed_obj, world = _find_discovery_task("mediator_set")
        from cpt_world import compute_query_truth

        truth = compute_query_truth(world, seed_obj)
        label_map = seed_obj["visible_schema"]["variable_labels"]
        raw = json.dumps(
            {
                "type": "answer",
                "mediators": [label_map[name] for name in truth["mediators"]],
                "order": [
                    [label_map[source], label_map[target]] for source, target in truth["order"]
                ],
            }
        )
        score = score_terminal_answer(raw, seed_obj, world)
        self.assertEqual(score["mediator_f1"], 1)
        self.assertEqual(score["order_f1"], 1)
        self.assertTrue(score["mediators_exact_match"])
        self.assertTrue(score["order_exact_match"])

    def test_mediator_partial_answer_gets_partial_variable_f1(self) -> None:
        world = WorldSpec(
            family="test_dag",
            topology="T-to-M1-to-M2-to-Y",
            variables=("T", "M1", "M2", "Y"),
            domains=(2, 2, 2, 2),
            state_names=(("0", "1"),) * 4,
            edges=((0, 1), (1, 2), (2, 3)),
            parents={0: (), 1: (0,), 2: (1,), 3: (2,)},
            cpt={
                0: ((Fraction(1, 2), Fraction(1, 2)),),
                1: ((Fraction(3, 4), Fraction(1, 4)), (Fraction(1, 4), Fraction(3, 4))),
                2: ((Fraction(3, 4), Fraction(1, 4)), (Fraction(1, 4), Fraction(3, 4))),
                3: ((Fraction(3, 4), Fraction(1, 4)), (Fraction(1, 4), Fraction(3, 4))),
            },
        )
        seed_obj = assemble_seed(
            world,
            _HIDING_MODES,
            "mediator_set",
            "discovery",
            anchors={"treatment": 0, "outcome": 3},
            seed_id="SCORE-PARTIAL-MEDIATOR",
        )
        labels = seed_obj["visible_schema"]["variable_labels"]
        score = score_terminal_answer(
            json.dumps(
                {
                    "type": "answer",
                    "mediators": [labels["M1"]],
                    "order": [],
                }
            ),
            seed_obj,
            world,
        )
        self.assertEqual(score["mediator_precision"], 1)
        self.assertEqual(score["mediator_recall"], Fraction(1, 2))
        self.assertEqual(score["mediator_f1"], Fraction(2, 3))


if __name__ == "__main__":
    unittest.main()
