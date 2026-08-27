from __future__ import annotations

import json
import unittest
from fractions import Fraction

from cpt_world import (
    TERMINAL_QUALITY_REWARD_VERSION,
    WorldGrammar,
    WorldSpec,
    assemble_seed,
    legal_query_anchors,
    parse_terminal_answer,
    sample_task_world,
    sample_world,
    score_terminal_answer,
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
        truth_value = 0.25
        raw = json.dumps({"type": "answer", "effect": truth_value})
        parsed = parse_terminal_answer(raw, seed_obj, world)
        self.assertEqual(parsed["kind"], "target_query")
        self.assertEqual(parsed["effect"], truth_value)
        score = score_terminal_answer(raw, seed_obj, world)
        self.assertEqual(score["prediction"], Fraction(truth_value))
        self.assertEqual(score["abs_error"], abs(score["prediction"] - score["truth"]))
        self.assertEqual(score["squared_error"], score["abs_error"] ** 2)
        self.assertEqual(score["reward_scalarization"], TERMINAL_QUALITY_REWARD_VERSION)

    def test_target_query_rejects_malformed_answer(self) -> None:
        seed_obj, world = _find_task("ate")
        with self.assertRaises(ValueError):
            parse_terminal_answer('{"type":"answer"}', seed_obj, world)
        with self.assertRaises(ValueError):
            parse_terminal_answer('{"type":"answer","effect":"not-a-number"}', seed_obj, world)

    def test_counterfactual_terminal_contract_is_scalar_only(self) -> None:
        seed_obj, world = _find_task("individual_counterfactual_probability")
        for answer in (
            {"type": "answer", "lower": 0.1, "upper": 0.2},
            {"type": "answer", "value": -0.1},
            {"type": "answer", "value": 1.1},
        ):
            with self.assertRaises(ValueError):
                parse_terminal_answer(json.dumps(answer), seed_obj, world)

    def test_individual_counterfactual_probability_uses_continuous_interval_distance(
        self,
    ) -> None:
        from cpt_world import compute_query_truth

        point_seed, world = _find_task("individual_counterfactual_probability")
        truth = compute_query_truth(world, point_seed)
        self.assertIn(truth["certification"], {"exact", "epsilon_sharp"})
        self.assertLessEqual(truth["endpoint_error"], truth["endpoint_tolerance"])
        midpoint = (truth["lower"] + truth["upper"]) / 2
        inside = score_terminal_answer(
            json.dumps({"type": "answer", "value": float(midpoint)}),
            point_seed,
            world,
        )
        self.assertEqual(inside["kind"], "individual_counterfactual_probability")
        self.assertTrue(inside["compatible"])
        self.assertEqual(inside["distance_to_interval"], 0)

        epsilon_truth = {
            "type": "individual_counterfactual_probability",
            "lower": Fraction(1, 5),
            "upper": Fraction(4, 5),
            "certification": "epsilon_sharp",
            "endpoint_error": 0.0015,
        }
        epsilon_score = score_terminal_answer(
            json.dumps({"type": "answer", "value": 0.5}),
            point_seed,
            world,
            terminal_truth=epsilon_truth,
        )
        self.assertEqual(epsilon_score["certification"], "epsilon_sharp")
        self.assertEqual(epsilon_score["endpoint_error"], 0.0015)
        self.assertEqual(
            epsilon_score["truth"]["certification"], "epsilon_sharp"
        )

        if truth["lower"] > 0:
            outside_value = 0.0
            expected_distance = truth["lower"]
        elif truth["upper"] < 1:
            outside_value = 1.0
            expected_distance = 1 - truth["upper"]
        else:
            self.fail("point-mode fixture unexpectedly has the unconstrained [0,1] interval")
        outside = score_terminal_answer(
            json.dumps({"type": "answer", "value": outside_value}),
            point_seed,
            world,
        )
        self.assertFalse(outside["compatible"])
        self.assertEqual(outside["distance_to_interval"], expected_distance)

        with self.assertRaises(ValueError):
            parse_terminal_answer(
                json.dumps({"type": "answer", "lower": 0.1, "upper": 0.2}),
                point_seed,
                world,
            )

    def test_decision_optimal_answer_has_zero_regret(self) -> None:
        seed_obj, world = _find_task("best_intervention")
        from cpt_world import compute_query_truth

        truth = compute_query_truth(world, seed_obj)
        label_map = seed_obj["visible_schema"]["variable_labels"]
        raw = json.dumps(
            {
                "type": "answer",
                "intervention": {
                    "target": label_map[truth["target"]],
                    "value": f"state_{truth['value']}",
                },
            }
        )
        score = score_terminal_answer(raw, seed_obj, world)
        self.assertEqual(score["regret"], 0)

    def test_decision_suboptimal_answer_has_positive_regret(self) -> None:
        seed_obj, world = _find_task("best_intervention")
        from cpt_world import compute_query_truth

        truth = compute_query_truth(world, seed_obj)
        label_map = seed_obj["visible_schema"]["variable_labels"]
        target_index = world.variables.index(str(truth["target"]))
        suboptimal_value = next(
            state for state in range(world.domains[target_index]) if state != truth["value"]
        )
        raw = json.dumps(
            {
                "type": "answer",
                "intervention": {
                    "target": label_map[truth["target"]],
                    "value": f"state_{suboptimal_value}",
                },
            }
        )
        score = score_terminal_answer(raw, seed_obj, world)
        self.assertGreater(score["regret"], 0)

    def test_decision_suboptimal_probability_and_regret_use_query_outcome(self) -> None:
        seed_obj, world = _decision_world()
        labels = seed_obj["visible_schema"]["variable_labels"]
        raw = json.dumps(
            {
                "type": "answer",
                "intervention": {"target": labels["A"], "value": "state_1"},
            }
        )
        score = score_terminal_answer(raw, seed_obj, world)
        self.assertEqual(score["optimal"], {"target": "A", "value": 0})
        self.assertEqual(score["optimal_probability"], Fraction(3, 20))
        self.assertEqual(score["chosen_probability"], Fraction(17, 20))
        self.assertEqual(score["regret"], Fraction(7, 10))

    def test_decision_parser_rejects_hidden_names_and_noncanonical_states(self) -> None:
        seed_obj, world = _decision_world()
        labels = seed_obj["visible_schema"]["variable_labels"]
        invalid_answers = (
            ("A", "state_0"),
            (labels["B"], "state_0"),
            (labels["A"], "state_00"),
            (labels["A"], "state_2"),
        )
        for target, value in invalid_answers:
            with self.assertRaises(ValueError):
                parse_terminal_answer(
                    json.dumps(
                        {
                            "type": "answer",
                            "intervention": {"target": target, "value": value},
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
    def test_backadj_correct_answer_has_full_f1(self) -> None:
        seed_obj, world = _find_discovery_task("backadj_minimal_sets")
        from cpt_world import compute_query_truth

        truth = compute_query_truth(world, seed_obj)
        label_map = seed_obj["visible_schema"]["variable_labels"]
        raw = json.dumps(
            {
                "type": "answer",
                "adjustment_sets": [
                    [label_map[name] for name in adjustment_set]
                    for adjustment_set in truth["adjustment_sets"]
                ],
            }
        )
        score = score_terminal_answer(raw, seed_obj, world)
        self.assertEqual(score["f1"], 1)
        self.assertTrue(score["exact_match"])

    def test_backadj_incomplete_answer_has_lower_f1(self) -> None:
        seed_obj, world = _find_discovery_task("backadj_minimal_sets")
        from cpt_world import compute_query_truth

        truth = compute_query_truth(world, seed_obj)
        if len(truth["adjustment_sets"]) <= 1:
            self.skipTest("no partial backadj answer available")
        label_map = seed_obj["visible_schema"]["variable_labels"]
        raw = json.dumps(
            {
                "type": "answer",
                "adjustment_sets": [[label_map[name] for name in truth["adjustment_sets"][0]]],
            }
        )
        score = score_terminal_answer(raw, seed_obj, world)
        self.assertLess(score["f1"], 1)

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
