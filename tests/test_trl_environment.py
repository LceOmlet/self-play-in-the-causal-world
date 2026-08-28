from __future__ import annotations

import inspect
import json
import math
import unittest
from collections import Counter
from itertools import islice
from unittest.mock import patch

from cpt_world import (
    CEILING_SENSITIVE_ADVANTAGE_QUERY_TYPES,
    DEFAULT_ADVANTAGE_UTILITY_EPSILON,
    TASK_FAMILY_QUERY_TYPES,
    CPTWorldEnvironment,
    bounded_negative_log_residual_utility,
    build_balanced_training_rows,
    build_cpt_world_advantage_utility,
    iter_random_balanced_training_rows,
    task_advantage_utility,
)


class TRLEnvironmentAdapterTests(unittest.TestCase):
    def test_bounded_log_residual_utility_preserves_endpoints_and_expands_ceiling(self) -> None:
        epsilon = DEFAULT_ADVANTAGE_UTILITY_EPSILON

        self.assertEqual(bounded_negative_log_residual_utility(0.0), 0.0)
        self.assertEqual(bounded_negative_log_residual_utility(1.0), 1.0)
        self.assertLess(
            bounded_negative_log_residual_utility(0.95),
            bounded_negative_log_residual_utility(0.99),
        )
        self.assertGreater(
            bounded_negative_log_residual_utility(0.99, epsilon=epsilon)
            - bounded_negative_log_residual_utility(0.95, epsilon=epsilon),
            0.99 - 0.95,
        )

    def test_decision_and_mediator_use_raw_quality_after_reward_v4(self) -> None:
        raw = 0.95

        self.assertEqual(task_advantage_utility(raw, "mediator_set"), raw)
        self.assertEqual(task_advantage_utility(raw, "best_intervention"), raw)
        for query_type in CEILING_SENSITIVE_ADVANTAGE_QUERY_TYPES:
            with self.subTest(query_type=query_type):
                self.assertNotEqual(task_advantage_utility(raw, query_type), raw)

    def test_trl_advantage_utility_reads_owner_rewards_and_logs_both_values(self) -> None:
        class RewardOwner:
            def __init__(self, reward: float) -> None:
                self.reward = reward

            def get_reward(self) -> float:
                return self.reward

        logged: list[tuple[str, float]] = []
        reward_func = build_cpt_world_advantage_utility(epsilon=0.02)

        utilities = reward_func(
            environments=[RewardOwner(0.95), RewardOwner(0.95)],
            query_type=["ate", "mediator_set"],
            log_metric=lambda name, value: logged.append((name, value)),
        )

        self.assertEqual(reward_func.__name__, "CPTWorldAdvantageUtility")
        self.assertEqual(utilities[1], 0.95)
        self.assertNotEqual(utilities[0], 0.95)
        self.assertIn(("task/ate/reward_raw", 0.95), logged)
        self.assertIn(("task/mediator_set/reward_utility", 0.95), logged)

    def test_trl_logs_owner_effect_metrics_and_terminal_coverage(self) -> None:
        class Episode:
            def __init__(self, score) -> None:
                self.terminal_score = score

        class RewardOwner:
            def __init__(self, reward: float, score) -> None:
                self.reward = reward
                self.episode = Episode(score)

            def get_reward(self) -> float:
                return self.reward

        environments = [
            RewardOwner(0.9, {"kind": "target_query", "squared_error": 0.09}),
            RewardOwner(0.8, {"kind": "target_query", "squared_error": 0.16}),
            RewardOwner(
                0.9,
                {
                    "kind": "counterfactual_roi",
                    "mean_squared_endpoint_error": 0.01,
                },
            ),
            RewardOwner(
                0.8,
                {
                    "kind": "counterfactual_roi",
                    "mean_squared_endpoint_error": 0.04,
                },
            ),
            RewardOwner(
                0.8,
                {"kind": "decision", "regret": 0.2, "normalized_regret": 0.25},
            ),
            RewardOwner(
                0.6,
                {"kind": "decision", "regret": 0.4, "normalized_regret": 0.75},
            ),
            RewardOwner(0.0, None),
        ]
        query_types = [
            "ate",
            "ate",
            "individual_counterfactual_probability",
            "individual_counterfactual_probability",
            "best_intervention",
            "best_intervention",
            "best_intervention",
        ]
        logged: list[tuple[str, float]] = []

        build_cpt_world_advantage_utility()(
            environments=environments,
            query_type=query_types,
            log_metric=lambda name, value: logged.append((name, value)),
        )
        metrics = dict(logged)

        self.assertAlmostEqual(metrics["effect/ate_mse"], 0.125)
        self.assertAlmostEqual(metrics["effect/ate_rmse"], math.sqrt(0.125))
        self.assertAlmostEqual(metrics["effect/cf_endpoint_mse"], 0.025)
        self.assertAlmostEqual(metrics["effect/cf_endpoint_rmse"], math.sqrt(0.025))
        self.assertAlmostEqual(metrics["effect/decision_regret"], 0.3)
        self.assertAlmostEqual(metrics["effect/decision_normalized_regret"], 0.5)
        self.assertEqual(metrics["effect/ate_coverage"], 1.0)
        self.assertEqual(metrics["effect/cf_coverage"], 1.0)
        self.assertAlmostEqual(metrics["effect/decision_coverage"], 2 / 3)
        self.assertEqual(metrics["effect/decision_count"], 2.0)

    def test_rows_are_exactly_balanced_and_carry_conversational_prompts(self) -> None:
        rows = build_balanced_training_rows(count_per_family=2, start_seed=7)

        self.assertEqual(len(rows), 2 * len(TASK_FAMILY_QUERY_TYPES))
        self.assertEqual(
            Counter(row["query_type"] for row in rows),
            Counter(dict.fromkeys(TASK_FAMILY_QUERY_TYPES, 2)),
        )
        self.assertTrue(all(row["prompt"][-1]["role"] == "user" for row in rows))

    def test_repeated_row_resets_to_common_randomness_and_zero_unfinished_reward(self) -> None:
        row = build_balanced_training_rows(count_per_family=1)[0]
        left = CPTWorldEnvironment()
        right = CPTWorldEnvironment()
        left_instruction = left.reset(**row)
        right_instruction = right.reset(**row)

        self.assertEqual(left_instruction, right_instruction)
        self.assertIn("`act` tool", left_instruction)
        self.assertEqual(left.get_reward(), 0.0)
        self.assertEqual(right.get_reward(), 0.0)
        self.assertIsNotNone(left.episode)
        episode = left.episode
        assert episode is not None
        world = episode.world
        labels = episode.seed["visible_schema"]["variable_labels"]
        target = next(name for name in world.variables if episode.seed["manipulability"][name])
        measure = next(
            name for name in world.variables if name != target and episode.seed["readable"][name]
        )
        command = {
            "type": "intervene",
            "target": labels[target],
            "value": "state_0",
            "measure": [labels[measure]],
            "batch_size": 8,
        }

        left_feedback = left.act(command)
        right_feedback = right.act(command)

        self.assertEqual(left_feedback, right_feedback)
        payload = json.loads(left_feedback.splitlines()[0])
        self.assertEqual(payload["type"], "batch_result")

    @patch("cpt_world.trl_environment.compute_query_truth")
    def test_random_stream_is_balanced_and_uses_fresh_sampler_seeds(self, truth_owner) -> None:
        truth_owner.return_value = {
            "type": "individual_counterfactual_probability",
            "lower": 0.25,
            "upper": 0.75,
        }

        rows = list(islice(iter_random_balanced_training_rows(), 10))

        self.assertEqual(
            Counter(row["query_type"] for row in rows),
            Counter(dict.fromkeys(TASK_FAMILY_QUERY_TYPES, 2)),
        )
        self.assertEqual(len({row["sample_index"] for row in rows}), len(rows))
        counterfactual_rows = [
            row for row in rows if row["query_type"] == "individual_counterfactual_probability"
        ]
        self.assertTrue(all(row["terminal_truth_json"] for row in counterfactual_rows))
        self.assertTrue(
            all(not row["terminal_truth_json"] for row in rows if row not in counterfactual_rows)
        )
        self.assertTrue(
            all(
                call.kwargs["counterfactual_endpoint_time_limit_seconds"] == 5.0
                for call in truth_owner.call_args_list
            )
        )

    @patch("cpt_world.trl_environment.compute_query_truth")
    def test_counterfactual_timeout_resamples_without_emitting_unscored_row(
        self,
        truth_owner,
    ) -> None:
        truth_owner.side_effect = [
            RuntimeError("simulated endpoint timeout"),
            {
                "type": "individual_counterfactual_probability",
                "lower": 0.1,
                "upper": 0.9,
            },
        ]
        stream = iter_random_balanced_training_rows()

        ate_row = next(stream)
        counterfactual_row = next(stream)

        self.assertEqual(ate_row["sample_index"], 0)
        self.assertEqual(counterfactual_row["sample_index"], 2)
        self.assertEqual(truth_owner.call_count, 2)
        self.assertEqual(
            json.loads(counterfactual_row["terminal_truth_json"])["type"],
            "individual_counterfactual_probability",
        )

    def test_cached_counterfactual_truth_scores_without_reopening_solver(self) -> None:
        row = next(
            row
            for row in build_balanced_training_rows(count_per_family=1)
            if row["query_type"] == "individual_counterfactual_probability"
        )
        row["terminal_truth_json"] = json.dumps(
            {
                "type": "individual_counterfactual_probability",
                "lower": 0.25,
                "upper": 0.75,
            }
        )
        environment = CPTWorldEnvironment()
        environment.reset(**row)

        with patch(
            "cpt_world.task_scoring.compute_query_truth",
            side_effect=AssertionError("cached truth must bypass the solver"),
        ):
            feedback = environment.act({"type": "answer", "lower": 0.25, "upper": 0.75})

        self.assertIn("Episode complete", feedback)
        self.assertEqual(environment.get_reward(), 1.0)

    def test_only_act_is_exposed_as_an_environment_tool(self) -> None:
        environment = CPTWorldEnvironment()
        methods = {
            name
            for name, member in inspect.getmembers(environment, predicate=inspect.ismethod)
            if name not in {"reset", "get_reward"} and not name.startswith("_")
        }
        self.assertEqual(methods, {"act"})

    def test_protocol_error_does_not_escape_the_tool_boundary(self) -> None:
        row = build_balanced_training_rows(count_per_family=1)[0]
        environment = CPTWorldEnvironment()
        environment.reset(**row)

        feedback = environment.act({"type": "not-a-command"})

        payload = json.loads(feedback.splitlines()[0])
        self.assertEqual(payload["type"], "protocol_error")
        self.assertEqual(payload["budget_consumed"], 0)


if __name__ == "__main__":
    unittest.main()
