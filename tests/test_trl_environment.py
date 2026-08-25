from __future__ import annotations

import inspect
import json
import unittest
from collections import Counter
from itertools import islice
from unittest.mock import patch

from cpt_world import (
    TASK_FAMILY_QUERY_TYPES,
    CPTWorldEnvironment,
    build_balanced_training_rows,
    iter_random_balanced_training_rows,
)


class TRLEnvironmentAdapterTests(unittest.TestCase):
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
            name
            for name in world.variables
            if name != target and episode.seed["readable"][name]
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
            row
            for row in rows
            if row["query_type"] == "individual_counterfactual_probability"
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
            feedback = environment.act({"type": "answer", "value": 0.5})

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
