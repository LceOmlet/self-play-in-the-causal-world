from __future__ import annotations

import inspect
import json
import unittest
from collections import Counter

from cpt_world import TASK_FAMILY_QUERY_TYPES, CPTWorldEnvironment, build_balanced_training_rows


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
