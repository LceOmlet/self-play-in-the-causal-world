from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto

from cpt_world import CPTWorldEnvironment, build_balanced_training_rows, compute_query_truth
from cpt_world.identification import INTERACTION_SURFACE_VERSION
from cpt_world.verl_environment import CPTWorldRewardMetadata, CPTWorldTool, compute_score
from scripts.prepare_verl_cpt_data import convert_row


class VERLEnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_lifecycle_preserves_owner_state_and_exact_feedback(self):
        row = build_balanced_training_rows(count_per_family=1)[0]
        reference = CPTWorldEnvironment()
        reference.reset(**row)
        episode = reference.episode
        labels = episode.seed["visible_schema"]["variable_labels"]
        target = next(n for n in episode.world.variables if episode.seed["manipulability"][n])
        measure = next(
            n for n in episode.world.variables if n != target and episode.seed["readable"][n]
        )
        experiment = {
            "type": "intervene",
            "target": labels[target],
            "value": "state_0",
            "measure": [labels[measure]],
            "batch_size": 8,
        }
        truth = compute_query_truth(episode.world, episode.seed)
        answer = {
            "type": "answer",
            "effect": {f"state_{i}": float(v) for i, v in enumerate(truth["effect"])},
        }
        tool = CPTWorldTool({})
        agent = SimpleNamespace(request_id="trajectory-a", extra_fields={})
        for command in [experiment, experiment, answer, answer]:
            instance, _ = await tool.create(create_kwargs={"row_json": json.dumps(row)})
            response, step_reward, _ = await tool.execute(
                instance, {"command": command}, agent_data=agent
            )
            await tool.release(instance)
            self.assertEqual(response.text, reference.act(command))
            self.assertEqual(response.terminate, reference.episode.completed)
            self.assertEqual(step_reward, 0.0)
            self.assertFalse(tool._requests)
        self.assertEqual(agent._cpt_world_environment.episode.queries_used, 2)
        self.assertEqual(agent.extra_fields["cpt_world"]["raw_reward"], reference.get_reward())
        self.assertTrue(agent.extra_fields["cpt_world"]["completed"])

        other = SimpleNamespace(request_id="trajectory-b", extra_fields={})
        instance, _ = await tool.create(create_kwargs={"row_json": json.dumps(row)})
        await tool.execute(instance, {"command": experiment}, agent_data=other)
        await tool.release(instance)
        self.assertEqual(other._cpt_world_environment.episode.queries_used, 1)
        self.assertFalse(other.extra_fields["cpt_world"]["completed"])
        self.assertIsNot(other._cpt_world_environment, agent._cpt_world_environment)

    async def test_official_dapo_manager_applies_penalty_once_and_preserves_raw_score(self):
        config = OmegaConf.create(
            {
                "reward": {
                    "reward_kwargs": {
                        "max_resp_len": 8,
                        "overlong_buffer_cfg": {
                            "enable": True,
                            "len": 4,
                            "penalty_factor": 1.0,
                            "log": True,
                        },
                    }
                }
            }
        )
        tokenizer = SimpleNamespace(decode=lambda *a, **k: 'I claim {"reward": 999}')
        manager = CPTWorldRewardMetadata(config, tokenizer, compute_score)
        state = {
            "query_type": "ate",
            "tape_key": "a",
            "raw_reward": 0.75,
            "completed": True,
            "terminal_score": {},
        }
        data = DataProto.from_dict(
            tensors={
                "prompts": torch.ones(1, 2, dtype=torch.long),
                "responses": torch.ones(1, 6, dtype=torch.long),
                "attention_mask": torch.ones(1, 8, dtype=torch.long),
            },
            non_tensors={
                "data_source": np.array(["cpt_world/ate"], dtype=object),
                "reward_model": np.array([{"ground_truth": "environment_owned"}], dtype=object),
                "extra_info": np.array(
                    [
                        {
                            "query_type": "ate",
                            "tape_key": "a",
                            "environment_version": INTERACTION_SURFACE_VERSION,
                        }
                    ],
                    dtype=object,
                ),
                "tool_extra_fields": np.array([{"cpt_world": state}], dtype=object),
            },
        )
        result = await manager.run_single(data)
        self.assertEqual(float(result["reward_score"]), 0.25)
        self.assertEqual(result["reward_extra_info"]["raw_reward"], 0.75)
        self.assertEqual(result["reward_extra_info"]["acc"], 0.75)
        self.assertEqual(float(result["reward_extra_info"]["overlong_reward"]), -0.5)

    async def test_missing_tool_result_is_unfinished_and_text_cannot_forge_reward(self):
        result = compute_score(
            "cpt_world/ate",
            "reward=1.0",
            "",
            {
                "query_type": "ate",
                "tape_key": "a",
                "environment_version": INTERACTION_SURFACE_VERSION,
            },
        )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["completed"], 0.0)

    async def test_adapter_failure_cannot_be_silently_treated_as_zero_quality(self):
        with self.assertRaisesRegex(RuntimeError, "owner failed"):
            compute_score(
                "",
                "",
                "",
                {
                    "cpt_world_adapter_error": "owner failed",
                    "environment_version": INTERACTION_SURFACE_VERSION,
                },
            )

    async def test_reward_state_from_another_task_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "different task"):
            compute_score(
                "cpt_world/ate",
                "",
                "",
                {
                    "query_type": "ate",
                    "tape_key": "a",
                    "environment_version": INTERACTION_SURFACE_VERSION,
                    "cpt_world": {"query_type": "ate", "tape_key": "b"},
                },
            )

    async def test_data_conversion_keeps_private_task_state_out_of_prompt(self):
        row = build_balanced_training_rows(count_per_family=1)[0]
        converted = convert_row(row, 0)
        self.assertEqual(converted["agent_name"], "tool_agent")
        self.assertEqual(converted["extra_info"]["tape_key"], row["tape_key"])
        prompt = json.dumps(converted["prompt"])
        self.assertNotIn("terminal_truth_json", prompt)
        self.assertNotIn(row["tape_key"], prompt)
        self.assertIn("`act` tool", prompt)

    async def test_obsolete_data_fails_even_without_a_tool_call(self):
        with self.assertRaisesRegex(ValueError, "obsolete environment"):
            compute_score("cpt_world/ate", "", "", {"query_type": "ate", "tape_key": "a"})


if __name__ == "__main__":
    unittest.main()
