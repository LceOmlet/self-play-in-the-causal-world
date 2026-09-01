from __future__ import annotations

import importlib.util
import json
import unittest
from fractions import Fraction
from pathlib import Path
from typing import Any
from unittest.mock import patch

from cpt_world.world_space import _best_intervention_is_observationally_discordant


def _load_runner() -> Any:
    path = Path(__file__).parents[1] / "scripts" / "run_deepseek_v4_flash_profile.py"
    spec = importlib.util.spec_from_file_location("run_deepseek_v4_flash_profile", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load DeepSeek profile runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


class _Response:
    status = 200

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class DeepSeekV4FlashProfileTests(unittest.TestCase):
    def test_best_intervention_schedule_uses_balanced_main_sampler(self) -> None:
        rng = runner.random.Random(1701)
        relations = []
        for repeat in range(5):
            entry = runner._schedule_entry(
                "best_intervention",
                repeat,
                rng,
                node_counts=(8,),
                max_domain_size=2,
            )
            world, seed = runner._materialize(entry)
            query = seed["query"]
            labels = seed["visible_schema"]["variable_labels"]
            inverse = {str(label): str(name) for name, label in labels.items()}
            anchors = {
                "decision_target": world.variables.index(inverse[str(query["decision_target"])]),
                "outcome": world.variables.index(inverse[str(query["outcome"])]),
                "objective": str(query["objective"]),
                "outcome_state": int(str(query["outcome_state"]).removeprefix("state_")),
            }
            relations.append(_best_intervention_is_observationally_discordant(world, anchors))
        self.assertEqual(relations, [False, True, True, True, True])

    def test_generated_exact_fraction_serializes_beyond_python_display_guard(self) -> None:
        value = Fraction(10**5000 + 1, 10**5000 + 3)
        encoded = runner._jsonable(value)

        self.assertIsInstance(encoded, str)
        self.assertGreater(len(encoded), 9000)
        self.assertEqual(runner._fraction_from_text(encoded), value)

    def test_summary_consumes_the_owner_target_query_score(self) -> None:
        summary = runner.summarize(
            {
                "schedule": {
                    "entries": [
                        {
                            "episode_id": "ate:00",
                            "query_type": "ate",
                            "node_count": 5,
                        },
                        {
                            "episode_id": "cf:00",
                            "query_type": "individual_counterfactual_probability",
                            "node_count": 6,
                        },
                        {
                            "episode_id": "decision:00",
                            "query_type": "best_intervention",
                            "node_count": 7,
                        },
                    ]
                },
                "episodes": [
                    {
                        "episode_id": "ate:00",
                        "status": "completed",
                        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                        "score": {
                            "kind": "target_query",
                            "l1_error": "1/2",
                            "total_variation_error": "1/4",
                            "squared_error": "1/16",
                        },
                    },
                    {
                        "episode_id": "cf:00",
                        "status": "completed",
                        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
                        "score": {
                            "kind": "counterfactual_roi",
                            "mean_absolute_endpoint_error": "1/10",
                            "mean_squared_endpoint_error": "1/50",
                        },
                    },
                    {
                        "episode_id": "decision:00",
                        "status": "completed",
                        "usage": {"prompt_tokens": 9, "completion_tokens": 2},
                        "score": {
                            "kind": "decision",
                            "regret": "1/5",
                            "normalized_regret": "2/5",
                        },
                    },
                ],
            }
        )

        self.assertEqual(summary["metric_means"]["ate"]["l1_error"]["mean"], 0.5)
        self.assertEqual(summary["metric_means"]["ate"]["total_variation_error"]["mean"], 0.25)
        self.assertEqual(summary["metric_means"]["ate"]["squared_error"]["mean"], 0.0625)
        self.assertEqual(
            summary["metric_means"]["individual_counterfactual_probability"][
                "mean_absolute_endpoint_error"
            ]["mean"],
            0.1,
        )
        self.assertEqual(
            summary["metric_means"]["individual_counterfactual_probability"][
                "mean_squared_endpoint_error"
            ]["mean"],
            0.02,
        )
        self.assertEqual(
            summary["metric_means"]["best_intervention"]["regret"]["mean"],
            0.2,
        )
        self.assertEqual(
            summary["metric_means"]["best_intervention"]["normalized_regret"]["mean"],
            0.4,
        )

    def test_small_schedule_and_outcome_tape_are_reproducible(self) -> None:
        first = runner.build_schedule(
            master_seed=1701,
            repeats=1,
            node_counts=(3, 4),
            max_domain_size=2,
        )
        second = runner.build_schedule(
            master_seed=1701,
            repeats=1,
            node_counts=(3, 4),
            max_domain_size=2,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first["entries"]), 5)
        self.assertEqual(first["budget"], runner._budget_contract())
        self.assertIsNone(first["budget"]["query_count_limit"])
        self.assertEqual(
            {entry["query_type"] for entry in first["entries"]},
            set(runner.QUERY_TYPES),
        )
        self.assertTrue(all(entry["node_count"] in {3, 4} for entry in first["entries"]))
        world, seed = runner._materialize(first["entries"][0])
        episode = runner.WorldSpecEpisode(
            world,
            seed,
            runner.OutcomeTape(first["entries"][0]["tape_key"]),
            budget=runner._budget_for_entry(first["entries"][0]),
        )
        user_prompt = episode.initial_messages()[1]["content"]
        expected_budget = first["entries"][0]["observation_bandwidth"] * (
            1 << first["entries"][0]["observation_budget_exponent"]
        )
        self.assertIn(f"total observation budget is {expected_budget}", user_prompt)
        self.assertIn("batch_size may be any positive integer", user_prompt)
        self.assertIn("no separate limit on the number of queries", user_prompt)
        self.assertIn("Solve the task through this multi-turn protocol", user_prompt)
        self.assertIn("inspect each batch_result", user_prompt)
        self.assertIn("You do not need to exhaust the budget", user_prompt)
        replay = runner.verify_schedule_replay(first)
        self.assertEqual(replay["verified_entries"], 5)
        self.assertEqual(replay["schedule_sha256"], runner._sha256(first))

    def test_provider_bridge_freezes_wire_contract_and_token_usage(self) -> None:
        captured: dict[str, object] = {}

        def opener(request: Any, *, timeout: float) -> _Response:
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["headers"] = dict(request.header_items())
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return _Response(
                {
                    "id": "response-1",
                    "created": 17,
                    "model": runner.MODEL,
                    "choices": [
                        {
                            "message": {"content": '{"type":"answer"}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 101,
                        "completion_tokens": 7,
                        "total_tokens": 108,
                    },
                }
            )

        content, metadata = runner.chat_completion(
            "test-secret",
            [{"role": "user", "content": "task"}],
            model_seed=123456,
            timeout_seconds=19.0,
            opener=opener,
        )

        self.assertEqual(content, '{"type":"answer"}')
        self.assertEqual(captured["url"], "https://api.ponderera.com/v1/chat/completions")
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["timeout"], 19.0)
        headers = captured["headers"]
        self.assertEqual(headers["Authorization"], "Bearer test-secret")
        self.assertEqual(headers["Content-type"], "application/json")
        self.assertEqual(
            captured["body"],
            {
                "enable_thinking": False,
                "messages": [{"role": "user", "content": "task"}],
                "model": "deepseek-v4-flash",
                "seed": 123456,
                "stream": False,
                "temperature": 0,
            },
        )
        self.assertEqual(
            metadata["usage"],
            {"prompt_tokens": 101, "completion_tokens": 7, "total_tokens": 108},
        )

    def test_invalid_command_returns_budget_preserving_feedback_and_can_be_corrected(
        self,
    ) -> None:
        entry = runner.build_schedule(
            master_seed=1701,
            repeats=1,
            node_counts=(3,),
            max_domain_size=2,
        )["entries"][0]
        world, seed = runner._materialize(entry)
        truth = runner.compute_query_truth(world, seed)
        answer = {
            "type": "answer",
            "effect": {
                f"state_{state}": float(component)
                for state, component in enumerate(truth["effect"])
            },
        }
        replies = iter(
            (
                '{"type":"observe","measure":[],"batch_size":4}',
                json.dumps(answer),
            )
        )

        def fake_chat_completion(*_args: object, **_kwargs: object) -> tuple[str, dict]:
            return next(replies), {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                }
            }

        with patch.object(runner, "chat_completion", side_effect=fake_chat_completion):
            result = runner.run_episode(entry, "test-secret", timeout_seconds=19.0)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["protocol_errors"], 1)
        self.assertEqual(result["queries_used"], 0)
        self.assertEqual(result["sample_rows_used"], 0)
        self.assertEqual(result["observations_used"], 0)
        self.assertEqual(result["usage"]["total_tokens"], 24)
        self.assertEqual(
            [turn["outcome"] for turn in result["turns"]], ["protocol_error", "terminal"]
        )
        payload = json.loads(result["turns"][0]["feedback"].splitlines()[0])
        self.assertFalse(payload["accepted"])
        self.assertEqual(payload["budget_consumed"], 0)
        self.assertEqual(
            payload["remaining_budget"],
            entry["observation_bandwidth"] * (1 << entry["observation_budget_exponent"]),
        )

    def test_repeated_invalid_commands_remain_correctable_without_a_turn_cap(self) -> None:
        entry = runner.build_schedule(
            master_seed=1701,
            repeats=1,
            node_counts=(3,),
            max_domain_size=2,
        )["entries"][0]

        world, seed = runner._materialize(entry)
        truth = runner.compute_query_truth(world, seed)
        answer = {
            "type": "answer",
            "effect": {
                f"state_{state}": float(component)
                for state, component in enumerate(truth["effect"])
            },
        }
        replies = iter(
            ['{"type":"observe","measure":[],"batch_size":4}'] * 6 + [json.dumps(answer)]
        )

        def fake_chat_completion(*_args: object, **_kwargs: object) -> tuple[str, dict]:
            return next(replies), {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                }
            }

        with patch.object(runner, "chat_completion", side_effect=fake_chat_completion):
            result = runner.run_episode(entry, "test-secret", timeout_seconds=19.0)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["protocol_errors"], 6)
        self.assertEqual(result["queries_used"], 0)
        self.assertEqual(result["observations_used"], 0)


if __name__ == "__main__":
    unittest.main()
