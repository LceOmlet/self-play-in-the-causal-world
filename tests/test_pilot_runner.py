from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from cpt_world import Direction, Variable
from scripts.run_gpt_ge_seed_pilot import (
    _load_document,
    _new_document,
    _validate_episode_record,
    _write_json_atomic,
    build_profile,
    chat_completion,
    pairing_sha256,
    pilot_layouts,
    run_config,
    run_episode,
    scheduled_episodes,
)


class PilotScheduleTests(unittest.TestCase):
    def test_six_layout_schedule_is_exact_and_marginally_balanced(self) -> None:
        layouts = pilot_layouts()
        self.assertEqual(
            [layout.layout_id for layout in layouts],
            [
                "labels-20260818-r0-t0-e0",
                "labels-20260818-r1-t1-e1",
                "labels-20260818-r2-t5-e1",
                "labels-20260818-r3-t3-e0",
                "labels-20260818-r4-t4-e0",
                "labels-20260818-r5-t2-e1",
            ],
        )
        self.assertEqual(len({tuple(layout.labels.values()) for layout in layouts}), 6)
        self.assertEqual(len({layout.target_order for layout in layouts}), 6)
        self.assertEqual(
            Counter(layout.reverse_effect_order for layout in layouts), {False: 3, True: 3}
        )

        for effect_order in (False, True):
            group = [layout for layout in layouts if layout.reverse_effect_order is effect_order]
            for role in Variable:
                self.assertEqual(len({layout.labels[role] for layout in group}), 3)
                self.assertEqual(
                    {layout.target_order.index(role) for layout in group},
                    {0, 1, 2},
                )

    def test_schedule_has_exactly_36_paired_episodes(self) -> None:
        episodes = scheduled_episodes()
        self.assertEqual(len(episodes), 36)
        self.assertEqual(
            Counter(item.seed.difficulty for item in episodes),
            {"easy": 12, "medium": 12, "hard": 12},
        )
        self.assertEqual(
            Counter(item.world.direction for item in episodes),
            {Direction.FORWARD: 18, Direction.REVERSE: 18},
        )
        self.assertEqual(len({item.episode_id for item in episodes}), 36)
        self.assertEqual(len({item.tape_key for item in episodes}), 1)
        difficulties = [item.seed.difficulty for item in episodes]
        self.assertNotEqual(difficulties, sorted(difficulties))

    def test_two_models_have_the_same_pairing_digest(self) -> None:
        schedule = scheduled_episodes()
        qwen = run_config("qwen3.5-27b", schedule, timeout=180.0)
        deepseek = run_config("DeepSeek-V4-Pro", schedule, timeout=180.0)
        self.assertEqual(pairing_sha256(qwen), pairing_sha256(deepseek))


class PilotRunnerTests(unittest.TestCase):
    def test_provider_bridge_matches_the_observed_chat_completion_contract(self) -> None:
        class FakeResponse:
            status = 200
            headers = {"x-request-id": "request-1"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [{"message": {"content": '{"type":"answer"}'}}],
                        "id": "response-1",
                        "model": "routed-model",
                        "system_fingerprint": "fingerprint-1",
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                    }
                ).encode()

        with patch(
            "scripts.run_gpt_ge_seed_pilot.urllib.request.urlopen",
            return_value=FakeResponse(),
        ) as urlopen:
            content, provider = chat_completion(
                "dummy-secret",
                "qwen3.5-27b",
                ({"role": "user", "content": "test"},),
                12.0,
            )
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(content, '{"type":"answer"}')
        self.assertEqual(provider["usage"], {"prompt_tokens": 3, "completion_tokens": 2})
        self.assertEqual(provider["response_id"], "response-1")
        self.assertEqual(provider["response_model"], "routed-model")
        self.assertEqual(provider["system_fingerprint"], "fingerprint-1")
        self.assertEqual(provider["request_id"], "request-1")
        self.assertEqual(provider["http_status"], 200)
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 12.0)
        self.assertEqual(request.full_url, "https://api.gpt.ge/v1/chat/completions")
        self.assertEqual(request.method, "POST")
        self.assertEqual(payload["model"], "qwen3.5-27b")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "test"}])
        self.assertEqual(payload["temperature"], 0)
        self.assertIs(payload["enable_thinking"], False)
        self.assertEqual(payload["max_tokens"], 512)
        self.assertIs(payload["stream"], False)
        self.assertEqual(request.get_header("Authorization"), "Bearer dummy-secret")
        self.assertEqual(request.get_header("Content-type"), "application/json")

    def test_episode_uses_owner_messages_sampler_and_terminal_parser(self) -> None:
        episode = scheduled_episodes()[0]
        layout = episode.task.layout
        calls: list[tuple[dict[str, str], ...]] = []

        def fake_chat(api_key, model, messages, timeout):
            self.assertEqual(api_key, "dummy")
            self.assertEqual(model, "qwen3.5-27b")
            self.assertGreater(timeout, 0)
            calls.append(messages)
            if len(calls) == 1:
                return (
                    json.dumps(
                        {
                            "type": "intervene",
                            "target": layout.first_label,
                            "value": 1,
                            "batch_size": 4,
                        }
                    ),
                    {"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
                )
            forward_field = f"effect_{layout.first_label.lower()}_to_{layout.second_label.lower()}"
            reverse_field = f"effect_{layout.second_label.lower()}_to_{layout.first_label.lower()}"
            return (
                json.dumps(
                    {
                        "type": "answer",
                        forward_field: 0.8,
                        reverse_field: 0.0,
                    }
                ),
                {"usage": {"prompt_tokens": 20, "completion_tokens": 6}},
            )

        result = run_episode("dummy", "qwen3.5-27b", episode, chat=fake_chat)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["rounds_used"], 1)
        self.assertEqual(result["samples_used"], 4)
        self.assertEqual(result["usage"], {"prompt_tokens": 30, "completion_tokens": 11})
        self.assertEqual(len(calls), 2)
        self.assertIn('"type":"batch_result"', calls[1][-1]["content"])
        self.assertNotIn(episode.seed.seed_id, json.dumps(calls))
        self.assertNotIn(episode.world.direction.value, json.dumps(calls).lower())
        self.assertNotIn("dummy", json.dumps(result))

    def test_protocol_failure_is_recorded_without_a_repair_call(self) -> None:
        episode = scheduled_episodes()[0]
        calls = 0

        def invalid_chat(api_key, model, messages, timeout):
            nonlocal calls
            calls += 1
            return "not-json", {}

        result = run_episode("dummy", "qwen3.5-27b", episode, chat=invalid_chat)
        self.assertEqual(calls, 1)
        self.assertEqual(result["status"], "protocol_failure")
        self.assertIsNone(result["terminal_effects"])

    def test_provider_error_cannot_persist_the_api_key(self) -> None:
        episode = scheduled_episodes()[0]

        def leaking_error(api_key, model, messages, timeout):
            raise OSError(f"Authorization: Bearer {api_key}")

        result = run_episode("dummy-secret", "qwen3.5-27b", episode, chat=leaking_error)
        self.assertEqual(result["status"], "infrastructure_failure")
        self.assertNotIn("dummy-secret", json.dumps(result))

    def test_environment_invariant_failure_fails_loud(self) -> None:
        episode = scheduled_episodes()[0]
        layout = episode.task.layout

        def intervention_chat(api_key, model, messages, timeout):
            return (
                json.dumps(
                    {
                        "type": "intervene",
                        "target": layout.first_label,
                        "value": 1,
                        "batch_size": 4,
                    }
                ),
                {"usage": {}},
            )

        with patch(
            "scripts.run_gpt_ge_seed_pilot.EpisodeSampler.intervene",
            side_effect=RuntimeError("evaluator invariant broke"),
        ):
            with self.assertRaisesRegex(RuntimeError, "evaluator invariant broke"):
                run_episode("dummy", "qwen3.5-27b", episode, chat=intervention_chat)

    def test_success_metadata_cannot_persist_the_api_key(self) -> None:
        episode = scheduled_episodes()[0]
        layout = episode.task.layout

        def echoing_metadata(api_key, model, messages, timeout):
            truth = episode.world.truth.effects
            return (
                json.dumps(
                    {
                        "type": "answer",
                        (
                            f"effect_{layout.first_label.lower()}_to_{layout.second_label.lower()}"
                        ): truth.first_to_second,
                        (
                            f"effect_{layout.second_label.lower()}_to_{layout.first_label.lower()}"
                        ): truth.second_to_first,
                    }
                ),
                {
                    "usage": {"prompt_tokens": 1, "note": api_key},
                    "response_id": api_key,
                },
            )

        result = run_episode(
            "dummy-secret",
            "qwen3.5-27b",
            episode,
            chat=echoing_metadata,
        )
        self.assertEqual(result["status"], "completed")
        self.assertNotIn("dummy-secret", json.dumps(result))
        self.assertEqual(result["turns"][0]["provider"]["response_id"], "[REDACTED_API_KEY]")

    def test_validator_replays_raw_terminal_answer_with_the_owner_parser(self) -> None:
        episode = scheduled_episodes()[0]
        layout = episode.task.layout

        def zero_chat(api_key, model, messages, timeout):
            return (
                json.dumps(
                    {
                        "type": "answer",
                        (
                            f"effect_{layout.first_label.lower()}_to_{layout.second_label.lower()}"
                        ): 0.0,
                        (
                            f"effect_{layout.second_label.lower()}_to_{layout.first_label.lower()}"
                        ): 0.0,
                    }
                ),
                {"usage": {}},
            )

        record = run_episode("dummy", "qwen3.5-27b", episode, chat=zero_chat)
        tampered = deepcopy(record)
        tampered["turns"][0]["parsed_command"]["second_to_first"] = 0.8
        tampered["terminal_effects"]["second_to_first"] = 0.8
        with self.assertRaisesRegex(ValueError, "owner parser"):
            _validate_episode_record("qwen3.5-27b", episode, tampered)

    def test_validator_rejects_drifted_usage_and_provider_provenance(self) -> None:
        episode = scheduled_episodes()[0]
        layout = episode.task.layout

        def terminal_chat(api_key, model, messages, timeout):
            return (
                json.dumps(
                    {
                        "type": "answer",
                        (
                            f"effect_{layout.first_label.lower()}_to_{layout.second_label.lower()}"
                        ): 0.0,
                        (
                            f"effect_{layout.second_label.lower()}_to_{layout.first_label.lower()}"
                        ): 0.0,
                    }
                ),
                {"usage": {"prompt_tokens": 2, "completion_tokens": 1}},
            )

        valid = run_episode("dummy", "qwen3.5-27b", episode, chat=terminal_chat)
        mutations = {
            "aggregate usage": lambda record: record.__setitem__(
                "usage", {"prompt_tokens": 999, "completion_tokens": 1}
            ),
            "response ID type": lambda record: record["turns"][0]["provider"].__setitem__(
                "response_id", {}
            ),
            "redaction type": lambda record: record["turns"][0]["provider"].__setitem__(
                "redacted_fields", "response_id"
            ),
            "HTTP status": lambda record: record["turns"][0]["provider"].__setitem__(
                "http_status", -1
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                tampered = deepcopy(valid)
                mutate(tampered)
                with self.assertRaises(ValueError):
                    _validate_episode_record("qwen3.5-27b", episode, tampered)

    def test_validator_rejects_unknown_raw_response_copy(self) -> None:
        episode = scheduled_episodes()[0]

        def invalid_chat(api_key, model, messages, timeout):
            return "not-json", {"usage": {}}

        record = run_episode("dummy", "qwen3.5-27b", episode, chat=invalid_chat)
        record["turns"][0]["raw_response"] = "secret duplicate"
        with self.assertRaisesRegex(ValueError, "field set"):
            _validate_episode_record("qwen3.5-27b", episode, record)

    def test_validator_binds_protocol_failure_metadata_to_replay(self) -> None:
        episode = scheduled_episodes()[0]

        def invalid_chat(api_key, model, messages, timeout):
            return "not-json", {"usage": {}}

        valid = run_episode("dummy", "qwen3.5-27b", episode, chat=invalid_chat)
        for key, value in (
            ("failure_turn", 7),
            ("failure_code", "WrongError"),
            ("failure_detail", "wrong detail"),
        ):
            with self.subTest(key=key):
                tampered = deepcopy(valid)
                tampered[key] = value
                with self.assertRaisesRegex(ValueError, "metadata"):
                    _validate_episode_record("qwen3.5-27b", episode, tampered)

    def test_validator_accepts_native_terminal_failure_records(self) -> None:
        episode = scheduled_episodes()[0]
        layout = episode.task.layout

        def completed_chat(api_key, model, messages, timeout):
            return (
                json.dumps(
                    {
                        "type": "answer",
                        (
                            f"effect_{layout.first_label.lower()}_to_{layout.second_label.lower()}"
                        ): 0.0,
                        (
                            f"effect_{layout.second_label.lower()}_to_{layout.first_label.lower()}"
                        ): 0.0,
                    }
                ),
                {"usage": {}},
            )

        records = (
            run_episode("dummy", "qwen3.5-27b", episode, chat=completed_chat),
            run_episode(
                "dummy",
                "qwen3.5-27b",
                episode,
                chat=lambda api_key, model, messages, timeout: ("not-json", {"usage": {}}),
            ),
            run_episode(
                "dummy",
                "qwen3.5-27b",
                episode,
                chat=lambda api_key, model, messages, timeout: (_ for _ in ()).throw(
                    OSError("transport")
                ),
            ),
        )
        for record in records:
            _validate_episode_record("qwen3.5-27b", episode, record)

    def test_assistant_credential_echo_is_redacted_and_replayable(self) -> None:
        episode = scheduled_episodes()[0]
        secret = "dummy-secret"
        record = run_episode(
            secret,
            "qwen3.5-27b",
            episode,
            chat=lambda api_key, model, messages, timeout: (api_key, {"usage": {}}),
        )
        self.assertEqual(record["status"], "protocol_failure")
        self.assertNotIn(secret, json.dumps(record))
        _validate_episode_record("qwen3.5-27b", episode, record)

    def test_profile_keeps_failure_coverage_explicit(self) -> None:
        schedule = scheduled_episodes()
        records = []
        for episode in schedule:
            truth = episode.world.truth.effects
            layout = episode.task.layout

            def perfect_chat(api_key, model, messages, timeout, *, truth=truth, layout=layout):
                return (
                    json.dumps(
                        {
                            "type": "answer",
                            (
                                f"effect_{layout.first_label.lower()}_to_"
                                f"{layout.second_label.lower()}"
                            ): truth.first_to_second,
                            (
                                f"effect_{layout.second_label.lower()}_to_"
                                f"{layout.first_label.lower()}"
                            ): truth.second_to_first,
                        }
                    ),
                    {"usage": {}},
                )

            records.append(run_episode("dummy", "qwen3.5-27b", episode, chat=perfect_chat))

        def invalid_chat(api_key, model, messages, timeout):
            return "not-json", {"usage": {}}

        records[-1] = run_episode(
            "dummy",
            "qwen3.5-27b",
            schedule[-1],
            chat=invalid_chat,
        )
        profile = build_profile("qwen3.5-27b", schedule, records)
        self.assertEqual(profile["valid_terminal_coverage"], 35 / 36)
        self.assertIsNone(profile["diagnostics_complete"])
        self.assertEqual(profile["diagnostics_valid_only"]["n_episodes"], 35)

    def test_resume_rejects_duplicate_wrong_model_and_over_budget_records(self) -> None:
        schedule = scheduled_episodes()
        config = run_config("qwen3.5-27b", schedule, timeout=180.0)
        episode = schedule[0]

        def invalid_chat(api_key, model, messages, timeout):
            return "not-json", {"usage": {}}

        valid = run_episode("dummy", "qwen3.5-27b", episode, chat=invalid_chat)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "resume.json")
            for mutate in ("duplicate", "wrong-model", "over-budget"):
                document = _new_document("qwen3.5-27b", schedule, timeout=180.0)
                record = deepcopy(valid)
                if mutate == "duplicate":
                    document["attempts"] = [record]
                    document["episodes"] = [record, deepcopy(record)]
                elif mutate == "wrong-model":
                    record["model"] = "wrong-model"
                    document["attempts"] = [record]
                    document["episodes"] = [record]
                else:
                    record["samples_used"] = 999
                    document["attempts"] = [record]
                    document["episodes"] = [record]
                _write_json_atomic(path, document)
                with self.assertRaises(ValueError, msg=mutate):
                    _load_document(path, config, schedule)

            missing_latest = _new_document("qwen3.5-27b", schedule, timeout=180.0)
            missing_latest["attempts"] = [valid]
            _write_json_atomic(path, missing_latest)
            with self.assertRaisesRegex(ValueError, "ordered attempted prefix"):
                _load_document(path, config, schedule)

            later_infrastructure = run_episode(
                "dummy",
                "qwen3.5-27b",
                episode,
                attempt_index=1,
                chat=lambda api_key, model, messages, timeout: (_ for _ in ()).throw(
                    OSError("transport")
                ),
            )
            illegal_retry = _new_document("qwen3.5-27b", schedule, timeout=180.0)
            illegal_retry["attempts"] = [valid, later_infrastructure]
            illegal_retry["episodes"] = [later_infrastructure]
            _write_json_atomic(path, illegal_retry)
            with self.assertRaisesRegex(ValueError, "cannot have a later attempt"):
                _load_document(path, config, schedule)

    def test_resume_accepts_infrastructure_then_completed_attempts(self) -> None:
        schedule = scheduled_episodes()
        config = run_config("qwen3.5-27b", schedule, timeout=180.0)
        episode = schedule[0]
        layout = episode.task.layout
        failed = run_episode(
            "dummy",
            "qwen3.5-27b",
            episode,
            attempt_index=0,
            chat=lambda api_key, model, messages, timeout: (_ for _ in ()).throw(
                OSError("transport")
            ),
        )

        def completed_chat(api_key, model, messages, timeout):
            return (
                json.dumps(
                    {
                        "type": "answer",
                        (
                            f"effect_{layout.first_label.lower()}_to_{layout.second_label.lower()}"
                        ): 0.0,
                        (
                            f"effect_{layout.second_label.lower()}_to_{layout.first_label.lower()}"
                        ): 0.0,
                    }
                ),
                {"usage": {}},
            )

        completed = run_episode(
            "dummy",
            "qwen3.5-27b",
            episode,
            attempt_index=1,
            chat=completed_chat,
        )
        document = _new_document("qwen3.5-27b", schedule, timeout=180.0)
        document["attempts"] = [failed, completed]
        document["episodes"] = [completed]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "resume.json")
            _write_json_atomic(path, document)
            loaded = _load_document(path, config, schedule)
        self.assertEqual(loaded["episodes"], [completed])

    def test_resume_rejects_active_api_key_anywhere_in_artifact(self) -> None:
        schedule = scheduled_episodes()
        config = run_config("qwen3.5-27b", schedule, timeout=180.0)
        secret = "dummy-active-secret"
        document = _new_document("qwen3.5-27b", schedule, timeout=180.0)
        document["profile"] = {"untrusted_note": secret}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "resume.json")
            _write_json_atomic(path, document)
            with self.assertRaisesRegex(ValueError, "active API credential"):
                _load_document(path, config, schedule, api_key=secret)

    def test_resume_rejects_nonprefix_reordered_and_unresolved_progress(self) -> None:
        schedule = scheduled_episodes()
        config = run_config("qwen3.5-27b", schedule, timeout=180.0)

        def invalid_chat(api_key, model, messages, timeout):
            return "not-json", {"usage": {}}

        first = run_episode("dummy", "qwen3.5-27b", schedule[0], chat=invalid_chat)
        second = run_episode("dummy", "qwen3.5-27b", schedule[1], chat=invalid_chat)
        unresolved = run_episode(
            "dummy",
            "qwen3.5-27b",
            schedule[0],
            chat=lambda api_key, model, messages, timeout: (_ for _ in ()).throw(
                OSError("transport")
            ),
        )
        cases = {
            "reordered": ([second, first], [second, first]),
            "nonprefix": ([second], [second]),
            "past unresolved": ([unresolved, second], [unresolved, second]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "resume.json")
            for label, (attempts, latest) in cases.items():
                with self.subTest(label=label):
                    document = _new_document("qwen3.5-27b", schedule, timeout=180.0)
                    document["attempts"] = attempts
                    document["episodes"] = latest
                    _write_json_atomic(path, document)
                    with self.assertRaises(ValueError):
                        _load_document(path, config, schedule)

    def test_resume_rejects_duplicate_json_keys_hiding_active_api_key(self) -> None:
        schedule = scheduled_episodes()
        config = run_config("qwen3.5-27b", schedule, timeout=180.0)
        secret = "dummy-active-secret"
        document = _new_document("qwen3.5-27b", schedule, timeout=180.0)
        encoded = json.dumps(document, ensure_ascii=False)
        escaped_secret = secret.replace("-", "\\u002d")
        duplicate = encoded.replace(
            '"profile": null',
            f'"profile": {{"hidden": "{escaped_secret}"}}, "profile": null',
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "resume.json")
            path.write_text(duplicate, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate object key"):
                _load_document(path, config, schedule, api_key=secret)


if __name__ == "__main__":
    unittest.main()
