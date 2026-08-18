"""Run the paired CPT-world seed pilot through the historical gpt.ge endpoint.

The provider bridge is intentionally thin: it sends the exact messages emitted
by ``cpt_world.protocol`` and returns the provider's message content unchanged.
It does not parse commands, sample worlds, or compute diagnostics itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cpt_world import (
    DEFAULT_LABEL_SEED,
    METRICS_SCHEMA_VERSION,
    RENDERER_VERSION,
    SEED_SUITE_VERSION,
    EffectVector,
    EpisodeSampler,
    InterventionCommand,
    SeedEpisode,
    TerminalAnswer,
    build_candidate_episodes,
    compute_terminal_diagnostics,
    factorial_layouts,
    parse_command,
    render_batch_message,
    render_initial_messages,
)

API_URL = "https://api.gpt.ge/v1/chat/completions"
API_KEY_ENV = "GPT_GE_API_KEY"
ADAPTER_VERSION = "gpt-ge-chat-completions-v1"
PILOT_VERSION = "paired-two-model-seed-pilot-v1"
RESULT_SCHEMA_VERSION = "cpt-world-model-pilot-result-v2"
REPLICATE_ID = "two-model-seed-v1-r0"
SUPPORTED_MODELS = ("qwen3.5-27b", "DeepSeek-V4-Pro")
TEMPERATURE = 0
THINKING_ENABLED = False
MAX_TOKENS = 512
STREAM = False

_PROVIDER_TEXT_FIELDS = (
    "response_id",
    "response_model",
    "system_fingerprint",
    "finish_reason",
    "request_id",
)
_PROVIDER_KEYS = {
    "usage",
    *_PROVIDER_TEXT_FIELDS,
    "http_status",
    "redacted_fields",
}
_RECORD_KEYS = {
    "model",
    "episode_id",
    "seed_id",
    "difficulty",
    "layout_id",
    "hidden_direction",
    "truth_effects",
    "initial_messages_sha256",
    "status",
    "turns",
    "messages",
    "rounds_used",
    "samples_used",
    "terminal_effects",
    "failure_turn",
    "failure_code",
    "failure_detail",
    "usage",
    "attempt_index",
}

# One occurrence of every role assignment and target order, with a 3/3 split
# of terminal effect-line order. This is a marginally balanced pilot subset,
# not the final benchmark sampling distribution.
_PILOT_LAYOUT_COORDINATES = (
    (0, 0, False),
    (1, 1, True),
    (2, 5, True),
    (3, 3, False),
    (4, 4, False),
    (5, 2, True),
)

ChatCompletion = Callable[
    [str, str, Sequence[Mapping[str, str]], float],
    tuple[str, dict[str, Any]],
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pilot_layouts(label_seed: int = DEFAULT_LABEL_SEED):
    """Return the preregistered six-layout marginally balanced pilot subset."""

    all_layouts = {layout.layout_id: layout for layout in factorial_layouts(label_seed)}
    selected = []
    for role_index, target_index, reverse_effect_order in _PILOT_LAYOUT_COORDINATES:
        layout_id = (
            f"labels-{label_seed}-r{role_index}-t{target_index}-e{int(reverse_effect_order)}"
        )
        selected.append(all_layouts[layout_id])
    return tuple(selected)


def scheduled_episodes() -> tuple[SeedEpisode, ...]:
    episodes = build_candidate_episodes(layouts=pilot_layouts(), replicate_id=REPLICATE_ID)
    return tuple(sorted(episodes, key=lambda episode: episode.episode_id))


def chat_completion(
    api_key: str,
    model: str,
    messages: Sequence[Mapping[str, str]],
    timeout: float = 180.0,
) -> tuple[str, dict[str, Any]]:
    """Call the one gpt.ge Chat Completions contract used by prior pilots."""

    payload = json.dumps(
        {
            "model": model,
            "messages": list(messages),
            "temperature": TEMPERATURE,
            "enable_thinking": THINKING_ENABLED,
            "max_tokens": MAX_TOKENS,
            "stream": STREAM,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        parsed = json.loads(response.read().decode("utf-8"))
        response_status = getattr(response, "status", None)
        response_headers = getattr(response, "headers", None)
        request_id = response_headers.get("x-request-id") if response_headers is not None else None
    try:
        choice = parsed["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("provider response has no assistant message content") from error
    if not isinstance(content, str):
        raise ValueError("provider assistant message content must be a string")
    usage = parsed.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    provider = {
        "usage": usage,
        "response_id": parsed.get("id") if isinstance(parsed.get("id"), str) else None,
        "response_model": (parsed.get("model") if isinstance(parsed.get("model"), str) else None),
        "system_fingerprint": (
            parsed.get("system_fingerprint")
            if isinstance(parsed.get("system_fingerprint"), str)
            else None
        ),
        "finish_reason": (
            choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None
        ),
        "request_id": request_id if isinstance(request_id, str) else None,
        "http_status": response_status if isinstance(response_status, int) else None,
    }
    return content, provider


def _base_episode_record(model: str, episode: SeedEpisode) -> dict[str, Any]:
    truth = episode.world.truth.effects
    initial_messages = [dict(message) for message in render_initial_messages(episode.task)]
    return {
        "model": model,
        "episode_id": episode.episode_id,
        "seed_id": episode.seed.seed_id,
        "difficulty": episode.seed.difficulty,
        "layout_id": episode.task.layout.layout_id,
        "hidden_direction": episode.world.direction.value,
        "truth_effects": {
            "first_to_second": truth.first_to_second,
            "second_to_first": truth.second_to_first,
        },
        "initial_messages_sha256": canonical_json_sha256(initial_messages),
        "status": None,
        "turns": [],
        "messages": initial_messages,
        "rounds_used": 0,
        "samples_used": 0,
        "terminal_effects": None,
        "failure_turn": None,
        "failure_code": None,
        "failure_detail": None,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }


def _add_usage(record: dict[str, Any], usage: Mapping[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens"):
        value = usage.get(key, 0)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            record["usage"][key] += value


def _safe_provider_metadata(provider: Mapping[str, Any], api_key: str) -> dict[str, Any]:
    """Keep only audited provider provenance and remove an echoed credential."""

    if not isinstance(provider, Mapping):
        raise ValueError("provider metadata must be a mapping")
    raw_usage = provider.get("usage", {})
    usage: dict[str, int] = {}
    if isinstance(raw_usage, Mapping):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = raw_usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                usage[key] = value

    result: dict[str, Any] = {"usage": usage}
    redacted_fields: list[str] = []
    for key in _PROVIDER_TEXT_FIELDS:
        value = provider.get(key)
        if isinstance(value, str):
            if api_key and api_key in value:
                result[key] = "[REDACTED_API_KEY]"
                redacted_fields.append(key)
            else:
                result[key] = value
        else:
            result[key] = None
    status = provider.get("http_status")
    result["http_status"] = (
        status if isinstance(status, int) and not isinstance(status, bool) else None
    )
    result["redacted_fields"] = redacted_fields
    return result


def _intervention_record(command: InterventionCommand) -> dict[str, Any]:
    return {
        "type": "intervene",
        "target": command.intervention.target.value,
        "value": command.intervention.value,
        "batch_size": command.batch_size,
    }


def run_episode(
    api_key: str,
    model: str,
    episode: SeedEpisode,
    *,
    timeout: float = 180.0,
    attempt_index: int = 0,
    chat: ChatCompletion = chat_completion,
) -> dict[str, Any]:
    """Run one model/world interaction without repairing invalid commands."""

    if isinstance(attempt_index, bool) or not isinstance(attempt_index, int) or attempt_index < 0:
        raise ValueError("attempt_index must be a nonnegative integer")
    record = _base_episode_record(model, episode)
    record["attempt_index"] = attempt_index
    messages: list[dict[str, str]] = record["messages"]
    sampler = EpisodeSampler(episode.world, episode.tape, episode.task.budget)

    for turn_index in range(episode.task.budget.max_rounds + 1):
        try:
            raw, provider = chat(api_key, model, tuple(messages), timeout)
        except OSError as error:
            record.update(
                {
                    "status": "infrastructure_failure",
                    "failure_turn": turn_index,
                    "failure_code": type(error).__name__,
                    "failure_detail": "provider transport request failed",
                    "rounds_used": sampler.rounds_used,
                    "samples_used": sampler.samples_used,
                }
            )
            return record
        except ValueError as error:
            record.update(
                {
                    "status": "infrastructure_failure",
                    "failure_turn": turn_index,
                    "failure_code": type(error).__name__,
                    "failure_detail": "provider response could not be decoded",
                    "rounds_used": sampler.rounds_used,
                    "samples_used": sampler.samples_used,
                }
            )
            return record

        try:
            provider = _safe_provider_metadata(provider, api_key)
        except ValueError as error:
            record.update(
                {
                    "status": "infrastructure_failure",
                    "failure_turn": turn_index,
                    "failure_code": type(error).__name__,
                    "failure_detail": "provider response metadata is invalid",
                    "rounds_used": sampler.rounds_used,
                    "samples_used": sampler.samples_used,
                }
            )
            return record
        usage = provider["usage"]
        _add_usage(record, usage if isinstance(usage, Mapping) else {})
        stored_raw = (
            raw.replace(api_key, "[REDACTED_API_KEY]") if api_key and api_key in raw else raw
        )
        messages.append({"role": "assistant", "content": stored_raw})
        turn_record: dict[str, Any] = {
            "turn": turn_index,
            "assistant_message_index": len(messages) - 1,
            "provider": provider,
        }
        try:
            command = parse_command(
                stored_raw,
                episode.task,
                remaining_rounds=sampler.remaining_rounds,
                remaining_samples=sampler.remaining_samples,
            )
        except (TypeError, ValueError) as error:
            turn_record["parse_error"] = str(error)
            record["turns"].append(turn_record)
            record.update(
                {
                    "status": "protocol_failure",
                    "failure_turn": turn_index,
                    "failure_code": type(error).__name__,
                    "failure_detail": str(error),
                    "rounds_used": sampler.rounds_used,
                    "samples_used": sampler.samples_used,
                }
            )
            return record

        if isinstance(command, TerminalAnswer):
            turn_record["parsed_command"] = {
                "type": "answer",
                "first_to_second": command.effects.first_to_second,
                "second_to_first": command.effects.second_to_first,
            }
            record["turns"].append(turn_record)
            record.update(
                {
                    "status": "completed",
                    "terminal_effects": {
                        "first_to_second": command.effects.first_to_second,
                        "second_to_first": command.effects.second_to_first,
                    },
                    "rounds_used": sampler.rounds_used,
                    "samples_used": sampler.samples_used,
                }
            )
            return record

        turn_record["parsed_command"] = _intervention_record(command)
        # The public parser has already established action and budget legality.
        # A sampler exception therefore signals an evaluator invariant bug and
        # must fail loudly instead of becoming a resumable provider failure.
        batch = sampler.intervene(command)
        observation = render_batch_message(
            batch,
            episode.task,
            remaining_rounds=sampler.remaining_rounds,
            remaining_samples=sampler.remaining_samples,
        )
        turn_record["batch_counts"] = list(batch.counts)
        record["turns"].append(turn_record)
        messages.append({"role": "user", "content": observation})

    raise RuntimeError("episode loop ended without a terminal result")


def _diagnostics_for_ids(
    episode_by_id: Mapping[str, SeedEpisode],
    record_by_id: Mapping[str, Mapping[str, Any]],
    episode_ids: Sequence[str],
) -> dict[str, Any] | None:
    if not episode_ids:
        return None
    truths = {episode_id: episode_by_id[episode_id].world.truth for episode_id in episode_ids}
    predictions = {
        episode_id: EffectVector(**record_by_id[episode_id]["terminal_effects"])
        for episode_id in episode_ids
    }
    diagnostics = compute_terminal_diagnostics(truths, predictions)
    return {
        "schema_version": diagnostics.schema_version,
        "vector_rmse": diagnostics.vector_rmse,
        "active_mae": diagnostics.active_mae,
        "inactive_mae": diagnostics.inactive_mae,
        "n_episodes": diagnostics.n_episodes,
    }


def build_profile(
    model: str,
    schedule: Sequence[SeedEpisode],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a coverage-explicit profile; failed episodes are never hidden."""

    scheduled_ids = [episode.episode_id for episode in schedule]
    if len(set(scheduled_ids)) != len(scheduled_ids):
        raise ValueError("scheduled episode IDs must be unique")
    record_ids = [record["episode_id"] for record in records]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("result episode IDs must be unique")
    if set(record_ids) != set(scheduled_ids):
        raise ValueError("records must cover the exact scheduled episode set")

    episode_by_id = {episode.episode_id: episode for episode in schedule}
    for record in records:
        _validate_episode_record(model, episode_by_id[str(record["episode_id"])], record)
    record_by_id = {str(record["episode_id"]): record for record in records}
    status_counts = Counter(str(record["status"]) for record in records)
    completed_ids = [
        episode_id
        for episode_id in scheduled_ids
        if record_by_id[episode_id]["status"] == "completed"
    ]
    valid_only = _diagnostics_for_ids(episode_by_id, record_by_id, completed_ids)
    complete = valid_only if len(completed_ids) == len(scheduled_ids) else None

    by_difficulty: dict[str, Any] = {}
    for difficulty in ("easy", "medium", "hard"):
        group_ids = [
            episode.episode_id for episode in schedule if episode.seed.difficulty == difficulty
        ]
        valid_group_ids = [
            episode_id
            for episode_id in group_ids
            if record_by_id[episode_id]["status"] == "completed"
        ]
        by_difficulty[difficulty] = {
            "scheduled": len(group_ids),
            "valid_terminal": len(valid_group_ids),
            "valid_terminal_coverage": len(valid_group_ids) / len(group_ids),
            "diagnostics_valid_only": _diagnostics_for_ids(
                episode_by_id, record_by_id, valid_group_ids
            ),
            "diagnostics_complete": (
                _diagnostics_for_ids(episode_by_id, record_by_id, group_ids)
                if len(valid_group_ids) == len(group_ids)
                else None
            ),
        }

    completed_records = [record_by_id[episode_id] for episode_id in completed_ids]
    return {
        "model": model,
        "scheduled": len(schedule),
        "status_counts": dict(sorted(status_counts.items())),
        "valid_terminal_coverage": len(completed_ids) / len(schedule),
        "diagnostics_schema_version": METRICS_SCHEMA_VERSION,
        "diagnostics_valid_only": valid_only,
        "diagnostics_complete": complete,
        "mean_rounds_used_valid": (
            sum(int(record["rounds_used"]) for record in completed_records) / len(completed_records)
            if completed_records
            else None
        ),
        "mean_samples_used_valid": (
            sum(int(record["samples_used"]) for record in completed_records)
            / len(completed_records)
            if completed_records
            else None
        ),
        "by_difficulty": by_difficulty,
    }


def run_config(
    model: str,
    schedule: Sequence[SeedEpisode],
    *,
    timeout: float,
) -> dict[str, Any]:
    budget = schedule[0].task.budget
    return {
        "pilot_version": PILOT_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "model": model,
        "endpoint": API_URL,
        "label_seed": DEFAULT_LABEL_SEED,
        "replicate_id": REPLICATE_ID,
        "renderer_version": RENDERER_VERSION,
        "seed_suite_version": SEED_SUITE_VERSION,
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "layout_ids": [layout.layout_id for layout in pilot_layouts()],
        "scheduled_episode_ids": [episode.episode_id for episode in schedule],
        "schedule_order": "ascending-opaque-episode-id-v1",
        "budget": {
            "max_rounds": budget.max_rounds,
            "max_samples": budget.max_samples,
            "batch_sizes": list(budget.batch_sizes),
        },
        "request": {
            "temperature": TEMPERATURE,
            "thinking_enabled": THINKING_ENABLED,
            "max_tokens": MAX_TOKENS,
            "stream": STREAM,
            "timeout_seconds": timeout,
        },
        "protocol_repairs": 0,
    }


def pairing_sha256(config: Mapping[str, Any]) -> str:
    """Digest the conditions that must match across the two requested models."""

    paired = dict(config)
    paired.pop("model", None)
    return canonical_json_sha256(paired)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _contains_secret(value: Any, secret: str) -> bool:
    if isinstance(value, str):
        return secret in value
    if isinstance(value, Mapping):
        return any(
            _contains_secret(key, secret) or _contains_secret(item, secret)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_secret(item, secret) for item in value)
    return False


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("resume JSON contains a duplicate object key")
        result[key] = value
    return result


def _validate_provider_metadata(provider: Any) -> dict[str, int]:
    if not isinstance(provider, dict) or set(provider) != _PROVIDER_KEYS:
        raise ValueError("turn contains invalid provider provenance")

    raw_usage = provider["usage"]
    if not isinstance(raw_usage, dict) or any(
        key not in {"prompt_tokens", "completion_tokens", "total_tokens"}
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for key, value in raw_usage.items()
    ):
        raise ValueError("turn contains invalid provider usage")

    for key in _PROVIDER_TEXT_FIELDS:
        if provider[key] is not None and not isinstance(provider[key], str):
            raise ValueError("turn contains invalid provider text provenance")
    status = provider["http_status"]
    if status is not None and (
        isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599
    ):
        raise ValueError("turn contains an invalid provider HTTP status")

    redacted = provider["redacted_fields"]
    if (
        not isinstance(redacted, list)
        or any(not isinstance(key, str) or key not in _PROVIDER_TEXT_FIELDS for key in redacted)
        or len(set(redacted)) != len(redacted)
    ):
        raise ValueError("turn contains invalid provider redaction provenance")
    redacted_set = set(redacted)
    placeholder_fields = {
        key for key in _PROVIDER_TEXT_FIELDS if provider[key] == "[REDACTED_API_KEY]"
    }
    if redacted_set != placeholder_fields:
        raise ValueError("provider redaction provenance does not match its fields")
    return raw_usage


def _validate_episode_record(
    model: str,
    episode: SeedEpisode,
    record: Mapping[str, Any],
) -> None:
    record_keys = set(record)
    if record_keys not in (_RECORD_KEYS, _RECORD_KEYS | {"elapsed_seconds"}):
        raise ValueError("episode record has an invalid field set")
    if "elapsed_seconds" in record:
        elapsed = record["elapsed_seconds"]
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError("episode record has invalid elapsed_seconds")

    expected_truth = episode.world.truth.effects
    expected_fields = {
        "model": model,
        "episode_id": episode.episode_id,
        "seed_id": episode.seed.seed_id,
        "difficulty": episode.seed.difficulty,
        "layout_id": episode.task.layout.layout_id,
        "hidden_direction": episode.world.direction.value,
        "truth_effects": {
            "first_to_second": expected_truth.first_to_second,
            "second_to_first": expected_truth.second_to_first,
        },
    }
    for key, expected in expected_fields.items():
        if record.get(key) != expected:
            raise ValueError(f"episode record has invalid {key}")

    expected_initial = [dict(message) for message in render_initial_messages(episode.task)]
    if record.get("initial_messages_sha256") != canonical_json_sha256(expected_initial):
        raise ValueError("episode record has an invalid initial-message digest")
    messages = record.get("messages")
    if not isinstance(messages, list) or messages[: len(expected_initial)] != expected_initial:
        raise ValueError("episode record does not preserve the exact initial messages")
    if any(
        not isinstance(message, dict)
        or set(message) != {"role", "content"}
        or message["role"] not in {"system", "user", "assistant"}
        or not isinstance(message["content"], str)
        for message in messages
    ):
        raise ValueError("episode record contains an invalid transcript message")

    attempt_index = record.get("attempt_index")
    if isinstance(attempt_index, bool) or not isinstance(attempt_index, int) or attempt_index < 0:
        raise ValueError("episode attempt_index must be a nonnegative integer")
    status = record.get("status")
    if status not in {"completed", "protocol_failure", "infrastructure_failure"}:
        raise ValueError("episode record has an unknown status")
    rounds_used = record.get("rounds_used")
    samples_used = record.get("samples_used")
    if (
        isinstance(rounds_used, bool)
        or not isinstance(rounds_used, int)
        or not 0 <= rounds_used <= episode.task.budget.max_rounds
    ):
        raise ValueError("episode record has invalid rounds_used")
    if (
        isinstance(samples_used, bool)
        or not isinstance(samples_used, int)
        or not 0 <= samples_used <= episode.task.budget.max_samples
    ):
        raise ValueError("episode record has invalid samples_used")

    turns = record.get("turns")
    if not isinstance(turns, list):
        raise ValueError("episode record turns must be a list")
    sampler = EpisodeSampler(episode.world, episode.tape, episode.task.budget)
    message_cursor = len(expected_initial)
    parsed_terminal: TerminalAnswer | None = None
    parse_failure: tuple[int, str, str] | None = None
    aggregate_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    common_turn_keys = {"turn", "assistant_message_index", "provider"}
    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError("episode record contains an invalid turn")
        if turn.get("turn") != turn_index:
            raise ValueError("episode turn indices are not contiguous")
        message_index = turn.get("assistant_message_index")
        if (
            isinstance(message_index, bool)
            or not isinstance(message_index, int)
            or message_index != message_cursor
            or not 0 <= message_index < len(messages)
            or messages[message_index]["role"] != "assistant"
        ):
            raise ValueError("turn does not reference a valid assistant message")
        raw_usage = _validate_provider_metadata(turn.get("provider"))
        for key in aggregate_usage:
            aggregate_usage[key] += raw_usage.get(key, 0)

        raw = messages[message_index]["content"]
        message_cursor += 1
        try:
            command = parse_command(
                raw,
                episode.task,
                remaining_rounds=sampler.remaining_rounds,
                remaining_samples=sampler.remaining_samples,
            )
        except (TypeError, ValueError) as error:
            if set(turn) != common_turn_keys | {"parse_error"}:
                raise ValueError("parse-failure turn has an invalid field set") from error
            if turn.get("parse_error") != str(error) or "parsed_command" in turn:
                raise ValueError("stored parse failure does not match the owner parser") from error
            if turn_index != len(turns) - 1:
                raise ValueError("a parse failure must end its attempt") from error
            parse_failure = (turn_index, type(error).__name__, str(error))
            continue

        if "parse_error" in turn:
            raise ValueError("successfully parsed turn cannot contain parse_error")
        if isinstance(command, TerminalAnswer):
            if set(turn) != common_turn_keys | {"parsed_command"}:
                raise ValueError("terminal turn has an invalid field set")
            expected_command = {
                "type": "answer",
                "first_to_second": command.effects.first_to_second,
                "second_to_first": command.effects.second_to_first,
            }
            if turn.get("parsed_command") != expected_command:
                raise ValueError("stored terminal command does not match the owner parser")
            if turn_index != len(turns) - 1:
                raise ValueError("a terminal answer must end its attempt")
            parsed_terminal = command
            continue

        if set(turn) != common_turn_keys | {"parsed_command", "batch_counts"}:
            raise ValueError("intervention turn has an invalid field set")
        if turn.get("parsed_command") != _intervention_record(command):
            raise ValueError("stored intervention does not match the owner parser")
        batch = sampler.intervene(command)
        if turn.get("batch_counts") != list(batch.counts):
            raise ValueError("stored batch does not match the owner sampler")
        expected_observation = {
            "role": "user",
            "content": render_batch_message(
                batch,
                episode.task,
                remaining_rounds=sampler.remaining_rounds,
                remaining_samples=sampler.remaining_samples,
            ),
        }
        if message_cursor >= len(messages) or messages[message_cursor] != expected_observation:
            raise ValueError("stored observation does not match the owner renderer")
        message_cursor += 1

    if message_cursor != len(messages):
        raise ValueError("episode transcript contains unowned trailing messages")
    if rounds_used != sampler.rounds_used:
        raise ValueError("rounds_used does not match the intervention transcript")
    if samples_used != sampler.samples_used:
        raise ValueError("samples_used does not match the intervention transcript")
    stored_usage = record.get("usage")
    if (
        not isinstance(stored_usage, dict)
        or set(stored_usage) != {"prompt_tokens", "completion_tokens"}
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in stored_usage.values()
        )
        or stored_usage != aggregate_usage
    ):
        raise ValueError("episode aggregate usage does not match provider turns")

    terminal = record.get("terminal_effects")
    if status == "completed":
        if not isinstance(terminal, dict) or set(terminal) != {
            "first_to_second",
            "second_to_first",
        }:
            raise ValueError("completed episode lacks an exact terminal vector")
        terminal_vector = EffectVector(**terminal)
        if parsed_terminal is None or parse_failure is not None:
            raise ValueError("completed episode must contain exactly one terminal command")
        if terminal_vector != parsed_terminal.effects:
            raise ValueError("terminal vector does not match the parsed terminal command")
        if any(
            record.get(key) is not None
            for key in ("failure_code", "failure_detail", "failure_turn")
        ):
            raise ValueError("completed episode cannot contain failure metadata")
    else:
        if terminal is not None:
            raise ValueError("failed episode cannot contain a terminal vector")
        if not isinstance(record.get("failure_code"), str) or not record["failure_code"]:
            raise ValueError("failed episode must contain a failure code")
        failure_turn = record.get("failure_turn")
        if isinstance(failure_turn, bool) or not isinstance(failure_turn, int):
            raise ValueError("failed episode must have an integer failure_turn")
        if status == "protocol_failure":
            if parse_failure is None:
                raise ValueError("protocol failure must be witnessed by the owner parser")
            expected_turn, expected_code, expected_detail = parse_failure
            if (
                failure_turn != expected_turn
                or record.get("failure_code") != expected_code
                or record.get("failure_detail") != expected_detail
            ):
                raise ValueError("protocol failure metadata does not match parser replay")
        else:
            if parse_failure is not None or parsed_terminal is not None:
                raise ValueError("infrastructure failure cannot follow a terminal protocol result")
            if failure_turn != len(turns):
                raise ValueError(
                    "infrastructure failure_turn must identify the failed provider call"
                )
            if record.get("failure_detail") not in {
                "provider transport request failed",
                "provider response could not be decoded",
                "provider response metadata is invalid",
            }:
                raise ValueError(
                    "infrastructure failure detail is not a registered provider failure"
                )


def _validate_document_records(
    document: Mapping[str, Any],
    model: str,
    schedule: Sequence[SeedEpisode],
) -> None:
    if set(document) != {
        "schema_version",
        "created_at",
        "updated_at",
        "config",
        "config_sha256",
        "pairing_sha256",
        "attempts",
        "episodes",
        "profile",
    }:
        raise ValueError("resume file has an invalid document field set")
    if not isinstance(document.get("created_at"), str) or not isinstance(
        document.get("updated_at"), str
    ):
        raise ValueError("resume file timestamps must be strings")
    scheduled_ids = [episode.episode_id for episode in schedule]
    episode_by_id = {episode.episode_id: episode for episode in schedule}
    attempts = document.get("attempts")
    latest = document.get("episodes")
    if not isinstance(attempts, list) or not isinstance(latest, list):
        raise ValueError("resume file attempts and episodes must be lists")

    attempt_groups: dict[str, list[Mapping[str, Any]]] = {}
    attempted_ids: list[str] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            raise ValueError("resume file contains an invalid attempt")
        episode_id = attempt.get("episode_id")
        if episode_id not in episode_by_id:
            raise ValueError("resume file contains an unknown episode ID")
        episode_id = str(episode_id)
        if episode_id not in attempt_groups:
            if (
                len(attempted_ids) >= len(scheduled_ids)
                or episode_id != scheduled_ids[len(attempted_ids)]
            ):
                raise ValueError("resume attempts must follow the scheduled episode prefix")
            attempted_ids.append(episode_id)
            attempt_groups[episode_id] = []
        elif attempted_ids[-1] != episode_id:
            raise ValueError("attempts for one episode must be contiguous")
        group = attempt_groups[episode_id]
        if attempt.get("attempt_index") != len(group):
            raise ValueError("resume attempt indices are not append-only")
        _validate_episode_record(model, episode_by_id[episode_id], attempt)
        group.append(attempt)

    latest_ids = [
        record.get("episode_id") if isinstance(record, dict) else None for record in latest
    ]
    if len(set(latest_ids)) != len(latest_ids):
        raise ValueError("resume file contains duplicate latest episode IDs")
    if latest_ids != attempted_ids:
        raise ValueError("latest episode records must equal the ordered attempted prefix")
    for record in latest:
        if not isinstance(record, dict):
            raise ValueError("resume file contains an invalid latest episode record")
        episode_id = record.get("episode_id")
        if episode_id not in episode_by_id:
            raise ValueError("resume file contains an unknown latest episode ID")
        _validate_episode_record(model, episode_by_id[str(episode_id)], record)
        if not attempt_groups.get(str(episode_id)) or record != attempt_groups[str(episode_id)][-1]:
            raise ValueError("latest episode record is not the final append-only attempt")
        if any(
            attempt["status"] != "infrastructure_failure"
            for attempt in attempt_groups[str(episode_id)][:-1]
        ):
            raise ValueError("terminal episode status cannot have a later attempt")
    if any(
        attempt_groups[episode_id][-1]["status"] == "infrastructure_failure"
        for episode_id in attempted_ids[:-1]
    ):
        raise ValueError("execution cannot progress past an unresolved infrastructure failure")


def _new_document(
    model: str,
    schedule: Sequence[SeedEpisode],
    *,
    timeout: float,
) -> dict[str, Any]:
    config = run_config(model, schedule, timeout=timeout)
    now = utc_now()
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at": now,
        "updated_at": now,
        "config": config,
        "config_sha256": canonical_json_sha256(config),
        "pairing_sha256": pairing_sha256(config),
        "attempts": [],
        "episodes": [],
        "profile": None,
    }


def _load_document(
    path: Path,
    expected_config: Mapping[str, Any],
    schedule: Sequence[SeedEpisode],
    *,
    api_key: str | None = None,
) -> dict[str, Any]:
    raw_document = path.read_text(encoding="utf-8")
    if api_key and api_key in raw_document:
        raise ValueError("resume file contains the active API credential")
    document = json.loads(
        raw_document,
        object_pairs_hook=_reject_duplicate_object_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"resume JSON contains nonstandard constant {value}")
        ),
    )
    if api_key and _contains_secret(document, api_key):
        raise ValueError("resume file contains the active API credential")
    if not isinstance(document, dict) or document.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("resume file has the wrong result schema")
    if document.get("config") != expected_config:
        raise ValueError("resume file does not match this exact pilot configuration")
    if document.get("config_sha256") != canonical_json_sha256(expected_config):
        raise ValueError("resume file configuration digest is invalid")
    if document.get("pairing_sha256") != pairing_sha256(expected_config):
        raise ValueError("resume file pairing digest is invalid")
    _validate_document_records(document, str(expected_config["model"]), schedule)
    return document


def _dry_run_manifest(schedule: Sequence[SeedEpisode]) -> dict[str, Any]:
    return {
        "pilot_version": PILOT_VERSION,
        "episodes_per_model": len(schedule),
        "layout_ids": [layout.layout_id for layout in pilot_layouts()],
        "difficulty_counts": dict(Counter(item.seed.difficulty for item in schedule)),
        "direction_counts": dict(Counter(item.world.direction.value for item in schedule)),
        "unique_episode_ids": len({item.episode_id for item in schedule}),
        "unique_tape_keys": len({item.tape_key for item in schedule}),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    schedule = scheduled_episodes()
    if args.dry_run:
        print(json.dumps(_dry_run_manifest(schedule), ensure_ascii=False, indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required unless --dry-run is used")
    if not args.timeout > 0:
        parser.error("--timeout must be positive")
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        print(f"{API_KEY_ENV} is required", file=sys.stderr)
        return 2

    config = run_config(args.model, schedule, timeout=args.timeout)
    if args.output.exists():
        if not args.resume:
            print("output already exists; pass --resume to continue it", file=sys.stderr)
            return 2
        document = _load_document(args.output, config, schedule, api_key=api_key)
    else:
        document = _new_document(args.model, schedule, timeout=args.timeout)
        _write_json_atomic(args.output, document)

    existing = {record["episode_id"]: record for record in document["episodes"]}
    for index, episode in enumerate(schedule, start=1):
        prior = existing.get(episode.episode_id)
        if prior is not None and prior.get("status") in {"completed", "protocol_failure"}:
            continue
        started = time.monotonic()
        attempt_index = sum(
            attempt["episode_id"] == episode.episode_id for attempt in document["attempts"]
        )
        record = run_episode(
            api_key,
            args.model,
            episode,
            timeout=args.timeout,
            attempt_index=attempt_index,
        )
        record["elapsed_seconds"] = time.monotonic() - started
        if _contains_secret(record, api_key):
            raise RuntimeError("refusing to persist a record containing the active API credential")
        _validate_episode_record(args.model, episode, record)
        document["attempts"].append(record)
        existing[episode.episode_id] = record
        document["episodes"] = [
            existing[item.episode_id] for item in schedule if item.episode_id in existing
        ]
        document["updated_at"] = utc_now()
        document["profile"] = None
        _write_json_atomic(args.output, document)
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(schedule)}",
                    "model": args.model,
                    "difficulty": episode.seed.difficulty,
                    "layout_id": episode.task.layout.layout_id,
                    "status": record["status"],
                    "rounds_used": record["rounds_used"],
                    "samples_used": record["samples_used"],
                },
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        if record["status"] == "infrastructure_failure":
            print("infrastructure failure recorded; rerun with --resume", file=sys.stderr)
            return 3

    ordered_records = [existing[episode.episode_id] for episode in schedule]
    document["episodes"] = ordered_records
    document["profile"] = build_profile(args.model, schedule, ordered_records)
    document["updated_at"] = utc_now()
    _write_json_atomic(args.output, document)
    print(json.dumps(document["profile"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
