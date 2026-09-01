"""Run a reproducible five-task DoLens profile through Ponderera.

The credential is read only from ``PONDERERA_API_KEY``. It is never printed or
persisted. World/task semantics remain owned by ``cpt_world``; this file only
freezes an evaluation schedule, drives the public episode API, and records the
provider's token accounting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from cpt_world import (
    DEFAULT_NODE_COUNTS,
    OBSERVATION_BUDGET_EXPONENTS,
    OBSERVATIONS_PER_BANDWIDTH_UNIT,
    TERMINAL_QUALITY_REWARD_VERSION,
    Budget,
    OutcomeTape,
    WorldGrammar,
    WorldSpec,
    WorldSpecEpisode,
    assemble_sampled_anchor_tasks,
    budget_for_observation_bandwidth,
    compute_query_truth,
    iter_sampled_seeds,
    legal_query_anchors,
    sample_world,
)

API_URL = "https://api.ponderera.com/v1/chat/completions"
MODEL = "deepseek-v4-flash"
RESULT_SCHEMA = "dolens-ponderera-profile-v7"
SCHEDULE_SCHEMA = "dolens-evaluation-schedule-v6"
MASTER_SEED = 2026082201
QUERY_TYPES = (
    "ate",
    "individual_counterfactual_probability",
    "best_intervention",
    "backadj_minimal_sets",
    "mediator_set",
)
DEFAULT_REPEATS = 30
DEFAULT_TIMEOUT_SECONDS = 180.0


def _budget_for_seed(seed: Mapping[str, Any]) -> Budget:
    return budget_for_observation_bandwidth(
        int(seed["observation_bandwidth"]),
        exponent=int(seed.get("observation_budget_exponent", 11)),
    )


def _budget_for_entry(entry: Mapping[str, Any]) -> Budget:
    return budget_for_observation_bandwidth(
        int(entry["observation_bandwidth"]),
        exponent=int(entry.get("observation_budget_exponent", 11)),
    )


def _budget_contract() -> Mapping[str, Any]:
    return {
        "kind": "scalar_observation_budget",
        "observations_per_bandwidth_unit": {
            "kind": "power_of_two",
            "exponents": list(OBSERVATION_BUDGET_EXPONENTS),
            "minimum": OBSERVATIONS_PER_BANDWIDTH_UNIT,
        },
        "query_cost": "batch_size*measure_width",
        "batch_size_domain": "any_positive_integer_fitting_remaining_budget",
        "query_count_limit": None,
    }


def _fraction_text(value: Fraction) -> str:
    """Serialize trusted generated exact rationals beyond Python's display guard."""

    previous_limit = sys.get_int_max_str_digits()
    if previous_limit:
        sys.set_int_max_str_digits(0)
    try:
        return str(value)
    finally:
        if previous_limit:
            sys.set_int_max_str_digits(previous_limit)


def _fraction_from_text(value: str) -> Fraction:
    """Parse a trusted generated exact rational beyond Python's input guard."""

    previous_limit = sys.get_int_max_str_digits()
    if previous_limit:
        sys.set_int_max_str_digits(0)
    try:
        return Fraction(value)
    finally:
        if previous_limit:
            sys.set_int_max_str_digits(previous_limit)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Fraction):
        return _fraction_text(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest().upper()


def _world_payload(world: WorldSpec) -> Mapping[str, Any]:
    return {
        "family": world.family,
        "topology": world.topology,
        "variables": world.variables,
        "domains": world.domains,
        "state_names": world.state_names,
        "edges": world.edges,
        "parents": world.parents,
        "cpt": world.cpt,
    }


def _model_seed(episode_id: str) -> int:
    digest = hashlib.sha256(f"dolens-model-seed-v1\0{episode_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _tape_key(episode_id: str) -> str:
    digest = hashlib.sha256(f"dolens-outcome-tape-v1\0{episode_id}".encode()).hexdigest()
    return f"dolens-profile:{digest}"


def _materialize(entry: Mapping[str, Any]) -> tuple[WorldSpec, Mapping[str, Any]]:
    grammar = WorldGrammar(
        node_counts=(int(entry["node_count"]),),
        max_domain_size=int(entry["max_domain_size"]),
    )
    tasks = assemble_sampled_anchor_tasks(
        grammar,
        int(entry["world_seed"]),
        str(entry["query_type"]),
        int(entry["anchor_index"]),
    )
    if len(tasks) != 1:
        raise ValueError("schedule entry did not resolve to exactly one task")
    return tasks[0]


def _frozen_schedule_entry(
    *,
    query_type: str,
    repeat: int,
    node_count: int,
    max_domain_size: int,
    world_seed: int,
    anchor_index: int,
    world: WorldSpec,
    seed: Mapping[str, Any],
) -> Mapping[str, Any]:
    episode_id = f"{query_type}:{repeat:02d}"
    episode = WorldSpecEpisode(
        world,
        seed,
        OutcomeTape(_tape_key(episode_id)),
        budget=_budget_for_seed(seed),
    )
    initial_messages = episode.initial_messages()
    return {
        "episode_id": episode_id,
        "query_type": query_type,
        "repeat": repeat,
        "node_count": node_count,
        "max_domain_size": max_domain_size,
        "world_seed": world_seed,
        "anchor_index": anchor_index,
        "seed_id": seed["seed_id"],
        "world_sha256": _sha256(_world_payload(world)),
        "seed_sha256": _sha256(seed),
        "initial_messages_sha256": _sha256(initial_messages),
        "tape_key": _tape_key(episode_id),
        "model_seed": _model_seed(episode_id),
        "edge_count": len(world.edges),
        "manipulable_width": sum(bool(value) for value in seed["manipulability"].values()),
        "observation_bandwidth": seed["observation_bandwidth"],
        "observation_budget_exponent": seed["observation_budget_exponent"],
    }


def _schedule_entry(
    query_type: str,
    repeat: int,
    rng: random.Random,
    *,
    node_counts: tuple[int, ...],
    max_domain_size: int,
    progress: bool = False,
) -> Mapping[str, Any]:
    if query_type == "best_intervention":
        node_count = rng.choice(node_counts)
        raw_slot = rng.randrange(0, 2**31)
        balanced_slot = raw_slot - raw_slot % 5 + repeat % 5
        grammar = WorldGrammar(
            node_counts=(node_count,),
            max_domain_size=max_domain_size,
        )
        (seed,) = iter_sampled_seeds(
            grammar,
            query_types=(query_type,),
            start_seed=balanced_slot,
            count=1,
        )
        seed_id = str(seed["seed_id"])
        world_seed = int(seed_id.split("-", 2)[1])
        anchor_index = int(seed_id.rsplit("-a", 1)[1])
        ((world, regenerated_seed),) = assemble_sampled_anchor_tasks(
            grammar,
            world_seed,
            query_type,
            anchor_index,
        )
        if regenerated_seed != seed:
            raise RuntimeError("profile task regeneration disagrees with the sampler owner")
        return _frozen_schedule_entry(
            query_type=query_type,
            repeat=repeat,
            node_count=node_count,
            max_domain_size=max_domain_size,
            world_seed=world_seed,
            anchor_index=anchor_index,
            world=world,
            seed=seed,
        )

    for candidate_index in range(10_000):
        node_count = rng.choice(node_counts)
        world_seed = rng.randrange(0, 2**31)
        grammar = WorldGrammar(
            node_counts=(node_count,),
            max_domain_size=max_domain_size,
        )
        structural = sample_world(grammar, world_seed)
        anchors = legal_query_anchors(structural, query_type)
        if not anchors:
            continue
        anchor_indices = list(range(len(anchors)))
        rng.shuffle(anchor_indices)
        for anchor_index in anchor_indices:
            candidate_started = time.monotonic()
            if progress and repeat >= 28:
                print(
                    json.dumps(
                        {
                            "event": "schedule_candidate_start",
                            "query_type": query_type,
                            "repeat": repeat,
                            "candidate_index": candidate_index,
                            "node_count": node_count,
                            "world_seed": world_seed,
                            "anchor_index": anchor_index,
                        }
                    ),
                    flush=True,
                )
            try:
                tasks = assemble_sampled_anchor_tasks(
                    grammar,
                    world_seed,
                    query_type,
                    anchor_index,
                )
            except (ValueError, NotImplementedError):
                continue
            if progress and repeat >= 28:
                print(
                    json.dumps(
                        {
                            "event": "schedule_candidate_complete",
                            "query_type": query_type,
                            "repeat": repeat,
                            "candidate_index": candidate_index,
                            "node_count": node_count,
                            "world_seed": world_seed,
                            "anchor_index": anchor_index,
                            "elapsed_seconds": time.monotonic() - candidate_started,
                        }
                    ),
                    flush=True,
                )
            if len(tasks) != 1:
                continue
            world, seed = tasks[0]
            return _frozen_schedule_entry(
                query_type=query_type,
                repeat=repeat,
                node_count=node_count,
                max_domain_size=max_domain_size,
                world_seed=world_seed,
                anchor_index=anchor_index,
                world=world,
                seed=seed,
            )
    raise RuntimeError(f"could not build a legal {query_type} task")


def build_schedule(
    *,
    master_seed: int = MASTER_SEED,
    repeats: int = DEFAULT_REPEATS,
    node_counts: tuple[int, ...] = DEFAULT_NODE_COUNTS,
    max_domain_size: int = 2,
    progress: bool = False,
) -> Mapping[str, Any]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if not node_counts:
        raise ValueError("node_counts must not be empty")
    rng = random.Random(master_seed)
    entries: list[Mapping[str, Any]] = []
    for repeat in range(repeats):
        for query_type in QUERY_TYPES:
            started = time.monotonic()
            if progress:
                print(
                    json.dumps(
                        {
                            "event": "schedule_entry_start",
                            "index": len(entries) + 1,
                            "total": repeats * len(QUERY_TYPES),
                            "query_type": query_type,
                            "repeat": repeat,
                        }
                    ),
                    flush=True,
                )
            entry = _schedule_entry(
                query_type,
                repeat,
                rng,
                node_counts=node_counts,
                max_domain_size=max_domain_size,
                progress=progress,
            )
            entries.append(entry)
            if progress:
                print(
                    json.dumps(
                        {
                            "event": "schedule_entry_complete",
                            "index": len(entries),
                            "query_type": query_type,
                            "repeat": repeat,
                            "node_count": entry["node_count"],
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
    return {
        "schema": SCHEDULE_SCHEMA,
        "master_seed": master_seed,
        "repeats_per_query": repeats,
        "query_types": QUERY_TYPES,
        "node_count_distribution": {
            "kind": "discrete_uniform",
            "support": node_counts,
        },
        "max_domain_size": max_domain_size,
        "budget": _budget_contract(),
        "entries": entries,
    }


def _probe_command(seed: Mapping[str, Any], world: WorldSpec) -> str:
    labels = seed["visible_schema"]["variable_labels"]
    for target_name in world.variables:
        if not seed["manipulability"][target_name]:
            continue
        measure_name = next(
            (name for name in world.variables if name != target_name and seed["readable"][name]),
            None,
        )
        if measure_name is None:
            continue
        return json.dumps(
            {
                "type": "intervene",
                "target": labels[target_name],
                "value": "state_0",
                "measure": [labels[measure_name]],
                "batch_size": 1,
            },
            separators=(",", ":"),
        )
    raise ValueError("task has no deterministic replay probe")


def verify_schedule_replay(
    schedule: Mapping[str, Any],
    *,
    duplicate_materialization: bool = True,
    progress: bool = False,
) -> Mapping[str, Any]:
    verified: list[Mapping[str, Any]] = []
    for index, entry in enumerate(schedule["entries"], start=1):
        started = time.monotonic()
        if progress:
            print(
                json.dumps(
                    {
                        "event": "replay_entry_start",
                        "index": index,
                        "total": len(schedule["entries"]),
                        "episode_id": entry["episode_id"],
                        "node_count": entry["node_count"],
                    }
                ),
                flush=True,
            )
        first_world, first_seed = _materialize(entry)
        second_world, second_seed = (
            _materialize(entry) if duplicate_materialization else (first_world, first_seed)
        )
        if _sha256(_world_payload(first_world)) != entry["world_sha256"]:
            raise ValueError("world fingerprint drift")
        if _sha256(first_seed) != entry["seed_sha256"]:
            raise ValueError("task-seed fingerprint drift")
        if first_world != second_world or first_seed != second_seed:
            raise ValueError("task materialization is not reproducible")
        first = WorldSpecEpisode(
            first_world,
            first_seed,
            OutcomeTape(str(entry["tape_key"])),
            budget=_budget_for_entry(entry),
        )
        second = WorldSpecEpisode(
            second_world,
            second_seed,
            OutcomeTape(str(entry["tape_key"])),
            budget=_budget_for_entry(entry),
        )
        if first.initial_messages() != second.initial_messages():
            raise ValueError("initial messages are not reproducible")
        if _sha256(first.initial_messages()) != entry["initial_messages_sha256"]:
            raise ValueError("initial-message fingerprint drift")
        command = _probe_command(first_seed, first_world)
        first_step = first.step(command)
        second_step = second.step(command)
        if first_step != second_step:
            raise ValueError("action-keyed batch replay drift")
        verified.append(
            {
                "episode_id": entry["episode_id"],
                "probe_command_sha256": _sha256(command),
                "feedback_sha256": _sha256(first_step.message),
            }
        )
        if progress:
            print(
                json.dumps(
                    {
                        "event": "replay_entry_complete",
                        "index": index,
                        "episode_id": entry["episode_id"],
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    return {
        "schedule_sha256": _sha256(schedule),
        "verified_entries": len(verified),
        "replay_sha256": _sha256(verified),
    }


def _clean_usage(value: Any) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            result[key] = item
    return result


def chat_completion(
    api_key: str,
    messages: Sequence[Mapping[str, str]],
    *,
    model_seed: int,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    opener: Any = urllib.request.urlopen,
) -> tuple[str, Mapping[str, Any]]:
    payload = {
        "model": MODEL,
        "messages": list(messages),
        "temperature": 0,
        "enable_thinking": False,
        "stream": False,
        "seed": model_seed,
    }
    request = urllib.request.Request(
        API_URL,
        data=_canonical_bytes(payload),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with opener(request, timeout=timeout_seconds) as response:
        http_status = getattr(response, "status", None)
        parsed = json.loads(response.read().decode("utf-8"))
    if not isinstance(parsed, Mapping):
        raise ValueError("provider response must be an object")
    choices = parsed.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("provider response must contain one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping) or not isinstance(choice.get("message"), Mapping):
        raise ValueError("provider choice is missing message")
    content = choice["message"].get("content")
    if not isinstance(content, str):
        raise ValueError("provider message content must be text")

    def optional_scalar(key: str) -> str | int | None:
        value = parsed.get(key)
        return value if isinstance(value, (str, int)) and not isinstance(value, bool) else None

    usage = _clean_usage(parsed.get("usage"))
    if "prompt_tokens" not in usage or "completion_tokens" not in usage:
        raise ValueError("provider response is missing input/output token accounting")
    return content, {
        "response_id": optional_scalar("id"),
        "created": optional_scalar("created"),
        "response_model": optional_scalar("model"),
        "system_fingerprint": optional_scalar("system_fingerprint"),
        "finish_reason": (
            choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None
        ),
        "http_status": (
            http_status
            if isinstance(http_status, int) and not isinstance(http_status, bool)
            else None
        ),
        "usage": usage,
    }


def _sum_usage(turns: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    totals: Counter[str] = Counter()
    for turn in turns:
        provider = turn.get("provider")
        if not isinstance(provider, Mapping):
            continue
        usage = provider.get("usage")
        if isinstance(usage, Mapping):
            for key, value in usage.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    totals[str(key)] += value
    return dict(totals)


def run_episode(
    entry: Mapping[str, Any],
    api_key: str,
    *,
    timeout_seconds: float,
) -> Mapping[str, Any]:
    world, seed = _materialize(entry)
    episode = WorldSpecEpisode(
        world,
        seed,
        OutcomeTape(str(entry["tape_key"])),
        budget=_budget_for_entry(entry),
    )
    messages: list[dict[str, str]] = [dict(message) for message in episode.initial_messages()]
    turns: list[Mapping[str, Any]] = []
    protocol_errors = 0
    started = time.monotonic()
    turn_index = 0
    while True:
        try:
            raw, provider = chat_completion(
                api_key,
                messages,
                model_seed=int(entry["model_seed"]),
                timeout_seconds=timeout_seconds,
            )
        except (OSError, UnicodeError, ValueError) as error:
            status = getattr(error, "code", None)
            return {
                "episode_id": entry["episode_id"],
                "status": "infrastructure_failure",
                "failure_code": type(error).__name__,
                "http_status": status if isinstance(status, int) else None,
                "turns": turns,
                "usage": _sum_usage(turns),
                "protocol_errors": protocol_errors,
                "elapsed_seconds": time.monotonic() - started,
            }
        messages.append({"role": "assistant", "content": raw})
        try:
            step = episode.step(raw)
        except (TypeError, ValueError, NotImplementedError) as error:
            protocol_errors += 1
            feedback = episode.render_protocol_error(error)
            turns.append(
                {
                    "turn": turn_index,
                    "assistant": raw,
                    "provider": provider,
                    "outcome": "protocol_error",
                    "failure_code": type(error).__name__,
                    "failure_detail": str(error),
                    "feedback": feedback,
                }
            )
            messages.append({"role": "user", "content": feedback})
            turn_index += 1
            continue
        turn: dict[str, Any] = {
            "turn": turn_index,
            "assistant": raw,
            "provider": provider,
            "outcome": step.kind,
        }
        if step.kind == "terminal":
            turns.append(turn)
            return {
                "episode_id": entry["episode_id"],
                "status": "completed",
                "turns": turns,
                "usage": _sum_usage(turns),
                "protocol_errors": protocol_errors,
                "queries_used": episode.queries_used,
                "sample_rows_used": episode.sample_rows_used,
                "observations_used": episode.observations_used,
                "remaining_budget": episode.remaining_budget,
                "truth": _jsonable(compute_query_truth(world, seed)),
                "score": _jsonable(step.score),
                "elapsed_seconds": time.monotonic() - started,
            }
        if step.message is None:
            raise RuntimeError("batch step is missing feedback")
        turn["feedback"] = step.message
        turns.append(turn)
        messages.append({"role": "user", "content": step.message})
        turn_index += 1


def _config(master_seed: int, repeats: int, timeout_seconds: float) -> Mapping[str, Any]:
    return {
        "model": MODEL,
        "endpoint": API_URL,
        "temperature": 0,
        "enable_thinking": False,
        "stream": False,
        "model_seed_field": "per_episode_recorded",
        "master_seed": master_seed,
        "repeats_per_query": repeats,
        "node_counts": DEFAULT_NODE_COUNTS,
        "node_count_distribution": "discrete_uniform",
        "max_domain_size": 2,
        "terminal_quality_reward": TERMINAL_QUALITY_REWARD_VERSION,
        "budget": _budget_contract(),
        "timeout_seconds": timeout_seconds,
        "automatic_output_repairs": 0,
        "transport_retries": 0,
    }


def _write_atomic(path: Path, value: Mapping[str, Any], api_key: str) -> None:
    encoded = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    if api_key and api_key in encoded:
        raise RuntimeError("credential appeared in the result artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_or_initialize(
    path: Path,
    config: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> dict[str, Any]:
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("result artifact must be an object")
        if value.get("config") != _jsonable(config):
            raise ValueError("result config does not match this run")
        if value.get("schedule_sha256") != _sha256(schedule):
            raise ValueError("result schedule does not match this run")
        return value
    return {
        "schema": RESULT_SCHEMA,
        "config": _jsonable(config),
        "config_sha256": _sha256(config),
        "schedule": _jsonable(schedule),
        "schedule_sha256": _sha256(schedule),
        "attempts": [],
        "episodes": [],
        "summary": None,
    }


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("boolean is not numeric")
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        result = float(_fraction_from_text(value))
    else:
        raise TypeError("value is not numeric")
    if not math.isfinite(result):
        raise ValueError("value is not finite")
    return result


def summarize(document: Mapping[str, Any]) -> Mapping[str, Any]:
    entries = {entry["episode_id"]: entry for entry in document["schedule"]["entries"]}
    episodes = document.get("episodes", [])
    status_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    tokens_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    node_counts: Counter[int] = Counter()
    for episode in episodes:
        entry = entries[episode["episode_id"]]
        query_type = str(entry["query_type"])
        status_by_task[query_type][str(episode["status"])] += 1
        node_counts[int(entry["node_count"])] += 1
        for key, value in episode.get("usage", {}).items():
            if isinstance(value, int):
                tokens_by_task[query_type][key] += value
        score = episode.get("score")
        if not isinstance(score, Mapping):
            continue
        kind = score.get("kind")
        if kind in {"effect", "target_query"}:
            metrics[query_type]["l1_error"].append(_as_float(score["l1_error"]))
            metrics[query_type]["total_variation_error"].append(
                _as_float(score["total_variation_error"])
            )
            metrics[query_type]["squared_error"].append(_as_float(score["squared_error"]))
        elif kind == "counterfactual_roi":
            metrics[query_type]["mean_absolute_endpoint_error"].append(
                _as_float(score["mean_absolute_endpoint_error"])
            )
            metrics[query_type]["mean_squared_endpoint_error"].append(
                _as_float(score["mean_squared_endpoint_error"])
            )
        elif kind == "decision":
            metrics[query_type]["regret"].append(_as_float(score["regret"]))
            metrics[query_type]["normalized_regret"].append(_as_float(score["normalized_regret"]))
        elif kind == "backadj":
            metrics[query_type]["f1"].append(_as_float(score["f1"]))
            metrics[query_type]["exact_match"].append(1.0 if score["exact_match"] else 0.0)
        elif kind == "mediator":
            metrics[query_type]["mediator_f1"].append(_as_float(score["mediator_f1"]))
            metrics[query_type]["order_f1"].append(_as_float(score["order_f1"]))
            metrics[query_type]["exact_match"].append(
                1.0 if score["mediators_exact_match"] and score["order_exact_match"] else 0.0
            )
    metric_means = {
        task: {
            name: {"n": len(values), "mean": sum(values) / len(values)}
            for name, values in names.items()
            if values
        }
        for task, names in metrics.items()
    }
    total_tokens = sum((Counter(counts) for counts in tokens_by_task.values()), Counter())
    return {
        "scheduled": len(document["schedule"]["entries"]),
        "recorded": len(episodes),
        "status_by_task": {task: dict(counts) for task, counts in status_by_task.items()},
        "tokens_by_task": {task: dict(counts) for task, counts in tokens_by_task.items()},
        "tokens_total": dict(total_tokens),
        "node_count_histogram": {str(node): count for node, count in sorted(node_counts.items())},
        "metric_means": metric_means,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-new-episodes", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/pilots/ponderera-deepseek-v4-flash-3to15-v1.json"),
    )
    args = parser.parse_args()

    config = _config(args.master_seed, args.repeats, args.timeout)
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if not isinstance(existing, Mapping) or existing.get("config") != _jsonable(config):
            raise ValueError("existing result config does not match this run")
        schedule = existing.get("schedule")
        if not isinstance(schedule, Mapping):
            raise ValueError("existing result is missing its frozen schedule")
        replay = verify_schedule_replay(
            schedule,
            duplicate_materialization=False,
            progress=True,
        )
    else:
        schedule = build_schedule(
            master_seed=args.master_seed,
            repeats=args.repeats,
            progress=True,
        )
        replay = verify_schedule_replay(schedule, progress=True)
    print(
        json.dumps(
            {
                "event": "environment_replay_verified",
                **replay,
                "node_histogram": dict(
                    sorted(Counter(entry["node_count"] for entry in schedule["entries"]).items())
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.prepare_only:
        return 0

    api_key = os.environ.get("PONDERERA_API_KEY", "").strip()
    if not api_key:
        print("PONDERERA_API_KEY is required", file=sys.stderr)
        return 2
    document = _load_or_initialize(args.output, config, schedule)
    completed_ids = {episode["episode_id"] for episode in document["episodes"]}
    new_episodes = 0
    for index, entry in enumerate(schedule["entries"], start=1):
        episode_id = entry["episode_id"]
        if episode_id in completed_ids:
            continue
        result = run_episode(entry, api_key, timeout_seconds=args.timeout)
        attempt = {
            **result,
            "attempt_index": sum(
                1 for item in document["attempts"] if item["episode_id"] == episode_id
            ),
        }
        document["attempts"].append(attempt)
        if result["status"] == "infrastructure_failure":
            document["summary"] = summarize(document)
            _write_atomic(args.output, document, api_key)
            print(
                json.dumps(
                    {
                        "event": "infrastructure_failure",
                        "episode_id": episode_id,
                        "failure_code": result["failure_code"],
                        "http_status": result["http_status"],
                    }
                ),
                flush=True,
            )
            return 3
        document["episodes"].append(result)
        completed_ids.add(episode_id)
        document["summary"] = summarize(document)
        _write_atomic(args.output, document, api_key)
        new_episodes += 1
        print(
            json.dumps(
                {
                    "event": "episode_complete",
                    "index": index,
                    "total": len(schedule["entries"]),
                    "episode_id": episode_id,
                    "status": result["status"],
                    "node_count": entry["node_count"],
                    "queries_used": result.get("queries_used"),
                    "sample_rows_used": result.get("sample_rows_used"),
                    "observations_used": result.get("observations_used"),
                    "usage": result.get("usage"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if args.max_new_episodes is not None and new_episodes >= args.max_new_episodes:
            break
    document["summary"] = summarize(document)
    _write_atomic(args.output, document, api_key)
    print(json.dumps({"event": "summary", **document["summary"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
