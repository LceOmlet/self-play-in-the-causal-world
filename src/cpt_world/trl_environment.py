"""Thin TRL ``environment_factory`` adapter for CPT-World episodes.

TRL owns rollout generation, tool dispatch, reward collection, vLLM weight
synchronization, and GRPO optimization. This module only binds those hooks to
the existing deterministic CPT-World semantic owners.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from typing import Any

from .query_truth import compute_query_truth
from .registry import TASK_FAMILY_QUERY_TYPES
from .world import OutcomeTape
from .world_runtime import WorldSpecEpisode
from .world_space import (
    WorldGrammar,
    WorldSpec,
    assemble_sampled_anchor_tasks,
    iter_sampled_seeds,
)

_TOOL_PROTOCOL_INSTRUCTION = """

TRAINING INTERFACE OVERRIDE:
Use the `act` tool for every experiment or terminal answer. Pass the exact JSON
object required above as the tool's `command` argument. Do not print that JSON
as ordinary assistant text. After `act` reports that the episode is complete,
make no more tool calls and finish with a brief acknowledgement.
""".rstrip()

COUNTERFACTUAL_ENDPOINT_TIME_LIMIT_SECONDS = 5.0
_COUNTERFACTUAL_QUERY_TYPE = "individual_counterfactual_probability"
_MAX_COUNTERFACTUAL_RESAMPLES = 10_000


def task_advantage_utility(
    terminal_quality: float,
    query_type: str,
) -> float:
    """Pass one owner-produced terminal quality unchanged to training."""

    if query_type not in TASK_FAMILY_QUERY_TYPES:
        raise ValueError(f"unsupported training query type: {query_type!r}")
    if not math.isfinite(terminal_quality) or not 0.0 <= terminal_quality <= 1.0:
        raise ValueError("terminal quality must be finite and lie in [0, 1]")
    return terminal_quality


def build_cpt_world_advantage_utility() -> Callable[..., list[float]]:
    """Build a TRL reward-function adapter over exact environment-owned rewards."""

    def cpt_world_advantage_utility(
        *,
        environments: Sequence[CPTWorldEnvironment],
        query_type: Sequence[str],
        log_metric: Callable[[str, float], None] | None = None,
        **_: object,
    ) -> list[float]:
        if len(environments) != len(query_type):
            raise ValueError("environment and query-type batches must have equal length")
        raw_rewards = [environment.get_reward() for environment in environments]
        utilities = [
            task_advantage_utility(raw, family)
            for raw, family in zip(raw_rewards, query_type, strict=True)
        ]
        if log_metric is not None:
            for family, raw, utility in zip(query_type, raw_rewards, utilities, strict=True):
                log_metric(f"task/{family}/reward_raw", raw)
                log_metric(f"task/{family}/reward_utility", utility)
            _log_terminal_effect_metrics(environments, query_type, log_metric)
        return utilities

    cpt_world_advantage_utility.__name__ = "CPTWorldAdvantageUtility"
    return cpt_world_advantage_utility


_EFFECT_METRIC_SPECS = {
    "ate": ("target_query", "squared_error", "ate"),
    "individual_counterfactual_probability": (
        "counterfactual_roi",
        "mean_squared_endpoint_error",
        "cf",
    ),
    "best_intervention": ("decision", "regret", "decision"),
}


def _terminal_score(environment: CPTWorldEnvironment) -> dict[str, Any] | None:
    episode = getattr(environment, "episode", None)
    score = getattr(episode, "terminal_score", None)
    if score is None:
        return None
    if not isinstance(score, dict):
        raise TypeError("terminal score owner returned a non-dict diagnostic")
    return score


def _log_terminal_effect_metrics(
    environments: Sequence[CPTWorldEnvironment],
    query_types: Sequence[str],
    log_metric: Callable[[str, float], None],
) -> None:
    """Log batch effect metrics from scorer-owned terminal diagnostics.

    Metrics are conditional on a valid terminal answer. Coverage and count are
    logged alongside them so an apparent improvement cannot be manufactured by
    silently dropping unfinished or invalid rollouts.
    """

    for family, (expected_kind, field, prefix) in _EFFECT_METRIC_SPECS.items():
        family_environments = [
            environment
            for environment, query_type in zip(environments, query_types, strict=True)
            if query_type == family
        ]
        if not family_environments:
            continue
        values: list[float] = []
        normalized_decision_values: list[float] = []
        for environment in family_environments:
            score = _terminal_score(environment)
            if score is None:
                continue
            if score.get("kind") != expected_kind:
                raise RuntimeError(
                    f"terminal diagnostic kind does not match {family}: {score.get('kind')!r}"
                )
            value = float(score[field])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"terminal diagnostic {field} must be finite and nonnegative")
            values.append(value)
            if family == "best_intervention":
                normalized_value = float(score["normalized_regret"])
                if not math.isfinite(normalized_value) or not 0.0 <= normalized_value <= 1.0:
                    raise ValueError(
                        "terminal diagnostic normalized_regret must be finite and lie in [0, 1]"
                    )
                normalized_decision_values.append(normalized_value)
        log_metric(f"effect/{prefix}_count", float(len(values)))
        log_metric(
            f"effect/{prefix}_coverage",
            len(values) / len(family_environments),
        )
        if not values:
            continue
        mean_value = sum(values) / len(values)
        if prefix == "ate":
            log_metric("effect/ate_mse", mean_value)
            log_metric("effect/ate_rmse", math.sqrt(mean_value))
        elif prefix == "cf":
            mean_squared_endpoint_error = sum(values) / len(values)
            log_metric("effect/cf_endpoint_mse", mean_squared_endpoint_error)
            log_metric("effect/cf_endpoint_rmse", math.sqrt(mean_squared_endpoint_error))
        else:
            log_metric("effect/decision_regret", mean_value)
            log_metric(
                "effect/decision_normalized_regret",
                sum(normalized_decision_values) / len(normalized_decision_values),
            )


def _training_row(
    grammar: WorldGrammar,
    sample_index: int,
    query_type: str,
) -> tuple[dict[str, Any], WorldSpec, dict[str, Any]]:
    (seed,) = iter_sampled_seeds(
        grammar,
        query_types=(query_type,),
        start_seed=sample_index,
        count=1,
    )
    anchor_index = int(str(seed["seed_id"]).rsplit("-a", 1)[1])
    ((world, regenerated_seed),) = assemble_sampled_anchor_tasks(
        grammar,
        sample_index,
        query_type,
        anchor_index,
    )
    if regenerated_seed != seed:
        raise RuntimeError("training row regeneration disagrees with the sampler owner")
    episode = WorldSpecEpisode(
        world,
        seed,
        OutcomeTape(f"prompt-only:{seed['seed_id']}"),
    )
    row = {
        "prompt": list(episode.initial_messages()),
        "sample_index": sample_index,
        "query_type": query_type,
        "anchor_index": anchor_index,
        "tape_key": f"trl-grpo:{seed['seed_id']}",
        "terminal_truth_json": "",
    }
    return row, world, dict(seed)


def build_balanced_training_rows(
    *,
    count_per_family: int,
    start_seed: int = 0,
    grammar: WorldGrammar | None = None,
) -> tuple[dict[str, Any], ...]:
    """Materialize exact-uniform task rows for TRL without duplicating sampler logic."""

    resolved_grammar = grammar or WorldGrammar()
    seeds = iter_sampled_seeds(
        resolved_grammar,
        query_types=TASK_FAMILY_QUERY_TYPES,
        start_seed=start_seed,
        count=count_per_family,
    )
    rows: list[dict[str, Any]] = []
    for offset in range(count_per_family):
        sample_index = start_seed + offset
        for family_offset, query_type in enumerate(TASK_FAMILY_QUERY_TYPES):
            seed = seeds[offset * len(TASK_FAMILY_QUERY_TYPES) + family_offset]
            row, _, regenerated_seed = _training_row(
                resolved_grammar,
                sample_index,
                query_type,
            )
            if regenerated_seed != seed:
                raise RuntimeError("balanced row disagrees with the sampler owner")
            rows.append(row)
    return tuple(rows)


def _iter_random_balanced_training_rows(
    *,
    start_seed: int = 0,
    grammar: WorldGrammar | None = None,
    counterfactual_endpoint_time_limit_seconds: float = (
        COUNTERFACTUAL_ENDPOINT_TIME_LIMIT_SECONDS
    ),
) -> Iterator[dict[str, Any]]:
    """Yield an infinite reproducible stream with exact-uniform task-family mixing.

    Each yielded row owns a fresh sampler seed. Counterfactual truth is solved
    before rollout and cached in the row. Exact and epsilon-sharp certificates
    are accepted; a larger unresolved endpoint gap rejects that candidate and
    advances to another sampled task.
    """

    if start_seed < 0:
        raise ValueError("start_seed must be nonnegative")
    if counterfactual_endpoint_time_limit_seconds <= 0:
        raise ValueError("counterfactual endpoint time limit must be positive")
    resolved_grammar = grammar or WorldGrammar()
    sample_index = start_seed
    while True:
        for query_type in TASK_FAMILY_QUERY_TYPES:
            attempts = 0
            while True:
                row, world, seed = _training_row(
                    resolved_grammar,
                    sample_index,
                    query_type,
                )
                sample_index += 1
                if query_type != _COUNTERFACTUAL_QUERY_TYPE:
                    break
                attempts += 1
                try:
                    truth = compute_query_truth(
                        world,
                        seed,
                        counterfactual_endpoint_time_limit_seconds=(
                            counterfactual_endpoint_time_limit_seconds
                        ),
                    )
                except RuntimeError as error:
                    if attempts >= _MAX_COUNTERFACTUAL_RESAMPLES:
                        raise RuntimeError(
                            "could not sample a counterfactual task with certified bounded truth"
                        ) from error
                    continue
                row["terminal_truth_json"] = json.dumps(
                    truth,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                break
            yield row


def iter_random_balanced_training_rows(
    *,
    start_seed: int = 0,
    grammar: WorldGrammar | None = None,
    counterfactual_endpoint_time_limit_seconds: float = (
        COUNTERFACTUAL_ENDPOINT_TIME_LIMIT_SECONDS
    ),
) -> Iterator[dict[str, Any]]:
    """Yield the deterministic training stream while preparing one row ahead.

    The single producer thread preserves the exact row order and prompt
    grouping expected by TRL.  It only overlaps CPU task construction and
    counterfactual certification with consumption of the preceding row.
    """

    source = _iter_random_balanced_training_rows(
        start_seed=start_seed,
        grammar=grammar,
        counterfactual_endpoint_time_limit_seconds=(counterfactual_endpoint_time_limit_seconds),
    )
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="cpt-world-row") as executor:
        pending = executor.submit(next, source)
        while True:
            row = pending.result()
            pending = executor.submit(next, source)
            yield row


class CPTWorldEnvironment:
    """Stateful environment instance reused by TRL for one rollout at a time."""

    def __init__(self, grammar: WorldGrammar | None = None) -> None:
        self._grammar = grammar or WorldGrammar()
        self._episode: WorldSpecEpisode | None = None

    def reset(
        self,
        *,
        prompt: object,
        sample_index: int,
        query_type: str,
        anchor_index: int,
        tape_key: str,
        terminal_truth_json: str = "",
        **_: object,
    ) -> str:
        """Reset the rollout to the task described by the repeated dataset row."""

        del prompt
        ((world, seed),) = assemble_sampled_anchor_tasks(
            self._grammar,
            sample_index,
            query_type,
            anchor_index,
        )
        terminal_truth = None
        if terminal_truth_json:
            decoded_truth = json.loads(terminal_truth_json)
            if not isinstance(decoded_truth, dict):
                raise ValueError("terminal_truth_json must encode an object")
            terminal_truth = decoded_truth
        self._episode = WorldSpecEpisode(
            world,
            seed,
            OutcomeTape(tape_key),
            terminal_truth=terminal_truth,
        )
        return _TOOL_PROTOCOL_INSTRUCTION

    def act(self, command: dict[str, object]) -> str:
        """Execute one CPT-World command.

        Args:
            command: One legal experiment or terminal-answer JSON object from
                the task protocol.

        Returns:
            Exact owner-rendered feedback, a zero-cost protocol error, or a
            terminal acknowledgement.
        """

        if self._episode is None:
            raise RuntimeError("environment must be reset before act")
        if self._episode.completed:
            return "Episode already complete. Make no more tool calls."
        try:
            raw = json.dumps(command, separators=(",", ":"), ensure_ascii=False)
            step = self._episode.step(raw)
        except (TypeError, ValueError) as error:
            return self._episode.render_protocol_error(error)
        if step.kind == "batch":
            if step.message is None:
                raise RuntimeError("batch step is missing owner-rendered feedback")
            return step.message
        if step.kind != "terminal" or step.reward is None:
            raise RuntimeError("episode returned an unknown step kind")
        return "Terminal answer accepted. Episode complete; make no more tool calls."

    def get_reward(self) -> float:
        """Return terminal quality, with unfinished rollouts receiving exactly zero."""

        if self._episode is None or self._episode.terminal_reward is None:
            return 0.0
        reward = self._episode.terminal_reward
        if not isinstance(reward, Fraction):
            raise TypeError("terminal reward owner returned a non-Fraction value")
        return float(reward)

    @property
    def episode(self) -> WorldSpecEpisode | None:
        """Expose current state for adapter tests and diagnostics, not as a TRL tool."""

        return self._episode
