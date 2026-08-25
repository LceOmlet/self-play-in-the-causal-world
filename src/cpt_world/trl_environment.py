"""Thin TRL ``environment_factory`` adapter for CPT-World episodes.

TRL owns rollout generation, tool dispatch, reward collection, vLLM weight
synchronization, and GRPO optimization. This module only binds those hooks to
the existing deterministic CPT-World semantic owners.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
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


def iter_random_balanced_training_rows(
    *,
    start_seed: int = 0,
    grammar: WorldGrammar | None = None,
    counterfactual_endpoint_time_limit_seconds: float = (
        COUNTERFACTUAL_ENDPOINT_TIME_LIMIT_SECONDS
    ),
) -> Iterator[dict[str, Any]]:
    """Yield an infinite reproducible stream with exact-uniform task-family mixing.

    Each yielded row owns a fresh sampler seed. Counterfactual truth is solved
    before rollout and cached in the row. A fail-closed endpoint timeout rejects
    that candidate and advances to another sampled task, so no unresolved truth
    is converted into a training reward.
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
                            "could not sample a counterfactual task with exact bounded truth"
                        ) from error
                    continue
                row["terminal_truth_json"] = json.dumps(
                    truth,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                break
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
