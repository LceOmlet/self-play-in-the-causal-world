"""Thin TRL ``environment_factory`` adapter for CPT-World episodes.

TRL owns rollout generation, tool dispatch, reward collection, vLLM weight
synchronization, and GRPO optimization. This module only binds those hooks to
the existing deterministic CPT-World semantic owners.
"""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any

from .registry import TASK_FAMILY_QUERY_TYPES
from .world import OutcomeTape
from .world_runtime import WorldSpecEpisode
from .world_space import (
    WorldGrammar,
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
            anchor_index = int(str(seed["seed_id"]).rsplit("-a", 1)[1])
            ((world, regenerated_seed),) = assemble_sampled_anchor_tasks(
                resolved_grammar,
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
            rows.append(
                {
                    "prompt": list(episode.initial_messages()),
                    "sample_index": sample_index,
                    "query_type": query_type,
                    "anchor_index": anchor_index,
                    "tape_key": f"trl-grpo:{seed['seed_id']}",
                }
            )
    return tuple(rows)


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
        self._episode = WorldSpecEpisode(world, seed, OutcomeTape(tape_key))
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
