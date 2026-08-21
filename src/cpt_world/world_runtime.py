"""Executable multi-valued interaction runtime for generic ``WorldSpec`` tasks.

The module composes existing semantic owners:

- ``query_truth.sample_worldspec_assignment`` owns scalable hard-do sampling;
- ``world.OutcomeTape`` owns action-keyed reproducible random draws;
- ``episode.Budget`` owns the public intervention budget;
- ``rendering`` and ``task_scoring`` own initial prompts and terminal answers.

This module only owns command validation, episode state, selected-measure
projection, and batch feedback for the generic world representation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .episode import Budget
from .query_truth import sample_worldspec_assignment
from .rendering import render_seed_initial_messages, resolve_observation_bandwidth
from .task_scoring import score_terminal_answer
from .world import OutcomeTape
from .world_space import WorldSpec


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def _json_object(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise TypeError("command must be a string")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("response is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    return value


@dataclass(frozen=True, slots=True)
class _RuntimeView:
    labels: Mapping[str, str]
    visible_to_internal: Mapping[str, str]
    manipulability: Mapping[str, bool]
    readable: Mapping[str, bool]


def _runtime_view(seed: Mapping[str, Any], world: WorldSpec) -> _RuntimeView:
    if not isinstance(seed, Mapping):
        raise TypeError("seed must be a mapping")
    if not isinstance(world, WorldSpec):
        raise TypeError("world must be a WorldSpec")
    visible_schema = seed.get("visible_schema")
    if not isinstance(visible_schema, Mapping):
        raise ValueError("seed is missing visible_schema")
    labels_value = visible_schema.get("variable_labels")
    variables_value = visible_schema.get("variables")
    manipulability_value = seed.get("manipulability")
    readable_value = seed.get("readable")
    if not isinstance(labels_value, Mapping):
        raise ValueError("visible_schema.variable_labels must be a mapping")
    if not isinstance(variables_value, (tuple, list)):
        raise ValueError("visible_schema.variables must be a sequence")
    if not isinstance(manipulability_value, Mapping) or not isinstance(readable_value, Mapping):
        raise ValueError("seed must contain manipulability and readable mappings")

    labels = {str(name): str(label) for name, label in labels_value.items()}
    manipulability = dict(manipulability_value)
    readable = dict(readable_value)
    expected = set(world.variables)
    if set(labels) != expected or set(manipulability) != expected or set(readable) != expected:
        raise ValueError("seed variable masks do not match the WorldSpec")
    if len(set(labels.values())) != len(labels):
        raise ValueError("visible variable labels must be unique")
    if any(not isinstance(value, bool) for value in manipulability.values()):
        raise ValueError("manipulability values must be bool")
    if any(not isinstance(value, bool) for value in readable.values()):
        raise ValueError("readable values must be bool")

    visible_states: dict[str, tuple[str, ...]] = {}
    for item in variables_value:
        if not isinstance(item, Mapping):
            raise ValueError("visible variables must be mappings")
        label = item.get("label")
        states = item.get("states")
        if not isinstance(label, str) or not isinstance(states, (tuple, list)):
            raise ValueError("visible variable must contain a label and states")
        if label in visible_states:
            raise ValueError("visible variable labels must be unique")
        visible_states[label] = tuple(str(state) for state in states)
    if set(visible_states) != set(labels.values()):
        raise ValueError("visible variable list does not match variable_labels")
    for node, name in enumerate(world.variables):
        expected_states = tuple(f"state_{state}" for state in range(world.domains[node]))
        if visible_states[labels[name]] != expected_states:
            raise ValueError("visible states do not match the WorldSpec domain")

    return _RuntimeView(
        labels=labels,
        visible_to_internal={label: name for name, label in labels.items()},
        manipulability=manipulability,
        readable=readable,
    )


def _state_index(value: object) -> int:
    if not isinstance(value, str) or not value.startswith("state_"):
        raise ValueError("intervention value must be a state_i string")
    suffix = value.removeprefix("state_")
    if not suffix.isascii() or not suffix.isdigit() or str(int(suffix)) != suffix:
        raise ValueError("intervention value must be a state_i string")
    return int(suffix)


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class WorldIntervention:
    target: int
    value: int

    def __post_init__(self) -> None:
        _nonnegative_int(self.target, field="intervention target")
        _nonnegative_int(self.value, field="intervention value")


@dataclass(frozen=True, slots=True)
class WorldInterventionCommand:
    intervention: WorldIntervention
    measure: tuple[int, ...]
    batch_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.intervention, WorldIntervention):
            raise TypeError("intervention must be a WorldIntervention")
        if not self.measure:
            raise ValueError("measure must not be empty")
        for node in self.measure:
            _nonnegative_int(node, field="measure node")
        if len(set(self.measure)) != len(self.measure):
            raise ValueError("measure variables must not contain duplicates")
        if self.intervention.target in self.measure:
            raise ValueError("the intervention target must not appear in measure")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError("batch_size must be an integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True, slots=True)
class WorldObservationCommand:
    measure: tuple[int, ...]
    batch_size: int

    def __post_init__(self) -> None:
        if not self.measure:
            raise ValueError("measure must not be empty")
        for node in self.measure:
            _nonnegative_int(node, field="measure node")
        if len(set(self.measure)) != len(self.measure):
            raise ValueError("measure variables must not contain duplicates")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError("batch_size must be an integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True, slots=True)
class WorldMeasuredBatch:
    intervention: WorldIntervention | None
    measure: tuple[int, ...]
    measure_domains: tuple[int, ...]
    start_index: int
    assignments: tuple[tuple[int, ...], ...]
    counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.intervention is not None and not isinstance(self.intervention, WorldIntervention):
            raise TypeError("intervention must be a WorldIntervention or None")
        if not self.measure or len(self.measure) != len(self.measure_domains):
            raise ValueError("measure and measure_domains must be nonempty and aligned")
        if any(domain <= 0 for domain in self.measure_domains):
            raise ValueError("measure domains must be positive")
        if len(self.assignments) != len(self.counts) or not self.assignments:
            raise ValueError("sparse assignments and counts must be nonempty and aligned")
        if len(set(self.assignments)) != len(self.assignments):
            raise ValueError("sparse assignments must not contain duplicates")
        for assignment in self.assignments:
            if len(assignment) != len(self.measure_domains) or any(
                isinstance(state, bool) or not isinstance(state, int) or not 0 <= state < domain
                for state, domain in zip(assignment, self.measure_domains, strict=True)
            ):
                raise ValueError("sparse assignment is outside the measured domain")
        _nonnegative_int(self.start_index, field="start_index")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in self.counts
        ):
            raise ValueError("sparse counts must contain positive integers")

    @property
    def sample_count(self) -> int:
        return sum(self.counts)

    def count(self, assignment: tuple[int, ...]) -> int:
        try:
            return self.counts[self.assignments.index(assignment)]
        except ValueError:
            return 0


@dataclass(frozen=True, slots=True)
class WorldEpisodeStep:
    kind: str
    message: str | None = None
    batch: WorldMeasuredBatch | None = None
    score: Mapping[str, Any] | None = None


def _parse_visible_measure(
    value: object,
    view: _RuntimeView,
    world: WorldSpec,
    *,
    measure_max: int | None,
    excluded_label: str | None = None,
) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("measure must be a nonempty JSON list")
    if any(not isinstance(label, str) for label in value):
        raise ValueError("measure labels must be strings")
    if len(set(value)) != len(value):
        raise ValueError("measure labels must not contain duplicates")
    if excluded_label is not None and excluded_label in value:
        raise ValueError("the intervention target must not appear in measure")
    if measure_max is not None and len(value) > measure_max:
        raise ValueError("measure exceeds the per-batch variable limit")
    measure: list[int] = []
    for visible_label in value:
        if visible_label not in view.visible_to_internal:
            raise ValueError("unknown measure variable")
        internal_name = view.visible_to_internal[visible_label]
        if not view.readable[internal_name]:
            raise ValueError("measure variable is not readable")
        measure.append(world.variables.index(internal_name))
    return tuple(measure)


def _parse_batch_size(value: object, budget: Budget, remaining_samples: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("batch_size must be an integer")
    if value not in budget.batch_sizes or value > remaining_samples:
        raise ValueError("batch_size is not legal with the remaining budget")
    return value


def parse_world_intervention(
    raw: str,
    seed: Mapping[str, Any],
    world: WorldSpec,
    *,
    budget: Budget,
    remaining_rounds: int,
    remaining_samples: int,
    measure_max: int | None = None,
) -> WorldInterventionCommand:
    """Parse one strict visible intervention for a generic WorldSpec episode."""

    if not isinstance(budget, Budget):
        raise TypeError("budget must be a Budget")
    remaining_rounds = _nonnegative_int(remaining_rounds, field="remaining_rounds")
    remaining_samples = _nonnegative_int(remaining_samples, field="remaining_samples")
    if remaining_rounds > budget.max_rounds or remaining_samples > budget.max_samples:
        raise ValueError("remaining budget exceeds the episode budget")
    if measure_max is not None and (
        isinstance(measure_max, bool) or not isinstance(measure_max, int) or measure_max <= 0
    ):
        raise ValueError("measure_max must be a positive integer or None")
    if remaining_rounds == 0 or remaining_samples < min(budget.batch_sizes):
        raise ValueError("no experiment is legal; a terminal answer is required")

    value = _json_object(raw)
    if set(value) != {"type", "target", "value", "measure", "batch_size"}:
        raise ValueError("intervention must contain exactly five protocol fields")
    if value.get("type") != "intervene":
        raise ValueError("type must be intervene")
    view = _runtime_view(seed, world)

    visible_target = value.get("target")
    if not isinstance(visible_target, str) or visible_target not in view.visible_to_internal:
        raise ValueError("unknown intervention target")
    target_name = view.visible_to_internal[visible_target]
    if not view.manipulability[target_name]:
        raise ValueError("intervention target is not manipulable")
    target = world.variables.index(target_name)
    state = _state_index(value.get("value"))
    if state >= world.domains[target]:
        raise ValueError("intervention state is outside the target domain")

    measure = _parse_visible_measure(
        value.get("measure"),
        view,
        world,
        measure_max=measure_max,
        excluded_label=visible_target,
    )
    batch_size = _parse_batch_size(value.get("batch_size"), budget, remaining_samples)
    return WorldInterventionCommand(
        intervention=WorldIntervention(target=target, value=state),
        measure=measure,
        batch_size=batch_size,
    )


def parse_world_observation(
    raw: str,
    seed: Mapping[str, Any],
    world: WorldSpec,
    *,
    budget: Budget,
    remaining_rounds: int,
    remaining_samples: int,
    measure_max: int | None = None,
) -> WorldObservationCommand:
    """Parse one strict passive-observation request for a WorldSpec episode."""

    if not isinstance(budget, Budget):
        raise TypeError("budget must be a Budget")
    remaining_rounds = _nonnegative_int(remaining_rounds, field="remaining_rounds")
    remaining_samples = _nonnegative_int(remaining_samples, field="remaining_samples")
    if remaining_rounds > budget.max_rounds or remaining_samples > budget.max_samples:
        raise ValueError("remaining budget exceeds the episode budget")
    if measure_max is not None and (
        isinstance(measure_max, bool) or not isinstance(measure_max, int) or measure_max <= 0
    ):
        raise ValueError("measure_max must be a positive integer or None")
    if remaining_rounds == 0 or remaining_samples < min(budget.batch_sizes):
        raise ValueError("no experiment is legal; a terminal answer is required")

    value = _json_object(raw)
    if set(value) != {"type", "measure", "batch_size"}:
        raise ValueError("observation must contain exactly three protocol fields")
    if value.get("type") != "observe":
        raise ValueError("type must be observe")
    view = _runtime_view(seed, world)
    measure = _parse_visible_measure(
        value.get("measure"),
        view,
        world,
        measure_max=measure_max,
    )
    batch_size = _parse_batch_size(value.get("batch_size"), budget, remaining_samples)
    return WorldObservationCommand(measure=measure, batch_size=batch_size)


def sample_worldspec_batch(
    world: WorldSpec,
    tape: OutcomeTape,
    command: WorldInterventionCommand | WorldObservationCommand,
    *,
    start_index: int,
) -> WorldMeasuredBatch:
    """Sample one hard-do or natural batch and retain its selected readout."""

    if not isinstance(world, WorldSpec):
        raise TypeError("world must be a WorldSpec")
    if not isinstance(tape, OutcomeTape):
        raise TypeError("tape must be an OutcomeTape")
    if not isinstance(command, (WorldInterventionCommand, WorldObservationCommand)):
        raise TypeError("command must be an intervention or observation command")
    start_index = _nonnegative_int(start_index, field="start_index")
    if start_index + command.batch_size > 2**64:
        raise ValueError("the requested arm stream exceeds its index space")
    if any(node >= len(world.variables) for node in command.measure):
        raise ValueError("measure node is outside the WorldSpec domain")

    if isinstance(command, WorldInterventionCommand):
        intervention = command.intervention
        target = intervention.target
        state = intervention.value
        if target >= len(world.variables) or state >= world.domains[target]:
            raise ValueError("intervention is outside the WorldSpec domain")
        interventions = {target: state}
    else:
        intervention = None
        interventions = {}

    counts: dict[tuple[int, ...], int] = {}
    for sample_index in range(start_index, start_index + command.batch_size):
        uniforms = tuple(
            (
                tape.worldspec_node_uniform(target, state, sample_index, node)
                if intervention is not None
                else tape.worldspec_observation_node_uniform(sample_index, node)
            )
            for node in range(len(world.variables))
        )
        selected = sample_worldspec_assignment(world, interventions, uniforms)
        projected = tuple(selected[node] for node in command.measure)
        counts[projected] = counts.get(projected, 0) + 1
    assignments = tuple(sorted(counts))
    return WorldMeasuredBatch(
        intervention=intervention,
        measure=command.measure,
        measure_domains=tuple(world.domains[node] for node in command.measure),
        start_index=start_index,
        assignments=assignments,
        counts=tuple(counts[assignment] for assignment in assignments),
    )


def render_world_batch_message(
    batch: WorldMeasuredBatch,
    seed: Mapping[str, Any],
    world: WorldSpec,
    *,
    budget: Budget,
    remaining_rounds: int,
    remaining_samples: int,
) -> str:
    """Render selected-measure counts without exposing any unrequested variable."""

    if not isinstance(batch, WorldMeasuredBatch):
        raise TypeError("batch must be a WorldMeasuredBatch")
    if not isinstance(budget, Budget):
        raise TypeError("budget must be a Budget")
    remaining_rounds = _nonnegative_int(remaining_rounds, field="remaining_rounds")
    remaining_samples = _nonnegative_int(remaining_samples, field="remaining_samples")
    view = _runtime_view(seed, world)
    measure_names = tuple(world.variables[node] for node in batch.measure)
    measure_labels = tuple(view.labels[name] for name in measure_names)
    joint_counts: dict[str, int] = {}
    for assignment, count in zip(batch.assignments, batch.counts, strict=True):
        key = ",".join(
            f"{label}=state_{state}"
            for label, state in zip(measure_labels, assignment, strict=True)
        )
        joint_counts[key] = count
    payload: dict[str, Any] = {
        "type": "batch_result",
        "batch": {
            "n": batch.sample_count,
            "measure": list(measure_labels),
            "joint_counts": joint_counts,
        },
        "remaining_rounds": remaining_rounds,
        "remaining_samples": remaining_samples,
    }
    if batch.intervention is None:
        payload["experiment"] = {"type": "observe"}
    else:
        target_name = world.variables[batch.intervention.target]
        payload["intervention"] = {
            "target": view.labels[target_name],
            "value": f"state_{batch.intervention.value}",
        }
    if remaining_rounds == 0 or remaining_samples < min(budget.batch_sizes):
        instruction = "No experiment remains legal. Return a terminal answer now."
    else:
        instruction = "Return the next legal JSON command."
    return json.dumps(payload, separators=(",", ":")) + "\n" + instruction


class WorldSpecEpisode:
    """Budgeted multi-turn owner for one hidden generic WorldSpec task."""

    def __init__(
        self,
        world: WorldSpec,
        seed: Mapping[str, Any],
        tape: OutcomeTape,
        *,
        budget: Budget | None = None,
        measure_max: int | None = None,
    ) -> None:
        if not isinstance(world, WorldSpec):
            raise TypeError("world must be a WorldSpec")
        if not isinstance(tape, OutcomeTape):
            raise TypeError("tape must be an OutcomeTape")
        if budget is not None and not isinstance(budget, Budget):
            raise TypeError("budget must be a Budget")
        _runtime_view(seed, world)
        resolved_budget = budget if budget is not None else Budget()
        resolved_measure_max = resolve_observation_bandwidth(seed, measure_max)
        # Reuse the renderer's public action-surface validation so an episode
        # cannot exist when every legal target is also the only readable measure.
        render_seed_initial_messages(
            seed,
            budget=resolved_budget,
            measure_max=resolved_measure_max,
        )
        self.world = world
        self.seed = seed
        self.tape = tape
        self.budget = resolved_budget
        self.measure_max = resolved_measure_max
        self._rounds_used = 0
        self._samples_used = 0
        self._arm_offsets: dict[WorldIntervention, int] = {}
        self._observation_offset = 0
        self._history: list[WorldMeasuredBatch] = []
        self._terminal_score: Mapping[str, Any] | None = None

    @property
    def rounds_used(self) -> int:
        return self._rounds_used

    @property
    def samples_used(self) -> int:
        return self._samples_used

    @property
    def remaining_rounds(self) -> int:
        return self.budget.max_rounds - self._rounds_used

    @property
    def remaining_samples(self) -> int:
        return self.budget.max_samples - self._samples_used

    @property
    def history(self) -> tuple[WorldMeasuredBatch, ...]:
        return tuple(self._history)

    @property
    def completed(self) -> bool:
        return self._terminal_score is not None

    @property
    def terminal_score(self) -> Mapping[str, Any] | None:
        return self._terminal_score

    def initial_messages(self) -> tuple[dict[str, str], dict[str, str]]:
        return render_seed_initial_messages(
            self.seed,
            budget=self.budget,
            measure_max=self.measure_max,
        )

    def parse_intervention(self, raw: str) -> WorldInterventionCommand:
        return parse_world_intervention(
            raw,
            self.seed,
            self.world,
            budget=self.budget,
            remaining_rounds=self.remaining_rounds,
            remaining_samples=self.remaining_samples,
            measure_max=self.measure_max,
        )

    def parse_observation(self, raw: str) -> WorldObservationCommand:
        return parse_world_observation(
            raw,
            self.seed,
            self.world,
            budget=self.budget,
            remaining_rounds=self.remaining_rounds,
            remaining_samples=self.remaining_samples,
            measure_max=self.measure_max,
        )

    def intervene(self, command: WorldInterventionCommand) -> WorldMeasuredBatch:
        if self.completed:
            raise ValueError("the episode is already complete")
        if not isinstance(command, WorldInterventionCommand):
            raise TypeError("command must be a WorldInterventionCommand")
        view = _runtime_view(self.seed, self.world)
        target = command.intervention.target
        state = command.intervention.value
        if target >= len(self.world.variables) or state >= self.world.domains[target]:
            raise ValueError("intervention is outside the WorldSpec domain")
        target_name = self.world.variables[target]
        if not view.manipulability[target_name]:
            raise ValueError("intervention target is not manipulable")
        if not command.measure or any(
            node >= len(self.world.variables) for node in command.measure
        ):
            raise ValueError("measure is outside the WorldSpec domain")
        if target in command.measure:
            raise ValueError("the intervention target must not appear in measure")
        if any(not view.readable[self.world.variables[node]] for node in command.measure):
            raise ValueError("measure contains an unreadable variable")
        if self.measure_max is not None and len(command.measure) > self.measure_max:
            raise ValueError("measure exceeds the per-batch variable limit")
        if command.batch_size not in self.budget.batch_sizes:
            raise ValueError("batch_size is not permitted by this episode")
        if self.remaining_rounds <= 0:
            raise ValueError("the experiment-round budget is exhausted")
        if command.batch_size > self.remaining_samples:
            raise ValueError("the experiment-sample budget is exhausted")

        start_index = self._arm_offsets.get(command.intervention, 0)
        batch = sample_worldspec_batch(
            self.world,
            self.tape,
            command,
            start_index=start_index,
        )
        self._arm_offsets[command.intervention] = start_index + command.batch_size
        self._rounds_used += 1
        self._samples_used += command.batch_size
        self._history.append(batch)
        return batch

    def observe(self, command: WorldObservationCommand) -> WorldMeasuredBatch:
        if self.completed:
            raise ValueError("the episode is already complete")
        if not isinstance(command, WorldObservationCommand):
            raise TypeError("command must be a WorldObservationCommand")
        view = _runtime_view(self.seed, self.world)
        if not command.measure or any(
            node >= len(self.world.variables) for node in command.measure
        ):
            raise ValueError("measure is outside the WorldSpec domain")
        if any(not view.readable[self.world.variables[node]] for node in command.measure):
            raise ValueError("measure contains an unreadable variable")
        if self.measure_max is not None and len(command.measure) > self.measure_max:
            raise ValueError("measure exceeds the per-batch variable limit")
        if command.batch_size not in self.budget.batch_sizes:
            raise ValueError("batch_size is not permitted by this episode")
        if self.remaining_rounds <= 0:
            raise ValueError("the experiment-round budget is exhausted")
        if command.batch_size > self.remaining_samples:
            raise ValueError("the experiment-sample budget is exhausted")

        batch = sample_worldspec_batch(
            self.world,
            self.tape,
            command,
            start_index=self._observation_offset,
        )
        self._observation_offset += command.batch_size
        self._rounds_used += 1
        self._samples_used += command.batch_size
        self._history.append(batch)
        return batch

    def render_feedback(self, batch: WorldMeasuredBatch) -> str:
        if batch not in self._history:
            raise ValueError("batch is not part of this episode history")
        return render_world_batch_message(
            batch,
            self.seed,
            self.world,
            budget=self.budget,
            remaining_rounds=self.remaining_rounds,
            remaining_samples=self.remaining_samples,
        )

    def step(self, raw: str) -> WorldEpisodeStep:
        """Execute one model command and return feedback or terminal diagnostics."""

        if self.completed:
            raise ValueError("the episode is already complete")
        command_type = _json_object(raw).get("type")
        if command_type == "intervene":
            batch = self.intervene(self.parse_intervention(raw))
            return WorldEpisodeStep(
                kind="batch",
                message=self.render_feedback(batch),
                batch=batch,
            )
        if command_type == "observe":
            batch = self.observe(self.parse_observation(raw))
            return WorldEpisodeStep(
                kind="batch",
                message=self.render_feedback(batch),
                batch=batch,
            )
        if command_type == "answer":
            score = score_terminal_answer(raw, self.seed, self.world)
            self._terminal_score = score
            return WorldEpisodeStep(kind="terminal", score=score)
        raise ValueError("type must be intervene, observe, or answer")
