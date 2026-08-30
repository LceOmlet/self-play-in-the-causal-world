"""Renderer for CPT-World seed tasks.

Boundary: this module only decides what becomes visible and builds the prompt.
It never computes query truth, scores answers, samples worlds, or parses model
commands. ``budget`` is a renderer argument. Sampled seeds own their fixed
``observation_bandwidth``; ``measure_max`` remains only as a legacy/manual-seed
fallback and cannot override a declared seed value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .episode import (
    OBSERVATIONS_PER_BANDWIDTH_UNIT,
    Budget,
    budget_for_observation_bandwidth,
)
from .registry import HIDING_MODES
from .world_space import world_state_names

RENDERER_VERSION = "cpt-world-seed-renderer-v3"
SYSTEM_MESSAGE = (
    "You are evaluated on active causal experimentation. "
    "Return exactly one legal JSON command and no prose."
)
DEFAULT_HIDING_MODES = tuple(sorted(HIDING_MODES))


@dataclass(frozen=True, slots=True)
class _RenderContext:
    seed_id: str
    label_map: Mapping[str, str]
    variables: tuple[Mapping[str, Any], ...]
    query: Mapping[str, Any]
    task_head: Mapping[str, Any]
    hiding_modes: tuple[str, ...]
    state_names: Mapping[str, tuple[str, ...]]
    manipulability: Mapping[str, bool]
    readable: Mapping[str, bool]
    budget: Budget
    measure_max: int | None

    def visible_labels(self) -> tuple[str, ...]:
        return tuple(str(item["label"]) for item in self.variables)

    def states_for_label(self, label: str) -> tuple[str, ...]:
        for item in self.variables:
            if item.get("label") == label:
                return tuple(str(state) for state in item.get("states", ()))
        raise ValueError(f"unknown visible label {label}")

    def internal_name_for_label(self, label: str) -> str:
        inverse = {visible: internal for internal, visible in self.label_map.items()}
        if label in inverse:
            return inverse[label]
        if label in self.state_names:
            return label
        raise ValueError(f"label {label} has no internal state declaration")

    def resolve_label(self, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(f"anchor {value!r} must be a string")
        if value in self.label_map:
            return str(self.label_map[value])
        if value in self.visible_labels():
            return value
        raise ValueError(f"anchor {value} is not present in the visible schema")

    def resolve_state_token(self, variable_value: object, state_value: object) -> str:
        variable_label = self.resolve_label(variable_value)
        internal_name = self.internal_name_for_label(variable_label)
        state_order = tuple(self.state_names.get(internal_name, ()))
        if not state_order:
            raise ValueError(f"no state order for {internal_name}")
        if isinstance(state_value, str) and state_value.startswith("state_"):
            return state_value
        if isinstance(state_value, int) and not isinstance(state_value, bool):
            index = state_value
        else:
            text = str(state_value)
            if text in state_order:
                index = state_order.index(text)
            elif text.isdigit():
                index = int(text)
            else:
                raise ValueError(f"cannot map state {state_value!r} for {internal_name}")
        if index < 0 or index >= len(state_order):
            raise ValueError(f"state index {index} out of range for {internal_name}")
        return f"state_{index}"

    def legal_targets(self) -> tuple[str, ...]:
        return tuple(
            str(self.label_map[name])
            for name, allowed in self.manipulability.items()
            if allowed and name in self.label_map
        )

    def readable_labels(self) -> tuple[str, ...]:
        return tuple(
            str(self.label_map[name])
            for name, allowed in self.readable.items()
            if allowed and name in self.label_map
        )


@dataclass(frozen=True, slots=True)
class RenderedAteQuerySurface:
    """Structured numerical treatment/outcome query surface.

    ``labels`` retains the exact visible-to-internal bridge needed by the
    evaluator.  ``semantic_key`` removes only the opaque label spellings; it
    keeps variable order, domains, query anchors, action order, readout order,
    and every other model-visible distinction.
    """

    labels: tuple[str, ...]
    domains: tuple[int, ...]
    query_type: str
    terminal_kind: str
    treatment: int
    outcome: int
    treatment_value: int
    baseline_value: int
    outcome_state: int | None
    factual_outcome_state: int | None
    legal_targets: tuple[int, ...]
    readable: tuple[int, ...]
    hiding_modes: tuple[str, ...]
    task_head: str
    measure_max: int | None

    @property
    def semantic_key(self) -> tuple[object, ...]:
        return (
            self.domains,
            self.query_type,
            self.terminal_kind,
            self.treatment,
            self.outcome,
            self.treatment_value,
            self.baseline_value,
            self.outcome_state,
            self.factual_outcome_state,
            self.legal_targets,
            self.readable,
            tuple(sorted(self.hiding_modes)),
            self.task_head,
            self.measure_max,
        )


@dataclass(frozen=True, slots=True)
class RenderedDecisionQuerySurface:
    """Structured best-intervention surface emitted by the renderer."""

    labels: tuple[str, ...]
    domains: tuple[int, ...]
    decision_target: int
    outcome: int
    outcome_state: int
    objective: str
    legal_targets: tuple[int, ...]
    readable: tuple[int, ...]
    hiding_modes: tuple[str, ...]
    task_head: str
    measure_max: int | None

    @property
    def semantic_key(self) -> tuple[object, ...]:
        return (
            self.domains,
            "best_intervention",
            self.decision_target,
            self.outcome,
            self.outcome_state,
            self.objective,
            self.legal_targets,
            self.readable,
            tuple(sorted(self.hiding_modes)),
            self.task_head,
            self.measure_max,
        )


@dataclass(frozen=True, slots=True)
class RenderedDiscoveryQuerySurface:
    """Structured backdoor- or mediator-discovery surface."""

    labels: tuple[str, ...]
    domains: tuple[int, ...]
    query_type: str
    treatment: int
    outcome: int
    legal_targets: tuple[int, ...]
    readable: tuple[int, ...]
    hiding_modes: tuple[str, ...]
    task_head: str
    measure_max: int | None

    @property
    def semantic_key(self) -> tuple[object, ...]:
        return (
            self.domains,
            self.query_type,
            self.treatment,
            self.outcome,
            self.legal_targets,
            self.readable,
            tuple(sorted(self.hiding_modes)),
            self.task_head,
            self.measure_max,
        )


def _hiding_modes(seed: Mapping[str, Any]) -> tuple[str, ...]:
    raw = seed.get("hiding_modes")
    if raw is None:
        return DEFAULT_HIDING_MODES
    if isinstance(raw, str):
        modes = (raw,)
    elif isinstance(raw, (tuple, list, set, frozenset)):
        modes = tuple(str(mode) for mode in raw)
    else:
        raise TypeError("hiding_modes must be a string or an iterable of strings")
    unknown = set(modes) - HIDING_MODES
    if unknown:
        raise ValueError(f"unsupported hiding modes: {sorted(unknown)}")
    return modes


def resolve_observation_bandwidth(
    seed: Mapping[str, Any],
    requested: int | None = None,
) -> int | None:
    """Resolve a seed-owned observation bandwidth without runner overrides."""

    if requested is not None and (
        isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0
    ):
        raise ValueError("measure_max must be a positive integer or None")
    declared = seed.get("observation_bandwidth")
    if declared is None:
        return requested
    visible_schema = seed.get("visible_schema")
    variables = visible_schema.get("variables") if isinstance(visible_schema, Mapping) else None
    variable_count = len(variables) if isinstance(variables, (list, tuple)) else 0
    if (
        isinstance(declared, bool)
        or not isinstance(declared, int)
        or not 1 <= declared <= variable_count
    ):
        raise ValueError("seed observation_bandwidth is outside its visible variable domain")
    if requested is not None and requested != declared:
        raise ValueError("measure_max cannot override seed observation_bandwidth")
    return declared


def _render_query_block(ctx: _RenderContext) -> tuple[list[str], list[str]]:
    query_type = str(ctx.query.get("type"))
    treatment_value = ctx.query.get("treatment")
    outcome_value = ctx.query.get("outcome")
    if query_type in {
        "ate",
        "individual_counterfactual_probability",
        "backadj_minimal_sets",
        "mediator_set",
    }:
        if treatment_value is None or outcome_value is None:
            raise ValueError(f"query {query_type} requires treatment and outcome anchors")
        treatment = ctx.resolve_label(treatment_value)
        outcome = ctx.resolve_label(outcome_value)

    if query_type == "ate":
        treatment_state = ctx.resolve_state_token(treatment, ctx.query.get("treatment_value", 1))
        baseline_state = ctx.resolve_state_token(treatment, ctx.query.get("baseline_value", 0))
        if treatment_state == baseline_state:
            raise ValueError("ATE treatment and baseline states must differ")
        outcome_states = ctx.states_for_label(outcome)
        lines = [
            f"Estimate the complete categorical treatment effect of {treatment} on {outcome}.",
            (
                f"For every state s of {outcome}, effect[s] = "
                f"P({outcome}=s | do({treatment}={treatment_state})) - "
                f"P({outcome}=s | do({treatment}={baseline_state}))."
            ),
            (
                "Return every listed outcome state exactly once. The effect components "
                "must form a valid difference of two categorical distributions: they sum "
                "to zero and their positive components sum to at most one."
            ),
        ]
        if treatment not in ctx.legal_targets() and outcome not in ctx.legal_targets():
            lines.append(
                f"{treatment} and {outcome} are readonly during experimentation. "
                "Use the available passive observations and experiments on legal do "
                "targets to estimate the effect."
            )
        effect_fields = ",".join(f'"{state}":<number in [-1,1]>' for state in outcome_states)
        answer_schema = [f'{{"type":"answer","effect":{{{effect_fields}}}}}']
    elif query_type == "individual_counterfactual_probability":
        treatment_states = ctx.states_for_label(treatment)
        factual_state = ctx.resolve_state_token(treatment, ctx.query.get("factual_value"))
        counterfactual_state = ctx.resolve_state_token(
            treatment, ctx.query.get("counterfactual_value")
        )
        factual_outcome_state = ctx.resolve_state_token(
            outcome, ctx.query.get("factual_outcome_state")
        )
        target_outcome_state = ctx.resolve_state_token(outcome, ctx.query.get("outcome_state"))
        if factual_state == counterfactual_state:
            raise ValueError("factual and counterfactual treatment states must differ")
        if factual_state not in treatment_states or counterfactual_state not in treatment_states:
            raise ValueError("counterfactual treatment states are outside the rendered domain")
        lines = [
            (
                f"One individual was assigned do({treatment}={factual_state}) and "
                f"their observed outcome was {outcome}={factual_outcome_state}."
            ),
            (
                f"Estimate q = P({outcome}_do({treatment}={counterfactual_state})="
                f"{target_outcome_state} | {outcome}_do({treatment}={factual_state})="
                f"{factual_outcome_state}) for this same individual."
            ),
            (
                "Across all causally sufficient structural mechanisms compatible with the "
                "hidden CPT-World, q has a sharp identified interval [lower, upper]. "
                "Estimate and return both endpoints of that interval."
            ),
        ]
        answer_schema = ['{"type":"answer","lower":<number in [0,1]>,"upper":<number in [0,1]>}']
        if treatment not in ctx.legal_targets() and outcome not in ctx.legal_targets():
            lines.append(
                f"{treatment} and {outcome} are readonly during experimentation. "
                "Use the available passive observations and experiments on legal do "
                "targets to estimate the identified interval."
            )
    elif query_type == "backadj_minimal_sets":
        lines = [
            f"Return all minimal adjustment sets for the total effect of {treatment} on {outcome}."
        ]
        answer_schema = [
            '{"type":"answer","adjustment_sets":[["LAB", ...], ...]}',
            (
                "Encode the singleton family containing the empty adjustment set as "
                '{"type":"answer","adjustment_sets":[[]]}.'
            ),
        ]
    elif query_type == "mediator_set":
        lines = [
            f"Return every observed intermediate variable, excluding {treatment} and "
            f"{outcome}, that lies on a directed path from {treatment} to {outcome}. "
            "Also return every directed edge that occurs consecutively on at least one "
            "such path, including edges incident to the two endpoints."
        ]
        answer_schema = [
            ('{"type":"answer","mediators":["LAB", ...],"order":[["LAB","LAB"], ...]}')
        ]
    elif query_type == "best_intervention":
        decision_target_value = ctx.query.get("decision_target")
        if outcome_value is None or decision_target_value is None:
            raise ValueError("best_intervention query requires outcome and decision_target anchors")
        outcome = ctx.resolve_label(outcome_value)
        decision_target = ctx.resolve_label(decision_target_value)
        if decision_target == outcome:
            raise ValueError("decision target must differ from the outcome")
        if decision_target in ctx.legal_targets():
            raise ValueError("decision target must be readonly during experimentation")
        objective = str(ctx.query.get("objective", "minimize"))
        outcome_state = ctx.resolve_state_token(
            outcome,
            ctx.query.get("target_state", ctx.query.get("outcome_state", 1)),
        )
        verb = "minimize" if objective == "minimize" else "maximize"
        decision_states = ctx.states_for_label(decision_target)
        lines = [
            f"Choose the final deployment state of {decision_target} that {verb}s "
            f"P({outcome}={outcome_state} | do({decision_target}=d)).",
            (
                "Final deployment candidates: "
                + " / ".join(f"do({decision_target}={state})" for state in decision_states)
            ),
            (
                f"{decision_target} is readonly during experimentation. The final choice "
                "is an answer, not an experiment."
            ),
        ]
        answer_schema = ['{"type":"answer","value":"state_i"}']
    else:
        raise ValueError(f"unsupported query type {query_type}")
    return lines, answer_schema


def _validated_action_surface(
    ctx: _RenderContext,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    legal_targets = ctx.legal_targets()
    if not legal_targets:
        raise ValueError(f"{ctx.seed_id}: action surface contains no legal target/measure pair")
    readable_labels = ctx.readable_labels()
    if not readable_labels:
        raise ValueError(f"{ctx.seed_id}: no readable variable is visible")
    dead_targets = [
        target
        for target in legal_targets
        if not any(target != measure for measure in readable_labels)
    ]
    if dead_targets:
        raise ValueError(
            f"{ctx.seed_id}: action surface contains no legal target/measure pair "
            f"for {dead_targets}"
        )
    return legal_targets, readable_labels


def _render_prompt(ctx: _RenderContext) -> str:
    lines: list[str] = ["DOLENS HIDDEN-MECHANISM TASK", ""]
    lines.append("Variables and finite states:")
    for item in ctx.variables:
        label = str(item["label"])
        states = " / ".join(str(state) for state in item["states"])
        lines.append(f"{label}: {states}")

    query_lines, answer_schema = _render_query_block(ctx)
    lines.append("")
    lines.append("Query:")
    lines.extend(f"- {line}" for line in query_lines)

    lines.append("")
    lines.append("Experiments:")
    legal_targets, readable_labels = _validated_action_surface(ctx)
    lines.append("Legal experimental do targets: " + json_dumps_list(legal_targets))
    for label in legal_targets:
        lines.append(f"{label} values: " + " / ".join(ctx.states_for_label(label)))
    lines.append(
        "Each intervention command is exactly "
        '{"type":"intervene","target":"LAB","value":"state_i",'
        '"measure":["LAB", ...],"batch_size":n}.'
    )
    lines.append(
        "Each passive observation command is exactly "
        '{"type":"observe","measure":["LAB", ...],"batch_size":n}.'
    )
    lines.append(
        "measure must be a nonempty subset of readable variables: "
        + json_dumps_list(readable_labels)
    )
    lines.append("The intervention target must not appear in measure.")
    lines.append(
        "An intervene command draws batch_size IID units from this same hidden world "
        "under the stated hard intervention."
    )
    lines.append(
        "An observe command draws batch_size IID units from the natural distribution "
        "of the same hidden world, with no intervention."
    )
    lines.append(
        "Feedback reports a lossless compact joint_histogram for only the requested "
        "measure variables; all unrequested values remain hidden."
    )
    lines.append(
        "joint_histogram.columns gives the variable order. Each joint_histogram.rows item "
        "is [[state_index,...],count]; omitted requested-measure assignments have count zero."
    )
    lines.append(
        "Commands are rejected without consuming budget when "
        "min(batch_size, product of measured domain cardinalities) exceeds 128."
    )
    if ctx.measure_max is not None:
        noun = "variable" if ctx.measure_max == 1 else "variables"
        lines.append(f"At most {ctx.measure_max} {noun} may be measured per batch.")
    if "no_full_joint" in ctx.hiding_modes:
        lines.append("The environment never returns full-joint counts automatically.")
    if "manipulability_via_action_legality" in ctx.hiding_modes:
        lines.append("Only the legal action feedback shown above is available.")
    if "evidence_by_intervention_only" in ctx.hiding_modes:
        lines.append(
            "There is no initial dataset; evidence comes only from requested batch experiments."
        )
    budget_line = f"The total observation budget is {ctx.budget.max_observations} scalar values."
    if (
        ctx.measure_max is not None
        and ctx.budget.max_observations == ctx.measure_max * OBSERVATIONS_PER_BANDWIDTH_UNIT
    ):
        budget_line = (
            f"The total observation budget is {ctx.budget.max_observations} scalar values "
            f"({ctx.measure_max} x {OBSERVATIONS_PER_BANDWIDTH_UNIT})."
        )
    lines.append(budget_line)
    lines.append(
        "Each query with batch_size b and r measured variables costs b*r from this budget. "
        "batch_size may be any positive integer whose cost fits the remaining budget. "
        "There is no separate limit on the number of queries."
    )
    lines.append(
        "Solve the task through this multi-turn protocol: first collect evidence with "
        "one or more legal observe/intervene commands, inspect each batch_result and "
        "the remaining budget, then return the terminal answer when the evidence is "
        "sufficient. You do not need to exhaust the budget."
    )
    lines.append(
        "If a command is invalid, it is not executed and consumes no experiment budget; "
        "use the protocol_error feedback to submit a corrected command."
    )

    lines.append("")
    lines.append("To finish, return exactly one JSON object and no prose:")
    lines.extend(answer_schema)
    lines.append("Return exactly one legal JSON command per turn.")
    return "\n".join(lines)


def json_dumps_list(values: object) -> str:
    import json

    return json.dumps(list(values), separators=(",", ":"))


def _render_context(
    seed: Mapping[str, Any],
    *,
    budget: Budget | None,
    measure_max: int | None,
) -> _RenderContext:
    if not isinstance(seed, Mapping):
        raise TypeError("seed must be a mapping")
    measure_max = resolve_observation_bandwidth(seed, measure_max)
    if budget is None:
        if measure_max is None:
            raise ValueError("a seed without observation_bandwidth requires an explicit Budget")
        budget = budget_for_observation_bandwidth(measure_max)
    if not isinstance(budget, Budget):
        raise TypeError("budget must be a Budget")

    visible_schema = seed.get("visible_schema")
    if not isinstance(visible_schema, Mapping):
        raise ValueError(f"{seed.get('seed_id', '<unknown>')}: missing visible_schema")
    variables = visible_schema.get("variables")
    if not isinstance(variables, (list, tuple)) or not variables:
        raise ValueError("visible_schema.variables must be a nonempty sequence")
    variables = tuple(dict(item) for item in variables)
    label_map = visible_schema.get("variable_labels")
    if not isinstance(label_map, Mapping):
        raise ValueError("visible_schema.variable_labels must be a mapping")
    query = seed.get("query")
    if not isinstance(query, Mapping):
        raise ValueError("seed must contain a query mapping")
    task_head = seed.get("task_head")
    if not isinstance(task_head, Mapping):
        raise ValueError("seed must contain a task_head mapping")
    manipulability = seed.get("manipulability")
    readable = seed.get("readable")
    if not isinstance(manipulability, Mapping) or not isinstance(readable, Mapping):
        raise ValueError("seed must contain manipulability and readable mappings")

    return _RenderContext(
        seed_id=str(seed.get("seed_id", "<unknown>")),
        label_map=dict(label_map),
        variables=tuple(variables),
        query=query,
        task_head=task_head,
        hiding_modes=_hiding_modes(seed),
        state_names=world_state_names(seed.get("world_source", {})),
        manipulability=manipulability,
        readable=readable,
        budget=budget,
        measure_max=measure_max,
    )


def rendered_ate_query_surface(
    seed: Mapping[str, Any],
    *,
    measure_max: int | None = None,
) -> RenderedAteQuerySurface:
    """Return the exact structured ATE query surface visible to the model."""

    return _rendered_target_query_surface(seed, "ate", measure_max=measure_max)


def rendered_counterfactual_query_surface(
    seed: Mapping[str, Any],
    *,
    measure_max: int | None = None,
) -> RenderedAteQuerySurface:
    """Return the structured individual-counterfactual surface visible to the model."""

    return _rendered_target_query_surface(
        seed,
        "individual_counterfactual_probability",
        measure_max=measure_max,
    )


def _rendered_target_query_surface(
    seed: Mapping[str, Any],
    expected_query_type: str,
    *,
    measure_max: int | None,
) -> RenderedAteQuerySurface:
    """Build the shared treatment/outcome surface without duplicating rendering."""

    ctx = _render_context(seed, budget=None, measure_max=measure_max)
    query_type = str(ctx.query.get("type"))
    if query_type != expected_query_type:
        raise ValueError(f"target query surface requires {expected_query_type}, got {query_type}")
    _render_query_block(ctx)
    legal_targets, readable_labels = _validated_action_surface(ctx)
    labels = ctx.visible_labels()
    label_positions = {label: index for index, label in enumerate(labels)}
    treatment = ctx.resolve_label(ctx.query.get("treatment"))
    outcome = ctx.resolve_label(ctx.query.get("outcome"))
    if query_type == "individual_counterfactual_probability":
        treatment_state = ctx.resolve_state_token(treatment, ctx.query.get("counterfactual_value"))
        baseline_state = ctx.resolve_state_token(treatment, ctx.query.get("factual_value"))
        factual_outcome_state = ctx.resolve_state_token(
            outcome, ctx.query.get("factual_outcome_state")
        )
        outcome_state = ctx.resolve_state_token(outcome, ctx.query.get("outcome_state"))
    else:
        treatment_state = ctx.resolve_state_token(treatment, ctx.query.get("treatment_value", 1))
        baseline_state = ctx.resolve_state_token(treatment, ctx.query.get("baseline_value", 0))
        factual_outcome_state = None
        outcome_state = None
    treatment_states = ctx.states_for_label(treatment)
    outcome_states = ctx.states_for_label(outcome)
    if outcome_state is not None and outcome_state not in outcome_states:
        raise ValueError("target-query outcome state is outside the rendered outcome domain")
    if treatment_state == baseline_state:
        raise ValueError("target-query treatment and baseline states must differ")
    return RenderedAteQuerySurface(
        labels=labels,
        domains=tuple(len(ctx.states_for_label(label)) for label in labels),
        query_type=query_type,
        terminal_kind=(
            "identified_interval"
            if query_type == "individual_counterfactual_probability"
            else "effect_vector"
        ),
        treatment=label_positions[treatment],
        outcome=label_positions[outcome],
        treatment_value=treatment_states.index(treatment_state),
        baseline_value=treatment_states.index(baseline_state),
        outcome_state=(outcome_states.index(outcome_state) if outcome_state is not None else None),
        factual_outcome_state=(
            outcome_states.index(factual_outcome_state)
            if factual_outcome_state is not None
            else None
        ),
        legal_targets=tuple(label_positions[label] for label in legal_targets),
        readable=tuple(label_positions[label] for label in readable_labels),
        hiding_modes=ctx.hiding_modes,
        task_head=str(ctx.task_head.get("head")),
        measure_max=ctx.measure_max,
    )


def rendered_decision_query_surface(
    seed: Mapping[str, Any],
    *,
    measure_max: int | None = None,
) -> RenderedDecisionQuerySurface:
    """Return the exact structured best-intervention surface visible to the model."""

    ctx = _render_context(seed, budget=None, measure_max=measure_max)
    if str(ctx.query.get("type")) != "best_intervention":
        raise ValueError("rendered_decision_query_surface requires a best_intervention seed")
    _render_query_block(ctx)
    legal_targets, readable_labels = _validated_action_surface(ctx)
    labels = ctx.visible_labels()
    label_positions = {label: index for index, label in enumerate(labels)}
    decision_target = ctx.resolve_label(ctx.query.get("decision_target"))
    outcome = ctx.resolve_label(ctx.query.get("outcome"))
    outcome_state = ctx.resolve_state_token(
        outcome,
        ctx.query.get("target_state", ctx.query.get("outcome_state", 1)),
    )
    outcome_states = ctx.states_for_label(outcome)
    if outcome_state not in outcome_states:
        raise ValueError("decision outcome state is outside the rendered outcome domain")
    objective = str(ctx.query.get("objective", "minimize"))
    if objective not in {"minimize", "maximize"}:
        raise ValueError("decision objective must be minimize or maximize")
    return RenderedDecisionQuerySurface(
        labels=labels,
        domains=tuple(len(ctx.states_for_label(label)) for label in labels),
        decision_target=label_positions[decision_target],
        outcome=label_positions[outcome],
        outcome_state=outcome_states.index(outcome_state),
        objective=objective,
        legal_targets=tuple(label_positions[label] for label in legal_targets),
        readable=tuple(label_positions[label] for label in readable_labels),
        hiding_modes=ctx.hiding_modes,
        task_head=str(ctx.task_head.get("head")),
        measure_max=ctx.measure_max,
    )


def rendered_discovery_query_surface(
    seed: Mapping[str, Any],
    *,
    measure_max: int | None = None,
) -> RenderedDiscoveryQuerySurface:
    """Return the exact structured discovery surface visible to the model."""

    ctx = _render_context(seed, budget=None, measure_max=measure_max)
    query_type = str(ctx.query.get("type"))
    if query_type not in {"backadj_minimal_sets", "mediator_set"}:
        raise ValueError("rendered_discovery_query_surface requires a discovery seed")
    _render_query_block(ctx)
    legal_targets, readable_labels = _validated_action_surface(ctx)
    labels = ctx.visible_labels()
    label_positions = {label: index for index, label in enumerate(labels)}
    treatment = ctx.resolve_label(ctx.query.get("treatment"))
    outcome = ctx.resolve_label(ctx.query.get("outcome"))
    return RenderedDiscoveryQuerySurface(
        labels=labels,
        domains=tuple(len(ctx.states_for_label(label)) for label in labels),
        query_type=query_type,
        treatment=label_positions[treatment],
        outcome=label_positions[outcome],
        legal_targets=tuple(label_positions[label] for label in legal_targets),
        readable=tuple(label_positions[label] for label in readable_labels),
        hiding_modes=ctx.hiding_modes,
        task_head=str(ctx.task_head.get("head")),
        measure_max=ctx.measure_max,
    )


def render_seed_task_prompt(
    seed: Mapping[str, Any],
    *,
    budget: Budget | None = None,
    measure_max: int | None = None,
) -> str:
    """Render a pinned or assembled seed into a truth-free LLM prompt.

    Only ``visible_schema``, the opaque query, task head, legal action masks,
    and explicit renderer parameters contribute text. The world source, graph,
    CPT, roles, and internal names are never serialized.
    """

    context = _render_context(
        seed,
        budget=budget,
        measure_max=measure_max,
    )
    return _render_prompt(context)


def render_seed_initial_messages(
    seed: Mapping[str, Any],
    *,
    budget: Budget | None = None,
    measure_max: int | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build complete truth-free initial model input for a seed task."""

    return (
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": render_seed_task_prompt(seed, budget=budget, measure_max=measure_max),
        },
    )
