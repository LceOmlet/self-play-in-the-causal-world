"""Strict terminal-answer parsing and raw task diagnostics.

This module is a task-mainline component, not a hidden audit layer. It parses
the terminal JSON documented by the renderer and reports raw scoring inputs:
full-vector treatment-effect error for ATE and exact regret for
single-intervention decisions. Reward scalarization is intentionally not
performed here.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from .query_truth import (
    compute_query_truth,
    interventional_probability,
    nearest_backdoor_adjustment_set,
    worldspec_projected_interventional_distribution,
)
from .rewards import TERMINAL_QUALITY_REWARD_VERSION
from .world_space import WorldSpec

_ATE_VECTOR_TOLERANCE = 1e-6


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


def _visible_label_map(seed: Mapping[str, Any]) -> dict[str, str]:
    visible_schema = seed.get("visible_schema")
    if not isinstance(visible_schema, Mapping):
        raise ValueError("seed missing visible_schema")
    label_map = visible_schema.get("variable_labels")
    if not isinstance(label_map, Mapping):
        raise ValueError("seed missing variable_labels")
    return {str(internal): str(label) for internal, label in label_map.items()}


def _resolve_visible_label(seed: Mapping[str, Any], label: str) -> str:
    inverse = {
        str(visible): str(internal) for internal, visible in _visible_label_map(seed).items()
    }
    if label in inverse:
        return inverse[label]
    raise ValueError(f"unknown visible label {label}")


def _resolve_seed_anchor(seed: Mapping[str, Any], label: str) -> str:
    inverse = {
        str(visible): str(internal) for internal, visible in _visible_label_map(seed).items()
    }
    if label in inverse:
        return inverse[label]
    internal_names = set(_visible_label_map(seed))
    if label in internal_names:
        return label
    raise ValueError(f"unknown visible label {label}")


def _state_index(text: str) -> int:
    if text.startswith("state_"):
        try:
            suffix = text.removeprefix("state_")
            index = int(suffix)
        except ValueError as error:
            raise ValueError(f"invalid state token {text}") from error
        if index < 0 or not suffix.isascii() or not suffix.isdigit() or str(index) != suffix:
            raise ValueError(f"invalid state token {text}")
        return index
    raise ValueError(f"state value must be a state_i token, got {text!r}")


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if not -1.0 <= result <= 1.0:
        raise ValueError(f"{field} must lie in [-1, 1]")
    return result


def _probability_float(value: object, *, field: str) -> float:
    result = _finite_float(value, field=field)
    if result < 0.0:
        raise ValueError(f"{field} must lie in [0, 1]")
    return result


def parse_terminal_answer(raw: str, seed: Mapping[str, Any], world: WorldSpec) -> Mapping[str, Any]:
    """Parse the terminal JSON documented for a seed's query/task head."""

    query = seed.get("query")
    task_head = seed.get("task_head")
    if not isinstance(query, Mapping) or not isinstance(task_head, Mapping):
        raise ValueError("seed must contain query and task_head")
    query_type = str(query.get("type"))
    head = str(task_head.get("head"))
    value = _json_object(raw)
    if value.get("type") != "answer":
        raise ValueError("terminal response type must be answer")

    if query_type == "ate" and head == "target_query":
        if set(value) != {"type", "effect"}:
            raise ValueError("answer must contain exactly type and effect")
        effect_value = value["effect"]
        if not isinstance(effect_value, Mapping):
            raise ValueError("effect must map every outcome state to a number")
        outcome_node = _outcome_node_index(seed, world)
        state_tokens = tuple(f"state_{state}" for state in range(world.domains[outcome_node]))
        if set(effect_value) != set(state_tokens):
            raise ValueError("effect must contain every outcome state exactly once")
        effect = tuple(
            _finite_float(effect_value[state], field=f"effect.{state}") for state in state_tokens
        )
        if abs(sum(effect)) > _ATE_VECTOR_TOLERANCE:
            raise ValueError("effect components must sum to zero")
        if sum(component for component in effect if component > 0) > 1 + _ATE_VECTOR_TOLERANCE:
            raise ValueError("positive effect components must sum to at most one")
        return {
            "kind": "target_query",
            "effect": effect,
        }

    if query_type == "individual_counterfactual_probability" and head == "target_query":
        if set(value) != {"type", "lower", "upper"}:
            raise ValueError("answer must contain exactly type, lower, and upper")
        lower = _finite_float(value["lower"], field="lower")
        upper = _finite_float(value["upper"], field="upper")
        if not 0.0 <= lower <= upper <= 1.0:
            raise ValueError("counterfactual ROI must satisfy 0 <= lower <= upper <= 1")
        return {"kind": "counterfactual_roi", "lower": lower, "upper": upper}

    if query_type == "best_intervention" and head == "decision":
        if set(value) != {"type", "value"}:
            raise ValueError("answer must contain exactly type and value")
        decision_target_value = query.get("decision_target")
        if decision_target_value is None:
            raise ValueError("best_intervention query missing decision_target")
        decision_target = _resolve_seed_anchor(seed, str(decision_target_value))
        target_index = world.variables.index(decision_target)
        raw_value = value["value"]
        if not isinstance(raw_value, str):
            raise ValueError("value must be a state_i token")
        chosen_value = _state_index(raw_value)
        if chosen_value >= world.domains[target_index]:
            raise ValueError("decision value is outside the deployment domain")
        return {
            "kind": "decision",
            "target": decision_target,
            "value": chosen_value,
        }

    if query_type == "backadj_minimal_sets" and head == "discovery":
        if set(value) != {"type", "adjustment_set"}:
            raise ValueError("answer must contain exactly type and adjustment_set")
        raw_set = value["adjustment_set"]
        if not isinstance(raw_set, list) or any(not isinstance(label, str) for label in raw_set):
            raise ValueError("adjustment_set must be a list of labels")
        if len(set(raw_set)) != len(raw_set):
            raise ValueError("adjustment_set must not contain duplicate labels")
        adjustment_set = frozenset(_resolve_visible_label(seed, label) for label in raw_set)
        treatment = _resolve_seed_anchor(seed, str(query.get("treatment")))
        outcome = _resolve_seed_anchor(seed, str(query.get("outcome")))
        if treatment in adjustment_set or outcome in adjustment_set:
            raise ValueError("adjustment_set must exclude treatment and outcome")
        return {"kind": "backadj", "adjustment_set": adjustment_set}

    if query_type == "mediator_set" and head == "discovery":
        if set(value) != {"type", "mediators", "order"}:
            raise ValueError("answer must contain exactly type, mediators, and order")
        raw_mediators = value["mediators"]
        raw_order = value["order"]
        if not isinstance(raw_mediators, list) or any(
            not isinstance(label, str) for label in raw_mediators
        ):
            raise ValueError("mediators must be a list of labels")
        if not isinstance(raw_order, list):
            raise ValueError("order must be a list")
        mediators = frozenset(_resolve_visible_label(seed, label) for label in raw_mediators)
        order: set[tuple[str, str]] = set()
        for pair in raw_order:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("each order entry must be a [source, target] pair")
            if not isinstance(pair[0], str) or not isinstance(pair[1], str):
                raise ValueError("order labels must be strings")
            order.add(
                (
                    _resolve_visible_label(seed, pair[0]),
                    _resolve_visible_label(seed, pair[1]),
                )
            )
        return {"kind": "mediator", "mediators": mediators, "order": frozenset(order)}

    raise NotImplementedError(f"terminal parser not implemented for {query_type}/{head}")


def _outcome_node_index(seed: Mapping[str, Any], world: WorldSpec) -> int:
    query = seed.get("query")
    if not isinstance(query, Mapping):
        raise ValueError("seed missing query")
    outcome_value = query.get("outcome")
    if outcome_value is None:
        raise ValueError("query missing outcome")
    outcome_name = _resolve_seed_anchor(seed, str(outcome_value))
    return world.variables.index(outcome_name)


def _outcome_state_index(seed: Mapping[str, Any], world: WorldSpec) -> int:
    query = seed.get("query")
    if not isinstance(query, Mapping):
        raise ValueError("seed missing query")
    outcome_index = _outcome_node_index(seed, world)
    state_value = query.get("target_state", query.get("outcome_state", 1))
    if isinstance(state_value, str) and state_value.startswith("state_"):
        return _state_index(state_value)
    if isinstance(state_value, int) and not isinstance(state_value, bool):
        return state_value
    state_names = world.state_names[outcome_index]
    if state_value in state_names:
        return state_names.index(state_value)
    return 1


def _query_node_index(seed: Mapping[str, Any], world: WorldSpec, field: str) -> int:
    query = seed.get("query")
    if not isinstance(query, Mapping) or query.get(field) is None:
        raise ValueError(f"query missing {field}")
    return world.variables.index(_resolve_seed_anchor(seed, str(query[field])))


def _query_state_index(
    seed: Mapping[str, Any],
    world: WorldSpec,
    node: int,
    field: str,
) -> int:
    query = seed.get("query")
    if not isinstance(query, Mapping) or query.get(field) is None:
        raise ValueError(f"query missing {field}")
    value = query[field]
    if isinstance(value, str) and value.startswith("state_"):
        state = _state_index(value)
    elif isinstance(value, int) and not isinstance(value, bool):
        state = value
    elif value in world.state_names[node]:
        state = world.state_names[node].index(value)
    else:
        raise ValueError(f"query field {field} has an unknown state")
    if not 0 <= state < world.domains[node]:
        raise ValueError(f"query field {field} is outside its state domain")
    return state


def _observational_conditional_distributions(
    world: WorldSpec,
    condition_node: int,
    outcome_node: int,
) -> tuple[tuple[Fraction, ...] | None, ...]:
    """Return every P(outcome | condition=state) from one projected law."""

    law = worldspec_projected_interventional_distribution(
        world,
        {},
        (condition_node, outcome_node),
    )
    masses = [
        [Fraction(0) for _ in range(world.domains[outcome_node])]
        for _ in range(world.domains[condition_node])
    ]
    totals = [Fraction(0) for _ in range(world.domains[condition_node])]
    for assignment, probability in law:
        mass = Fraction(probability)
        masses[assignment[0]][assignment[1]] += mass
        totals[assignment[0]] += mass
    return tuple(
        None if total == 0 else tuple(mass / total for mass in row)
        for row, total in zip(masses, totals, strict=True)
    )


def _distance_to_interval(value: Fraction, interval: tuple[Fraction, Fraction]) -> Fraction:
    return max(interval[0] - value, Fraction(0), value - interval[1])


def _set_f1(truth: set[Any], predicted: set[Any]) -> tuple[Fraction, Fraction, Fraction]:
    if not predicted and not truth:
        return Fraction(1), Fraction(1), Fraction(1)
    if not predicted or not truth:
        return Fraction(0), Fraction(0), Fraction(0)
    overlap = len(truth & predicted)
    precision = Fraction(overlap, len(predicted))
    recall = Fraction(overlap, len(truth))
    if precision + recall == 0:
        return Fraction(0), Fraction(0), Fraction(0)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _pair_f1(
    truth: frozenset[tuple[str, str]],
    predicted: frozenset[tuple[str, str]],
) -> tuple[Fraction, Fraction, Fraction]:
    truth_pairs = set(truth)
    predicted_pairs = set(predicted)
    if not predicted_pairs and not truth_pairs:
        return Fraction(1), Fraction(1), Fraction(1)
    if not predicted_pairs or not truth_pairs:
        return Fraction(0), Fraction(0), Fraction(0)
    overlap = len(truth_pairs & predicted_pairs)
    precision = Fraction(overlap, len(predicted_pairs))
    recall = Fraction(overlap, len(truth_pairs))
    if precision + recall == 0:
        return Fraction(0), Fraction(0), Fraction(0)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def score_terminal_answer(
    raw: str,
    seed: Mapping[str, Any],
    world: WorldSpec,
    *,
    terminal_truth: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Return raw, scalarization-free scoring inputs for one terminal answer."""

    parsed = parse_terminal_answer(raw, seed, world)
    truth = compute_query_truth(world, seed) if terminal_truth is None else terminal_truth
    query = seed.get("query")
    query_type = str(query.get("type")) if isinstance(query, Mapping) else ""
    if str(truth.get("type")) != query_type:
        raise ValueError("cached terminal truth does not match the task query type")

    if parsed["kind"] == "target_query":
        truth_effect = tuple(Fraction(component) for component in truth["effect"])
        predicted = tuple(Fraction(component) for component in parsed["effect"])
        if len(predicted) != len(truth_effect):
            raise ValueError("ATE prediction and truth domains do not match")
        component_errors = tuple(
            abs(predicted_component - truth_component)
            for predicted_component, truth_component in zip(predicted, truth_effect, strict=True)
        )
        l1_error = sum(component_errors, start=Fraction(0))
        squared_error = sum(
            (error**2 for error in component_errors),
            start=Fraction(0),
        ) / len(component_errors)
        treatment_index = _query_node_index(seed, world, "treatment")
        outcome_index = _outcome_node_index(seed, world)
        baseline_state = _query_state_index(seed, world, treatment_index, "baseline_value")
        treatment_state = _query_state_index(seed, world, treatment_index, "treatment_value")
        observational_laws = _observational_conditional_distributions(
            world, treatment_index, outcome_index
        )
        observed_baseline = observational_laws[baseline_state]
        observed_treatment = observational_laws[treatment_state]
        observational_effect = None
        observational_shortcut_error = None
        if observed_baseline is not None and observed_treatment is not None:
            observational_effect = tuple(
                treated - baseline
                for treated, baseline in zip(observed_treatment, observed_baseline, strict=True)
            )
            observational_shortcut_error = (
                sum(
                    (
                        abs(observed - causal)
                        for observed, causal in zip(observational_effect, truth_effect, strict=True)
                    ),
                    start=Fraction(0),
                )
                / 2
            )
        return {
            "kind": "target_query",
            "truth": truth_effect,
            "prediction": predicted,
            "component_errors": component_errors,
            "l1_error": l1_error,
            "total_variation_error": l1_error / 2,
            "squared_error": squared_error,
            "observational_shortcut": observational_effect,
            "observational_shortcut_error": observational_shortcut_error,
            "reward_scalarization": TERMINAL_QUALITY_REWARD_VERSION,
        }

    if parsed["kind"] == "counterfactual_roi":
        truth_lower = Fraction(truth["lower"])
        truth_upper = Fraction(truth["upper"])
        certification = str(truth.get("certification", "exact"))
        endpoint_error = float(truth.get("endpoint_error", 0.0))
        if not math.isfinite(endpoint_error) or endpoint_error < 0.0:
            raise ValueError("counterfactual endpoint error must be finite and nonnegative")
        predicted_lower = Fraction(parsed["lower"])
        predicted_upper = Fraction(parsed["upper"])
        error_bound = Fraction(endpoint_error)
        certified_lower_range = (
            truth_lower,
            min(Fraction(1), truth_lower + error_bound),
        )
        certified_upper_range = (
            max(Fraction(0), truth_upper - error_bound),
            truth_upper,
        )
        lower_error = _distance_to_interval(predicted_lower, certified_lower_range)
        upper_error = _distance_to_interval(predicted_upper, certified_upper_range)
        mean_absolute_endpoint_error = (lower_error + upper_error) / 2
        mean_squared_endpoint_error = (lower_error**2 + upper_error**2) / 2
        treatment_index = _query_node_index(seed, world, "treatment")
        outcome_index = _outcome_node_index(seed, world)
        factual_value = _query_state_index(seed, world, treatment_index, "factual_value")
        counterfactual_value = _query_state_index(
            seed, world, treatment_index, "counterfactual_value"
        )
        factual_outcome_state = _query_state_index(
            seed, world, outcome_index, "factual_outcome_state"
        )
        target_outcome_state = _query_state_index(seed, world, outcome_index, "outcome_state")
        observational_laws = _observational_conditional_distributions(
            world, treatment_index, outcome_index
        )
        factual_law = observational_laws[factual_value]
        counterfactual_law = observational_laws[counterfactual_value]
        observational_roi = None
        observational_shortcut_error = None
        if factual_law is not None and counterfactual_law is not None:
            factual_probability = factual_law[factual_outcome_state]
            counterfactual_probability = counterfactual_law[target_outcome_state]
            if factual_probability > 0:
                observational_lower = (
                    max(Fraction(0), factual_probability + counterfactual_probability - 1)
                    / factual_probability
                )
                observational_upper = (
                    min(factual_probability, counterfactual_probability) / factual_probability
                )
                observational_roi = {
                    "lower": observational_lower,
                    "upper": observational_upper,
                }
                observational_shortcut_error = (
                    _distance_to_interval(observational_lower, certified_lower_range)
                    + _distance_to_interval(observational_upper, certified_upper_range)
                ) / 2
        return {
            "kind": "counterfactual_roi",
            "truth": {
                "lower": truth_lower,
                "upper": truth_upper,
                "certification": certification,
                "endpoint_error": endpoint_error,
                "certified_lower_range": certified_lower_range,
                "certified_upper_range": certified_upper_range,
            },
            "prediction": {"lower": predicted_lower, "upper": predicted_upper},
            "lower_endpoint_error": lower_error,
            "upper_endpoint_error": upper_error,
            "mean_absolute_endpoint_error": mean_absolute_endpoint_error,
            "mean_squared_endpoint_error": mean_squared_endpoint_error,
            "observational_shortcut": observational_roi,
            "observational_shortcut_error": observational_shortcut_error,
            "certification": certification,
            "endpoint_error": endpoint_error,
            "reward_scalarization": TERMINAL_QUALITY_REWARD_VERSION,
        }

    if parsed["kind"] == "decision":
        objective = str(seed["query"].get("objective", "minimize"))
        optimal_name = truth["target"]
        optimal_value = int(truth["value"])
        optimal_probability = truth["probability"]
        chosen_name = parsed["target"]
        chosen_value = int(parsed["value"])
        target_index = world.variables.index(chosen_name)
        outcome_index = _outcome_node_index(seed, world)
        outcome_state = _outcome_state_index(seed, world)
        candidate_probabilities = tuple(
            interventional_probability(
                world,
                {target_index: state},
                outcome_index,
                outcome_state,
            )
            for state in range(world.domains[target_index])
        )
        chosen_probability = candidate_probabilities[chosen_value]
        minimum_probability = min(candidate_probabilities)
        maximum_probability = max(candidate_probabilities)
        probability_span = maximum_probability - minimum_probability
        regret = (
            optimal_probability - chosen_probability
            if objective == "maximize"
            else chosen_probability - optimal_probability
        )
        if probability_span == 0:
            normalized_regret = Fraction(0)
        else:
            normalized_regret = Fraction(regret) / Fraction(probability_span)
        observational_laws = _observational_conditional_distributions(
            world, target_index, outcome_index
        )
        observed_probabilities: list[Fraction] = []
        observational_shortcut_error = None
        observational_shortcut_normalized_regret = None
        observational_choice = None
        for law in observational_laws:
            if law is None:
                break
            observed_probabilities.append(law[outcome_state])
        if len(observed_probabilities) == len(candidate_probabilities):
            choose = min if objective == "minimize" else max
            observational_optimum = choose(observed_probabilities)
            observational_choice = next(
                state
                for state, probability in enumerate(observed_probabilities)
                if probability == observational_optimum
            )
            observational_causal_probability = candidate_probabilities[observational_choice]
            observational_shortcut_error = (
                optimal_probability - observational_causal_probability
                if objective == "maximize"
                else observational_causal_probability - optimal_probability
            )
            observational_shortcut_normalized_regret = (
                Fraction(0)
                if probability_span == 0
                else Fraction(observational_shortcut_error) / Fraction(probability_span)
            )
        return {
            "kind": "decision",
            "optimal": {"target": optimal_name, "value": optimal_value},
            "chosen": {"target": chosen_name, "value": chosen_value},
            "prediction": {"target": chosen_name, "value": chosen_value},
            "optimal_probability": optimal_probability,
            "chosen_probability": chosen_probability,
            "candidate_probabilities": candidate_probabilities,
            "minimum_probability": minimum_probability,
            "maximum_probability": maximum_probability,
            "probability_span": probability_span,
            "regret": regret,
            "normalized_regret": normalized_regret,
            "observational_shortcut": (
                tuple(observed_probabilities)
                if len(observed_probabilities) == len(candidate_probabilities)
                else None
            ),
            "observational_choice": observational_choice,
            "observational_shortcut_error": observational_shortcut_error,
            "observational_shortcut_normalized_regret": (observational_shortcut_normalized_regret),
            "optimal_action": regret == 0,
            "reward_scalarization": TERMINAL_QUALITY_REWARD_VERSION,
        }

    if parsed["kind"] == "backadj":
        treatment_index = _query_node_index(seed, world, "treatment")
        outcome_index = _outcome_node_index(seed, world)
        predicted_set = frozenset(parsed["adjustment_set"])
        edit_distance, nearest_set = nearest_backdoor_adjustment_set(
            world,
            treatment_index,
            outcome_index,
            predicted_set,
        )
        return {
            "kind": "backadj",
            "prediction": tuple(sorted(predicted_set)),
            "nearest_valid_adjustment_set": nearest_set,
            "edit_distance": edit_distance,
            "valid_adjustment_set": edit_distance == 0,
            "reward_scalarization": TERMINAL_QUALITY_REWARD_VERSION,
        }

    if parsed["kind"] == "mediator":
        truth_mediators = frozenset(truth["mediators"])
        truth_order = frozenset(truth["order"])
        mediator_precision, mediator_recall, mediator_f1 = _set_f1(
            set(truth_mediators), set(parsed["mediators"])
        )
        order_precision, order_recall, order_f1 = _pair_f1(truth_order, parsed["order"])
        return {
            "kind": "mediator",
            "mediator_precision": mediator_precision,
            "mediator_recall": mediator_recall,
            "mediator_f1": mediator_f1,
            "order_precision": order_precision,
            "order_recall": order_recall,
            "order_f1": order_f1,
            "mediators_exact_match": truth_mediators == parsed["mediators"],
            "order_exact_match": truth_order == parsed["order"],
            "reward_scalarization": TERMINAL_QUALITY_REWARD_VERSION,
        }

    raise NotImplementedError("scorer not implemented for this parsed answer kind")
