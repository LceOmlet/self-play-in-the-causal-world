"""Strict terminal-answer parsing and raw task diagnostics.

This module is a task-mainline component, not a hidden audit layer. It parses
the terminal JSON documented by the renderer and reports raw scoring inputs:
squared/absolute effect error for target queries and exact regret for
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
)
from .rewards import TERMINAL_QUALITY_REWARD_VERSION
from .world_space import WorldSpec

_NUMERICAL_TOLERANCE = 1e-9


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
        return {
            "kind": "target_query",
            "effect": _finite_float(value["effect"], field="effect"),
        }

    if query_type == "individual_counterfactual_probability" and head == "target_query":
        if set(value) != {"type", "value"}:
            raise ValueError("answer must contain exactly type and value")
        point = _finite_float(value["value"], field="value")
        if not 0.0 <= point <= 1.0:
            raise ValueError("counterfactual value must lie in [0, 1]")
        return {"kind": "individual_counterfactual_probability", "value": point}

    if query_type == "best_intervention" and head == "decision":
        if set(value) != {"type", "intervention"}:
            raise ValueError("answer must contain exactly type and intervention")
        intervention = value["intervention"]
        if not isinstance(intervention, Mapping) or set(intervention) != {"target", "value"}:
            raise ValueError("intervention must contain exactly target and value")
        target = intervention["target"]
        if not isinstance(target, str):
            raise ValueError("intervention target must be a string")
        if not isinstance(intervention["value"], str):
            raise ValueError("intervention value must be a state_i token")
        internal_target = _resolve_visible_label(seed, target)
        decision_target_value = query.get("decision_target")
        if decision_target_value is None:
            raise ValueError("best_intervention query missing decision_target")
        decision_target = _resolve_seed_anchor(seed, str(decision_target_value))
        if internal_target != decision_target:
            raise ValueError("intervention target is not a final decision candidate")
        state = _state_index(intervention["value"])
        target_index = world.variables.index(internal_target)
        if state >= world.domains[target_index]:
            raise ValueError("intervention value is outside the target domain")
        return {
            "kind": "decision",
            "target": internal_target,
            "value": state,
        }

    if query_type == "backadj_minimal_sets" and head == "discovery":
        if set(value) != {"type", "adjustment_sets"}:
            raise ValueError("answer must contain exactly type and adjustment_sets")
        raw_sets = value["adjustment_sets"]
        if not isinstance(raw_sets, list):
            raise ValueError("adjustment_sets must be a list")
        adjustment_sets: set[frozenset[str]] = set()
        for raw_set in raw_sets:
            if not isinstance(raw_set, list) or any(
                not isinstance(label, str) for label in raw_set
            ):
                raise ValueError("each adjustment set must be a list of labels")
            adjustment_sets.add(frozenset(_resolve_visible_label(seed, label) for label in raw_set))
        return {"kind": "backadj", "adjustment_sets": tuple(adjustment_sets)}

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


def score_terminal_answer(raw: str, seed: Mapping[str, Any], world: WorldSpec) -> Mapping[str, Any]:
    """Return raw, scalarization-free scoring inputs for one terminal answer."""

    parsed = parse_terminal_answer(raw, seed, world)
    truth = compute_query_truth(world, seed)

    if parsed["kind"] == "target_query":
        truth_effect = truth["effect"]
        predicted = Fraction(parsed["effect"])
        absolute_error = abs(predicted - truth_effect)
        return {
            "kind": "target_query",
            "truth": truth_effect,
            "prediction": predicted,
            "abs_error": absolute_error,
            "squared_error": absolute_error**2,
            "reward_scalarization": TERMINAL_QUALITY_REWARD_VERSION,
        }

    if parsed["kind"] == "individual_counterfactual_probability":
        truth_lower = truth["lower"]
        truth_upper = truth["upper"]
        prediction = Fraction(parsed["value"])
        distance = max(truth_lower - prediction, Fraction(0), prediction - truth_upper)
        return {
            "kind": "individual_counterfactual_probability",
            "truth": {"lower": truth_lower, "upper": truth_upper},
            "prediction": prediction,
            "compatible": float(distance) <= _NUMERICAL_TOLERANCE,
            "distance_to_interval": distance,
            "numerical_tolerance": _NUMERICAL_TOLERANCE,
            "reward_scalarization": TERMINAL_QUALITY_REWARD_VERSION,
        }

    if parsed["kind"] == "decision":
        objective = str(seed["query"].get("objective", "minimize"))
        optimal_name = truth["target"]
        optimal_value = int(truth["value"])
        optimal_probability = truth["probability"]
        chosen_name = parsed["target"]
        chosen_value = int(parsed["value"])
        chosen_probability = (
            optimal_probability
            if chosen_name == optimal_name and chosen_value == optimal_value
            else interventional_probability(
                world,
                {world.variables.index(chosen_name): chosen_value},
                _outcome_node_index(seed, world),
                _outcome_state_index(seed, world),
            )
        )
        regret = (
            optimal_probability - chosen_probability
            if objective == "maximize"
            else chosen_probability - optimal_probability
        )
        return {
            "kind": "decision",
            "optimal": {"target": optimal_name, "value": optimal_value},
            "chosen": {"target": chosen_name, "value": chosen_value},
            "optimal_probability": optimal_probability,
            "chosen_probability": chosen_probability,
            "regret": regret,
            "reward_scalarization": TERMINAL_QUALITY_REWARD_VERSION,
        }

    if parsed["kind"] == "backadj":
        truth_sets = set(frozenset(item) for item in truth["adjustment_sets"])
        predicted_sets = set(parsed["adjustment_sets"])
        precision, recall, f1 = _set_f1(truth_sets, predicted_sets)
        return {
            "kind": "backadj",
            "truth": tuple(sorted(tuple(sorted(item)) for item in truth_sets)),
            "prediction": tuple(sorted(tuple(sorted(item)) for item in predicted_sets)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": truth_sets == predicted_sets,
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
