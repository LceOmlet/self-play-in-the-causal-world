"""Frozen scalar rewards derived from terminal task diagnostics.

``task_scoring`` remains the sole owner of parsing, truth lookup, and raw
diagnostics.  This module consumes those exact diagnostics without reopening
the model answer or recomputing task truth.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from math import sqrt
from typing import Any

from .episode import OBSERVATIONS_PER_BANDWIDTH_UNIT

TERMINAL_QUALITY_REWARD_VERSION = "terminal-quality-v6"
UNFINISHED_TERMINAL_QUALITY = Fraction(0)
TERMINAL_SAMPLING_RESOLUTION = Fraction.from_float(
    1.0 / sqrt(OBSERVATIONS_PER_BANDWIDTH_UNIT)
)


def _fraction_metric(
    score: Mapping[str, Any],
    field: str,
    *,
    upper: Fraction = Fraction(1),
) -> Fraction:
    value = score.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise ValueError(f"terminal diagnostic {field} must be a real number")
    result = Fraction(value)
    if not 0 <= result <= upper:
        raise ValueError(f"terminal diagnostic {field} must lie in [0, {upper}]")
    return result


def _adjustment_family(value: object, *, field: str) -> tuple[frozenset[str], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"terminal diagnostic {field} must be an adjustment-set family")
    family: list[frozenset[str]] = []
    seen: set[frozenset[str]] = set()
    for raw_set in value:
        if isinstance(raw_set, (str, bytes)) or not isinstance(raw_set, Sequence):
            raise ValueError(f"terminal diagnostic {field} contains a non-set entry")
        if any(not isinstance(name, str) for name in raw_set):
            raise ValueError(f"terminal diagnostic {field} contains a non-string variable")
        adjustment_set = frozenset(raw_set)
        if len(adjustment_set) != len(raw_set):
            raise ValueError(f"terminal diagnostic {field} contains a duplicate variable")
        if adjustment_set in seen:
            raise ValueError(f"terminal diagnostic {field} contains a duplicate adjustment set")
        seen.add(adjustment_set)
        family.append(adjustment_set)
    return tuple(family)


def _set_dice(left: frozenset[str], right: frozenset[str]) -> Fraction:
    if not left and not right:
        return Fraction(1)
    return Fraction(2 * len(left & right), len(left) + len(right))


def _minimum_assignment_cost(costs: tuple[tuple[Fraction, ...], ...]) -> Fraction:
    """Return the exact rectangular Hungarian assignment cost for rows <= columns."""

    row_count = len(costs)
    if row_count == 0:
        return Fraction(0)
    column_count = len(costs[0])
    if row_count > column_count or any(len(row) != column_count for row in costs):
        raise ValueError("assignment costs must be rectangular with rows <= columns")

    row_potential = [Fraction(0)] * (row_count + 1)
    column_potential = [Fraction(0)] * (column_count + 1)
    column_match = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)

    for row in range(1, row_count + 1):
        column_match[0] = row
        current_column = 0
        minimum_slack: list[Fraction | None] = [None] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[current_column] = True
            current_row = column_match[current_column]
            delta: Fraction | None = None
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                slack = (
                    costs[current_row - 1][column - 1]
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if minimum_slack[column] is None or slack < minimum_slack[column]:
                    minimum_slack[column] = slack
                    predecessor[column] = current_column
                if delta is None or minimum_slack[column] < delta:
                    delta = minimum_slack[column]
                    next_column = column
            if delta is None:
                raise ValueError("assignment matrix has no augmenting column")
            for column in range(column_count + 1):
                if used[column]:
                    row_potential[column_match[column]] += delta
                    column_potential[column] -= delta
                elif column > 0:
                    slack = minimum_slack[column]
                    if slack is None:
                        raise ValueError("assignment matrix contains an unvisited column")
                    minimum_slack[column] = slack - delta
            current_column = next_column
            if column_match[current_column] == 0:
                break
        while True:
            previous_column = predecessor[current_column]
            column_match[current_column] = column_match[previous_column]
            current_column = previous_column
            if current_column == 0:
                break

    return sum(
        (
            costs[column_match[column] - 1][column - 1]
            for column in range(1, column_count + 1)
            if column_match[column] != 0
        ),
        start=Fraction(0),
    )


def soft_adjustment_family_f1(predicted: object, truth: object) -> Fraction:
    """Score two adjustment-set families by optimal one-to-one Dice matching."""

    predicted_family = _adjustment_family(predicted, field="prediction")
    truth_family = _adjustment_family(truth, field="truth")
    if not predicted_family and not truth_family:
        return Fraction(1)
    if not predicted_family or not truth_family:
        return Fraction(0)

    if len(predicted_family) <= len(truth_family):
        rows = predicted_family
        columns = truth_family
    else:
        rows = truth_family
        columns = predicted_family
    costs = tuple(tuple(Fraction(1) - _set_dice(left, right) for right in columns) for left in rows)
    matched_similarity = Fraction(len(rows)) - _minimum_assignment_cost(costs)
    return 2 * matched_similarity / (len(predicted_family) + len(truth_family))


def _shortcut_calibrated_quality(
    score: Mapping[str, Any],
    *,
    error_field: str,
    shortcut_error_field: str,
    absolute_error_upper: Fraction,
) -> Fraction:
    """Return continuous accuracy at the fixed-budget sampling resolution."""

    error = _fraction_metric(score, error_field, upper=absolute_error_upper)
    raw_shortcut_error = score.get(shortcut_error_field)
    if raw_shortcut_error is None:
        return Fraction(1) - error / absolute_error_upper
    shortcut_error = _fraction_metric(
        {shortcut_error_field: raw_shortcut_error},
        shortcut_error_field,
        upper=absolute_error_upper,
    )
    scale = shortcut_error + TERMINAL_SAMPLING_RESOLUTION
    return scale / (scale + error)


def terminal_quality_reward(score: Mapping[str, Any]) -> Fraction:
    """Map one owner-produced terminal diagnostic record to the current reward."""

    if not isinstance(score, Mapping):
        raise TypeError("score must be a terminal diagnostic mapping")
    kind = score.get("kind")
    if kind == "target_query":
        quality = _shortcut_calibrated_quality(
            score,
            error_field="total_variation_error",
            shortcut_error_field="observational_shortcut_error",
            absolute_error_upper=Fraction(2),
        )
    elif kind == "counterfactual_roi":
        quality = _shortcut_calibrated_quality(
            score,
            error_field="mean_absolute_endpoint_error",
            shortcut_error_field="observational_shortcut_error",
            absolute_error_upper=Fraction(1),
        )
    elif kind == "decision":
        quality = _shortcut_calibrated_quality(
            score,
            error_field="regret",
            shortcut_error_field="observational_shortcut_error",
            absolute_error_upper=Fraction(1),
        )
    elif kind == "backadj":
        quality = soft_adjustment_family_f1(score.get("prediction"), score.get("truth"))
    elif kind == "mediator":
        quality = (_fraction_metric(score, "mediator_f1") + _fraction_metric(score, "order_f1")) / 2
    else:
        raise ValueError(f"unsupported terminal diagnostic kind: {kind!r}")
    if not 0 <= quality <= 1:
        raise ValueError("terminal quality must lie in [0, 1]")
    return quality
