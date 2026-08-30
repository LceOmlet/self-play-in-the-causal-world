"""Frozen scalar rewards derived from terminal task diagnostics.

``task_scoring`` remains the sole owner of parsing, truth lookup, and raw
diagnostics.  This module consumes those exact diagnostics without reopening
the model answer or recomputing task truth.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from math import sqrt
from typing import Any

from .episode import OBSERVATIONS_PER_BANDWIDTH_UNIT

TERMINAL_QUALITY_REWARD_VERSION = "terminal-quality-v9"
UNFINISHED_TERMINAL_QUALITY = Fraction(0)
TERMINAL_SAMPLING_RESOLUTION = Fraction.from_float(1.0 / sqrt(OBSERVATIONS_PER_BANDWIDTH_UNIT))


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
            error_field="normalized_regret",
            shortcut_error_field="observational_shortcut_normalized_regret",
            absolute_error_upper=Fraction(1),
        )
    elif kind == "backadj":
        edit_distance = score.get("edit_distance")
        if isinstance(edit_distance, bool) or not isinstance(edit_distance, int):
            raise ValueError("terminal diagnostic edit_distance must be a nonnegative integer")
        if edit_distance < 0:
            raise ValueError("terminal diagnostic edit_distance must be a nonnegative integer")
        quality = Fraction(1, 1 + edit_distance)
    elif kind == "mediator":
        quality = (_fraction_metric(score, "mediator_f1") + _fraction_metric(score, "order_f1")) / 2
    else:
        raise ValueError(f"unsupported terminal diagnostic kind: {kind!r}")
    if not 0 <= quality <= 1:
        raise ValueError("terminal quality must lie in [0, 1]")
    return quality
