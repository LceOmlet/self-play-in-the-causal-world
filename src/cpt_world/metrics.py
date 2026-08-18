"""Frozen terminal diagnostics for the bidirectional effect task.

These continuous metrics are diagnostics, not a training reward.  They consume
canonical effect vectors produced by the protocol decoder and active-direction
certificates produced by the world generator.  This module never parses visible
labels and never infers activity with a threshold or an argmax.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from .episode import EffectVector, EpisodeTruth, opposite

METRICS_SCHEMA_VERSION = "cpt-world-terminal-diagnostics-v1"


@dataclass(frozen=True, slots=True)
class TerminalDiagnostics:
    """The three fixed, equally episode-weighted terminal diagnostics."""

    vector_rmse: float
    active_mae: float
    inactive_mae: float
    n_episodes: int
    schema_version: str = field(init=False, default=METRICS_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        for field_name in ("vector_rmse", "active_mae", "inactive_mae"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{field_name} must be a finite nonnegative number")
        if (
            isinstance(self.n_episodes, bool)
            or not isinstance(self.n_episodes, int)
            or self.n_episodes <= 0
        ):
            raise ValueError("n_episodes must be a positive integer")


def compute_terminal_diagnostics(
    truths: Mapping[str, EpisodeTruth],
    predictions: Mapping[str, EffectVector],
) -> TerminalDiagnostics:
    """Aggregate full-vector, active, and inactive terminal errors.

    ``vector_rmse`` is the RMSE over all ``2N`` scalar components, not the
    average Euclidean norm of the per-episode vectors.  Episode identifiers
    must match exactly so failed or missing episodes cannot be silently
    filtered from one model's profile.
    """

    if not truths:
        raise ValueError("at least one episode is required")
    if any(not isinstance(episode_id, str) or not episode_id for episode_id in truths):
        raise TypeError("truth episode IDs must be nonempty strings")
    if any(not isinstance(episode_id, str) or not episode_id for episode_id in predictions):
        raise TypeError("prediction episode IDs must be nonempty strings")
    if set(truths) != set(predictions):
        missing = sorted(set(truths) - set(predictions))
        extra = sorted(set(predictions) - set(truths))
        raise ValueError(f"prediction episode IDs differ: missing={missing}, extra={extra}")

    squared_errors: list[float] = []
    active_errors: list[float] = []
    inactive_errors: list[float] = []

    for episode_id in sorted(truths):
        truth = truths[episode_id]
        prediction = predictions[episode_id]
        if not isinstance(truth, EpisodeTruth):
            raise TypeError(f"truths[{episode_id!r}] must be an EpisodeTruth")
        if not isinstance(prediction, EffectVector):
            raise TypeError(f"predictions[{episode_id!r}] must be an EffectVector")

        forward_error = prediction.first_to_second - truth.effects.first_to_second
        reverse_error = prediction.second_to_first - truth.effects.second_to_first
        squared_errors.extend((forward_error**2, reverse_error**2))

        active = truth.active_direction
        inactive = opposite(active)
        active_errors.append(abs(prediction.component(active) - truth.effects.component(active)))
        inactive_errors.append(
            abs(prediction.component(inactive) - truth.effects.component(inactive))
        )

    n_episodes = len(truths)
    return TerminalDiagnostics(
        vector_rmse=math.sqrt(math.fsum(squared_errors) / (2 * n_episodes)),
        active_mae=math.fsum(active_errors) / n_episodes,
        inactive_mae=math.fsum(inactive_errors) / n_episodes,
        n_episodes=n_episodes,
    )
