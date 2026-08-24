"""Shared observation-budget type for the generic WorldSpec runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Budget:
    """The sole episode budget, measured in returned scalar observations."""

    max_observations: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_observations, bool)
            or not isinstance(self.max_observations, int)
            or self.max_observations <= 0
        ):
            raise ValueError("max_observations must be a positive integer")


OBSERVATIONS_PER_BANDWIDTH_UNIT = 2048


def budget_for_observation_bandwidth(observation_bandwidth: int) -> Budget:
    """Apply the research contract ``B = observation_bandwidth * 2048``."""

    if (
        isinstance(observation_bandwidth, bool)
        or not isinstance(observation_bandwidth, int)
        or observation_bandwidth <= 0
    ):
        raise ValueError("observation_bandwidth must be a positive integer")
    return Budget(max_observations=observation_bandwidth * OBSERVATIONS_PER_BANDWIDTH_UNIT)
