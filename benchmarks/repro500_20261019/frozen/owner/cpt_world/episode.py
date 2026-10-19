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


OBSERVATION_BUDGET_EXPONENTS = (11, 12, 13, 14)
OBSERVATIONS_PER_BANDWIDTH_UNIT = 1 << min(OBSERVATION_BUDGET_EXPONENTS)


def observations_per_bandwidth_unit(exponent: int) -> int:
    """Resolve one supported power-of-two sample unit from its exponent."""

    if (
        isinstance(exponent, bool)
        or not isinstance(exponent, int)
        or exponent not in OBSERVATION_BUDGET_EXPONENTS
    ):
        raise ValueError(
            f"observation budget exponent must be one of {OBSERVATION_BUDGET_EXPONENTS}"
        )
    return 1 << exponent


def budget_for_observation_bandwidth(
    observation_bandwidth: int,
    *,
    exponent: int = min(OBSERVATION_BUDGET_EXPONENTS),
) -> Budget:
    """Apply ``B = observation_bandwidth * 2**exponent``."""

    if (
        isinstance(observation_bandwidth, bool)
        or not isinstance(observation_bandwidth, int)
        or observation_bandwidth <= 0
    ):
        raise ValueError("observation_bandwidth must be a positive integer")
    return Budget(
        max_observations=(observation_bandwidth * observations_per_bandwidth_unit(exponent))
    )
