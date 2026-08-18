"""Small, shared data types for CPT-world episodes.

This module contains no sampler, renderer, or scoring logic.  Keeping the
canonical effect vector here lets those components exchange typed values
without learning about one another's implementation details.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class Direction(StrEnum):
    """The active causal direction between the two canonical focal roles."""

    FORWARD = "forward"
    REVERSE = "reverse"


class Variable(StrEnum):
    """Canonical roles hidden behind opaque labels in the rendered task."""

    FIRST = "first"
    SECOND = "second"
    ISOLATED = "isolated"


def _bounded_effect(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not -1.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be finite and lie in [-1, 1]")
    return result


@dataclass(frozen=True, slots=True)
class EffectVector:
    """Both canonical interventional effects returned at the terminal step."""

    first_to_second: float
    second_to_first: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "first_to_second",
            _bounded_effect(self.first_to_second, field_name="first_to_second"),
        )
        object.__setattr__(
            self,
            "second_to_first",
            _bounded_effect(self.second_to_first, field_name="second_to_first"),
        )

    def component(self, direction: Direction) -> float:
        if direction is Direction.FORWARD:
            return self.first_to_second
        if direction is Direction.REVERSE:
            return self.second_to_first
        raise TypeError("direction must be a Direction")


@dataclass(frozen=True, slots=True)
class EpisodeTruth:
    """Ground-truth effects plus the generator-certified active coordinate."""

    effects: EffectVector
    active_direction: Direction

    def __post_init__(self) -> None:
        if not isinstance(self.active_direction, Direction):
            raise TypeError("active_direction must be a Direction")
        active = self.effects.component(self.active_direction)
        inactive = self.effects.component(opposite(self.active_direction))
        if active == 0.0:
            raise ValueError("the certified active effect must be nonzero")
        if inactive != 0.0:
            raise ValueError("the certified inactive effect must be exactly zero")


def opposite(direction: Direction) -> Direction:
    if direction is Direction.FORWARD:
        return Direction.REVERSE
    if direction is Direction.REVERSE:
        return Direction.FORWARD
    raise TypeError("direction must be a Direction")


@dataclass(frozen=True, slots=True)
class HardIntervention:
    target: Variable
    value: int

    def __post_init__(self) -> None:
        if not isinstance(self.target, Variable):
            raise TypeError("target must be a Variable")
        if isinstance(self.value, bool) or self.value not in (0, 1):
            raise ValueError("an intervention value must be 0 or 1")


@dataclass(frozen=True, slots=True)
class InterventionCommand:
    intervention: HardIntervention
    batch_size: int

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError("batch_size must be an integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True, slots=True)
class TerminalAnswer:
    effects: EffectVector


@dataclass(frozen=True, slots=True)
class Budget:
    max_rounds: int = 4
    max_samples: int = 64
    batch_sizes: tuple[int, ...] = (4, 8, 16, 32)

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_rounds, bool)
            or not isinstance(self.max_rounds, int)
            or self.max_rounds <= 0
        ):
            raise ValueError("max_rounds must be a positive integer")
        if (
            isinstance(self.max_samples, bool)
            or not isinstance(self.max_samples, int)
            or self.max_samples <= 0
        ):
            raise ValueError("max_samples must be a positive integer")
        if not self.batch_sizes:
            raise ValueError("batch_sizes must not be empty")
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in self.batch_sizes
        ):
            raise ValueError("batch_sizes must contain positive integers")
        if tuple(sorted(set(self.batch_sizes))) != self.batch_sizes:
            raise ValueError("batch_sizes must be sorted and contain no duplicates")
        if self.batch_sizes[-1] > self.max_samples:
            raise ValueError("batch_sizes cannot exceed max_samples")
