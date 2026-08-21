"""Shared episode-budget type for the generic WorldSpec runtime."""

from __future__ import annotations

from dataclasses import dataclass


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
