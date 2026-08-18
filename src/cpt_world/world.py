"""Exact binary CPT-world semantics and an action-keyed outcome sampler."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

from .episode import (
    Budget,
    Direction,
    EffectVector,
    EpisodeTruth,
    HardIntervention,
    InterventionCommand,
    Variable,
)

Assignment = tuple[int, int, int]
ASSIGNMENTS: tuple[Assignment, ...] = tuple(product((0, 1), repeat=3))
_ROLE_INDEX = {Variable.FIRST: 0, Variable.SECOND: 1, Variable.ISOLATED: 2}
_HALF = Fraction(1, 2)
_TAPE_DOMAIN = b"cpt-world-outcome-tape-v1\0"


def assignment_value(assignment: Assignment, variable: Variable) -> int:
    return assignment[_ROLE_INDEX[variable]]


def _bernoulli_mass(value: int, probability_one: Fraction) -> Fraction:
    return probability_one if value == 1 else 1 - probability_one


@dataclass(frozen=True, slots=True)
class CptWorld:
    """A hidden world with one active focal edge and one isolated variable."""

    seed_id: str
    effect: Fraction
    direction: Direction

    def __post_init__(self) -> None:
        if not self.seed_id:
            raise ValueError("seed_id must not be empty")
        if not isinstance(self.effect, Fraction):
            raise TypeError("effect must be a Fraction")
        if not 0 < self.effect < _HALF:
            raise ValueError("effect must lie strictly between 0 and 1/2")
        if not isinstance(self.direction, Direction):
            raise TypeError("direction must be a Direction")

    @property
    def low(self) -> Fraction:
        return _HALF - self.effect

    @property
    def high(self) -> Fraction:
        return _HALF + self.effect

    @property
    def truth(self) -> EpisodeTruth:
        magnitude = float(2 * self.effect)
        effects = (
            EffectVector(magnitude, 0.0)
            if self.direction is Direction.FORWARD
            else EffectVector(0.0, magnitude)
        )
        return EpisodeTruth(effects=effects, active_direction=self.direction)


def interventional_distribution(
    world: CptWorld,
    intervention: HardIntervention | None = None,
) -> tuple[tuple[Assignment, Fraction], ...]:
    """Return the exact observational or hard-do distribution.

    This is the single probability-law owner used by both certificates and the
    sampler.  Intervening replaces the target mechanism; it does not condition
    on the target value.
    """

    distribution: list[tuple[Assignment, Fraction]] = []
    for assignment in ASSIGNMENTS:
        if intervention is not None and (
            assignment_value(assignment, intervention.target) != intervention.value
        ):
            distribution.append((assignment, Fraction(0)))
            continue

        first, second, isolated = assignment
        mass = Fraction(1)

        if intervention is None or intervention.target is not Variable.FIRST:
            first_probability = (
                _HALF
                if world.direction is Direction.FORWARD
                else (world.high if second == 1 else world.low)
            )
            mass *= _bernoulli_mass(first, first_probability)

        if intervention is None or intervention.target is not Variable.SECOND:
            second_probability = (
                (world.high if first == 1 else world.low)
                if world.direction is Direction.FORWARD
                else _HALF
            )
            mass *= _bernoulli_mass(second, second_probability)

        if intervention is None or intervention.target is not Variable.ISOLATED:
            mass *= _bernoulli_mass(isolated, _HALF)

        distribution.append((assignment, mass))

    if sum((mass for _, mass in distribution), start=Fraction(0)) != 1:
        raise RuntimeError("internal error: CPT distribution is not normalized")
    return tuple(distribution)


@dataclass(frozen=True, slots=True)
class OutcomeTape:
    """A deterministic potential-outcome stream keyed by intervention arm.

    Each arm has its own index space, so interleaving actions or splitting a
    batch cannot reassign random draws.  This property is essential for fair
    paired comparisons of different policies.
    """

    tape_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.tape_key, str) or not self.tape_key:
            raise ValueError("tape_key must be a nonempty string")
        if len(self.tape_key.encode("utf-8")) >= 2**32:
            raise ValueError("tape_key is too long")

    def uniform(self, intervention: HardIntervention, sample_index: int) -> Fraction:
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise TypeError("sample_index must be an integer")
        if not 0 <= sample_index < 2**64:
            raise ValueError("sample_index must lie in [0, 2^64)")
        key = self.tape_key.encode("utf-8")
        payload = b"".join(
            (
                _TAPE_DOMAIN,
                len(key).to_bytes(4, "big"),
                key,
                intervention.target.value.encode("ascii"),
                bytes((intervention.value,)),
                sample_index.to_bytes(8, "big"),
            )
        )
        digest = hashlib.sha256(payload).digest()
        return Fraction(int.from_bytes(digest, "big"), 2**256)


@dataclass(frozen=True, slots=True)
class SampleBatch:
    intervention: HardIntervention
    start_index: int
    counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.counts) != len(ASSIGNMENTS):
            raise ValueError("counts must contain one entry per binary assignment")
        if (
            isinstance(self.start_index, bool)
            or not isinstance(self.start_index, int)
            or self.start_index < 0
        ):
            raise ValueError("start_index must be a nonnegative integer")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in self.counts
        ):
            raise ValueError("counts must contain nonnegative integers")

    @property
    def sample_count(self) -> int:
        return sum(self.counts)

    def count(self, assignment: Assignment) -> int:
        return self.counts[ASSIGNMENTS.index(assignment)]


def sample_batch(
    world: CptWorld,
    tape: OutcomeTape,
    intervention: HardIntervention,
    *,
    start_index: int,
    sample_count: int,
) -> SampleBatch:
    """Sample a batch from one intervention arm without mutable global RNG."""

    if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 0:
        raise ValueError("start_index must be a nonnegative integer")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    if start_index + sample_count > 2**64:
        raise ValueError("the requested arm stream exceeds its index space")

    distribution = interventional_distribution(world, intervention)
    counts = [0] * len(ASSIGNMENTS)
    for sample_index in range(start_index, start_index + sample_count):
        draw = tape.uniform(intervention, sample_index)
        cumulative = Fraction(0)
        selected = len(ASSIGNMENTS) - 1
        for index, (_, mass) in enumerate(distribution):
            cumulative += mass
            if draw < cumulative:
                selected = index
                break
        counts[selected] += 1

    return SampleBatch(intervention, start_index, tuple(counts))


class EpisodeSampler:
    """Concrete budget/state owner for one hidden-world episode."""

    def __init__(
        self,
        world: CptWorld,
        tape: OutcomeTape,
        budget: Budget | None = None,
    ) -> None:
        if not isinstance(world, CptWorld):
            raise TypeError("world must be a CptWorld")
        if not isinstance(tape, OutcomeTape):
            raise TypeError("tape must be an OutcomeTape")
        if budget is not None and not isinstance(budget, Budget):
            raise TypeError("budget must be a Budget")
        self.world = world
        self.tape = tape
        self.budget = budget if budget is not None else Budget()
        self._rounds_used = 0
        self._samples_used = 0
        self._arm_offsets: dict[HardIntervention, int] = {}
        self._history: list[SampleBatch] = []

    @property
    def rounds_used(self) -> int:
        return self._rounds_used

    @property
    def samples_used(self) -> int:
        return self._samples_used

    @property
    def remaining_rounds(self) -> int:
        return self.budget.max_rounds - self._rounds_used

    @property
    def remaining_samples(self) -> int:
        return self.budget.max_samples - self._samples_used

    @property
    def history(self) -> tuple[SampleBatch, ...]:
        return tuple(self._history)

    def intervene(self, command: InterventionCommand) -> SampleBatch:
        if not isinstance(command, InterventionCommand):
            raise TypeError("command must be an InterventionCommand")
        if command.batch_size not in self.budget.batch_sizes:
            raise ValueError("batch_size is not permitted by this episode")
        if self.remaining_rounds <= 0:
            raise ValueError("the intervention-round budget is exhausted")
        if command.batch_size > self.remaining_samples:
            raise ValueError("the intervention-sample budget is exhausted")

        start_index = self._arm_offsets.get(command.intervention, 0)
        batch = sample_batch(
            self.world,
            self.tape,
            command.intervention,
            start_index=start_index,
            sample_count=command.batch_size,
        )
        self._arm_offsets[command.intervention] = start_index + command.batch_size
        self._rounds_used += 1
        self._samples_used += command.batch_size
        self._history.append(batch)
        return batch
