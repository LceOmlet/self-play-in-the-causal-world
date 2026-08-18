"""Exact candidate seeds and deterministic episode construction."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from .episode import Direction
from .protocol import DEFAULT_LABEL_SEED, VisibleLayout, VisibleTask, factorial_layouts
from .world import CptWorld, OutcomeTape

SEED_SUITE_VERSION = "qn-candidate-seeds-v1"
_IDENTIFIER_DOMAIN = b"cpt-world-seed-identifiers-v1\0"


def _opaque_identifier(kind: str, *parts: str) -> str:
    payload = "\0".join((kind, *parts)).encode("utf-8")
    digest = hashlib.sha256(_IDENTIFIER_DOMAIN + payload).hexdigest()[:24]
    return f"{kind}-{digest}"


@dataclass(frozen=True, slots=True)
class SeedSpec:
    seed_id: str
    difficulty: str
    effect: Fraction

    def __post_init__(self) -> None:
        if not self.seed_id:
            raise ValueError("seed_id must not be empty")
        if self.difficulty not in {"easy", "medium", "hard"}:
            raise ValueError("difficulty must be easy, medium, or hard")
        if not isinstance(self.effect, Fraction) or not 0 < self.effect < Fraction(1, 2):
            raise ValueError("effect must be an exact Fraction in (0, 1/2)")

    @property
    def active_effect(self) -> Fraction:
        return 2 * self.effect

    def world(self, direction: Direction) -> CptWorld:
        return CptWorld(seed_id=self.seed_id, effect=self.effect, direction=direction)


SEED_SPECS: tuple[SeedSpec, ...] = (
    SeedSpec("QN-EASY", "easy", Fraction(2, 5)),
    SeedSpec("QN-MEDIUM", "medium", Fraction(1, 5)),
    SeedSpec("QN-HARD", "hard", Fraction(1, 20)),
)


def seed_by_id(seed_id: str) -> SeedSpec:
    for seed in SEED_SPECS:
        if seed.seed_id == seed_id:
            return seed
    raise KeyError(f"unknown seed_id: {seed_id}")


@dataclass(frozen=True, slots=True)
class SeedEpisode:
    """Internal world, truth-free visible task, and paired outcome-tape key."""

    episode_id: str
    seed: SeedSpec
    world: CptWorld
    task: VisibleTask
    tape_key: str

    @property
    def tape(self) -> OutcomeTape:
        return OutcomeTape(self.tape_key)


def build_candidate_episodes(
    *,
    label_seed: int = DEFAULT_LABEL_SEED,
    layouts: Sequence[VisibleLayout] | None = None,
    replicate_id: str = "r0",
) -> tuple[SeedEpisode, ...]:
    """Cross exact difficulty seeds, both truths, and selected surface layouts.

    The default uses the full surface factorial.  Callers running a pilot may
    pass a preregistered subset; this function does not silently subsample it.
    All truth and surface variants with the same seed/replicate share a tape key
    for paired potential-outcome comparisons.
    """

    selected_layouts = tuple(layouts) if layouts is not None else factorial_layouts(label_seed)
    if not selected_layouts:
        raise ValueError("at least one visible layout is required")
    if len({layout.layout_id for layout in selected_layouts}) != len(selected_layouts):
        raise ValueError("layout IDs must be unique")
    if not isinstance(replicate_id, str) or not replicate_id:
        raise ValueError("replicate_id must be a nonempty string")

    episodes: list[SeedEpisode] = []
    tape_key = _opaque_identifier("tape", SEED_SUITE_VERSION, replicate_id)
    for seed in SEED_SPECS:
        for layout in selected_layouts:
            for direction in Direction:
                episode_id = _opaque_identifier(
                    "episode",
                    SEED_SUITE_VERSION,
                    seed.seed_id,
                    replicate_id,
                    layout.layout_id,
                    direction.value,
                )
                episodes.append(
                    SeedEpisode(
                        episode_id=episode_id,
                        seed=seed,
                        world=seed.world(direction),
                        task=VisibleTask(layout=layout),
                        tape_key=tape_key,
                    )
                )
    return tuple(episodes)
