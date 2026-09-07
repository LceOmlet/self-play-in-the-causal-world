"""Check sufficient conditions for population identification by legal experiments.

This checks the actual world and permissions, not a finite list of sampled
alternatives. The constructive factor-ratio proof is in
docs/population-identification-v1.md. It makes no finite-sample accuracy claim.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .world_space import WorldSpec

INTERACTION_SURFACE_VERSION = "full-nonanchor-identification-v1"


def validate_population_identification(
    world: WorldSpec,
    source: int,
    outcome: int,
    manipulability: Mapping[str, bool],
    readable: Mapping[str, bool],
    observation_bandwidth: int,
) -> None:
    """Fail closed unless the constructive identification theorem applies."""
    if not world.path_exists(source, outcome):
        raise ValueError("Identification requires the public source-ancestor-of-outcome prior")
    for node, name in enumerate(world.variables):
        if bool(manipulability[name]) != (node not in {source, outcome}):
            raise ValueError("Identification requires every non-anchor intervention")
        if not readable[name]:
            raise ValueError("Identification requires all variables to be readable")
    if observation_bandwidth != len(world.variables):
        raise ValueError("Identification requires full joint observation bandwidth")
    for node, rows in world.cpt.items():
        if any(not math.isfinite(float(p)) or p <= 0 for row in rows for p in row):
            raise ValueError("Identification requires strictly positive CPT entries")
        parents = world.parents[node]
        for position, parent in enumerate(parents):
            stride = math.prod(world.domains[p] for p in parents[position + 1 :])
            states = world.domains[parent]
            # The existing CPT owner orders rows with the last parent fastest.
            # Hold all other parents fixed and vary this parent. A structural
            # edge must change the child distribution in at least one context.
            active = any(
                rows[base + offset] != rows[base + offset + value * stride]
                for base in range(0, len(rows), states * stride)
                for offset in range(stride)
                for value in range(1, states)
            )
            if not active:
                raise ValueError(
                    "Identification requires causal minimality; inactive structural edge "
                    f"{world.variables[parent]} -> {world.variables[node]}"
                )
