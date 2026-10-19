"""Lossless snapshot restoration. No sampler is called on the restore path."""
from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = HERE / "frozen" / "owner"
sys.path.insert(0, str(OWNER))
# Fail closed if an embedding process imported another owner before us.
if "cpt_world" in sys.modules:
    if Path(sys.modules["cpt_world"].__file__).resolve().parent != OWNER / "cpt_world":
        raise RuntimeError("A different cpt_world owner is already imported")

from cpt_world import Budget, CPTWorldEnvironment, OutcomeTape, WorldSpec, WorldSpecEpisode
from cpt_world.world_space import legal_world


def canonical(obj) -> bytes:
    """Canonical UTF-8 JSON; Python JSON round-trips finite binary64 exactly."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256(obj) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def _probability_data(value):
    if isinstance(value, Fraction):
        return {"fraction": [value.numerator, value.denominator]}
    return value


def _probability(value):
    if isinstance(value, dict) and set(value) == {"fraction"}:
        return Fraction(*value["fraction"])
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("CPT probability must be numeric or tagged Fraction")
    return value


def world_to_data(world: WorldSpec) -> dict:
    return {"family": world.family, "topology": world.topology,
            "variables": list(world.variables), "domains": list(world.domains),
            "state_names": [list(states) for states in world.state_names],
            "edges": [list(edge) for edge in world.edges],
            "parents": {str(k): list(v) for k, v in world.parents.items()},
            "cpt": {str(k): [[_probability_data(p) for p in row] for row in table]
                    for k, table in world.cpt.items()}}


def world_from_data(data: dict) -> WorldSpec:
    count = len(data["variables"])
    if (not count or any(not isinstance(v, str) for v in data["variables"])
            or any(isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in data["domains"])
            or set(data["cpt"]) != {str(i) for i in range(count)}):
        raise ValueError("invalid variable/domain/CPT index schema")
    if any(len(set(states)) != len(states) or any(not isinstance(s, str) for s in states)
           for states in data["state_names"]):
        raise ValueError("state names must be unique strings within each domain")
    if any(isinstance(i, bool) or not isinstance(i, int)
           for edge in data["edges"] for i in edge):
        raise ValueError("DAG indices must be integers")
    world = WorldSpec(
        family=data["family"], topology=data["topology"],
        variables=tuple(data["variables"]), domains=tuple(data["domains"]),
        state_names=tuple(tuple(states) for states in data["state_names"]),
        edges=tuple(tuple(edge) for edge in data["edges"]),
        parents={int(k): tuple(v) for k, v in data["parents"].items()},
        cpt={int(k): tuple(tuple(_probability(p) for p in row) for row in table)
             for k, table in data["cpt"].items()},
    )
    if not legal_world(world):
        raise ValueError("snapshot is not a legal DAG/domain/CPT world (owner tolerance 1e-12)")
    if canonical(world_to_data(world)) != canonical(data):
        raise ValueError("snapshot world is not in canonical lossless schema")
    return world


def restore_episode(capsule: dict) -> WorldSpecEpisode:
    if capsule["format"] != "cpt-world-capsule-v1":
        raise ValueError("unsupported capsule format")
    # Copy via JSON so two resets can never share mutable seed/truth state.
    data = json.loads(canonical(capsule))
    world = world_from_data(data["world"])
    # Native sampler masks are inserted in WorldSpec variable order. Canonical
    # JSON sorts object keys, so explicitly restore that owner-declared order:
    # rendering includes permission lists in mask iteration order.
    for field in ("readable", "manipulability"):
        mask = data["seed"][field]
        if set(mask) != set(world.variables):
            raise ValueError("seed permission mask does not match world variables")
        data["seed"][field] = {name: mask[name] for name in world.variables}
    budget_data = data.get("budget")
    budget = Budget(**budget_data) if budget_data is not None else None
    truth = data["terminal_truth"]
    # The native mediator owner returns tuple pairs and the scorer places
    # them in a frozenset. JSON arrays must regain that native container type.
    if truth is not None and truth.get("type") == "mediator_set":
        truth["mediators"] = tuple(truth["mediators"])
        truth["order"] = tuple(tuple(pair) for pair in truth["order"])
    return WorldSpecEpisode(
        world, data["seed"], OutcomeTape(data["tape_key"]),
        terminal_truth=truth, budget=budget,
        measure_max=data.get("measure_max"), max_graph_nodes=data.get("max_graph_nodes", 16),
    )


class FrozenEnvironment(CPTWorldEnvironment):
    """Same native act/reward; reset reads the full snapshot instead of sampling."""

    def reset(self, *, capsule: dict, prompt: object, **kwargs) -> str:
        if canonical(prompt) != canonical(capsule["row"]["prompt"]):
            raise ValueError("reset prompt differs from frozen prepared prompt")
        self._episode = restore_episode(capsule)
        return ""
