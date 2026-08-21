"""Pinned candidate-seed manifest types and validation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import query_task_compatible

CANDIDATE_SEED_MANIFEST_SCHEMA = "cpt-world-seed-manifest-v1"
_CANDIDATE_SEED_MANIFEST_RELPATH = Path("data") / "seeds" / "candidate-v1.json"
ALLOWED_CANDIDATE_QUERY_TYPES = frozenset(
    {
        "ate",
        "counterfactual_transition_bounds",
        "backadj_minimal_sets",
        "best_intervention",
        "mediator_set",
    }
)
ALLOWED_CANDIDATE_TASK_HEADS = frozenset({"target_query", "discovery", "decision"})


@dataclass(frozen=True, slots=True)
class CandidateSeedSpec:
    """A real-world candidate seed pinned from data/seeds/candidate-v1.json.

    The manifest holds the hidden world source and the anonymous visible schema.
    This type only pins the seed structure; world/query/task owners still need
    their own exact implementation before any episode can be sampled.
    """

    seed_id: str
    world_source: Mapping[str, Any]
    query: Mapping[str, Any]
    task_head: Mapping[str, Any]
    manipulability: Mapping[str, bool]
    readable: Mapping[str, bool]
    visible_schema: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.seed_id:
            raise ValueError("seed_id must not be empty")
        if self.query.get("type") not in ALLOWED_CANDIDATE_QUERY_TYPES:
            raise ValueError(f"unsupported query type in seed {self.seed_id}")
        if self.task_head.get("head") not in ALLOWED_CANDIDATE_TASK_HEADS:
            raise ValueError(f"unsupported task head in seed {self.seed_id}")
        if set(self.manipulability) != set(self.readable):
            raise ValueError(f"manipulability/readable variable sets differ in {self.seed_id}")
        if self.query.get("type") == "best_intervention":
            decision_target = self.query.get("decision_target")
            if not isinstance(decision_target, str) or decision_target not in self.manipulability:
                raise ValueError(f"invalid decision target in seed {self.seed_id}")
            if self.manipulability[decision_target]:
                raise ValueError(
                    f"decision target must be readonly during experimentation in {self.seed_id}"
                )

    def seed_triple(self) -> tuple[str, str, str]:
        """Return the defining (world, query, task) triple for this seed."""

        world = str(self.world_source.get("graph_id") or self.world_source.get("type") or "")
        query = str(self.query.get("type") or "")
        task = str(self.task_head.get("head") or "")
        if not world or not query or not task:
            raise ValueError(f"seed {self.seed_id} has an incomplete (world, query, task) triple")
        return world, query, task

    def visible_labels(self) -> Mapping[str, str]:
        labels = self.visible_schema.get("variable_labels")
        if not isinstance(labels, Mapping):
            raise ValueError(f"seed {self.seed_id} has no variable_labels mapping")
        return labels

    def visible_variables(self) -> tuple[Mapping[str, Any], ...]:
        variables = self.visible_schema.get("variables")
        if not isinstance(variables, list) or not all(
            isinstance(item, Mapping) and isinstance(item.get("label"), str) for item in variables
        ):
            raise ValueError(f"seed {self.seed_id} has no model-visible variable list")
        return tuple(variables)

    def internal_variable_names(self) -> tuple[str, ...]:
        source_type = self.world_source.get("type")
        if source_type == "cladder_meta_model":
            structure = str(self.world_source.get("structure", ""))
            names: list[str] = []
            for token in structure.replace(" ", "").split(","):
                for name in token.split("->"):
                    if name and name not in names:
                        names.append(name)
            return tuple(names)
        if source_type == "bnlearn_bif":
            bif = str(self.world_source.get("bif", ""))
            return tuple(re.findall(r"variable\s+(\w+)\s*\{", bif))
        if source_type == "sampled_motif":
            variables = self.world_source.get("variables", ())
            if isinstance(variables, (list, tuple)):
                return tuple(str(name) for name in variables)
            raise ValueError(f"sampled_motif world missing variables in {self.seed_id}")
        raise ValueError(f"unsupported world_source type in {self.seed_id}")


def candidate_seed_manifest_path() -> Path:
    return Path(__file__).resolve().parents[2] / _CANDIDATE_SEED_MANIFEST_RELPATH


def load_candidate_seed_manifest(path: Path | None = None) -> tuple[CandidateSeedSpec, ...]:
    manifest_path = path or candidate_seed_manifest_path()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema") != CANDIDATE_SEED_MANIFEST_SCHEMA:
        raise ValueError(f"unexpected seed manifest schema in {manifest_path}")
    seeds = tuple(CandidateSeedSpec(**item) for item in raw.get("seeds", []))
    validate_candidate_seed_manifest(seeds)
    return seeds


def validate_candidate_seed_manifest(seeds: Sequence[CandidateSeedSpec]) -> None:
    if not seeds:
        raise ValueError("candidate seed manifest must contain at least one seed")
    seed_ids = [seed.seed_id for seed in seeds]
    if len(set(seed_ids)) != len(seed_ids):
        raise ValueError("candidate seed IDs must be unique")

    for seed in seeds:
        if not query_task_compatible(str(seed.query.get("type")), str(seed.task_head.get("head"))):
            raise ValueError(f"{seed.seed_id}: query/task head mismatch")
    repo = Path(__file__).resolve().parents[2]
    for seed in seeds:
        names = seed.internal_variable_names()
        labels = seed.visible_labels()
        visible_variables = seed.visible_variables()
        if set(labels) != set(names):
            raise ValueError(f"{seed.seed_id}: visible labels do not match internal variables")
        if {item["label"] for item in visible_variables} != set(labels.values()):
            raise ValueError(f"{seed.seed_id}: visible variable list does not match label map")
        if any(len(label) != 3 or not label.isalpha() for label in labels.values()):
            raise ValueError(f"{seed.seed_id}: visible labels must be three-letter tokens")
        for name, label in labels.items():
            if len(name) >= 3 and name.lower() in label.lower():
                raise ValueError(f"{seed.seed_id}: visible label leaks internal name {name}")
        if len(set(labels.values())) != len(labels.values()):
            raise ValueError(f"{seed.seed_id}: visible labels duplicate within seed")
        source_type = seed.world_source.get("type")
        if source_type == "cladder_meta_model":
            source_file = Path(str(seed.world_source.get("file", "")))
            if not (repo / source_file).exists():
                raise ValueError(f"{seed.seed_id}: missing CLadder source file")
        elif source_type == "bnlearn_bif":
            source_file = Path(str(seed.world_source.get("file", "")))
            bif_path = repo / source_file
            if not bif_path.exists():
                raise ValueError(f"{seed.seed_id}: missing BIF source file")
            if seed.world_source.get("bif") != bif_path.read_text(encoding="utf-8"):
                raise ValueError(f"{seed.seed_id}: embedded BIF does not match source file")
        elif source_type == "sampled_motif":
            if not seed.world_source.get("topology"):
                raise ValueError(f"{seed.seed_id}: sampled_motif missing topology")
        else:
            raise ValueError(f"{seed.seed_id}: unsupported world_source type")


def cladder_ate_is_degenerate(seed: CandidateSeedSpec) -> bool:
    """Detect an observational shortcut for a CLadder confounding ATE seed.

    Returns True when P(Y|do(X=x)) equals P(Y|X=x), which means interventions
    add no information beyond the observational conditional.
    """

    if seed.query.get("type") != "ate":
        return False
    source = seed.world_source
    if source.get("type") != "cladder_meta_model" or source.get("graph_id") != "confounding":
        return False
    params = source.get("params", {})
    groundtruth = source.get("groundtruth", {})
    try:
        p_v1 = float(params["p(V1)"])
        y = params["p(Y | V1, X)"]
        do_1 = sum((p_v1 if v1 else 1.0 - p_v1) * y[v1][1] for v1 in (0, 1))
        do_0 = sum((p_v1 if v1 else 1.0 - p_v1) * y[v1][0] for v1 in (0, 1))
        obs_1 = float(groundtruth["P(Y=1 | X=1)"])
        obs_0 = float(groundtruth["P(Y=1 | X=0)"])
    except (KeyError, TypeError, IndexError):
        return False
    return abs((do_1 - do_0) - (obs_1 - obs_0)) < 1e-12
