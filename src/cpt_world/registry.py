"""Registries for CPT-World seed assembly.

These registries keep the currently supported options in one module so that
world loaders, renderers, query solvers, task scorers, and seed assemblers do
not each maintain their own lists.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

WORLD_SOURCE_TYPES = frozenset(
    {
        "cladder_meta_model",
        "bnlearn_bif",
        "qn_builtin",
        "sampled_dag",
        # Backwards-compatible fixture alias generated before the grammar was
        # replaced by topological-order + forward-edge-subset sampling.
        "sampled_motif",
    }
)

HIDING_MODES = frozenset(
    {
        "mechanism_hidden",
        "role_hidden",
        "relevant_set_hidden",
        "evidence_by_intervention_only",
        "no_full_joint",
        "manipulability_via_action_legality",
    }
)

# Owner state vocabulary. Registration means the option is part of the design;
# it does not mean a truth owner, scorer, parser, or planner is implemented.
OWNER_STATUS_REGISTERED = "registered"
OWNER_STATUS_IMPLEMENTED = "implemented"
OWNER_STATUS_DIAGNOSTIC_ONLY = "diagnostic_only"
OWNER_STATUSES = frozenset(
    {
        OWNER_STATUS_REGISTERED,
        OWNER_STATUS_IMPLEMENTED,
        OWNER_STATUS_DIAGNOSTIC_ONLY,
    }
)

QUERY_TYPES: dict[str, Mapping[str, Any]] = {
    "ate": {
        "anchors": ("treatment", "outcome"),
        "answer_kind": "single_effect",
        "task_heads": frozenset({"target_query"}),
        "truth_owner_status": OWNER_STATUS_IMPLEMENTED,
    },
    "individual_counterfactual_probability": {
        "anchors": ("treatment", "outcome"),
        "answer_kind": "individual_probability",
        "task_heads": frozenset({"target_query"}),
        "truth_owner_status": OWNER_STATUS_IMPLEMENTED,
    },
    "backadj_minimal_sets": {
        "anchors": ("treatment", "outcome"),
        "answer_kind": "set",
        "task_heads": frozenset({"discovery"}),
        "truth_owner_status": OWNER_STATUS_IMPLEMENTED,
    },
    "best_intervention": {
        "anchors": ("decision_target", "outcome"),
        "answer_kind": "intervention",
        "task_heads": frozenset({"decision"}),
        "truth_owner_status": OWNER_STATUS_IMPLEMENTED,
    },
    "mediator_set": {
        "anchors": ("treatment", "outcome"),
        "answer_kind": "set_with_order",
        "task_heads": frozenset({"discovery"}),
        "truth_owner_status": OWNER_STATUS_IMPLEMENTED,
    },
}

TASK_HEADS: dict[str, Mapping[str, Any]] = {
    "target_query": {
        "answer_kind": "numeric_value",
        "scorer_owner_status": OWNER_STATUS_DIAGNOSTIC_ONLY,
    },
    "discovery": {
        "answer_kind": "set_or_set_with_order",
        "scorer_owner_status": OWNER_STATUS_DIAGNOSTIC_ONLY,
    },
    "decision": {
        "answer_kind": "intervention",
        "scorer_owner_status": OWNER_STATUS_DIAGNOSTIC_ONLY,
    },
}


def query_task_compatible(query_type: str, task_head: str) -> bool:
    return task_head in QUERY_TYPES[query_type]["task_heads"]


def query_truth_owner_status(query_type: str) -> str:
    return str(QUERY_TYPES[query_type]["truth_owner_status"])


def task_scorer_owner_status(task_head: str) -> str:
    return str(TASK_HEADS[task_head]["scorer_owner_status"])


def seed_triple_is_registered(world: str, query_type: str, task_head: str) -> bool:
    """Return True when a seed triple uses registered options that are compatible."""

    if query_type not in QUERY_TYPES:
        return False
    if task_head not in TASK_HEADS:
        return False
    if not query_task_compatible(query_type, task_head):
        return False
    return world in WORLD_SOURCE_TYPES or world in {
        "confounding",
        "diamondcut",
        "collision",
        "chain",
        "fork",
        "mediation",
        "diamond",
    }


def check_seed_legality(seed: Mapping[str, Any]) -> list[str]:
    """Return legality failures for a seed mapping.

    This is the assembly-level legality check. World-specific checks such as
    P(Y|do X) != P(Y|X) remain in the degeneration owner.
    """

    errors: list[str] = []
    seed_id = str(seed.get("seed_id", "<unknown>"))
    world_source = seed.get("world_source")
    query = seed.get("query")
    task_head = seed.get("task_head")

    if not isinstance(world_source, Mapping):
        errors.append(f"{seed_id}: missing world_source")
    elif world_source.get("type") not in WORLD_SOURCE_TYPES:
        errors.append(f"{seed_id}: unsupported world_source type")

    if not isinstance(query, Mapping):
        errors.append(f"{seed_id}: missing query")
    else:
        query_type = query.get("type")
        if query_type not in QUERY_TYPES:
            errors.append(f"{seed_id}: unsupported query type")
        else:
            for anchor in QUERY_TYPES[query_type]["anchors"]:
                if anchor not in query:
                    errors.append(f"{seed_id}: query missing anchor {anchor}")
    if not isinstance(task_head, Mapping):
        errors.append(f"{seed_id}: missing task_head")
    else:
        head = task_head.get("head")
        if head not in TASK_HEADS:
            errors.append(f"{seed_id}: unsupported task head")
        elif isinstance(query, Mapping) and not query_task_compatible(str(query.get("type")), head):
            errors.append(f"{seed_id}: query/task head mismatch")

    manipulability = seed.get("manipulability")
    readable = seed.get("readable")
    if not isinstance(manipulability, Mapping) or not isinstance(readable, Mapping):
        errors.append(f"{seed_id}: missing manipulability/readable mappings")
    elif set(manipulability) != set(readable):
        errors.append(f"{seed_id}: manipulability/readable variable sets differ")

    if (
        isinstance(query, Mapping)
        and query.get("type") == "best_intervention"
        and isinstance(manipulability, Mapping)
    ):
        decision_value = query.get("decision_target")
        visible_schema = seed.get("visible_schema")
        labels = (
            visible_schema.get("variable_labels") if isinstance(visible_schema, Mapping) else None
        )
        if isinstance(labels, Mapping):
            inverse = {str(visible): str(internal) for internal, visible in labels.items()}
            decision_name = inverse.get(str(decision_value), str(decision_value))
            if decision_name not in manipulability:
                errors.append(f"{seed_id}: decision target is not a world variable")
            elif manipulability[decision_name]:
                errors.append(f"{seed_id}: decision target must be readonly during experimentation")

    return errors
