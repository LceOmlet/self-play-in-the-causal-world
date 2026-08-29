"""Aggregate structural coverage for response-marginal message RLT lifts.

The profiler never reports sample identities.  It inspects the original
quadratic circuit before optimization, then aggregates only the failed owner
whose unchanged five-second solve determines each unresolved task.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from measure_counterfactual_optimizer import (
    CONDITIONAL_ENDPOINT_TOLERANCE,
    DISTRIBUTION_COUNT,
    DISTRIBUTION_START_SEED,
    ENDPOINT_SECONDS,
    _query_indices,
)

import cpt_world.counterfactual_solver as solver
from cpt_world import (
    WorldGrammar,
    iter_sampled_seeds,
    sample_task_world,
    sparse_individual_counterfactual_probability_bounds,
)


def _role(name: str) -> str:
    if name.startswith("ve_"):
        return "message"
    if name.startswith("k_"):
        return "kernel"
    return "other"


def _model_rlt_coverage(owner: Any) -> dict[str, Any]:
    kernel_keys = {variable.name: key for key, variable in owner.kernel_cache.items()}
    entries_by_table: dict[tuple[int, int, int], set[str]] = {}
    for name, key in kernel_keys.items():
        entries_by_table.setdefault(key[:3], set()).add(name)

    quadratic_edges: set[tuple[str, str]] = set()
    for constraint in owner.model.getConss():
        if not constraint.isNonlinear():
            continue
        quadratic, _, _ = owner.model.getTermsQuadratic(constraint)
        for left, right, coefficient in quadratic:
            if coefficient == 0.0:
                continue
            quadratic_edges.add(tuple(sorted((left.name, right.name))))

    groups: set[tuple[tuple[int, int, int], str]] = set()
    groups_by_role = Counter[str]()
    for edge in quadratic_edges:
        for kernel_name, other_name in (edge, edge[::-1]):
            if kernel_name not in kernel_keys:
                continue
            other_key = kernel_keys.get(other_name)
            if other_key is not None and other_key[0] == kernel_keys[kernel_name][0]:
                continue
            group = (kernel_keys[kernel_name][:3], other_name)
            if group not in groups:
                groups.add(group)
                groups_by_role[_role(other_name)] += 1

    required_edges: set[tuple[str, str]] = set()
    required_by_role = Counter[str]()
    existing_by_role = Counter[str]()
    complete_groups = Counter[str]()
    rlt_equalities_by_role = Counter[str]()
    existing_products_per_group = Counter[str]()
    projected_subset_relations = 0
    projected_subset_terms = 0
    for table, other_name in groups:
        role = _role(other_name)
        table_edges = {
            tuple(sorted((kernel_name, other_name)))
            for kernel_name in entries_by_table[table]
        }
        required_edges.update(table_edges)
        required_by_role[role] += len(table_edges)
        existing = len(table_edges & quadratic_edges)
        existing_by_role[role] += existing
        existing_products_per_group[str(existing)] += 1
        complete_groups[role] += existing == len(table_edges)
        domain = owner.world.domains[table[0]]
        rlt_equalities_by_role[role] += 2 * domain - 1
        existing_kernel_names = {
            kernel_name
            for kernel_name in entries_by_table[table]
            if tuple(sorted((kernel_name, other_name))) in quadratic_edges
        }
        for position in (3, 4):
            for state in range(domain):
                subset_size = sum(
                    kernel_keys[kernel_name][position] == state
                    for kernel_name in existing_kernel_names
                )
                if subset_size >= 2:
                    projected_subset_relations += 1
                    projected_subset_terms += subset_size

    return {
        "groups": len(groups),
        "groups_by_role": dict(groups_by_role),
        "required_products": len(required_edges),
        "existing_products": len(required_edges & quadratic_edges),
        "missing_products": len(required_edges - quadratic_edges),
        "required_by_role": dict(required_by_role),
        "existing_by_role": dict(existing_by_role),
        "complete_groups": dict(complete_groups),
        "rlt_equalities_by_role": dict(rlt_equalities_by_role),
        "existing_products_per_group": dict(existing_products_per_group),
        "projected_subset_relations": projected_subset_relations,
        "projected_subset_terms": projected_subset_terms,
    }


@contextmanager
def _trace(calls: list[dict[str, Any]]) -> Iterator[None]:
    original = solver._SparseResponseModel.optimize

    def traced(self: Any, *args: Any, **kwargs: Any) -> tuple[float, float]:
        record = _model_rlt_coverage(self)
        try:
            result = original(self, *args, **kwargs)
        except RuntimeError:
            record["outcome"] = "failed"
            calls.append(record)
            raise
        record["outcome"] = "closed"
        calls.append(record)
        return result

    solver._SparseResponseModel.optimize = traced
    try:
        yield
    finally:
        solver._SparseResponseModel.optimize = original


def _percentile(values: list[int], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(percentile * len(ordered) + 0.999999) - 1))
    return float(ordered[rank])


def main() -> None:
    grammar = WorldGrammar()
    totals = Counter[str]()
    sums = Counter[str]()
    role_sums: dict[str, Counter[str]] = {
        role: Counter() for role in ("message", "kernel", "other")
    }
    existing_products_per_group = Counter[str]()
    missing_per_owner: list[int] = []
    for sample_index in range(
        DISTRIBUTION_START_SEED,
        DISTRIBUTION_START_SEED + DISTRIBUTION_COUNT,
    ):
        seed = iter_sampled_seeds(
            grammar,
            start_seed=sample_index,
            count=1,
            query_types=("individual_counterfactual_probability",),
        )[0]
        world = sample_task_world(
            grammar,
            sample_index,
            "individual_counterfactual_probability",
        )
        query = _query_indices(world, seed)
        calls: list[dict[str, Any]] = []
        try:
            with _trace(calls):
                sparse_individual_counterfactual_probability_bounds(
                    world,
                    query[0],
                    query[1],
                    factual_value=query[2],
                    counterfactual_value=query[3],
                    factual_outcome_state=query[4],
                    target_outcome_state=query[5],
                    time_limit_seconds=ENDPOINT_SECONDS,
                    conditional_endpoint_tolerance=CONDITIONAL_ENDPOINT_TOLERANCE,
                )
        except RuntimeError:
            totals["unresolved"] += 1
            failed = next(
                (record for record in reversed(calls) if record["outcome"] == "failed"),
                None,
            )
            if failed is None:
                totals["unprofiled_failure"] += 1
                continue
            missing_per_owner.append(failed["missing_products"])
            for key in (
                "groups",
                "required_products",
                "existing_products",
                "missing_products",
                "projected_subset_relations",
                "projected_subset_terms",
            ):
                sums[key] += failed[key]
            existing_products_per_group.update(failed["existing_products_per_group"])
            for role in role_sums:
                role_sums[role]["groups"] += failed["groups_by_role"].get(role, 0)
                role_sums[role]["required_products"] += failed["required_by_role"].get(
                    role, 0
                )
                role_sums[role]["existing_products"] += failed["existing_by_role"].get(
                    role, 0
                )
                role_sums[role]["complete_groups"] += failed["complete_groups"].get(
                    role, 0
                )
                role_sums[role]["rlt_equalities"] += failed[
                    "rlt_equalities_by_role"
                ].get(role, 0)
        else:
            totals["closed"] += 1

    payload = {
        "cohort_size": DISTRIBUTION_COUNT,
        "totals": dict(totals),
        "failed_owner_sums": dict(sums),
        "failed_owner_role_sums": {
            role: dict(values) for role, values in role_sums.items()
        },
        "failed_owner_existing_products_per_group": dict(
            sorted(existing_products_per_group.items(), key=lambda item: int(item[0]))
        ),
        "failed_owner_missing_products_p50": statistics.median(missing_per_owner),
        "failed_owner_missing_products_p95": _percentile(missing_per_owner, 0.95),
        "failed_owner_missing_products_max": max(missing_per_owner),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
