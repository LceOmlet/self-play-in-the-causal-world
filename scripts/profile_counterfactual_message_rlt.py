"""Aggregate structural coverage for response-marginal message RLT lifts.

The profiler never reports sample identities.  It inspects the original
quadratic circuit before optimization, then aggregates only the failed owner
whose unchanged five-second solve determines each unresolved task.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
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
    response_owner_by_kernel: dict[str, tuple[str, int, int]] = {}
    for name, key in kernel_keys.items():
        node, left, right, _, _ = key
        block = owner.pricing_block_by_edge.get((node, left, right))
        response_owner_by_kernel[name] = (
            ("block", node, block.block_id)
            if block is not None
            else ("forest", node, left * len(owner.contexts[node]) + right)
        )
    entries_by_table: dict[tuple[int, int, int], set[str]] = {}
    for name, key in kernel_keys.items():
        entries_by_table.setdefault(key[:3], set()).add(name)

    quadratic_edges: set[tuple[str, str]] = set()
    local_owner_groups: list[dict[tuple[tuple[str, int, int], str], set[str]]] = []
    for constraint in owner.model.getConss():
        if not constraint.isNonlinear():
            continue
        constraint_groups: dict[tuple[tuple[str, int, int], str], set[str]] = {}
        quadratic, _, _ = owner.model.getTermsQuadratic(constraint)
        for left, right, coefficient in quadratic:
            if coefficient == 0.0:
                continue
            edge = tuple(sorted((left.name, right.name)))
            quadratic_edges.add(edge)
            for kernel_name, other_name in (edge, edge[::-1]):
                if kernel_name not in kernel_keys:
                    continue
                other_key = kernel_keys.get(other_name)
                if other_key is not None and other_key[0] == kernel_keys[kernel_name][0]:
                    continue
                constraint_groups.setdefault(
                    (response_owner_by_kernel[kernel_name], other_name),
                    set(),
                ).add(kernel_name)
        if constraint_groups:
            local_owner_groups.append(constraint_groups)

    groups: set[tuple[tuple[int, int, int], str]] = set()
    owner_groups: dict[tuple[tuple[str, int, int], str], set[str]] = {}
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
            owner_groups.setdefault(
                (response_owner_by_kernel[kernel_name], other_name),
                set(),
            ).add(kernel_name)

    required_edges: set[tuple[str, str]] = set()
    required_by_role = Counter[str]()
    existing_by_role = Counter[str]()
    complete_groups = Counter[str]()
    rlt_equalities_by_role = Counter[str]()
    existing_products_per_group = Counter[str]()
    projected_subset_relations = 0
    projected_subset_terms = 0
    owner_group_size_histogram = Counter[str]()
    cyclic_owner_group_size_histogram = Counter[str]()
    local_owner_group_size_histogram = Counter[str]()
    local_cyclic_owner_group_size_histogram = Counter[str]()
    for constraint_groups in local_owner_groups:
        for (response_owner, _), kernel_names in constraint_groups.items():
            local_owner_group_size_histogram[str(len(kernel_names))] += 1
            if response_owner[0] == "block":
                local_cyclic_owner_group_size_histogram[str(len(kernel_names))] += 1
    cyclic_signatures: dict[
        tuple[int, tuple[tuple[int, int, int, int], ...]],
        int,
    ] = {}
    for (response_owner, _), kernel_names in owner_groups.items():
        size = len(kernel_names)
        owner_group_size_histogram[str(size)] += 1
        if response_owner[0] == "block":
            cyclic_owner_group_size_histogram[str(size)] += 1
            block = owner.pricing_blocks[response_owner[2]]
            signature_entries: list[tuple[int, int, int, int]] = []
            for kernel_name in kernel_names:
                node, left, right, left_state, right_state = kernel_keys[kernel_name]
                contexts = owner.contexts[node]
                signature_entries.append(
                    (
                        block.context_indices[contexts[left]],
                        block.context_indices[contexts[right]],
                        left_state,
                        right_state,
                    )
                )
            signature = (block.block_id, tuple(sorted(signature_entries)))
            cyclic_signatures[signature] = cyclic_signatures.get(signature, 0) + 1

    cyclic_signature_specs = []
    for (block_id, entries), occurrences in cyclic_signatures.items():
        block = owner.pricing_blocks[block_id]
        marginals = tuple(
            tuple(float(value) for value in owner.context_rows[block.node][context])
            for context in block.contexts
        )
        cyclic_signature_specs.append((block_id, marginals, entries, occurrences))
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
        "owner_group_size_histogram": dict(owner_group_size_histogram),
        "cyclic_owner_group_size_histogram": dict(
            cyclic_owner_group_size_histogram
        ),
        "local_owner_group_size_histogram": dict(local_owner_group_size_histogram),
        "local_cyclic_owner_group_size_histogram": dict(
            local_cyclic_owner_group_size_histogram
        ),
        "cyclic_signature_specs": cyclic_signature_specs,
    }


def _cyclic_signature_tightening(
    specs: list[
        tuple[
            int,
            tuple[tuple[float, ...], ...],
            tuple[tuple[int, int, int, int], ...],
            int,
        ]
    ],
    *,
    support_mode: str,
) -> dict[str, float | int]:
    if support_mode not in {"cycle-exact", "exact", "warm"}:
        raise ValueError("support mode must be cycle-exact, exact, or warm")
    strict_upper_occurrences = 0
    strict_lower_occurrences = 0
    strict_upper_unique = 0
    strict_lower_unique = 0
    upper_improvement_sum = 0.0
    lower_improvement_sum = 0.0
    star_strict_upper_unique = 0
    star_strict_upper_occurrences = 0
    star_upper_improvement_sum = 0.0
    cyclic_graph_unique = 0
    cyclic_graph_occurrences = 0
    cyclic_graph_strict_upper_unique = 0
    cyclic_graph_strict_lower_unique = 0
    forest_graph_strict_upper_unique = 0
    forest_graph_strict_lower_unique = 0
    cyclic_product_occurrences = 0
    strict_cyclic_group_occurrences = 0
    strict_cyclic_product_occurrences = 0
    strict_cyclic_perspective_rows = 0
    support_started = time.perf_counter()
    specs_by_block: dict[
        int,
        list[
            tuple[
                tuple[tuple[float, ...], ...],
                tuple[tuple[int, int, int, int], ...],
                int,
            ]
        ],
    ] = {}
    for block_id, marginals, entries, occurrences in specs:
        specs_by_block.setdefault(block_id, []).append(
            (marginals, entries, occurrences)
        )
    evaluated: list[
        tuple[
            tuple[tuple[float, ...], ...],
            tuple[tuple[int, int, int, int], ...],
            int,
            float,
            float,
        ]
    ] = []
    for block_id in sorted(specs_by_block):
        block_specs = sorted(specs_by_block[block_id], key=lambda item: item[1])
        if support_mode == "cycle-exact":
            block_specs = [
                item
                for item in block_specs
                if not solver._edges_form_forest(
                    len(item[0]),
                    tuple(
                        sorted(
                            {tuple(sorted(entry[:2])) for entry in item[1]}
                        )
                    ),
                )
            ]
            if not block_specs:
                continue
        marginals = block_specs[0][0]
        if support_mode == "warm":
            aggregate: Counter[tuple[int, int, int, int]] = Counter()
            for _, entries, _ in block_specs:
                aggregate.update(entries)
            aggregate_objective = {
                entry: float(coefficient) for entry, coefficient in aggregate.items()
            }
            maximum_owner = solver._ExactPricedResponseLP(
                marginals, aggregate_objective, sense="maximize"
            )
            minimum_owner = solver._ExactPricedResponseLP(
                marginals, aggregate_objective, sense="minimize"
            )
            maximum_owner.optimize(time_limit_seconds=None)
            minimum_owner.optimize(time_limit_seconds=None)
        else:
            maximum_owner = None
            minimum_owner = None
        for index, (_, entries, occurrences) in enumerate(block_specs):
            objective = {entry: 1.0 for entry in entries}
            if support_mode == "warm":
                maximum, _ = maximum_owner.transformed_upper_bound(
                    objective, time_limit_seconds=None
                )
                transformed_minimum, _ = minimum_owner.transformed_upper_bound(
                    objective, time_limit_seconds=None
                )
                minimum = -transformed_minimum
            else:
                if index == 0:
                    maximum_owner = solver._ExactPricedResponseLP(
                        marginals, objective, sense="maximize"
                    )
                    minimum_owner = solver._ExactPricedResponseLP(
                        marginals, objective, sense="minimize"
                    )
                else:
                    maximum_owner.restart_objective(objective)
                    minimum_owner.restart_objective(objective)
                maximum, _, _ = maximum_owner.optimize(time_limit_seconds=None)
                minimum, _, _ = minimum_owner.optimize(time_limit_seconds=None)
            evaluated.append((marginals, entries, occurrences, maximum, minimum))

    for marginals, entries, occurrences, maximum, minimum in evaluated:
        context_edges = tuple(sorted({tuple(sorted(entry[:2])) for entry in entries}))
        graph_is_forest = solver._edges_form_forest(len(marginals), context_edges)
        if not graph_is_forest:
            cyclic_graph_unique += 1
            cyclic_graph_occurrences += occurrences
            cyclic_product_occurrences += occurrences * len(entries)
        individual_upper = sum(
            min(
                marginals[left][left_state],
                marginals[right][right_state],
            )
            for left, right, left_state, right_state in entries
        )
        individual_lower = sum(
            max(
                0.0,
                marginals[left][left_state]
                + marginals[right][right_state]
                - 1.0,
            )
            for left, right, left_state, right_state in entries
        )
        upper_improvement = max(0.0, individual_upper - maximum)
        lower_improvement = max(0.0, minimum - individual_lower)
        star_upper = individual_upper
        contexts = {context for entry in entries for context in entry[:2]}
        for context in contexts:
            counts = [0] * len(marginals[context])
            nonincident = 0
            for left, right, left_state, right_state in entries:
                if context == left:
                    counts[left_state] += 1
                elif context == right:
                    counts[right_state] += 1
                else:
                    nonincident += 1
            star_upper = min(
                star_upper,
                nonincident
                + sum(
                    probability * counts[state]
                    for state, probability in enumerate(marginals[context])
                ),
            )
        star_improvement = max(0.0, individual_upper - star_upper)
        if upper_improvement > 1e-10:
            strict_upper_unique += 1
            strict_upper_occurrences += occurrences
            upper_improvement_sum += occurrences * upper_improvement
            if graph_is_forest:
                forest_graph_strict_upper_unique += 1
            else:
                cyclic_graph_strict_upper_unique += 1
        if lower_improvement > 1e-10:
            strict_lower_unique += 1
            strict_lower_occurrences += occurrences
            lower_improvement_sum += occurrences * lower_improvement
            if graph_is_forest:
                forest_graph_strict_lower_unique += 1
            else:
                cyclic_graph_strict_lower_unique += 1
        if not graph_is_forest and (
            upper_improvement > 1e-10 or lower_improvement > 1e-10
        ):
            strict_cyclic_group_occurrences += occurrences
            strict_cyclic_product_occurrences += occurrences * len(entries)
            strict_cyclic_perspective_rows += occurrences * (
                int(upper_improvement > 1e-10) + int(lower_improvement > 1e-10)
            )
        if star_improvement > 1e-10:
            star_strict_upper_unique += 1
            star_strict_upper_occurrences += occurrences
            star_upper_improvement_sum += occurrences * star_improvement
    return {
        "cyclic_unique_signatures": len(specs),
        "cyclic_analyzed_signatures": len(evaluated),
        "cyclic_strict_upper_unique": strict_upper_unique,
        "cyclic_strict_lower_unique": strict_lower_unique,
        "cyclic_strict_upper_occurrences": strict_upper_occurrences,
        "cyclic_strict_lower_occurrences": strict_lower_occurrences,
        "cyclic_upper_improvement_sum": upper_improvement_sum,
        "cyclic_lower_improvement_sum": lower_improvement_sum,
        "cyclic_star_strict_upper_unique": star_strict_upper_unique,
        "cyclic_star_strict_upper_occurrences": star_strict_upper_occurrences,
        "cyclic_star_upper_improvement_sum": star_upper_improvement_sum,
        "cyclic_support_seconds": time.perf_counter() - support_started,
        "cyclic_graph_unique": cyclic_graph_unique,
        "cyclic_graph_occurrences": cyclic_graph_occurrences,
        "cyclic_graph_strict_upper_unique": cyclic_graph_strict_upper_unique,
        "cyclic_graph_strict_lower_unique": cyclic_graph_strict_lower_unique,
        "forest_graph_strict_upper_unique": forest_graph_strict_upper_unique,
        "forest_graph_strict_lower_unique": forest_graph_strict_lower_unique,
        "cyclic_product_occurrences": cyclic_product_occurrences,
        "strict_cyclic_group_occurrences": strict_cyclic_group_occurrences,
        "strict_cyclic_product_occurrences": strict_cyclic_product_occurrences,
        "strict_cyclic_perspective_rows": strict_cyclic_perspective_rows,
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--support-mode",
        choices=("cycle-exact", "exact", "warm"),
        default="exact",
    )
    args = parser.parse_args()
    grammar = WorldGrammar()
    totals = Counter[str]()
    sums = Counter[str]()
    role_sums: dict[str, Counter[str]] = {
        role: Counter() for role in ("message", "kernel", "other")
    }
    existing_products_per_group = Counter[str]()
    owner_group_size_histogram = Counter[str]()
    cyclic_owner_group_size_histogram = Counter[str]()
    local_owner_group_size_histogram = Counter[str]()
    local_cyclic_owner_group_size_histogram = Counter[str]()
    missing_per_owner: list[int] = []
    strict_cyclic_products_per_owner: list[int] = []
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
            failed.update(
                _cyclic_signature_tightening(
                    failed.pop("cyclic_signature_specs"),
                    support_mode=args.support_mode,
                )
            )
            missing_per_owner.append(failed["missing_products"])
            for key in (
                "groups",
                "required_products",
                "existing_products",
                "missing_products",
                "projected_subset_relations",
                "projected_subset_terms",
                "cyclic_unique_signatures",
                "cyclic_analyzed_signatures",
                "cyclic_strict_upper_unique",
                "cyclic_strict_lower_unique",
                "cyclic_strict_upper_occurrences",
                "cyclic_strict_lower_occurrences",
                "cyclic_star_strict_upper_unique",
                "cyclic_star_strict_upper_occurrences",
                "cyclic_graph_unique",
                "cyclic_graph_occurrences",
                "cyclic_graph_strict_upper_unique",
                "cyclic_graph_strict_lower_unique",
                "forest_graph_strict_upper_unique",
                "forest_graph_strict_lower_unique",
                "cyclic_product_occurrences",
                "strict_cyclic_group_occurrences",
                "strict_cyclic_product_occurrences",
                "strict_cyclic_perspective_rows",
            ):
                sums[key] += failed[key]
            strict_cyclic_products_per_owner.append(
                failed["strict_cyclic_product_occurrences"]
            )
            sums["cyclic_upper_improvement_sum"] += failed[
                "cyclic_upper_improvement_sum"
            ]
            sums["cyclic_lower_improvement_sum"] += failed[
                "cyclic_lower_improvement_sum"
            ]
            sums["cyclic_star_upper_improvement_sum"] += failed[
                "cyclic_star_upper_improvement_sum"
            ]
            sums["cyclic_support_seconds"] += failed["cyclic_support_seconds"]
            existing_products_per_group.update(failed["existing_products_per_group"])
            owner_group_size_histogram.update(failed["owner_group_size_histogram"])
            cyclic_owner_group_size_histogram.update(
                failed["cyclic_owner_group_size_histogram"]
            )
            local_owner_group_size_histogram.update(
                failed["local_owner_group_size_histogram"]
            )
            local_cyclic_owner_group_size_histogram.update(
                failed["local_cyclic_owner_group_size_histogram"]
            )
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
        "support_mode": args.support_mode,
        "totals": dict(totals),
        "failed_owner_sums": dict(sums),
        "failed_owner_role_sums": {
            role: dict(values) for role, values in role_sums.items()
        },
        "failed_owner_existing_products_per_group": dict(
            sorted(existing_products_per_group.items(), key=lambda item: int(item[0]))
        ),
        "failed_response_owner_product_group_sizes": dict(
            sorted(owner_group_size_histogram.items(), key=lambda item: int(item[0]))
        ),
        "failed_cyclic_response_owner_product_group_sizes": dict(
            sorted(
                cyclic_owner_group_size_histogram.items(),
                key=lambda item: int(item[0]),
            )
        ),
        "failed_local_response_owner_product_group_sizes": dict(
            sorted(local_owner_group_size_histogram.items(), key=lambda item: int(item[0]))
        ),
        "failed_local_cyclic_response_owner_product_group_sizes": dict(
            sorted(
                local_cyclic_owner_group_size_histogram.items(),
                key=lambda item: int(item[0]),
            )
        ),
        "failed_owner_missing_products_p50": statistics.median(missing_per_owner),
        "failed_owner_missing_products_p95": _percentile(missing_per_owner, 0.95),
        "failed_owner_missing_products_max": max(missing_per_owner),
        "failed_owner_strict_cyclic_products_p50": statistics.median(
            strict_cyclic_products_per_owner
        ),
        "failed_owner_strict_cyclic_products_p95": _percentile(
            strict_cyclic_products_per_owner, 0.95
        ),
        "failed_owner_strict_cyclic_products_max": max(
            strict_cyclic_products_per_owner
        ),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
