"""Aggregate audit of exact response-pricing elimination width.

The production pricer uses deterministic greedy min-fill and falls back to a
generic SCIP MAP model when its induced table exceeds the existing memory
guard.  This probe computes the true minimum elimination width for every such
non-forbidden pricing graph reached by a frozen unresolved task.  It emits no
task or graph identities and does not change the pricing decision.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from typing import Any

from measure_counterfactual_optimizer import (
    CONDITIONAL_ENDPOINT_TOLERANCE,
    DISTRIBUTION_COUNT,
    DISTRIBUTION_START_SEED,
    ENDPOINT_SECONDS,
    _query_indices,
)

from cpt_world import (
    WorldGrammar,
    iter_sampled_seeds,
    sample_task_world,
    sparse_individual_counterfactual_probability_bounds,
)
from cpt_world import counterfactual_solver as solver


def _filled_degree(adjacency: tuple[int, ...], variable: int, eliminated: int) -> int:
    """Degree of one variable after exactly the selected vertices are eliminated."""

    variable_bit = 1 << variable
    boundary = adjacency[variable] & ~eliminated & ~variable_bit
    frontier = adjacency[variable] & eliminated
    seen = 0
    while frontier:
        bit = frontier & -frontier
        frontier ^= bit
        if seen & bit:
            continue
        seen |= bit
        node = bit.bit_length() - 1
        neighbors = adjacency[node]
        boundary |= neighbors & ~eliminated & ~variable_bit
        frontier |= neighbors & eliminated & ~seen
    return boundary.bit_count()


@cache
def _exact_width(
    variable_count: int, edges: tuple[tuple[int, int], ...]
) -> tuple[int, tuple[int, ...]]:
    """Return exact treewidth and one optimal elimination order by subset DP."""

    adjacency = [0] * variable_count
    for left, right in edges:
        adjacency[left] |= 1 << right
        adjacency[right] |= 1 << left
    adjacency_tuple = tuple(adjacency)
    size = 1 << variable_count
    unreachable = variable_count + 1
    best = [unreachable] * size
    predecessor = [-1] * size
    best[0] = 0
    full = size - 1
    for eliminated in range(size):
        current = best[eliminated]
        if current == unreachable:
            continue
        remaining = full ^ eliminated
        while remaining:
            bit = remaining & -remaining
            remaining ^= bit
            variable = bit.bit_length() - 1
            width = max(
                current,
                _filled_degree(adjacency_tuple, variable, eliminated),
            )
            successor = eliminated | bit
            if width < best[successor]:
                best[successor] = width
                predecessor[successor] = variable
    order_reversed: list[int] = []
    mask = full
    while mask:
        variable = predecessor[mask]
        if variable < 0:
            raise RuntimeError("treewidth subset DP has no predecessor")
        order_reversed.append(variable)
        mask ^= 1 << variable
    return best[full], tuple(reversed(order_reversed))


@contextmanager
def _trace_pricing_width(calls: list[dict[str, Any]]) -> Iterator[None]:
    original = solver._exact_pairwise_map

    def traced(
        unary: Any,
        pairwise: Any,
        *,
        domain_size: int,
        constant: float,
        time_limit_seconds: float | None = None,
        forbidden_responses: frozenset[tuple[int, ...]] = frozenset(),
    ) -> Any:
        edges = tuple(sorted(pairwise))
        _, greedy_width = solver._minimum_fill_order(len(unary), set(edges))
        greedy_cells = domain_size ** (greedy_width + 1)
        record: dict[str, Any] = {
            "contexts": len(unary),
            "domain": domain_size,
            "edges": len(edges),
            "greedy_width": greedy_width,
            "greedy_cells": greedy_cells,
            "forbidden": bool(forbidden_responses),
            "forbidden_count": len(forbidden_responses),
            "greedy_fallback": bool(
                forbidden_responses
                or greedy_cells > solver._MAX_EXACT_MIN_SUM_TABLE_ENTRIES
            ),
        }
        if not forbidden_responses and len(unary) <= 20:
            exact_width, _ = _exact_width(len(unary), edges)
            exact_cells = domain_size ** (exact_width + 1)
            record.update(
                {
                    "exact_width": exact_width,
                    "exact_cells": exact_cells,
                    "exact_recoverable": (
                        greedy_cells > solver._MAX_EXACT_MIN_SUM_TABLE_ENTRIES
                        and exact_cells
                        <= solver._MAX_EXACT_MIN_SUM_TABLE_ENTRIES
                    ),
                }
            )
        result = original(
            unary,
            pairwise,
            domain_size=domain_size,
            constant=constant,
            time_limit_seconds=time_limit_seconds,
            forbidden_responses=forbidden_responses,
        )
        record.update(
            {
                "backend": result.backend,
                "status": result.status,
                "solve_seconds": result.solve_seconds,
            }
        )
        calls.append(record)
        return result

    solver._exact_pairwise_map = traced
    try:
        yield
    finally:
        solver._exact_pairwise_map = original


def _bucket(value: int, limits: tuple[int, ...]) -> str:
    for limit in limits:
        if value <= limit:
            return f"le_{limit}"
    return f"gt_{limits[-1]}"


def main() -> None:
    grammar = WorldGrammar()
    totals: Counter[str] = Counter()
    failed_width_pairs: Counter[str] = Counter()
    failed_fallback_reason: Counter[str] = Counter()
    failed_exact_cell_ratio: Counter[str] = Counter()
    failed_forbidden_sizes: Counter[str] = Counter()
    failed_forbidden_status: Counter[str] = Counter()
    failed_forbidden_seconds: list[float] = []
    for sample_index in range(
        DISTRIBUTION_START_SEED,
        DISTRIBUTION_START_SEED + DISTRIBUTION_COUNT,
    ):
        rendered = iter_sampled_seeds(
            grammar,
            start_seed=sample_index,
            count=1,
            query_types=("individual_counterfactual_probability",),
        )[0]
        world = sample_task_world(
            grammar, sample_index, "individual_counterfactual_probability"
        )
        treatment, outcome, baseline, treated, factual_outcome, target_outcome = (
            _query_indices(world, rendered)
        )
        calls: list[dict[str, Any]] = []
        try:
            with _trace_pricing_width(calls):
                sparse_individual_counterfactual_probability_bounds(
                    world,
                    treatment,
                    outcome,
                    factual_value=baseline,
                    counterfactual_value=treated,
                    factual_outcome_state=factual_outcome,
                    target_outcome_state=target_outcome,
                    time_limit_seconds=ENDPOINT_SECONDS,
                    conditional_endpoint_tolerance=CONDITIONAL_ENDPOINT_TOLERANCE,
                )
        except RuntimeError:
            totals["unresolved"] += 1
            for record in calls:
                totals["unresolved_pricing_calls"] += 1
                if not record["greedy_fallback"]:
                    totals["unresolved_min_sum_calls"] += 1
                    continue
                totals["unresolved_fallback_calls"] += 1
                if record["forbidden"]:
                    failed_fallback_reason["forbidden_response"] += 1
                    failed_forbidden_sizes[
                        _bucket(record["forbidden_count"], (8, 32, 128, 512, 2048))
                    ] += 1
                    failed_forbidden_status[record["status"]] += 1
                    failed_forbidden_seconds.append(record["solve_seconds"])
                    continue
                failed_fallback_reason["greedy_width"] += 1
                exact_width = record["exact_width"]
                failed_width_pairs[
                    f"greedy_{record['greedy_width']}_exact_{exact_width}"
                ] += 1
                if record["exact_recoverable"]:
                    totals["unresolved_exact_order_recoverable_calls"] += 1
                ratio = record["exact_cells"] / record["greedy_cells"]
                if ratio <= 0.001:
                    ratio_bucket = "le_0.001"
                elif ratio <= 0.01:
                    ratio_bucket = "le_0.01"
                elif ratio <= 0.1:
                    ratio_bucket = "le_0.1"
                elif ratio <= 0.5:
                    ratio_bucket = "le_0.5"
                else:
                    ratio_bucket = "gt_0.5"
                failed_exact_cell_ratio[ratio_bucket] += 1
                failed_fallback_reason[
                    "contexts_" + _bucket(record["contexts"], (8, 12, 16, 20))
                ] += 1
        else:
            totals["closed"] += 1

    print(
        json.dumps(
            {
                "cohort_size": DISTRIBUTION_COUNT,
                "totals": dict(sorted(totals.items())),
                "unresolved_fallback_reasons": dict(
                    sorted(failed_fallback_reason.items())
                ),
                "unresolved_greedy_exact_width_pairs": dict(
                    sorted(failed_width_pairs.items())
                ),
                "unresolved_exact_to_greedy_cell_ratio": dict(
                    sorted(failed_exact_cell_ratio.items())
                ),
                "unresolved_forbidden_response_sizes": dict(
                    sorted(failed_forbidden_sizes.items())
                ),
                "unresolved_forbidden_response_status": dict(
                    sorted(failed_forbidden_status.items())
                ),
                "unresolved_forbidden_response_seconds": {
                    "total": sum(failed_forbidden_seconds),
                    "max": max(failed_forbidden_seconds, default=0.0),
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
