"""Aggregate profile of exact one-world projections for twin-message boxes.

The profiler uses the frozen optimization cohort and emits only aggregate
counts.  It never reports sample identities, labels, topology strings, CPT
values, or per-instance outcomes.
"""

from __future__ import annotations

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
from profile_counterfactual_failure_classes import _error_fields

from cpt_world import (
    WorldGrammar,
    iter_sampled_seeds,
    sample_task_world,
    sparse_individual_counterfactual_probability_bounds,
)
from cpt_world import counterfactual_solver as solver


@contextmanager
def _capture_models(
    models: list[Any],
    failure_ratios: list[float],
) -> Iterator[None]:
    original = solver._SparseResponseModel.optimize

    def traced(self: Any, *args: Any, **kwargs: Any) -> tuple[float, float]:
        models.append(self)
        try:
            return original(self, *args, **kwargs)
        except RuntimeError as exc:
            _, _, _, primal, dual = _error_fields(str(exc))
            accepted_gap = float(kwargs.get("accepted_absolute_gap", 0.0))
            if primal is not None and dual is not None and accepted_gap > 0.0:
                failure_ratios.append(abs(primal - dual) / accepted_gap)
            raise

    solver._SparseResponseModel.optimize = traced
    try:
        yield
    finally:
        solver._SparseResponseModel.optimize = original


def main() -> None:
    grammar = WorldGrammar()
    outcomes: Counter[str] = Counter()
    diagnostics: Counter[str] = Counter()
    failure_ratios: list[float] = []
    wall_seconds: list[float] = []
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
            grammar,
            sample_index,
            "individual_counterfactual_probability",
        )
        treatment, outcome, factual, counterfactual, factual_outcome, target_outcome = (
            _query_indices(world, rendered)
        )
        models: list[Any] = []
        started = time.perf_counter()
        try:
            with _capture_models(models, failure_ratios):
                result = sparse_individual_counterfactual_probability_bounds(
                    world,
                    treatment,
                    outcome,
                    factual_value=factual,
                    counterfactual_value=counterfactual,
                    factual_outcome_state=factual_outcome,
                    target_outcome_state=target_outcome,
                    time_limit_seconds=ENDPOINT_SECONDS,
                    conditional_endpoint_tolerance=CONDITIONAL_ENDPOINT_TOLERANCE,
                )
        except RuntimeError:
            outcomes["unresolved"] += 1
        else:
            outcomes[result.certification] += 1
        wall_seconds.append(time.perf_counter() - started)
        for model in models:
            diagnostics.update(model.projected_bound_diagnostics)

    ordered = sorted(wall_seconds)
    p95_index = max(
        0,
        min(len(ordered) - 1, (95 * len(ordered) + 99) // 100 - 1),
    )
    cells = diagnostics["cells"]
    tolerance_buckets: Counter[str] = Counter()
    for ratio in failure_ratios:
        if ratio <= 1.0:
            bucket = "le_1"
        elif ratio <= 2.0:
            bucket = "le_2"
        elif ratio <= 5.0:
            bucket = "le_5"
        elif ratio <= 10.0:
            bucket = "le_10"
        elif ratio <= 100.0:
            bucket = "le_100"
        else:
            bucket = "gt_100"
        tolerance_buckets[bucket] += 1
    print(
        json.dumps(
            {
                "cohort_size": DISTRIBUTION_COUNT,
                "outcomes": dict(sorted(outcomes.items())),
                "message_cells": cells,
                "positive_lower_cells": diagnostics["positive_lower"],
                "strict_upper_cells": diagnostics["strict_upper"],
                "positive_lower_rate": (
                    diagnostics["positive_lower"] / cells if cells else 0.0
                ),
                "strict_upper_rate": (
                    diagnostics["strict_upper"] / cells if cells else 0.0
                ),
                "lower_sum": diagnostics["lower_sum"],
                "upper_reduction_sum": diagnostics["upper_reduction_sum"],
                "failure_tolerance_gap_ratio": dict(sorted(tolerance_buckets.items())),
                "p50_wall_seconds": statistics.median(ordered),
                "p95_wall_seconds": ordered[p95_index],
                "total_wall_seconds": sum(ordered),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
