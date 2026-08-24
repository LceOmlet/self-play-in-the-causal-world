"""Fixed-seed feasibility probe for the exact counterfactual truth owner."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import time
from ctypes import wintypes
from typing import Any

from cpt_world import (
    WorldGrammar,
    counterfactual_transition_bounds,
    iter_sampled_seeds,
    sample_task_world,
    sparse_counterfactual_transition_bounds,
)


def _query_indices(world: Any, seed: dict[str, Any]) -> tuple[int, int, int, int, int]:
    query = seed["query"]
    visible_to_internal = {
        visible: internal for internal, visible in seed["visible_schema"]["variable_labels"].items()
    }
    treatment = world.variables.index(visible_to_internal[query["treatment"]])
    outcome = world.variables.index(visible_to_internal[query["outcome"]])
    baseline = int(str(query["baseline_value"]).removeprefix("state_"))
    comparison = int(str(query["treatment_value"]).removeprefix("state_"))
    outcome_state = int(str(query["outcome_state"]).removeprefix("state_"))
    return treatment, outcome, baseline, comparison, outcome_state


def _direct_only(world: Any, treatment: int, outcome: int) -> bool:
    descendants: set[int] = set()
    stack = [treatment]
    while stack:
        current = stack.pop()
        for parent, child in world.edges:
            if parent == current and child not in descendants:
                descendants.add(child)
                stack.append(child)
    return (treatment, outcome) in world.edges and all(
        parent == treatment or parent not in descendants for parent in world.parents[outcome]
    )


def _peak_working_set_mb() -> float | None:
    if os.name != "nt":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = wintypes.HANDLE
    get_process_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_process_memory.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    )
    get_process_memory.restype = wintypes.BOOL
    process = get_current_process()
    if not get_process_memory(
        process,
        ctypes.byref(counters),
        counters.cb,
    ):
        return None
    return counters.PeakWorkingSetSize / (1024 * 1024)


def run_probe(start_seed: int, count: int, endpoint_seconds: float) -> dict[str, Any]:
    grammar = WorldGrammar()
    records: list[dict[str, Any]] = []
    for sample_index in range(start_seed, start_seed + count):
        seed = iter_sampled_seeds(
            grammar,
            start_seed=sample_index,
            count=1,
            query_types=("counterfactual_transition_bounds",),
        )[0]
        world = sample_task_world(grammar, sample_index, "counterfactual_transition_bounds")
        treatment, outcome, baseline, comparison, outcome_state = _query_indices(world, seed)
        started = time.perf_counter()
        if _direct_only(world, treatment, outcome):
            lower, upper = counterfactual_transition_bounds(
                world,
                treatment,
                outcome,
                treatment_value=comparison,
                baseline_value=baseline,
                outcome_state=outcome_state,
            )
            record = {
                "seed": sample_index,
                "nodes": len(world.variables),
                "path": "direct_closed_form",
                "closed": True,
                "lower": float(lower),
                "upper": float(upper),
                "generated_columns": 0,
            }
        else:
            try:
                result = sparse_counterfactual_transition_bounds(
                    world,
                    treatment,
                    outcome,
                    treatment_value=comparison,
                    baseline_value=baseline,
                    outcome_state=outcome_state,
                    time_limit_seconds=endpoint_seconds,
                )
            except RuntimeError as error:
                record = {
                    "seed": sample_index,
                    "nodes": len(world.variables),
                    "path": "on_demand_columns",
                    "closed": False,
                    "error": str(error),
                }
            else:
                record = {
                    "seed": sample_index,
                    "nodes": len(world.variables),
                    "path": "on_demand_columns",
                    "closed": True,
                    "lower": result.lower,
                    "upper": result.upper,
                    "build_seconds": result.build_seconds,
                    "solve_seconds": result.solve_seconds,
                    "generated_columns": result.generated_columns,
                    "response_blocks": result.response_blocks,
                    "dynamic_response_blocks": result.dynamic_response_blocks,
                    "max_response_contexts": result.max_response_contexts,
                    "auxiliary_variables": result.auxiliary_variables,
                }
        record["wall_seconds"] = time.perf_counter() - started
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    times = sorted(float(record["wall_seconds"]) for record in records)
    closed = sum(bool(record["closed"]) for record in records)
    direct_records = [record for record in records if record["path"] == "direct_closed_form"]
    on_demand_records = [record for record in records if record["path"] == "on_demand_columns"]
    on_demand_closed = sum(bool(record["closed"]) for record in on_demand_records)
    p95_index = max(0, min(len(times) - 1, int(0.95 * len(times) + 0.999999) - 1))
    return {
        "contract": {
            "node_counts": [3, 15],
            "start_seed": start_seed,
            "count": count,
            "requested_scip_endpoint_seconds": endpoint_seconds,
            "strict_wall_limit": False,
            "truth_requires_optimal": True,
        },
        "closed": closed,
        "closed_rate": closed / count,
        "direct_closed_form": len(direct_records),
        "on_demand": len(on_demand_records),
        "on_demand_closed": on_demand_closed,
        "on_demand_closed_rate": (
            on_demand_closed / len(on_demand_records) if on_demand_records else 1.0
        ),
        "p50_wall_seconds": statistics.median(times),
        "p95_wall_seconds": times[p95_index],
        "max_wall_seconds": max(times),
        "peak_working_set_mb": _peak_working_set_mb(),
        "generated_columns": sum(int(record.get("generated_columns", 0)) for record in records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument(
        "--endpoint-seconds",
        type=float,
        default=10.0,
        help="requested SCIP time limit per endpoint; not a strict process wall limit",
    )
    args = parser.parse_args()
    if args.start_seed < 0 or args.count <= 0 or args.endpoint_seconds <= 0:
        parser.error("start-seed must be nonnegative; count and endpoint-seconds must be positive")
    summary = run_probe(args.start_seed, args.count, args.endpoint_seconds)
    print(json.dumps({"summary": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
