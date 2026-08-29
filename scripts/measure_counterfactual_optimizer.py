"""Aggregate-only measurement harness for counterfactual solver optimization.

The optimization cohort is disjoint from the formal fixed-seed regression
cohort.  Neither cohort emits per-instance records: optimizer feedback is only
the distribution-level aggregate declared in the ce-optimize specification.
"""

from __future__ import annotations

import ast
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from cpt_world import (
    WorldGrammar,
    iter_sampled_seeds,
    sample_task_world,
    sparse_individual_counterfactual_probability_bounds,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOLVER_PATH = REPO_ROOT / "src" / "cpt_world" / "counterfactual_solver.py"
FORMAL_START_SEED = 0
FORMAL_COUNT = 30
DISTRIBUTION_START_SEED = 10_000
DISTRIBUTION_COUNT = 30
ENDPOINT_SECONDS = 5.0
CONDITIONAL_ENDPOINT_TOLERANCE = 1e-3

IMMUTABLE_PATHS = (
    "scripts/measure_counterfactual_optimizer.py",
    "scripts/probe_counterfactual_solver.py",
    "src/cpt_world/world_space.py",
    "src/cpt_world/query_truth.py",
    "src/cpt_world/rendering.py",
    "src/cpt_world/rewards.py",
    "src/cpt_world/registry.py",
    "tests",
    "pyproject.toml",
)


def _query_indices(world: Any, seed: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    query = seed["query"]
    visible_to_internal = {
        visible: internal for internal, visible in seed["visible_schema"]["variable_labels"].items()
    }
    treatment = world.variables.index(visible_to_internal[query["treatment"]])
    outcome = world.variables.index(visible_to_internal[query["outcome"]])
    factual = int(str(query["factual_value"]).removeprefix("state_"))
    counterfactual = int(str(query["counterfactual_value"]).removeprefix("state_"))
    factual_outcome = int(str(query["factual_outcome_state"]).removeprefix("state_"))
    target_outcome = int(str(query["outcome_state"]).removeprefix("state_"))
    return treatment, outcome, factual, counterfactual, factual_outcome, target_outcome


def _measure_cohort(start_seed: int, count: int) -> dict[str, float | int]:
    grammar = WorldGrammar()
    exact = 0
    epsilon_sharp = 0
    unresolved = 0
    wall_seconds: list[float] = []
    for sample_index in range(start_seed, start_seed + count):
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
        treatment, outcome, factual, counterfactual, factual_outcome, target_outcome = (
            _query_indices(world, seed)
        )
        started = time.perf_counter()
        try:
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
            unresolved += 1
        else:
            if result.certification == "epsilon_sharp":
                epsilon_sharp += 1
            else:
                exact += 1
        wall_seconds.append(time.perf_counter() - started)

    ordered = sorted(wall_seconds)
    p95_index = max(0, min(len(ordered) - 1, (95 * len(ordered) + 99) // 100 - 1))
    closed = exact + epsilon_sharp
    return {
        "closed_rate": closed / count,
        "exact_count": exact,
        "epsilon_sharp_count": epsilon_sharp,
        "unresolved_count": unresolved,
        "p50_wall_seconds": statistics.median(ordered),
        "p95_wall_seconds": ordered[p95_index],
        "total_wall_seconds": sum(ordered),
    }


def _focused_tests_pass() -> bool:
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_counterfactual_solver"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def _immutable_contract_passes() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--", *IMMUTABLE_PATHS],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0 and not completed.stdout.strip()


def _anti_case_specific_scan_passes() -> bool:
    source = SOLVER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_names = {
        "sample_index",
        "seed",
        "seed_id",
        "rng_seed",
        "world_seed",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            return False
        if isinstance(node, ast.Attribute) and node.attr == "topology":
            parent = next(
                (
                    candidate
                    for candidate in ast.walk(tree)
                    if isinstance(candidate, (ast.If, ast.IfExp, ast.Match, ast.Compare))
                    and node in ast.walk(candidate)
                ),
                None,
            )
            if parent is not None:
                return False
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            if "sampled-" in lowered or "seed-specific" in lowered:
                return False
    return True


def main() -> None:
    tests_passed = _focused_tests_pass()
    immutable_passed = _immutable_contract_passes()
    anti_case_passed = _anti_case_specific_scan_passes()
    if not (tests_passed and immutable_passed and anti_case_passed):
        payload = {
            "focused_tests_passed": int(tests_passed),
            "semantic_parity_passed": int(tests_passed),
            "immutable_contract_passed": int(immutable_passed),
            "anti_case_specific_scan_passed": int(anti_case_passed),
        }
        print(json.dumps(payload, sort_keys=True))
        raise SystemExit(1)

    distribution = _measure_cohort(DISTRIBUTION_START_SEED, DISTRIBUTION_COUNT)
    formal = _measure_cohort(FORMAL_START_SEED, FORMAL_COUNT)
    payload = {
        "focused_tests_passed": 1,
        "semantic_parity_passed": 1,
        "immutable_contract_passed": 1,
        "anti_case_specific_scan_passed": 1,
        "distribution_closed_rate": distribution["closed_rate"],
        "formal_regression_closed_rate": formal["closed_rate"],
        "distribution_exact_count": distribution["exact_count"],
        "distribution_epsilon_sharp_count": distribution["epsilon_sharp_count"],
        "distribution_unresolved_count": distribution["unresolved_count"],
        "distribution_p50_wall_seconds": distribution["p50_wall_seconds"],
        "distribution_p95_wall_seconds": distribution["p95_wall_seconds"],
        "distribution_total_wall_seconds": distribution["total_wall_seconds"],
        "formal_exact_count": formal["exact_count"],
        "formal_epsilon_sharp_count": formal["epsilon_sharp_count"],
        "formal_unresolved_count": formal["unresolved_count"],
        "formal_p50_wall_seconds": formal["p50_wall_seconds"],
        "formal_p95_wall_seconds": formal["p95_wall_seconds"],
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
