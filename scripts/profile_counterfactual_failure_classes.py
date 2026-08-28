"""Aggregate-only failure profiler for exact counterfactual closure.

This diagnostic uses the frozen optimization cohort but never emits sample
indices, variable labels, graph names, CPT values, or per-instance records.
It records which exact owner failed and coarse, relabeling-invariant structural
histograms for closed and unresolved instances.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import combinations
from typing import Any

from pyscipopt import SCIP_BRANCHDIR

from cpt_world import (
    WorldGrammar,
    iter_sampled_seeds,
    sample_task_world,
    sparse_individual_counterfactual_probability_bounds,
)
from cpt_world import counterfactual_solver as solver

DISTRIBUTION_START_SEED = 10_000
DISTRIBUTION_COUNT = 30
ENDPOINT_SECONDS = 5.0
CONDITIONAL_ENDPOINT_TOLERANCE = 2e-3


def _query_indices(
    world: Any,
    seed: dict[str, Any],
) -> tuple[int, int, int, int, int, int]:
    query = seed["query"]
    visible_to_internal = {
        visible: internal
        for internal, visible in seed["visible_schema"]["variable_labels"].items()
    }
    treatment = world.variables.index(visible_to_internal[query["treatment"]])
    outcome = world.variables.index(visible_to_internal[query["outcome"]])
    factual = int(str(query["factual_value"]).removeprefix("state_"))
    counterfactual = int(str(query["counterfactual_value"]).removeprefix("state_"))
    factual_outcome = int(
        str(query["factual_outcome_state"]).removeprefix("state_")
    )
    target_outcome = int(str(query["outcome_state"]).removeprefix("state_"))
    return (
        treatment,
        outcome,
        factual,
        counterfactual,
        factual_outcome,
        target_outcome,
    )


def _bucket(value: int, cuts: tuple[int, ...]) -> str:
    for cut in cuts:
        if value <= cut:
            return f"le_{cut}"
    return f"gt_{cuts[-1]}"


def _structure_record(
    world: Any,
    treatment: int,
    outcome: int,
    *,
    baseline_value: int,
    treatment_value: int,
    factual_outcome_state: int,
    target_outcome_state: int,
) -> dict[str, Any]:
    ancestors = solver._ancestors(world, outcome) | {outcome}
    affected = tuple(
        node
        for node in solver._topological_order(world)
        if node in (solver._descendants(world, treatment) & ancestors)
    )
    affected_set = set(affected)
    changed_parent_counts = tuple(
        sum(parent == treatment or parent in affected_set for parent in world.parents[node])
        for node in affected
    )
    changed_context_sizes = tuple(
        max(
            1,
            _product(
                world.domains[parent]
                for parent in world.parents[node]
                if parent == treatment or parent in affected_set
            ),
        )
        for node in affected
    )
    shared_roots = tuple(
        node
        for node in ancestors
        if node not in affected_set
        and node != treatment
        and not world.parents[node]
        and any(node in world.parents[affected_node] for affected_node in affected)
    )
    context_divisors = tuple(
        _product(
            world.domains[parent]
            for parent in world.parents[node]
            if parent in shared_roots
        )
        for node in affected
    )
    response_log10_reductions = tuple(
        (
            len(
                solver._active_contexts(
                    world,
                    node,
                    treatment,
                    baseline_value=0,
                    treatment_value=min(1, world.domains[treatment] - 1),
                )
            )
            * (1.0 - 1.0 / divisor)
            * math.log10(world.domains[node])
        )
        if divisor > 1
        else 0.0
        for node, divisor in zip(affected, context_divisors, strict=True)
    )
    single_root_log10_reductions = tuple(
        max(
            (
                len(
                    solver._active_contexts(
                        world,
                        node,
                        treatment,
                        baseline_value=0,
                        treatment_value=min(1, world.domains[treatment] - 1),
                    )
                )
                * (1.0 - 1.0 / world.domains[root])
                * math.log10(world.domains[node])
            )
            for node in affected
            if root in world.parents[node]
        )
        for root in shared_roots
    )
    best_single_root = (
        max(
            range(len(shared_roots)),
            key=lambda index: (
                single_root_log10_reductions[index],
                -world.domains[shared_roots[index]],
                -shared_roots[index],
            ),
        )
        if shared_roots
        else None
    )
    outcome_events = ((factual_outcome_state,), (target_outcome_state,))
    both_terminal_endpoints = all(
        solver._terminal_event_endpoint_is_jointly_attainable(
            world,
            treatment,
            outcome,
            outcome_events,
            endpoint,
            baseline_value=baseline_value,
            treatment_value=treatment_value,
        )
        for endpoint in ("lower", "upper")
    )
    terminal_reduced_affected = tuple(
        node for node in affected if node != outcome
    )
    terminal_reduced_root_separator = bool(
        both_terminal_endpoints
        and terminal_reduced_affected
        and any(
            not world.parents[root]
            and root not in affected_set
            and root != treatment
            and all(
                root in world.parents[node]
                for node in terminal_reduced_affected
            )
            for root in ancestors
        )
    )
    generalized_one_mediator = False
    indirect_layered_one_mediator = False
    two_mechanism_signature = "not_two_mechanisms"
    if len(affected) == 2 and affected[-1] == outcome:
        mediator = affected[0]
        two_mechanism_signature = (
            f"direct_{str(treatment in world.parents[outcome]).lower()}"
            f"_x_to_m_{str(treatment in world.parents[mediator]).lower()}"
            f"_m_to_y_{str(mediator in world.parents[outcome]).lower()}"
            f"_shared_m_parents_{sum(parent != treatment for parent in world.parents[mediator])}"
            f"_m_domain_{world.domains[mediator]}"
        )
        generalized_one_mediator = bool(
            treatment in world.parents[mediator]
            and treatment in world.parents[outcome]
            and mediator in world.parents[outcome]
            and all(
                parent == treatment or parent not in affected_set
                for parent in world.parents[mediator]
            )
        )
        indirect_layered_one_mediator = bool(
            world.parents[mediator] == (treatment,)
            and treatment not in world.parents[outcome]
            and mediator in world.parents[outcome]
            and world.domains[mediator] <= 4
        )
    return {
        "affected_mechanisms": len(affected),
        "direct_terminal": treatment in world.parents[outcome],
        "max_changed_parents": max(changed_parent_counts, default=0),
        "max_changed_contexts": max(changed_context_sizes, default=1),
        "has_convergence": any(count >= 2 for count in changed_parent_counts),
        "legacy_root_separator": solver._root_separator(
            world, treatment, outcome
        )
        is not None,
        "shared_root_cutset_size": len(shared_roots),
        "shared_root_strata": _product(
            world.domains[node] for node in shared_roots
        ),
        "shared_root_affected_coverage": sum(
            divisor > 1 for divisor in context_divisors
        ),
        "max_context_divisor": max(context_divisors, default=1),
        "max_response_log10_reduction": max(
            response_log10_reductions, default=0.0
        ),
        "best_single_root_strata": (
            world.domains[shared_roots[best_single_root]]
            if best_single_root is not None
            else 1
        ),
        "best_single_root_log10_reduction": (
            single_root_log10_reductions[best_single_root]
            if best_single_root is not None
            else 0.0
        ),
        "both_terminal_endpoints": both_terminal_endpoints,
        "terminal_reduced_root_separator": terminal_reduced_root_separator,
        "generalized_one_mediator": generalized_one_mediator,
        "indirect_layered_one_mediator": indirect_layered_one_mediator,
        "two_mechanism_signature": two_mechanism_signature,
    }


def _product(values: Iterator[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _elimination_profile(
    world: Any,
    treatment: int,
    outcome: int,
    *,
    terminal_endpoint: str,
    order_mode: str,
) -> tuple[int, int]:
    """Return total and peak symbolic message cells for one exact order."""

    ancestors = solver._ancestors(world, outcome) | {outcome}
    affected_set = solver._descendants(world, treatment) & ancestors
    order = solver._topological_order(world)
    mechanism_affected = tuple(
        node
        for node in order
        if node in affected_set
        and not (terminal_endpoint in {"lower", "upper"} and node == outcome)
    )
    shared = tuple(
        node
        for node in order
        if node in ancestors and node not in affected_set and node != treatment
    )
    relevant = set(mechanism_affected) | set(shared)

    def group(node: int) -> tuple[tuple[int, int], ...]:
        return (
            ((node, 0), (node, 1))
            if node in affected_set
            else ((node, -1),)
        )

    groups = {node: group(node) for node in relevant}

    def parent_tokens(node: int) -> list[tuple[int, int]]:
        tokens: list[tuple[int, int]] = []
        for parent in world.parents[node]:
            if parent == treatment:
                continue
            if parent in affected_set:
                tokens.extend(((parent, 0), (parent, 1)))
            else:
                tokens.append((parent, -1))
        return tokens

    factors: list[set[tuple[int, int]]] = []
    if terminal_endpoint in {"lower", "upper"}:
        factors.append(set(parent_tokens(outcome)))
    for node in relevant:
        factors.append(set((*parent_tokens(node), *groups[node])))

    def token_domain(token: tuple[int, int]) -> int:
        node, world_index = token
        if node == outcome and world_index in {0, 1}:
            return 1
        return world.domains[node]

    remaining = set(relevant)
    total_cells = 0
    peak_cells = 0
    reverse_order = tuple(node for node in reversed(order) if node in relevant)
    while remaining:
        if order_mode == "reverse_topological":
            node = next(item for item in reverse_order if item in remaining)
        elif order_mode == "min_fill":
            candidates = []
            for item in remaining:
                item_group = set(groups[item])
                involving = [factor for factor in factors if factor & item_group]
                output_scope = set().union(*involving) - item_group
                cells = _product(token_domain(token) for token in output_scope)
                fill_pairs = len(output_scope) * max(0, len(output_scope) - 1) // 2
                candidates.append((cells, fill_pairs, len(output_scope), item))
            node = min(candidates)[-1]
        else:
            raise ValueError(f"unknown elimination order: {order_mode}")
        item_group = set(groups[node])
        involving = [factor for factor in factors if factor & item_group]
        output_scope = set().union(*involving) - item_group
        cells = _product(token_domain(token) for token in output_scope)
        total_cells += cells
        peak_cells = max(peak_cells, cells)
        factors = [factor for factor in factors if not factor & item_group]
        factors.append(output_scope)
        remaining.remove(node)
    return total_cells, peak_cells


def _ratio_bucket(value: float) -> str:
    if value <= 0.1:
        return "le_0.1"
    if value <= 0.25:
        return "le_0.25"
    if value <= 0.5:
        return "le_0.5"
    if value <= 0.75:
        return "le_0.75"
    if value <= 1.0:
        return "le_1.0"
    return "gt_1.0"


def _progress_bucket(value: float) -> str:
    if value <= 0.01:
        return "le_0.01"
    if value <= 0.1:
        return "le_0.1"
    if value <= 0.5:
        return "le_0.5"
    if value <= 0.9:
        return "le_0.9"
    return "gt_0.9"


def _error_fields(message: str) -> tuple[str, str, str, float | None, float | None]:
    status = re.search(r"status=([^, )]+)", message)
    pricing_closed = re.search(r"pricing_closed=([^, )]+)", message)
    pricing_timed_out = re.search(r"pricing_timed_out=([^, )]+)", message)
    primal = re.search(r"primal=([^, )]+)", message)
    dual = re.search(r"dual=([^, )]+)", message)

    def parsed(match: re.Match[str] | None) -> float | None:
        if match is None:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    return (
        status.group(1) if status else "unreported",
        pricing_closed.group(1) if pricing_closed else "unreported",
        pricing_timed_out.group(1) if pricing_timed_out else "unreported",
        parsed(primal),
        parsed(dual),
    )


@contextmanager
def _trace_sparse_optimization(
    calls: list[dict[str, Any]],
) -> Iterator[None]:
    original = solver._SparseResponseModel.optimize

    def traced(self: Any, *args: Any, **kwargs: Any) -> tuple[float, float]:
        static_column_counts = tuple(
            len(block.columns) for block in self.pricing_blocks if not block.dynamic
        )
        kernel_entries_by_node = Counter(
            key[0] for key in self.kernel_cache
        )
        dynamic_complete_column_counts = tuple(
            self.world.domains[block.node] ** len(block.contexts)
            for block in self.dynamic_pricing_blocks
        )
        record = {
            "sense": self.sense,
            "terminal_endpoint": self.terminal_event_endpoint or "joint",
            "affected_mechanisms": len(self.affected),
            "response_blocks": len(self.pricing_blocks),
            "dynamic_blocks": len(self.dynamic_pricing_blocks),
            "max_response_contexts": max(
                (len(block.contexts) for block in self.pricing_blocks),
                default=0,
            ),
            "static_response_columns": sum(static_column_counts),
            "max_static_response_columns": max(static_column_counts, default=0),
            "pairwise_static_column_cells": sum(
                left * right
                for left, right in combinations(static_column_counts, 2)
            ),
            "pairwise_kernel_cells": sum(
                left * right
                for left, right in combinations(
                    tuple(kernel_entries_by_node.values()), 2
                )
            ),
            "small_dynamic_blocks": sum(
                count <= solver._LEGACY_MAX_EXPLICIT_RESPONSE_COLUMNS
                for count in dynamic_complete_column_counts
            ),
            "all_dynamic_blocks_small": bool(dynamic_complete_column_counts)
            and all(
                count <= solver._LEGACY_MAX_EXPLICIT_RESPONSE_COLUMNS
                for count in dynamic_complete_column_counts
            ),
            "max_dynamic_complete_columns": max(
                dynamic_complete_column_counts, default=0
            ),
        }
        try:
            result = original(self, *args, **kwargs)
        except RuntimeError as exc:
            status, pricing_closed, pricing_timed_out, primal, dual = _error_fields(
                str(exc)
            )
            record.update(
                {
                    "outcome": "failed",
                    "status": status,
                    "pricing_closed": pricing_closed,
                    "pricing_timed_out": pricing_timed_out,
                    "primal": primal,
                    "dual": dual,
                    "outer_width": max(
                        0.0,
                        self.target_outer_bounds[1] - self.target_outer_bounds[0],
                    ),
                    "outer_lower": self.target_outer_bounds[0],
                    "outer_upper": self.target_outer_bounds[1],
                    "auxiliary_variables": len(self.auxiliary_values),
                    "master_variables": self.model.getNVars(),
                    "master_constraints": self.model.getNConss(),
                    "generated_columns": self.pricer.generated_columns,
                    "pricing_rounds": len(self.pricer.rounds),
                    "completed_pricing_rounds": sum(
                        round_state.completed for round_state in self.pricer.rounds
                    ),
                    "min_sum_calls": self.pricer.min_sum_calls,
                    "max_min_sum_width": self.pricer.max_min_sum_width,
                    "pricing_scip_fallback_calls": self.pricer.scip_fallback_calls,
                    "solving_nodes": self.model.getNNodes(),
                    "lp_iterations": self.model.getNLPIterations(),
                    "lp_solves": self.model.getNLPs(),
                    "cuts_found": self.model.getNCuts(),
                    "cuts_applied": self.model.getNCutsApplied(),
                    "presolving_seconds": self.model.getPresolvingTime(),
                    "solving_seconds": self.model.getSolvingTime(),
                    "branchings_by_role": dict(
                        Counter(
                            (
                                "response_kernel"
                                if variable.name.startswith("k_")
                                else "elimination_message"
                                if variable.name.startswith("ve_")
                                else "response_column"
                                if variable.name.startswith("lambda_")
                                else "target"
                                if variable.name == "counterfactual_target"
                                else "other"
                            )
                            for variable in self.model.getVars(transformed=True)
                            for _ in range(
                                variable.getNBranchingsCurrentRun(
                                    SCIP_BRANCHDIR.DOWNWARDS
                                )
                                + variable.getNBranchingsCurrentRun(
                                    SCIP_BRANCHDIR.UPWARDS
                                )
                            )
                        )
                    ),
                }
            )
            calls.append(record)
            raise
        record.update(
            {
                "outcome": "closed",
                "certification": self.last_certification,
            }
        )
        calls.append(record)
        return result

    solver._SparseResponseModel.optimize = traced
    try:
        yield
    finally:
        solver._SparseResponseModel.optimize = original


def _increment_histograms(
    histograms: dict[str, Counter[str]],
    prefix: str,
    structure: dict[str, Any],
) -> None:
    histograms[f"{prefix}_affected_mechanisms"][
        _bucket(structure["affected_mechanisms"], (1, 2, 3, 4))
    ] += 1
    histograms[f"{prefix}_max_changed_parents"][
        _bucket(structure["max_changed_parents"], (1, 2, 3))
    ] += 1
    histograms[f"{prefix}_max_changed_contexts"][
        _bucket(structure["max_changed_contexts"], (2, 4, 8, 16, 32))
    ] += 1
    histograms[f"{prefix}_direct_terminal"][
        str(structure["direct_terminal"]).lower()
    ] += 1
    histograms[f"{prefix}_has_convergence"][
        str(structure["has_convergence"]).lower()
    ] += 1
    histograms[f"{prefix}_legacy_root_separator"][
        str(structure["legacy_root_separator"]).lower()
    ] += 1
    histograms[f"{prefix}_shared_root_cutset_size"][
        _bucket(structure["shared_root_cutset_size"], (0, 1, 2, 3))
    ] += 1
    histograms[f"{prefix}_shared_root_strata"][
        _bucket(structure["shared_root_strata"], (1, 2, 4, 8, 16))
    ] += 1
    histograms[f"{prefix}_shared_root_affected_coverage"][
        _bucket(structure["shared_root_affected_coverage"], (0, 1, 2, 3))
    ] += 1
    histograms[f"{prefix}_max_context_divisor"][
        _bucket(structure["max_context_divisor"], (1, 2, 3, 4, 5))
    ] += 1
    log_reduction = structure["max_response_log10_reduction"]
    if log_reduction <= 0.0:
        reduction_bucket = "le_0"
    elif log_reduction <= 1.0:
        reduction_bucket = "le_1"
    elif log_reduction <= 3.0:
        reduction_bucket = "le_3"
    elif log_reduction <= 6.0:
        reduction_bucket = "le_6"
    elif log_reduction <= 12.0:
        reduction_bucket = "le_12"
    else:
        reduction_bucket = "gt_12"
    histograms[f"{prefix}_max_response_log10_reduction"][reduction_bucket] += 1
    histograms[f"{prefix}_best_single_root_strata"][
        _bucket(structure["best_single_root_strata"], (1, 2, 3, 4, 5))
    ] += 1
    single_reduction = structure["best_single_root_log10_reduction"]
    if single_reduction <= 0.0:
        single_bucket = "le_0"
    elif single_reduction <= 1.0:
        single_bucket = "le_1"
    elif single_reduction <= 3.0:
        single_bucket = "le_3"
    elif single_reduction <= 6.0:
        single_bucket = "le_6"
    elif single_reduction <= 12.0:
        single_bucket = "le_12"
    else:
        single_bucket = "gt_12"
    histograms[f"{prefix}_best_single_root_log10_reduction"][single_bucket] += 1
    histograms[f"{prefix}_both_terminal_endpoints"][
        str(structure["both_terminal_endpoints"]).lower()
    ] += 1
    histograms[f"{prefix}_terminal_reduced_root_separator"][
        str(structure["terminal_reduced_root_separator"]).lower()
    ] += 1
    histograms[f"{prefix}_generalized_one_mediator"][
        str(structure["generalized_one_mediator"]).lower()
    ] += 1
    histograms[f"{prefix}_indirect_layered_one_mediator"][
        str(structure["indirect_layered_one_mediator"]).lower()
    ] += 1
    histograms[f"{prefix}_two_mechanism_signature"][
        structure["two_mechanism_signature"]
    ] += 1


def main() -> None:
    grammar = WorldGrammar()
    totals = Counter[str]()
    histograms: dict[str, Counter[str]] = {
        name: Counter()
        for prefix in ("closed", "unresolved")
        for name in (
            f"{prefix}_affected_mechanisms",
            f"{prefix}_max_changed_parents",
            f"{prefix}_max_changed_contexts",
            f"{prefix}_direct_terminal",
            f"{prefix}_has_convergence",
            f"{prefix}_legacy_root_separator",
            f"{prefix}_shared_root_cutset_size",
            f"{prefix}_shared_root_strata",
            f"{prefix}_shared_root_affected_coverage",
            f"{prefix}_max_context_divisor",
            f"{prefix}_max_response_log10_reduction",
            f"{prefix}_best_single_root_strata",
            f"{prefix}_best_single_root_log10_reduction",
            f"{prefix}_both_terminal_endpoints",
            f"{prefix}_terminal_reduced_root_separator",
            f"{prefix}_generalized_one_mediator",
            f"{prefix}_indirect_layered_one_mediator",
            f"{prefix}_two_mechanism_signature",
        )
    }
    failure_owner = Counter[str]()
    failure_status = Counter[str]()
    failure_pricing = Counter[str]()
    failure_model_width = Counter[str]()
    failure_auxiliaries = Counter[str]()
    failure_master_variables = Counter[str]()
    failure_generated_columns = Counter[str]()
    failure_pricing_rounds = Counter[str]()
    failure_pricing_backend = Counter[str]()
    failure_normalized_gap = Counter[str]()
    failure_min_fill_total_ratio = Counter[str]()
    failure_min_fill_peak_ratio = Counter[str]()
    failure_static_response_columns = Counter[str]()
    failure_pairwise_static_column_cells = Counter[str]()
    failure_pairwise_kernel_cells = Counter[str]()
    failure_primal_progress = Counter[str]()
    failure_dual_progress = Counter[str]()
    failure_lagging_side = Counter[str]()
    failure_solving_nodes = Counter[str]()
    failure_lp_iterations = Counter[str]()
    failure_branchings_by_role = Counter[str]()
    failure_lp_solves = Counter[str]()
    failure_cuts_applied = Counter[str]()
    failure_small_dynamic_blocks = Counter[str]()
    failure_all_dynamic_blocks_small = Counter[str]()
    failure_max_dynamic_complete_columns = Counter[str]()

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
        (
            treatment,
            outcome,
            factual,
            counterfactual,
            factual_outcome,
            target_outcome,
        ) = _query_indices(world, seed)
        structure = _structure_record(
            world,
            treatment,
            outcome,
            baseline_value=factual,
            treatment_value=counterfactual,
            factual_outcome_state=factual_outcome,
            target_outcome_state=target_outcome,
        )
        calls: list[dict[str, Any]] = []
        try:
            with _trace_sparse_optimization(calls):
                sparse_individual_counterfactual_probability_bounds(
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
            prefix = "unresolved"
            totals["unresolved"] += 1
            failed = next(
                (record for record in reversed(calls) if record["outcome"] == "failed"),
                None,
            )
            if failed is None:
                failure_owner["non_sparse_owner"] += 1
            else:
                owner = f"{failed['terminal_endpoint']}:{failed['sense']}"
                failure_owner[owner] += 1
                failure_status[failed["status"]] += 1
                failure_pricing[
                    "closed_"
                    + failed["pricing_closed"]
                    + "_timedout_"
                    + failed["pricing_timed_out"]
                ] += 1
                failure_model_width[
                    _bucket(failed["max_response_contexts"], (2, 4, 8, 16, 32))
                ] += 1
                failure_auxiliaries[
                    _bucket(failed["auxiliary_variables"], (10, 50, 100, 250, 500))
                ] += 1
                failure_master_variables[
                    _bucket(failed["master_variables"], (100, 500, 1000, 2500, 5000))
                ] += 1
                failure_generated_columns[
                    _bucket(failed["generated_columns"], (0, 10, 50, 100, 250))
                ] += 1
                failure_pricing_rounds[
                    _bucket(failed["pricing_rounds"], (0, 1, 2, 4, 8))
                ] += 1
                pricing_backend = (
                    "scip_fallback"
                    if failed["pricing_scip_fallback_calls"]
                    else "min_sum_only"
                    if failed["min_sum_calls"]
                    else "no_pricing_map"
                )
                failure_pricing_backend[pricing_backend] += 1
                failure_static_response_columns[
                    _bucket(
                        failed["static_response_columns"],
                        (0, 32, 128, 512, 2048),
                    )
                ] += 1
                failure_pairwise_static_column_cells[
                    _bucket(
                        failed["pairwise_static_column_cells"],
                        (0, 1_000, 10_000, 100_000, 1_000_000),
                    )
                ] += 1
                failure_pairwise_kernel_cells[
                    _bucket(
                        failed["pairwise_kernel_cells"],
                        (0, 1_000, 10_000, 100_000, 1_000_000),
                    )
                ] += 1
                current_total, current_peak = _elimination_profile(
                    world,
                    treatment,
                    outcome,
                    terminal_endpoint=failed["terminal_endpoint"],
                    order_mode="reverse_topological",
                )
                min_fill_total, min_fill_peak = _elimination_profile(
                    world,
                    treatment,
                    outcome,
                    terminal_endpoint=failed["terminal_endpoint"],
                    order_mode="min_fill",
                )
                failure_min_fill_total_ratio[
                    _ratio_bucket(min_fill_total / max(current_total, 1))
                ] += 1
                failure_min_fill_peak_ratio[
                    _ratio_bucket(min_fill_peak / max(current_peak, 1))
                ] += 1
                primal = failed["primal"]
                dual = failed["dual"]
                failure_solving_nodes[
                    _bucket(failed["solving_nodes"], (1, 10, 100, 1_000, 10_000))
                ] += 1
                failure_lp_iterations[
                    _bucket(
                        failed["lp_iterations"],
                        (100, 1_000, 10_000, 100_000, 1_000_000),
                    )
                ] += 1
                failure_branchings_by_role.update(failed["branchings_by_role"])
                failure_lp_solves[
                    _bucket(failed["lp_solves"], (1, 5, 10, 25, 50, 100))
                ] += 1
                failure_cuts_applied[
                    _bucket(failed["cuts_applied"], (0, 10, 100, 1_000, 10_000))
                ] += 1
                failure_small_dynamic_blocks[
                    _bucket(failed["small_dynamic_blocks"], (0, 1, 2, 3))
                ] += 1
                failure_all_dynamic_blocks_small[
                    str(failed["all_dynamic_blocks_small"]).lower()
                ] += 1
                failure_max_dynamic_complete_columns[
                    _bucket(
                        failed["max_dynamic_complete_columns"],
                        (32, 256, 3_125, 65_536, 1_000_000),
                    )
                ] += 1
                if primal is None or dual is None:
                    failure_normalized_gap["unreported"] += 1
                else:
                    width = max(failed["outer_width"], 1e-15)
                    outer_lower, outer_upper = (
                        failed.get("outer_lower"), failed.get("outer_upper")
                    )
                    if outer_lower is None or outer_upper is None:
                        outer_lower = 0.0
                        outer_upper = width
                    if failed["sense"] == "maximize":
                        primal_progress = (primal - outer_lower) / width
                        dual_progress = (outer_upper - dual) / width
                    else:
                        primal_progress = (outer_upper - primal) / width
                        dual_progress = (dual - outer_lower) / width
                    failure_primal_progress[_progress_bucket(primal_progress)] += 1
                    failure_dual_progress[_progress_bucket(dual_progress)] += 1
                    failure_lagging_side[
                        "primal" if primal_progress < dual_progress else "dual"
                    ] += 1
                    normalized_gap = abs(primal - dual) / width
                    if normalized_gap <= 0.002:
                        gap_bucket = "le_0.002"
                    elif normalized_gap <= 0.01:
                        gap_bucket = "le_0.01"
                    elif normalized_gap <= 0.1:
                        gap_bucket = "le_0.1"
                    elif normalized_gap <= 0.5:
                        gap_bucket = "le_0.5"
                    else:
                        gap_bucket = "gt_0.5"
                    failure_normalized_gap[gap_bucket] += 1
        else:
            prefix = "closed"
            totals["closed"] += 1
        _increment_histograms(histograms, prefix, structure)

    payload = {
        "cohort_size": DISTRIBUTION_COUNT,
        "totals": dict(sorted(totals.items())),
        "histograms": {
            name: dict(sorted(counter.items()))
            for name, counter in sorted(histograms.items())
        },
        "unresolved_failure_owner": dict(sorted(failure_owner.items())),
        "unresolved_failure_status": dict(sorted(failure_status.items())),
        "unresolved_pricing_state": dict(sorted(failure_pricing.items())),
        "unresolved_model_context_width": dict(sorted(failure_model_width.items())),
        "unresolved_auxiliary_variables": dict(sorted(failure_auxiliaries.items())),
        "unresolved_master_variables": dict(sorted(failure_master_variables.items())),
        "unresolved_generated_columns": dict(sorted(failure_generated_columns.items())),
        "unresolved_pricing_rounds": dict(sorted(failure_pricing_rounds.items())),
        "unresolved_pricing_backend": dict(sorted(failure_pricing_backend.items())),
        "unresolved_normalized_gap": dict(sorted(failure_normalized_gap.items())),
        "unresolved_min_fill_total_ratio": dict(
            sorted(failure_min_fill_total_ratio.items())
        ),
        "unresolved_min_fill_peak_ratio": dict(
            sorted(failure_min_fill_peak_ratio.items())
        ),
        "unresolved_static_response_columns": dict(
            sorted(failure_static_response_columns.items())
        ),
        "unresolved_pairwise_static_column_cells": dict(
            sorted(failure_pairwise_static_column_cells.items())
        ),
        "unresolved_pairwise_kernel_cells": dict(
            sorted(failure_pairwise_kernel_cells.items())
        ),
        "unresolved_primal_progress": dict(sorted(failure_primal_progress.items())),
        "unresolved_dual_progress": dict(sorted(failure_dual_progress.items())),
        "unresolved_lagging_side": dict(sorted(failure_lagging_side.items())),
        "unresolved_solving_nodes": dict(sorted(failure_solving_nodes.items())),
        "unresolved_lp_iterations": dict(sorted(failure_lp_iterations.items())),
        "unresolved_branchings_by_role": dict(
            sorted(failure_branchings_by_role.items())
        ),
        "unresolved_lp_solves": dict(sorted(failure_lp_solves.items())),
        "unresolved_cuts_applied": dict(sorted(failure_cuts_applied.items())),
        "unresolved_small_dynamic_blocks": dict(
            sorted(failure_small_dynamic_blocks.items())
        ),
        "unresolved_all_dynamic_blocks_small": dict(
            sorted(failure_all_dynamic_blocks_small.items())
        ),
        "unresolved_max_dynamic_complete_columns": dict(
            sorted(failure_max_dynamic_complete_columns.items())
        ),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
