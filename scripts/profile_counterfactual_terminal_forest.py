"""Aggregate coverage probe for exact terminal-forest elimination.

For a forest of queried terminal contexts, arbitrary edge couplings with the
fixed CPT marginals glue to one joint response law.  Hence every edgewise
Frechet endpoint is jointly attainable.  This probe reports whether that
complete structural class occurs among frozen unresolved owners without
emitting task, variable, topology, or CPT identities.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from itertools import product
from typing import Any

from measure_counterfactual_optimizer import (
    CONDITIONAL_ENDPOINT_TOLERANCE,
    DISTRIBUTION_COUNT,
    DISTRIBUTION_START_SEED,
    ENDPOINT_SECONDS,
    _query_indices,
)
from profile_counterfactual_failure_classes import _trace_sparse_optimization
from pyscipopt import SCIP_PARAMSETTING, Model, quicksum

from cpt_world import (
    WorldGrammar,
    iter_sampled_seeds,
    sample_task_world,
    sparse_individual_counterfactual_probability_bounds,
)
from cpt_world import counterfactual_solver as solver


def _terminal_context_edges(
    world: Any,
    treatment: int,
    outcome: int,
    baseline: int,
    treated: int,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    ancestors = solver._ancestors(world, outcome) | {outcome}
    affected = solver._descendants(world, treatment) & ancestors
    contexts = solver._active_contexts(
        world, outcome, treatment, baseline, treated
    )
    context_indices = {context: index for index, context in enumerate(contexts)}
    parent_tokens: list[tuple[int, int]] = []
    for parent in world.parents[outcome]:
        if parent == treatment:
            continue
        if parent in affected:
            parent_tokens.extend(((parent, 0), (parent, 1)))
        else:
            parent_tokens.append((parent, -1))
    edges: set[tuple[int, int]] = set()
    for assignment in product(
        *(range(world.domains[token[0]]) for token in parent_tokens)
    ):
        values = dict(zip(parent_tokens, assignment, strict=True))
        left_context: list[int] = []
        right_context: list[int] = []
        for parent in world.parents[outcome]:
            if parent == treatment:
                left_context.append(baseline)
                right_context.append(treated)
            elif parent in affected:
                left_context.append(values[(parent, 0)])
                right_context.append(values[(parent, 1)])
            else:
                state = values[(parent, -1)]
                left_context.append(state)
                right_context.append(state)
        left_index = context_indices[tuple(left_context)]
        right_index = context_indices[tuple(right_context)]
        if left_index != right_index:
            edges.add(tuple(sorted((left_index, right_index))))
    return len(contexts), tuple(sorted(edges))


def _cycle_rank(context_count: int, edges: tuple[tuple[int, int], ...]) -> int:
    components = solver._context_components(context_count, edges)
    covered = sum(len(component.indices) for component in components)
    isolated = context_count - covered
    component_count = len(components) + isolated
    return len(edges) - context_count + component_count


def _oriented_terminal_requirements(
    world: Any,
    treatment: int,
    outcome: int,
    baseline: int,
    treated: int,
    outcome_events: tuple[tuple[int, ...], tuple[int, ...]],
    endpoint: str,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, int, tuple[int, ...], tuple[int, ...], float], ...],
]:
    ancestors = solver._ancestors(world, outcome) | {outcome}
    affected = solver._descendants(world, treatment) & ancestors
    contexts = solver._active_contexts(world, outcome, treatment, baseline, treated)
    context_indices = {context: index for index, context in enumerate(contexts)}
    parent_tokens: list[tuple[int, int]] = []
    for parent in world.parents[outcome]:
        if parent == treatment:
            continue
        if parent in affected:
            parent_tokens.extend(((parent, 0), (parent, 1)))
        else:
            parent_tokens.append((parent, -1))
    requirements: set[
        tuple[int, int, tuple[int, ...], tuple[int, ...], float]
    ] = set()
    for assignment in product(
        *(range(world.domains[token[0]]) for token in parent_tokens)
    ):
        values = dict(zip(parent_tokens, assignment, strict=True))
        left_context: list[int] = []
        right_context: list[int] = []
        for parent in world.parents[outcome]:
            if parent == treatment:
                left_context.append(baseline)
                right_context.append(treated)
            elif parent in affected:
                left_context.append(values[(parent, 0)])
                right_context.append(values[(parent, 1)])
            else:
                state = values[(parent, -1)]
                left_context.append(state)
                right_context.append(state)
        left_index = context_indices[tuple(left_context)]
        right_index = context_indices[tuple(right_context)]
        left_event, right_event = outcome_events
        left_row = world.cpt[outcome][
            solver._row_index(world, outcome, tuple(left_context))
        ]
        right_row = world.cpt[outcome][
            solver._row_index(world, outcome, tuple(right_context))
        ]
        left_mass = sum(float(left_row[state]) for state in left_event)
        right_mass = sum(float(right_row[state]) for state in right_event)
        target = (
            max(0.0, left_mass + right_mass - 1.0)
            if endpoint == "lower"
            else min(left_mass, right_mass)
        )
        if left_index == right_index:
            # Consistency fixes this joint event.  The terminal cost used by
            # the candidate elimination keeps that exact intersection rather
            # than asking the same context to attain a two-context Frechet
            # endpoint.
            continue
        if left_index > right_index:
            left_index, right_index = right_index, left_index
            left_event, right_event = right_event, left_event
        requirements.add(
            (left_index, right_index, left_event, right_event, target)
        )
    return contexts, tuple(sorted(requirements))


def _terminal_endpoint_feasibility(
    world: Any,
    treatment: int,
    outcome: int,
    baseline: int,
    treated: int,
    outcome_events: tuple[tuple[int, ...], tuple[int, ...]],
    endpoint: str,
) -> tuple[str, float, int]:
    quotient = solver._coarsen_terminal_event_outcome(
        world, outcome, outcome_events
    )
    if quotient is not None:
        world, outcome_events = quotient
    requirements = _oriented_terminal_requirements(
        world,
        treatment,
        outcome,
        baseline,
        treated,
        outcome_events,
        endpoint,
    )
    contexts, endpoint_rows = requirements
    domain_size = world.domains[outcome]
    marginals = tuple(
        tuple(
            float(value)
            for value in world.cpt[outcome][
                solver._row_index(world, outcome, context)
            ]
        )
        for context in contexts
    )
    initial_weights = solver._comonotone_response_weights(marginals)
    model = Model("terminal-endpoint-response-feasibility")
    model.hideOutput()
    model.setPresolve(SCIP_PARAMSETTING.OFF)
    model.setIntParam("parallel/maxnthreads", 1)
    model.setIntParam("randomization/randomseedshift", 0)
    model.setIntParam("randomization/permutationseed", 0)
    model.setRealParam("numerics/feastol", solver._SCIP_NUMERICAL_TOLERANCE)
    model.setRealParam("limits/time", ENDPOINT_SECONDS)
    columns = [
        (
            response,
            model.addVar(name=f"lambda_{index}", lb=0.0),
        )
        for index, response in enumerate(initial_weights)
    ]
    normalization = model.addCons(
        quicksum(variable for _, variable in columns) == 1.0,
        separate=False,
        modifiable=True,
    )
    marginal_rows: dict[tuple[int, int], Any] = {}
    for context, marginal in enumerate(marginals):
        for state, probability in enumerate(marginal):
            marginal_rows[(context, state)] = model.addCons(
                quicksum(
                    variable
                    for response, variable in columns
                    if response[context] == state
                )
                == probability,
                separate=False,
                modifiable=True,
            )
    kernel_rows: dict[tuple[int, int, int, int], Any] = {}
    kernel_variables: dict[tuple[int, int, int, int], Any] = {}
    for left, right, _, _, _ in endpoint_rows:
        for left_state in range(domain_size):
            for right_state in range(domain_size):
                key = (left, right, left_state, right_state)
                if key in kernel_variables:
                    continue
                variable = model.addVar(name="terminal_kernel", lb=0.0, ub=1.0)
                kernel_variables[key] = variable
                kernel_rows[key] = model.addCons(
                    variable
                    - quicksum(
                        column
                        for response, column in columns
                        if response[left] == left_state
                        and response[right] == right_state
                    )
                    == 0.0,
                    separate=False,
                    modifiable=True,
                )
    for left, right, left_event, right_event, target in endpoint_rows:
        model.addCons(
            quicksum(
                kernel_variables[(left, right, left_state, right_state)]
                for left_state in left_event
                for right_state in right_event
            )
            == target
        )
    block = solver._PricingBlock(
        block_id=0,
        node=outcome,
        contexts=contexts,
        context_indices={context: index for index, context in enumerate(contexts)},
        columns=columns,
        initial_weights=initial_weights,
        normalization=normalization,
        marginals=marginal_rows,
        kernels=kernel_rows,
        dynamic=True,
    )
    pricer = solver._ResponsePricer([block], {outcome: domain_size})
    model.includePricer(
        pricer,
        "TerminalEndpointResponsePricer",
        "exact response columns for terminal endpoint feasibility",
    )
    deadline = time.perf_counter() + ENDPOINT_SECONDS
    pricer.begin_solve(deadline=deadline)
    started = time.perf_counter()
    try:
        model.optimize()
    except Exception:
        return (
            "unresolved_pricing_abort",
            time.perf_counter() - started,
            pricer.generated_columns,
        )
    elapsed = time.perf_counter() - started
    status = str(model.getStatus())
    if status == "optimal":
        return "certified_attainable", elapsed, pricer.generated_columns
    farkas_closed = bool(
        pricer.rounds
        and pricer.rounds[-1].farkas
        and pricer.rounds[-1].completed
        and pricer.rounds[-1].generated_columns == 0
        and not pricer.timed_out
    )
    if status == "infeasible" and farkas_closed:
        return "certified_infeasible", elapsed, pricer.generated_columns
    return "unresolved", elapsed, pricer.generated_columns


def main() -> None:
    grammar = WorldGrammar()
    totals: Counter[str] = Counter()
    joint_cycle_ranks: Counter[str] = Counter()
    joint_context_counts: Counter[str] = Counter()
    feasibility_status: Counter[str] = Counter()
    feasibility_seconds: list[float] = []
    feasibility_columns: list[int] = []
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
            with _trace_sparse_optimization(calls):
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
            failed = next(
                (record for record in reversed(calls) if record["outcome"] == "failed"),
                None,
            )
            if failed is None or failed["terminal_endpoint"] != "joint":
                totals["unresolved_nonjoint_owner"] += 1
                continue
            totals["unresolved_joint_owner"] += 1
            context_count, edges = _terminal_context_edges(
                world, treatment, outcome, baseline, treated
            )
            rank = _cycle_rank(context_count, edges)
            if rank == 0:
                totals["unresolved_joint_terminal_forest"] += 1
            else:
                totals["unresolved_joint_terminal_cyclic"] += 1
            joint_cycle_ranks[str(rank)] += 1
            joint_context_counts[str(context_count)] += 1
            endpoint = "lower" if failed["sense"] == "minimize" else "upper"
            status, elapsed, columns = _terminal_endpoint_feasibility(
                world,
                treatment,
                outcome,
                baseline,
                treated,
                ((factual_outcome,), (target_outcome,)),
                endpoint,
            )
            feasibility_status[status] += 1
            feasibility_seconds.append(elapsed)
            feasibility_columns.append(columns)
        else:
            totals["closed"] += 1

    print(
        json.dumps(
            {
                "cohort_size": DISTRIBUTION_COUNT,
                "totals": dict(sorted(totals.items())),
                "unresolved_joint_cycle_ranks": dict(sorted(joint_cycle_ranks.items())),
                "unresolved_joint_context_counts": dict(
                    sorted(joint_context_counts.items())
                ),
                "terminal_endpoint_feasibility": dict(
                    sorted(feasibility_status.items())
                ),
                "terminal_endpoint_feasibility_seconds": {
                    "total": sum(feasibility_seconds),
                    "max": max(feasibility_seconds, default=0.0),
                },
                "terminal_endpoint_generated_columns": {
                    "total": sum(feasibility_columns),
                    "max": max(feasibility_columns, default=0),
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
