"""Exact structural reduction; this file builds an isolated production candidate."""

from itertools import product
from math import prod
import time

import numpy as np

from cpt_world import counterfactual_solver as s
from cpt_world import query_truth as q


def _indirect_mediator_joint_bounds(
    world,
    treatment,
    outcome,
    *,
    mediator,
    baseline_value,
    treatment_value,
    outcome_events,
    time_limit_seconds,
):
    """Solve the class with simultaneously extremal transport diagonals.

    For two mediator states, disjoint terminal events read only the two
    off-diagonal entries; identical events read the diagonal or fixed masses.
    For three mediator states and a binary terminal-event quotient, every
    response event set is a singleton or its complement, so its rectangle
    also depends on just one diagonal entry and fixed marginals. All diagonal
    minima (or maxima) are simultaneously attainable by a transport. One
    such transport therefore optimizes every terminal response, for every
    shared terminal context. The remaining complete-response problems are
    linear, while the original mechanism independence is retained.
    """
    states = world.domains[mediator]
    if states > 3 or treatment not in world.parents[mediator]:
        return None
    left_event, right_event = map(frozenset, outcome_events)
    if left_event != right_event and not left_event.isdisjoint(right_event):
        return None
    if baseline_value == treatment_value:
        return None
    quotient = s._coarsen_terminal_event_outcome(world, outcome, outcome_events)
    if quotient is not None:
        world, outcome_events = quotient
    if states == 3 and world.domains[outcome] > 2:
        return None
    same_event = outcome_events[0] == outcome_events[1]
    mediator_shared = tuple(p for p in world.parents[mediator] if p != treatment)
    outcome_shared = tuple(p for p in world.parents[outcome] if p != mediator)
    shared = tuple(sorted(set(mediator_shared) | set(outcome_shared)))
    response_count = world.domains[outcome] ** states
    assignment_count = prod(world.domains[p] for p in shared)
    if assignment_count * response_count * states**2 > s._MAX_LAYERED_OBJECTIVE_EVALUATIONS:
        return None
    started = time.perf_counter()
    law = (
        q.worldspec_projected_interventional_distribution(world, {}, shared)
        if shared
        else (((), 1.0),)
    )
    records = []
    transports = {}
    terminals = {}
    for assignment, probability in law:
        if float(probability) <= 0.0:
            continue
        values = dict(zip(shared, assignment, strict=True))
        upstream = tuple(values[p] for p in mediator_shared)
        downstream = tuple(values[p] for p in outcome_shared)
        records.append((upstream, downstream, float(probability)))
        if upstream not in transports:
            marginals = []
            for action in (baseline_value, treatment_value):
                context = tuple(
                    action if p == treatment else values[p] for p in world.parents[mediator]
                )
                marginals.append(
                    tuple(map(float, world.cpt[mediator][s._row_index(world, mediator, context)]))
                )
            transports[upstream] = s._ExactResponseLP(tuple(marginals))
        if downstream not in terminals:
            marginals = []
            for state in range(states):
                context = tuple(
                    state if p == mediator else values[p] for p in world.parents[outcome]
                )
                marginals.append(
                    tuple(map(float, world.cpt[outcome][s._row_index(world, outcome, context)]))
                )
            terminals[downstream] = s._ExactResponseLP(tuple(marginals))
    responses = tuple(product(range(world.domains[outcome]), repeat=states))
    array = np.asarray(responses, dtype=np.intp)
    left = np.isin(array, outcome_events[0]).astype(float)
    right = np.isin(array, outcome_events[1]).astype(float)
    trace_cost = {(i, j): float(i == j) for i in range(states) for j in range(states)}
    solve_seconds = 0.0
    bounds = []
    for sense in ("minimize", "maximize"):
        deadline = None if time_limit_seconds is None else time.perf_counter() + time_limit_seconds
        transport_sense = (
            sense if same_event else ("maximize" if sense == "minimize" else "minimize")
        )
        response_costs = {}
        for upstream, owner in transports.items():
            remaining = None if deadline is None else deadline - time.perf_counter()
            _, elapsed = owner.optimize(
                trace_cost, sense=transport_sense, time_limit_seconds=remaining
            )
            solve_seconds += elapsed
            coupling = np.asarray(
                [
                    [float(owner.model.getVal(owner.weights[(i, j)])) for j in range(states)]
                    for i in range(states)
                ]
            )
            response_costs[upstream] = np.einsum("ri,ij,rj->r", left, coupling, right)
        objectives = {key: np.zeros(response_count) for key in terminals}
        for upstream, downstream, probability in records:
            objectives[downstream] += probability * response_costs[upstream]
        endpoint = 0.0
        for downstream, owner in terminals.items():
            remaining = None if deadline is None else deadline - time.perf_counter()
            costs = dict(zip(responses, map(float, objectives[downstream]), strict=True))
            value, elapsed = owner.optimize(costs, sense=sense, time_limit_seconds=remaining)
            solve_seconds += elapsed
            endpoint += value
        bounds.append(endpoint)
    return s.CounterfactualBoundsResult(
        lower=bounds[0],
        upper=bounds[1],
        build_seconds=max(0.0, time.perf_counter() - started - solve_seconds),
        solve_seconds=solve_seconds,
        affected_nodes=2,
        pair_kernel_entries=len(transports) * states**2,
        generated_columns=0,
        response_blocks=len(transports) + len(terminals),
        dynamic_response_blocks=0,
        max_response_contexts=max(2, states),
        auxiliary_variables=0,
        backend="indirect_mediator_diagonal_transport",
    )
