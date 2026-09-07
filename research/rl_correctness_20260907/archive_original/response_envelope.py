"""Proven response-level transport envelope, diagnostic candidate only.

For fixed terminal response r, a mediator transport C contributes the mass
of a rectangle A(r) x B(r). Its extrema are max(0,u(A)+v(B)-1) and
min(u(A),v(B)). Averaging these pointwise bounds over a *jointly feasible*
terminal response distribution gives valid global outer bounds. It does not
claim that the separate optimal C choices can all be attained by one SCM.
"""

from dataclasses import dataclass
from itertools import product
from math import prod
import time

import numpy as np

from cpt_world import counterfactual_solver as s
from cpt_world import query_truth as q


@dataclass(frozen=True)
class Envelope:
    lower: float
    upper: float
    build_seconds: float
    solve_seconds: float
    response_columns: int
    response_blocks: int
    shared_assignments: int
    attained_lower: float | None = None
    attained_upper: float | None = None


def response_envelope(
    world,
    treatment,
    outcome,
    *,
    baseline_value,
    treatment_value,
    outcome_events,
    time_limit_seconds=None,
):
    affected = q._descendants(world, treatment) & (q._ancestors(world, outcome) | {outcome})
    if len(affected) != 2 or treatment in world.parents[outcome]:
        return None
    mediator = next(iter(affected - {outcome}))
    if treatment not in world.parents[mediator] or mediator not in world.parents[outcome]:
        return None
    mediator_shared = tuple(p for p in world.parents[mediator] if p != treatment)
    outcome_shared = tuple(p for p in world.parents[outcome] if p != mediator)
    shared = tuple(sorted(set(mediator_shared) | set(outcome_shared)))
    m, d = world.domains[mediator], world.domains[outcome]
    columns = d**m
    assignment_count = prod(world.domains[p] for p in shared)
    if columns > s._MAX_LAYERED_RESPONSE_COLUMNS:
        return None
    if assignment_count * columns * m > s._MAX_LAYERED_OBJECTIVE_EVALUATIONS:
        return None
    started = time.perf_counter()
    responses = tuple(product(range(d), repeat=m))
    response_array = np.asarray(responses, dtype=np.intp)
    left_indicator = np.isin(response_array, outcome_events[0]).astype(float)
    right_indicator = np.isin(response_array, outcome_events[1]).astype(float)
    objectives = {}
    law_records = []
    shared_law = (
        q.worldspec_projected_interventional_distribution(world, {}, shared)
        if shared
        else (((), 1.0),)
    )
    for assignment, mass in shared_law:
        if float(mass) <= 0.0:
            continue
        values = dict(zip(shared, assignment, strict=True))
        rows = []
        for action in (baseline_value, treatment_value):
            context = tuple(
                action if p == treatment else values[p] for p in world.parents[mediator]
            )
            rows.append(
                np.asarray(world.cpt[mediator][s._row_index(world, mediator, context)], dtype=float)
            )
        left_mass = left_indicator @ rows[0]
        right_mass = right_indicator @ rows[1]
        block = tuple(values[p] for p in outcome_shared)
        upstream = tuple(values[p] for p in mediator_shared)
        law_records.append((upstream, block, float(mass), rows))
        if block not in objectives:
            objectives[block] = [np.zeros(columns), np.zeros(columns)]
        objectives[block][0] += float(mass) * np.maximum(0.0, left_mass + right_mass - 1.0)
        objectives[block][1] += float(mass) * np.minimum(left_mass, right_mass)
    owners = {}
    for block in objectives:
        values = dict(zip(outcome_shared, block, strict=True))
        marginals = []
        for state in range(m):
            context = tuple(state if p == mediator else values[p] for p in world.parents[outcome])
            marginals.append(
                tuple(map(float, world.cpt[outcome][s._row_index(world, outcome, context)]))
            )
        owners[block] = s._ExactResponseLP(tuple(marginals))
    build_seconds = time.perf_counter() - started
    bounds = []
    attained = []
    solve_seconds = 0.0
    for position, sense in enumerate(("minimize", "maximize")):
        deadline = None if time_limit_seconds is None else time.perf_counter() + time_limit_seconds
        bound = 0.0
        response_weights = {}
        for block, owner in owners.items():
            costs = dict(zip(responses, map(float, objectives[block][position]), strict=True))
            remaining = None if deadline is None else deadline - time.perf_counter()
            value, elapsed = owner.optimize(costs, sense=sense, time_limit_seconds=remaining)
            solve_seconds += elapsed
            bound += value
            response_weights[block] = {
                response: float(owner.model.getVal(variable))
                for response, variable in owner.weights.items()
                if float(owner.model.getVal(variable)) > 0.0
            }
        bounds.append(bound)
        certificate_started = time.perf_counter()
        value = _common_transport_attainment(
            law_records,
            response_weights,
            outcome_events,
            sense=sense,
            deadline=deadline,
            same_action=baseline_value == treatment_value,
        )
        solve_seconds += time.perf_counter() - certificate_started
        attained.append(
            value
            if value is not None and s._global_bounds_numerically_closed(value, bound)
            else None
        )
    build_seconds = max(0.0, time.perf_counter() - started - solve_seconds)
    return Envelope(
        bounds[0],
        bounds[1],
        build_seconds,
        solve_seconds,
        columns,
        len(owners),
        assignment_count,
        attained[0],
        attained[1],
    )


def _common_transport_attainment(
    law_records, response_weights, outcome_events, *, sense, deadline, same_action
):
    """Construct an SCM attaining the outer LP optimum, or return no certificate.

    The envelope is an average of nonnegative transport slacks. Equality
    requires every response of positive weight to attain its rectangle bound.
    One common transport per upstream context must satisfy all these linear
    equations, including those from different downstream shared contexts.
    Feasibility gives independent mechanism distributions that can be glued
    across disjoint context blocks. Infeasibility says only that this selected
    optimal terminal response solution is not attainable by that construction.
    """
    transports = {}
    seen = {}
    for upstream, downstream, _mass, rows in law_records:
        if upstream not in transports:
            owner = s._ExactResponseLP(tuple(tuple(map(float, row)) for row in rows))
            transports[upstream] = owner
            seen[upstream] = set()
            if same_action:
                for (left, right), variable in owner.weights.items():
                    if left != right:
                        owner.model.addCons(variable == 0.0)
        owner = transports[upstream]
        for response in response_weights[downstream]:
            left = tuple(i for i, state in enumerate(response) if state in outcome_events[0])
            right = tuple(i for i, state in enumerate(response) if state in outcome_events[1])
            key = (left, right)
            if not left or not right or key in seen[upstream]:
                continue
            seen[upstream].add(key)
            a = sum(float(rows[0][i]) for i in left)
            b = sum(float(rows[1][i]) for i in right)
            target = max(0.0, a + b - 1.0) if sense == "minimize" else min(a, b)
            owner.model.addCons(sum(owner.weights[(i, j)] for i in left for j in right) == target)
    couplings = {}
    for upstream, owner in transports.items():
        remaining = None if deadline is None else deadline - time.perf_counter()
        if remaining is not None and remaining <= 0.0:
            return None
        try:
            owner.optimize(
                {response: 0.0 for response in owner.responses},
                sense="minimize",
                time_limit_seconds=remaining,
            )
        except RuntimeError:
            return None
        couplings[upstream] = {
            pair: float(owner.model.getVal(variable)) for pair, variable in owner.weights.items()
        }
    value = 0.0
    for upstream, downstream, mass, _rows in law_records:
        for response, weight in response_weights[downstream].items():
            value += (
                mass
                * weight
                * sum(
                    probability
                    for (i, j), probability in couplings[upstream].items()
                    if response[i] in outcome_events[0] and response[j] in outcome_events[1]
                )
            )
    return value
