"""Exact sparse-response solver for Markovian counterfactual bounds.

Only context pairs that occur in the pruned twin world enter the model.
Terminal states are first quotiented by their two queried event-membership
bits; when that makes every local response domain explicit, SCIP presolve is
re-enabled instead of retaining an unnecessary branch-and-price interface.
One- and two-mediator chains use exact layered elimination: terminal event
couplings collapse to jointly attainable Frechet costs, after which small
transport and response linear programs close both endpoints.
Disconnected context components are coupled independently and then glued;
forest components use exact edge-transport tables and cyclic components use
an exact SCIP MAP pricer.  Reverse-topological paired elimination compiles
the twin probability into a bounded arithmetic circuit without expanding one
giant polynomial.

Cross-node mechanism independence is retained by the twin-network product.
The same original SCIP circuit is reused for the lower and upper endpoints.
Global optimality is checked internally before either endpoint is returned;
a time limit can never become a task label.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations, pairwise, product
from typing import Any

import numpy as np
from pyscipopt import SCIP_PARAMSETTING, SCIP_RESULT, Model, Pricer, quicksum

from .world_space import WorldSpec

_SCIP_NUMERICAL_TOLERANCE = 1e-9
# Retained only by the private legacy path used for implementation parity.
_LEGACY_MAX_EXPLICIT_RESPONSE_CONTEXTS = 5
_LEGACY_MAX_EXPLICIT_RESPONSE_COLUMNS = 5**5
# Above this many entries in one induced min-sum table, pricing falls back to
# SCIP.  Both paths solve the same deterministic-response MAP problem exactly;
# this guard only prevents the structure-aware fast path from overusing memory.
_MAX_EXACT_MIN_SUM_TABLE_ENTRIES = 5_000_000
_MAX_LAYERED_RESPONSE_COLUMNS = 100_000
_MAX_LAYERED_UPSTREAM_VERTICES = 4_096
_MAX_LAYERED_OBJECTIVE_EVALUATIONS = 10_000_000


def _global_bounds_numerically_closed(primal: float, dual: float) -> bool:
    """Return whether SCIP's rigorous bracket is below numerical precision."""

    if not np.isfinite(primal) or not np.isfinite(dual):
        return False
    scale = max(1.0, abs(primal), abs(dual))
    return abs(primal - dual) <= 10.0 * _SCIP_NUMERICAL_TOLERANCE * scale


@dataclass(frozen=True, slots=True)
class CounterfactualBoundsResult:
    """Certified endpoints plus non-semantic performance statistics."""

    lower: float
    upper: float
    build_seconds: float
    solve_seconds: float
    affected_nodes: int
    pair_kernel_entries: int
    generated_columns: int
    response_blocks: int
    dynamic_response_blocks: int
    max_response_contexts: int
    auxiliary_variables: int
    certification: str = "exact"
    endpoint_error: float = 0.0
    backend: str = "sparse_response_branch_price"


@dataclass(slots=True)
class _SymbolicFactor:
    scope: tuple[tuple[int, int], ...]
    values: Any
    upper_bounds: Any
    initial_values: Any


@dataclass(slots=True)
class _PricingBlock:
    block_id: int
    node: int
    contexts: tuple[tuple[int, ...], ...]
    columns: list[tuple[tuple[int, ...], Any]]
    initial_weights: dict[tuple[int, ...], float]
    normalization: Any
    marginals: dict[tuple[int, int], Any]
    kernels: dict[tuple[int, int, int, int], Any]
    dynamic: bool
    direct_objective: Mapping[tuple[int, int, int, int], float] | None = None


@dataclass(slots=True)
class _PricingRoundState:
    """Completion evidence for one SCIP pricing callback."""

    farkas: bool
    completed: bool = False
    generated_columns: int = 0


@dataclass(frozen=True, slots=True)
class _ContextComponent:
    indices: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    forest: bool


@dataclass(frozen=True, slots=True)
class _MinSumFactor:
    """Dense factor used by exact response-MAP variable elimination."""

    scope: tuple[int, ...]
    values: Any


@dataclass(frozen=True, slots=True)
class _MinSumResult:
    value: float
    response: tuple[int, ...]
    induced_width: int


@dataclass(frozen=True, slots=True)
class _PairwiseMapOptimization:
    value: float
    response: tuple[int, ...] | None
    status: str
    solve_seconds: float
    variables: int
    constraints: int
    backend: str


def _exact_transport_bounds(
    left_marginal: tuple[float, ...],
    right_marginal: tuple[float, ...],
    lower_cost: Mapping[tuple[int, int], float],
    upper_cost: Mapping[tuple[int, int], float],
    *,
    time_limit_seconds: float | None,
) -> tuple[float, float, float, float]:
    """Optimize two linear costs over one exact transportation polytope."""

    model = Model("exact-one-mediator-counterfactual-transport")
    model.hideOutput()
    model.setIntParam("parallel/maxnthreads", 1)
    model.setIntParam("randomization/randomseedshift", 0)
    model.setIntParam("randomization/permutationseed", 0)
    model.setRealParam("limits/gap", 0.0)
    model.setRealParam("limits/absgap", 0.0)
    model.setRealParam("numerics/feastol", _SCIP_NUMERICAL_TOLERANCE)
    if time_limit_seconds is not None:
        model.setRealParam("limits/time", time_limit_seconds)
    transport = {
        (left, right): model.addVar(lb=0.0, ub=min(left_mass, right_marginal[right]))
        for left, left_mass in enumerate(left_marginal)
        for right in range(len(right_marginal))
    }
    for left, probability in enumerate(left_marginal):
        model.addCons(
            quicksum(
                transport[(left, right)] for right in range(len(right_marginal))
            )
            == probability
        )
    for right, probability in enumerate(right_marginal):
        model.addCons(
            quicksum(
                transport[(left, right)] for left in range(len(left_marginal))
            )
            == probability
        )

    lower_expression = quicksum(
        lower_cost[pair] * variable for pair, variable in transport.items()
    )
    model.setObjective(lower_expression, "minimize")
    started = time.perf_counter()
    model.optimize()
    lower_seconds = time.perf_counter() - started
    if str(model.getStatus()) != "optimal":
        raise RuntimeError("one-mediator lower transport did not certify optimality")
    lower = float(model.getPrimalbound())

    model.freeTransform()
    upper_expression = quicksum(
        upper_cost[pair] * variable for pair, variable in transport.items()
    )
    model.setObjective(upper_expression, "maximize")
    started = time.perf_counter()
    model.optimize()
    upper_seconds = time.perf_counter() - started
    if str(model.getStatus()) != "optimal":
        raise RuntimeError("one-mediator upper transport did not certify optimality")
    return lower, float(model.getPrimalbound()), lower_seconds, upper_seconds


def _binary_transport_vertices(
    left_marginal: tuple[float, float],
    right_marginal: tuple[float, float],
) -> tuple[dict[tuple[int, int], float], ...]:
    """Return every vertex of a two-by-two transportation polytope."""

    lower = max(0.0, left_marginal[0] + right_marginal[0] - 1.0)
    upper = min(left_marginal[0], right_marginal[0])
    vertices: list[dict[tuple[int, int], float]] = []
    for top_left in (lower, upper):
        vertex = {
            (0, 0): top_left,
            (0, 1): left_marginal[0] - top_left,
            (1, 0): right_marginal[0] - top_left,
            (1, 1): 1.0 - left_marginal[0] - right_marginal[0] + top_left,
        }
        if not vertices or any(
            abs(vertex[pair] - vertices[0][pair]) > _SCIP_NUMERICAL_TOLERANCE
            for pair in vertex
        ):
            vertices.append(vertex)
    return tuple(vertices)


def _comonotone_response_weights(
    marginals: tuple[tuple[float, ...], ...],
) -> dict[tuple[int, ...], float]:
    """Construct one sparse feasible coupling from a shared uniform variable."""

    breakpoints = {0.0, 1.0}
    for row in marginals:
        cumulative = 0.0
        for probability in row[:-1]:
            cumulative += probability
            breakpoints.add(min(1.0, max(0.0, cumulative)))
    ordered = sorted(breakpoints)
    weights: dict[tuple[int, ...], float] = {}
    for left, right in pairwise(ordered):
        weight = right - left
        if weight <= _SCIP_NUMERICAL_TOLERANCE:
            continue
        midpoint = (left + right) / 2.0
        response: list[int] = []
        for row in marginals:
            cumulative = 0.0
            selected = len(row) - 1
            for state, probability in enumerate(row):
                cumulative += probability
                if midpoint < cumulative:
                    selected = state
                    break
            response.append(selected)
        response_tuple = tuple(response)
        weights[response_tuple] = weights.get(response_tuple, 0.0) + weight
    return weights


class _ExactResponseLP:
    """Reusable exact LP owner for one fixed response-marginal polytope."""

    def __init__(self, marginals: tuple[tuple[float, ...], ...]) -> None:
        if not marginals:
            raise ValueError("response LP needs at least one context")
        domain_size = len(marginals[0])
        self.responses = tuple(product(range(domain_size), repeat=len(marginals)))
        self.model = Model("exact-layered-response-lp")
        self.model.hideOutput()
        self.model.setIntParam("parallel/maxnthreads", 1)
        self.model.setIntParam("randomization/randomseedshift", 0)
        self.model.setIntParam("randomization/permutationseed", 0)
        self.model.setRealParam("limits/gap", 0.0)
        self.model.setRealParam("limits/absgap", 0.0)
        self.model.setRealParam("numerics/feastol", _SCIP_NUMERICAL_TOLERANCE)
        self.weights = {
            response: self.model.addVar(lb=0.0, ub=1.0)
            for response in self.responses
        }
        self.model.addCons(quicksum(self.weights.values()) == 1.0)
        for context, marginal in enumerate(marginals):
            for state, probability in enumerate(marginal):
                self.model.addCons(
                    quicksum(
                        variable
                        for response, variable in self.weights.items()
                        if response[context] == state
                    )
                    == probability
                )
        self.solved = False

    def optimize(
        self,
        objective: Mapping[tuple[int, ...], float],
        *,
        sense: str,
        time_limit_seconds: float | None,
    ) -> tuple[float, float]:
        if set(objective) != set(self.responses):
            raise ValueError("response LP objective does not cover its response space")
        if self.solved:
            self.model.freeTransform()
        if time_limit_seconds is not None:
            if time_limit_seconds <= 0.0:
                raise RuntimeError("layered response LP exhausted its endpoint time limit")
            self.model.setRealParam("limits/time", time_limit_seconds)
        self.model.setObjective(
            quicksum(
                objective[response] * variable
                for response, variable in self.weights.items()
            ),
            sense,
        )
        started = time.perf_counter()
        self.model.optimize()
        elapsed = time.perf_counter() - started
        self.solved = True
        if str(self.model.getStatus()) != "optimal":
            raise RuntimeError("layered response LP did not certify optimality")
        return float(self.model.getPrimalbound()), elapsed


class _ExactPricedResponseLP:
    """Exact response LP using dual constraints and the owner's MAP pricer."""

    def __init__(
        self,
        marginals: tuple[tuple[float, ...], ...],
        objective: Mapping[tuple[int, int, int, int], float],
        *,
        sense: str,
    ) -> None:
        if not marginals:
            raise ValueError("priced response LP needs at least one context")
        if sense not in {"minimize", "maximize"}:
            raise ValueError("priced response LP sense must be minimize or maximize")
        domain_size = len(marginals[0])
        if any(len(row) != domain_size for row in marginals):
            raise ValueError("priced response LP marginals have inconsistent domains")
        initial_weights = _comonotone_response_weights(marginals)

        self.sense = sense
        self.multiplier = 1.0 if sense == "maximize" else -1.0
        self.transformed_objective = {
            key: self.multiplier * coefficient for key, coefficient in objective.items()
        }
        self.domain_size = domain_size
        self.context_count = len(marginals)
        self.model = Model("exact-priced-layered-response-dual")
        self.model.hideOutput()
        self.model.setPresolve(SCIP_PARAMSETTING.OFF)
        self.model.setIntParam("parallel/maxnthreads", 1)
        self.model.setIntParam("randomization/randomseedshift", 0)
        self.model.setIntParam("randomization/permutationseed", 0)
        self.model.setRealParam("limits/gap", 0.0)
        self.model.setRealParam("limits/absgap", 0.0)
        self.model.setRealParam("numerics/feastol", _SCIP_NUMERICAL_TOLERANCE)
        infinity = self.model.infinity()
        self.constant_dual = self.model.addVar(
            name="dual_normalization", lb=-infinity, ub=infinity
        )
        self.marginal_duals = {
            (context, state): self.model.addVar(
                name=f"dual_marginal_{context}_{state}",
                lb=-infinity,
                ub=infinity,
            )
            for context in range(self.context_count)
            for state in range(domain_size - 1)
        }
        self.model.setObjective(
            self.constant_dual
            + quicksum(
                sum(
                    weight
                    for response, weight in initial_weights.items()
                    if response[context] == state
                )
                * variable
                for (context, state), variable in self.marginal_duals.items()
            ),
            "minimize",
        )
        self.response_constraints: dict[tuple[int, ...], Any] = {}
        self.solved = False
        for response in initial_weights:
            self._add_response_constraint(response)

    def _add_response_constraint(self, response: tuple[int, ...]) -> None:
        if response in self.response_constraints:
            return
        expression = self.constant_dual + quicksum(
            self.marginal_duals[(context, state)]
            for context, state in enumerate(response)
            if state < self.domain_size - 1
        )
        constraint = self.model.addCons(
            expression >= _response_objective_value(
                self.transformed_objective, response
            )
        )
        self.response_constraints[response] = constraint

    def restart_objective(
        self,
        objective: Mapping[tuple[int, int, int, int], float],
    ) -> None:
        """Reuse the exact dual and its response pool for a new linear cost."""

        if self.solved:
            self.model.freeTransform()
        self.transformed_objective = {
            key: self.multiplier * coefficient for key, coefficient in objective.items()
        }
        for response, constraint in self.response_constraints.items():
            self.model.chgLhs(
                constraint,
                _response_objective_value(self.transformed_objective, response),
            )
        self.solved = False

    def _pricing_problem(
        self,
        transformed_objective: Mapping[tuple[int, int, int, int], float] | None = None,
    ) -> tuple[
        dict[int, tuple[float, ...]],
        dict[tuple[int, int], tuple[float, ...]],
        float,
    ]:
        constant = float(self.model.getVal(self.constant_dual))
        unary_values = {
            context: [
                (
                    float(self.model.getVal(self.marginal_duals[(context, state)]))
                    if state < self.domain_size - 1
                    else 0.0
                )
                for state in range(self.domain_size)
            ]
            for context in range(self.context_count)
        }
        pairwise_values: dict[tuple[int, int], list[float]] = {}
        objective = (
            self.transformed_objective
            if transformed_objective is None
            else transformed_objective
        )
        for (
            left,
            right,
            left_state,
            right_state,
        ), coefficient in objective.items():
            if abs(coefficient) <= _SCIP_NUMERICAL_TOLERANCE:
                continue
            if left == right:
                if left_state == right_state:
                    unary_values[left][left_state] -= coefficient
                continue
            if left > right:
                left, right = right, left
                left_state, right_state = right_state, left_state
            table = pairwise_values.setdefault(
                (left, right), [0.0] * self.domain_size**2
            )
            table[left_state * self.domain_size + right_state] -= coefficient
        return (
            {context: tuple(values) for context, values in unary_values.items()},
            {edge: tuple(values) for edge, values in pairwise_values.items()},
            constant,
        )

    def transformed_upper_bound(
        self,
        objective: Mapping[tuple[int, int, int, int], float],
        *,
        time_limit_seconds: float | None,
    ) -> tuple[float, float]:
        """Bound a new objective from the current exact dual solution.

        Raising the normalization dual by the largest response-constraint
        violation makes the current dual feasible for the new objective.
        The resulting value is therefore a rigorous upper bound on the new
        response LP optimum.
        """

        if not self.solved:
            raise RuntimeError("a solved response dual is required for warm bounding")
        if time_limit_seconds is not None and time_limit_seconds <= 0.0:
            raise RuntimeError("warm response bound exhausted its endpoint time limit")
        transformed = {
            key: self.multiplier * coefficient for key, coefficient in objective.items()
        }
        started = time.perf_counter()
        unary, pairwise_costs, constant = self._pricing_problem(transformed)
        pricing = _exact_pairwise_map(
            unary,
            pairwise_costs,
            domain_size=self.domain_size,
            constant=constant,
            time_limit_seconds=time_limit_seconds,
        )
        elapsed = time.perf_counter() - started
        if pricing.status != "optimal" or pricing.response is None:
            raise RuntimeError(
                "warm response bound MAP did not certify optimality: "
                f"status={pricing.status}"
            )
        correction = max(0.0, -pricing.value)
        return float(self.model.getPrimalbound()) + correction, elapsed

    def optimize(
        self,
        *,
        time_limit_seconds: float | None,
    ) -> tuple[float, float, int]:
        deadline: float | None = None
        if time_limit_seconds is not None:
            if time_limit_seconds <= 0.0:
                raise RuntimeError("priced response LP exhausted its endpoint time limit")
            deadline = time.perf_counter() + time_limit_seconds
        started = time.perf_counter()
        initial_responses = len(self.response_constraints)
        while True:
            remaining = None if deadline is None else deadline - time.perf_counter()
            if remaining is not None:
                if remaining <= 0.0:
                    raise RuntimeError("priced response LP exhausted its endpoint time limit")
                self.model.setRealParam("limits/time", remaining)
            self.model.optimize()
            self.solved = True
            status = str(self.model.getStatus())
            if status != "optimal":
                raise RuntimeError(
                    "priced response dual did not certify optimality: "
                    f"status={status}"
                )
            unary, pairwise_costs, constant = self._pricing_problem()
            pricing = _exact_pairwise_map(
                unary,
                pairwise_costs,
                domain_size=self.domain_size,
                constant=constant,
                time_limit_seconds=remaining,
            )
            if pricing.status != "optimal" or pricing.response is None:
                raise RuntimeError(
                    "priced response dual MAP did not certify optimality: "
                    f"status={pricing.status}"
                )
            if pricing.value >= -10.0 * _SCIP_NUMERICAL_TOLERANCE:
                break
            if pricing.response in self.response_constraints:
                raise RuntimeError(
                    "priced response dual returned a violated existing constraint"
                )
            self.model.freeTransform()
            self.solved = False
            self._add_response_constraint(pricing.response)

        elapsed = time.perf_counter() - started
        transformed_value = float(self.model.getPrimalbound())
        value = transformed_value if self.sense == "maximize" else -transformed_value
        return value, elapsed, len(self.response_constraints) - initial_responses


def _one_mediator_joint_bounds(
    world: WorldSpec,
    treatment: int,
    outcome: int,
    *,
    baseline_value: int,
    treatment_value: int,
    outcome_events: tuple[tuple[int, ...], tuple[int, ...]],
    time_limit_seconds: float | None,
) -> CounterfactualBoundsResult | None:
    """Close the exact two-world query with one unconditioned mediator.

    The supported twin graph is ``X -> M -> Y`` with the direct ``X -> Y``
    edge, no additional parent of ``M``, and arbitrary unaffected parents of
    ``Y``.  For each shared-parent assignment, all cross-arm outcome-event
    Frechet endpoints are jointly attainable.  Eliminating the outcome
    mechanism therefore leaves one exact optimal-transport problem over the
    mediator's two potential states.
    """

    ancestors = _ancestors(world, outcome) | {outcome}
    affected = tuple(
        node
        for node in _topological_order(world)
        if node in (_descendants(world, treatment) & ancestors)
    )
    if len(affected) != 2 or affected[-1] != outcome:
        return None
    mediator = affected[0]
    if world.parents[mediator] != (treatment,):
        return None
    if treatment not in world.parents[outcome] or mediator not in world.parents[outcome]:
        return None

    from .query_truth import worldspec_projected_interventional_distribution

    shared_parents = tuple(
        parent for parent in world.parents[outcome] if parent not in {treatment, mediator}
    )
    parent_law = (
        worldspec_projected_interventional_distribution(world, {}, shared_parents)
        if shared_parents
        else (((), 1.0),)
    )
    left_event, right_event = outcome_events
    lower_cost = {
        (left, right): 0.0
        for left in range(world.domains[mediator])
        for right in range(world.domains[mediator])
    }
    upper_cost = dict(lower_cost)

    def event_probability(
        treatment_state: int,
        mediator_state: int,
        shared_values: Mapping[int, int],
        event: tuple[int, ...],
    ) -> float:
        context = tuple(
            treatment_state
            if parent == treatment
            else mediator_state
            if parent == mediator
            else shared_values[parent]
            for parent in world.parents[outcome]
        )
        row = world.cpt[outcome][_row_index(world, outcome, context)]
        return sum(float(row[state]) for state in event)

    for assignment, probability in parent_law:
        shared_values = dict(zip(shared_parents, assignment, strict=True))
        weight = float(probability)
        for left in range(world.domains[mediator]):
            left_probability = event_probability(
                baseline_value,
                left,
                shared_values,
                left_event,
            )
            for right in range(world.domains[mediator]):
                right_probability = event_probability(
                    treatment_value,
                    right,
                    shared_values,
                    right_event,
                )
                pair = (left, right)
                lower_cost[pair] += weight * max(
                    0.0, left_probability + right_probability - 1.0
                )
                upper_cost[pair] += weight * min(
                    left_probability, right_probability
                )

    left_marginal = tuple(
        float(value)
        for value in world.cpt[mediator][
            _row_index(world, mediator, (baseline_value,))
        ]
    )
    right_marginal = tuple(
        float(value)
        for value in world.cpt[mediator][
            _row_index(world, mediator, (treatment_value,))
        ]
    )
    build_started = time.perf_counter()
    lower, upper, lower_seconds, upper_seconds = _exact_transport_bounds(
        left_marginal,
        right_marginal,
        lower_cost,
        upper_cost,
        time_limit_seconds=time_limit_seconds,
    )
    return CounterfactualBoundsResult(
        lower=lower,
        upper=upper,
        build_seconds=time.perf_counter() - build_started - lower_seconds - upper_seconds,
        solve_seconds=lower_seconds + upper_seconds,
        affected_nodes=2,
        pair_kernel_entries=len(lower_cost),
        generated_columns=0,
        response_blocks=0,
        dynamic_response_blocks=0,
        max_response_contexts=2 * world.domains[mediator],
        auxiliary_variables=0,
        backend="one_mediator_transport",
    )


def _two_mediator_joint_bounds(
    world: WorldSpec,
    treatment: int,
    outcome: int,
    *,
    baseline_value: int,
    treatment_value: int,
    outcome_events: tuple[tuple[int, ...], tuple[int, ...]],
    time_limit_seconds: float | None,
    endpoint_only: str | None = None,
) -> CounterfactualBoundsResult | None:
    """Exactly eliminate a layered two-mediator treatment structure.

    Without a direct treatment-to-outcome edge, the elimination remains exact
    for one endpoint only when the terminal response has a shared Frechet
    construction for that endpoint.  ``endpoint_only`` exposes precisely that
    certified composition to the terminal-endpoint dispatcher.
    """

    if endpoint_only not in {None, "lower", "upper"}:
        raise ValueError("endpoint_only must be lower, upper, or None")

    ancestors = _ancestors(world, outcome) | {outcome}
    affected = tuple(
        node
        for node in _topological_order(world)
        if node in (_descendants(world, treatment) & ancestors)
    )
    if len(affected) != 3 or affected[-1] != outcome:
        return None
    first, second, _ = affected
    if world.domains[first] > 4:
        return None
    if treatment not in world.parents[first]:
        return None
    if first not in world.parents[second]:
        return None
    if second not in world.parents[outcome]:
        return None
    direct_terminal = treatment in world.parents[outcome]
    if endpoint_only is None and not direct_terminal:
        return None
    if endpoint_only is not None and not (
        _terminal_event_endpoint_is_jointly_attainable(
            world,
            treatment,
            outcome,
            outcome_events,
            endpoint_only,
            baseline_value=baseline_value,
            treatment_value=treatment_value,
        )
    ):
        return None
    if any(parent in affected for parent in world.parents[first]):
        return None
    if {parent for parent in world.parents[second] if parent in affected} != {first}:
        return None
    outcome_affected = {
        parent for parent in world.parents[outcome] if parent in affected
    }
    if outcome_affected not in ({second}, {first, second}):
        return None

    first_shared = tuple(
        parent for parent in world.parents[first] if parent != treatment
    )
    second_shared = tuple(
        parent
        for parent in world.parents[second]
        if parent not in {treatment, first}
    )
    outcome_shared = tuple(
        parent
        for parent in world.parents[outcome]
        if parent not in {treatment, first, second}
    )
    all_shared = tuple(sorted(set(first_shared) | set(second_shared) | set(outcome_shared)))
    second_context_count = world.domains[first] * (
        2 if treatment in world.parents[second] else 1
    )
    response_count = world.domains[second] ** second_context_count

    from .query_truth import (
        _response_coupling_vertices,
        worldspec_projected_interventional_distribution,
    )

    shared_law = (
        worldspec_projected_interventional_distribution(world, {}, all_shared)
        if all_shared
        else (((), 1.0),)
    )
    law_records: list[tuple[dict[int, int], float]] = [
        (dict(zip(all_shared, assignment, strict=True)), float(probability))
        for assignment, probability in shared_law
        if float(probability) > 0.0
    ]
    shared_assignments = tuple(
        product(*(range(world.domains[parent]) for parent in first_shared))
    )
    second_assignments = tuple(
        product(*(range(world.domains[parent]) for parent in second_shared))
    )

    def parent_context(
        node: int,
        *,
        treatment_state: int,
        affected_state: int | None,
        affected_parent: int | None,
        shared_values: Mapping[int, int],
    ) -> tuple[int, ...]:
        return tuple(
            treatment_state
            if parent == treatment
            else affected_state
            if parent == affected_parent
            else shared_values[parent]
            for parent in world.parents[node]
        )

    first_vertices: dict[
        tuple[int, ...], tuple[dict[tuple[int, int], float], ...]
    ] = {}
    for assignment in shared_assignments:
        shared_values = dict(zip(first_shared, assignment, strict=True))
        left_row_exact = world.cpt[first][
            _row_index(
                world,
                first,
                parent_context(
                    first,
                    treatment_state=baseline_value,
                    affected_state=None,
                    affected_parent=None,
                    shared_values=shared_values,
                ),
            )
        ]
        right_row_exact = world.cpt[first][
            _row_index(
                world,
                first,
                parent_context(
                    first,
                    treatment_state=treatment_value,
                    affected_state=None,
                    affected_parent=None,
                    shared_values=shared_values,
                ),
            )
        ]
        left_row = tuple(float(value) for value in left_row_exact)
        right_row = tuple(float(value) for value in right_row_exact)
        if world.domains[first] == 2:
            vertices = _binary_transport_vertices(left_row, right_row)
        else:
            vertices = tuple(
                {
                    (response[0], response[1]): float(weight)
                    for response, weight in vertex
                }
                for vertex in _response_coupling_vertices(
                    (left_row_exact, right_row_exact)
                )
            )
        first_vertices[assignment] = vertices

    # The upstream transport choices and downstream response owners form a
    # bipartite dependency graph.  Disconnected components have Cartesian
    # feasible sets and additive objectives, so their extrema can be solved
    # independently.  Enumerating one global Cartesian product is exact but
    # needlessly exponential in the number of independent shared contexts.
    dependencies: dict[
        tuple[str, tuple[int, ...]], set[tuple[str, tuple[int, ...]]]
    ] = {}
    for shared_values, _ in law_records:
        first_assignment = tuple(shared_values[parent] for parent in first_shared)
        second_assignment = tuple(shared_values[parent] for parent in second_shared)
        first_key = ("first", first_assignment)
        second_key = ("second", second_assignment)
        dependencies.setdefault(first_key, set()).add(second_key)
        dependencies.setdefault(second_key, set()).add(first_key)
    components: list[tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]] = []
    unseen = set(dependencies)
    while unseen:
        start = min(unseen)
        stack = [start]
        component: set[tuple[str, tuple[int, ...]]] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            unseen.discard(current)
            stack.extend(dependencies[current] - component)
        component_first = tuple(
            sorted(assignment for kind, assignment in component if kind == "first")
        )
        component_second = tuple(
            sorted(assignment for kind, assignment in component if kind == "second")
        )
        if component_first and component_second:
            components.append((component_first, component_second))
    if not components:
        raise ValueError("shared-parent law has no positive-probability assignment")

    estimated_objective_evaluations = 0
    for component_first, component_second in components:
        selection_count = 1
        for assignment in component_first:
            selection_count *= len(first_vertices[assignment])
        if selection_count > _MAX_LAYERED_UPSTREAM_VERTICES:
            return None
        component_law_count = sum(
            1
            for shared_values, _ in law_records
            if tuple(shared_values[parent] for parent in second_shared)
            in component_second
        )
        estimated_objective_evaluations += (
            selection_count
            * len(component_second)
            * max(1, component_law_count)
            * response_count
            * world.domains[first] ** 2
        )
    if (
        response_count <= _MAX_LAYERED_RESPONSE_COLUMNS
        and estimated_objective_evaluations > _MAX_LAYERED_OBJECTIVE_EVALUATIONS
    ):
        return None

    explicit_responses = response_count <= _MAX_LAYERED_RESPONSE_COLUMNS
    responses = (
        tuple(product(range(world.domains[second]), repeat=second_context_count))
        if explicit_responses
        else ()
    )
    left_event, right_event = outcome_events

    def outcome_event_probability(
        treatment_state: int,
        first_state: int,
        second_state: int,
        shared_values: Mapping[int, int],
        event: tuple[int, ...],
    ) -> float:
        context = tuple(
            treatment_state
            if parent == treatment
            else first_state
            if parent == first
            else second_state
            if parent == second
            else shared_values[parent]
            for parent in world.parents[outcome]
        )
        row = world.cpt[outcome][_row_index(world, outcome, context)]
        return sum(float(row[state]) for state in event)

    terminal_costs: list[
        tuple[
            dict[int, int],
            float,
            dict[tuple[int, int, int, int], float],
            dict[tuple[int, int, int, int], float],
        ]
    ] = []
    for shared_values, probability in law_records:
        lower_cost: dict[tuple[int, int, int, int], float] = {}
        upper_cost: dict[tuple[int, int, int, int], float] = {}
        for left_first in range(world.domains[first]):
            for right_first in range(world.domains[first]):
                for left_second in range(world.domains[second]):
                    left_probability = outcome_event_probability(
                        baseline_value,
                        left_first,
                        left_second,
                        shared_values,
                        left_event,
                    )
                    for right_second in range(world.domains[second]):
                        right_probability = outcome_event_probability(
                            treatment_value,
                            right_first,
                            right_second,
                            shared_values,
                            right_event,
                        )
                        states = (
                            left_first,
                            right_first,
                            left_second,
                            right_second,
                        )
                        lower_cost[states] = max(
                            0.0, left_probability + right_probability - 1.0
                        )
                        upper_cost[states] = min(
                            left_probability, right_probability
                        )
        terminal_costs.append((shared_values, probability, lower_cost, upper_cost))

    total_started = time.perf_counter()
    response_specs: dict[
        tuple[int, ...],
        tuple[
            tuple[tuple[float, ...], ...],
            tuple[int, ...],
            tuple[int, ...],
            _ExactResponseLP | None,
        ],
    ] = {}
    for second_assignment in second_assignments:
        second_shared_values = dict(zip(second_shared, second_assignment, strict=True))
        contexts: list[tuple[int, ...]] = []
        context_indices: dict[tuple[int, int], int] = {}
        for current_treatment in (baseline_value, treatment_value):
            for first_state in range(world.domains[first]):
                context = parent_context(
                    second,
                    treatment_state=current_treatment,
                    affected_state=first_state,
                    affected_parent=first,
                    shared_values=second_shared_values,
                )
                if context not in contexts:
                    contexts.append(context)
                context_indices[(current_treatment, first_state)] = contexts.index(
                    context
                )
        marginals = tuple(
            tuple(
                float(value)
                for value in world.cpt[second][
                    _row_index(world, second, context)
                ]
            )
            for context in contexts
        )
        left_indices = tuple(
            context_indices[(baseline_value, first_state)]
            for first_state in range(world.domains[first])
        )
        right_indices = tuple(
            context_indices[(treatment_value, first_state)]
            for first_state in range(world.domains[first])
        )
        response_specs[second_assignment] = (
            marginals,
            left_indices,
            right_indices,
            _ExactResponseLP(marginals) if explicit_responses else None,
        )
    solve_seconds = 0.0
    generated_columns = 0
    priced_response_owners: dict[
        tuple[str, tuple[int, ...]], _ExactPricedResponseLP
    ] = {}

    def objectives_for_selection(
        selected: Mapping[tuple[int, ...], Mapping[tuple[int, int], float]],
        *,
        sense: str,
        second_scope: tuple[tuple[int, ...], ...],
    ) -> dict[tuple[int, ...], dict[tuple[int, int, int, int], float]]:
        objectives: dict[
            tuple[int, ...], dict[tuple[int, int, int, int], float]
        ] = {assignment: {} for assignment in second_scope}
        second_scope_set = set(second_scope)
        for second_assignment in second_scope:
            _, left_indices, right_indices, _ = response_specs[second_assignment]
            for shared_values, probability, lower_cost, upper_cost in terminal_costs:
                current_second_assignment = tuple(
                    shared_values[parent] for parent in second_shared
                )
                if (
                    current_second_assignment != second_assignment
                    or current_second_assignment not in second_scope_set
                ):
                    continue
                first_assignment = tuple(
                    shared_values[parent] for parent in first_shared
                )
                first_transport = selected[first_assignment]
                direct_objective = objectives[second_assignment]
                terminal = lower_cost if sense == "minimize" else upper_cost
                for (left_first, right_first), mass in first_transport.items():
                    left_context = left_indices[left_first]
                    right_context = right_indices[right_first]
                    for left_second in range(world.domains[second]):
                        for right_second in range(world.domains[second]):
                            key = (
                                left_context,
                                right_context,
                                left_second,
                                right_second,
                            )
                            direct_objective[key] = direct_objective.get(
                                key, 0.0
                            ) + probability * mass * terminal[
                                (
                                    left_first,
                                    right_first,
                                    left_second,
                                    right_second,
                                )
                            ]
        return objectives

    def optimize_endpoint(*, sense: str) -> float:
        nonlocal generated_columns, solve_seconds
        endpoint_deadline = (
            None
            if time_limit_seconds is None
            else time.perf_counter() + time_limit_seconds
        )
        if explicit_responses:
            endpoint_total = 0.0
            for component_first, component_second in components:
                best = float("inf") if sense == "minimize" else -float("inf")
                vertex_lists = tuple(
                    first_vertices[assignment] for assignment in component_first
                )
                for selected_vertices in product(*vertex_lists):
                    selected = dict(
                        zip(component_first, selected_vertices, strict=True)
                    )
                    total = 0.0
                    direct_objectives = objectives_for_selection(
                        selected,
                        sense=sense,
                        second_scope=component_second,
                    )
                    for second_assignment in component_second:
                        _, _, _, owner = response_specs[second_assignment]
                        if owner is None:
                            raise RuntimeError("explicit response LP owner is missing")
                        direct_objective = direct_objectives[second_assignment]
                        objective = {
                            response: _response_objective_value(
                                direct_objective, response
                            )
                            for response in responses
                        }
                        remaining = (
                            None
                            if endpoint_deadline is None
                            else endpoint_deadline - time.perf_counter()
                        )
                        value, elapsed = owner.optimize(
                            objective,
                            sense=sense,
                            time_limit_seconds=remaining,
                        )
                        solve_seconds += elapsed
                        total += value
                    if sense == "minimize":
                        best = min(best, total)
                    else:
                        best = max(best, total)
                endpoint_total += best
            return endpoint_total

        multiplier = 1.0 if sense == "maximize" else -1.0
        feasible_weights = {
            second_assignment: _comonotone_response_weights(
                response_specs[second_assignment][0]
            )
            for second_assignment in second_assignments
        }
        endpoint_transformed_total = 0.0
        for component_first, component_second in components:
            candidates: list[
                tuple[
                    float,
                    dict[tuple[int, ...], dict[tuple[int, int, int, int], float]],
                ]
            ] = []
            vertex_lists = tuple(
                first_vertices[assignment] for assignment in component_first
            )
            for selected_vertices in product(*vertex_lists):
                selected = dict(zip(component_first, selected_vertices, strict=True))
                direct_objectives = objectives_for_selection(
                    selected,
                    sense=sense,
                    second_scope=component_second,
                )
                feasible = 0.0
                for second_assignment, objective in direct_objectives.items():
                    feasible += multiplier * sum(
                        weight * _response_objective_value(objective, response)
                        for response, weight in feasible_weights[
                            second_assignment
                        ].items()
                    )
                candidates.append((feasible, direct_objectives))
            candidates.sort(key=lambda item: item[0], reverse=True)

            best_transformed = -float("inf")
            for feasible, direct_objectives in candidates:
                best_transformed = max(best_transformed, feasible)
                if all(
                    (sense, second_assignment) in priced_response_owners
                    for second_assignment in component_second
                ):
                    upper_bound = 0.0
                    for second_assignment, objective in direct_objectives.items():
                        remaining = (
                            None
                            if endpoint_deadline is None
                            else endpoint_deadline - time.perf_counter()
                        )
                        bound, elapsed = priced_response_owners[
                            (sense, second_assignment)
                        ].transformed_upper_bound(
                            objective,
                            time_limit_seconds=remaining,
                        )
                        solve_seconds += elapsed
                        upper_bound += bound
                    if (
                        upper_bound
                        <= best_transformed + 10.0 * _SCIP_NUMERICAL_TOLERANCE
                    ):
                        continue

                transformed_total = 0.0
                for second_assignment, objective in direct_objectives.items():
                    marginals = response_specs[second_assignment][0]
                    owner_key = (sense, second_assignment)
                    priced_owner = priced_response_owners.get(owner_key)
                    if priced_owner is None:
                        priced_owner = _ExactPricedResponseLP(
                            marginals,
                            objective,
                            sense=sense,
                        )
                        priced_response_owners[owner_key] = priced_owner
                    else:
                        priced_owner.restart_objective(objective)
                    remaining = (
                        None
                        if endpoint_deadline is None
                        else endpoint_deadline - time.perf_counter()
                    )
                    value, elapsed, added = priced_owner.optimize(
                        time_limit_seconds=remaining
                    )
                    solve_seconds += elapsed
                    generated_columns += added
                    transformed_total += multiplier * value
                best_transformed = max(best_transformed, transformed_total)
            endpoint_transformed_total += best_transformed
        return (
            endpoint_transformed_total
            if sense == "maximize"
            else -endpoint_transformed_total
        )

    if endpoint_only == "lower":
        lower = optimize_endpoint(sense="minimize")
        upper = lower
    elif endpoint_only == "upper":
        upper = optimize_endpoint(sense="maximize")
        lower = upper
    else:
        lower = optimize_endpoint(sense="minimize")
        upper = optimize_endpoint(sense="maximize")
    total_seconds = time.perf_counter() - total_started
    return CounterfactualBoundsResult(
        lower=lower,
        upper=upper,
        build_seconds=max(0.0, total_seconds - solve_seconds),
        solve_seconds=solve_seconds,
        affected_nodes=3,
        pair_kernel_entries=world.domains[first] ** 2 * world.domains[second] ** 2,
        generated_columns=generated_columns,
        response_blocks=len(second_assignments),
        dynamic_response_blocks=(
            len(second_assignments) if not explicit_responses else 0
        ),
        max_response_contexts=second_context_count,
        auxiliary_variables=0,
        backend=(
            "two_mediator_layered_elimination"
            if endpoint_only is None
            else f"two_mediator_layered_{endpoint_only}_endpoint"
        ),
    )


def _minimum_fill_order(
    variable_count: int,
    edges: set[tuple[int, int]],
) -> tuple[tuple[int, ...], int]:
    """Return a deterministic greedy min-fill order and its induced width."""

    graph = {variable: set() for variable in range(variable_count)}
    for left, right in edges:
        graph[left].add(right)
        graph[right].add(left)
    remaining = set(graph)
    order: list[int] = []
    induced_width = 0
    while remaining:
        candidates: list[tuple[int, int, int]] = []
        for variable in remaining:
            neighbors = sorted(graph[variable] & remaining)
            missing = sum(right not in graph[left] for left, right in combinations(neighbors, 2))
            candidates.append((missing, len(neighbors), variable))
        _, _, variable = min(candidates)
        neighbors = sorted(graph[variable] & remaining)
        induced_width = max(induced_width, len(neighbors))
        for left, right in combinations(neighbors, 2):
            graph[left].add(right)
            graph[right].add(left)
        remaining.remove(variable)
        order.append(variable)
    return tuple(order), induced_width


def _exact_pairwise_min_sum(
    unary: Mapping[int, tuple[float, ...]],
    pairwise: Mapping[tuple[int, int], tuple[float, ...]],
    *,
    domain_size: int,
    constant: float,
) -> _MinSumResult | None:
    """Exactly minimize a pairwise categorical energy by variable elimination.

    ``None`` requests the exact SCIP fallback when the greedy induced table is
    too large.  It never changes or relaxes the pricing objective.
    """

    variable_count = len(unary)
    order, induced_width = _minimum_fill_order(variable_count, set(pairwise))
    if domain_size ** (induced_width + 1) > _MAX_EXACT_MIN_SUM_TABLE_ENTRIES:
        return None

    factors: list[_MinSumFactor] = [
        _MinSumFactor((variable,), np.asarray(values, dtype=float))
        for variable, values in sorted(unary.items())
    ]
    factors.extend(
        _MinSumFactor(
            edge,
            np.asarray(values, dtype=float).reshape((domain_size,) * len(edge)),
        )
        for edge, values in sorted(pairwise.items())
    )
    decisions: list[tuple[int, tuple[int, ...], Any]] = []
    scalar = float(constant)

    for variable in order:
        bucket = [factor for factor in factors if variable in factor.scope]
        factors = [factor for factor in factors if variable not in factor.scope]
        if not bucket:
            decisions.append((variable, (), (0,)))
            continue
        retained_scope = tuple(
            sorted({item for factor in bucket for item in factor.scope if item != variable})
        )
        full_scope = (variable, *retained_scope)
        joint = np.zeros((domain_size,) * len(full_scope), dtype=float)
        for factor in bucket:
            ordered_scope = tuple(item for item in full_scope if item in factor.scope)
            permutation = tuple(factor.scope.index(item) for item in ordered_scope)
            aligned = factor.values
            if permutation != tuple(range(len(factor.scope))):
                aligned = np.transpose(aligned, permutation)
            shape = tuple(domain_size if item in factor.scope else 1 for item in full_scope)
            joint += aligned.reshape(shape)
        reduced_values = np.min(joint, axis=0)
        argmins = np.argmin(joint, axis=0)
        decisions.append((variable, retained_scope, argmins))
        if retained_scope:
            factors.append(_MinSumFactor(retained_scope, reduced_values))
        else:
            scalar += float(reduced_values)

    if factors:
        raise RuntimeError("min-sum elimination left an uneliminated factor")
    assignment: dict[int, int] = {}
    for variable, retained_scope, argmins in reversed(decisions):
        retained = tuple(assignment[item] for item in retained_scope)
        assignment[variable] = int(argmins[retained] if retained else argmins)
    return _MinSumResult(
        value=scalar,
        response=tuple(assignment[variable] for variable in range(variable_count)),
        induced_width=induced_width,
    )


def _response_objective_value(
    coefficients: Mapping[tuple[int, int, int, int], float],
    response: tuple[int, ...],
) -> float:
    return sum(
        coefficient
        for (left, right, left_state, right_state), coefficient in coefficients.items()
        if response[left] == left_state and response[right] == right_state
    )


def _off_diagonal_additive_decomposition(
    table: tuple[float, ...],
    domain_size: int,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]] | None:
    """Decompose ``P[a,b]`` into row + column + a nonnegative diagonal.

    When this succeeds, the corresponding categorical pair potential needs
    only one same-label variable per state.  The check is numerical and exact
    up to the solver tolerance; unsupported potentials use the generic
    transportation formulation.
    """

    if domain_size < 3:
        return None
    row: list[float | None] = [None] * domain_size
    column: list[float | None] = [None] * domain_size
    row[0] = 0.0
    changed = True
    while changed:
        changed = False
        for left_state in range(domain_size):
            for right_state in range(domain_size):
                if left_state == right_state:
                    continue
                value = table[left_state * domain_size + right_state]
                if row[left_state] is not None and column[right_state] is None:
                    column[right_state] = value - row[left_state]
                    changed = True
                elif column[right_state] is not None and row[left_state] is None:
                    row[left_state] = value - column[right_state]
                    changed = True
    if any(value is None for value in row) or any(value is None for value in column):
        return None
    resolved_row = tuple(float(value) for value in row if value is not None)
    resolved_column = tuple(float(value) for value in column if value is not None)
    scale = max(1.0, max(abs(value) for value in table))
    tolerance = 10.0 * _SCIP_NUMERICAL_TOLERANCE * scale
    for left_state in range(domain_size):
        for right_state in range(domain_size):
            if left_state == right_state:
                continue
            expected = resolved_row[left_state] + resolved_column[right_state]
            if abs(table[left_state * domain_size + right_state] - expected) > tolerance:
                return None
    diagonal = tuple(
        table[state * domain_size + state] - resolved_row[state] - resolved_column[state]
        for state in range(domain_size)
    )
    if any(value < -tolerance for value in diagonal):
        return None
    return (
        resolved_row,
        resolved_column,
        tuple(max(0.0, value) for value in diagonal),
    )


def _exact_pairwise_map(
    unary: Mapping[int, tuple[float, ...]],
    pairwise: Mapping[tuple[int, int], tuple[float, ...]],
    *,
    domain_size: int,
    constant: float,
    time_limit_seconds: float | None = None,
    forbidden_responses: frozenset[tuple[int, ...]] = frozenset(),
) -> _PairwiseMapOptimization:
    """Solve the categorical pairwise MAP used by response pricing exactly."""

    started = time.perf_counter()
    min_sum = (
        None
        if forbidden_responses
        else _exact_pairwise_min_sum(
            unary,
            pairwise,
            domain_size=domain_size,
            constant=constant,
        )
    )
    if min_sum is not None:
        return _PairwiseMapOptimization(
            value=min_sum.value,
            response=min_sum.response,
            status="optimal",
            solve_seconds=time.perf_counter() - started,
            variables=0,
            constraints=0,
            backend=f"min_sum_width_{min_sum.induced_width}",
        )

    model = Model("exact-pairwise-response-map")
    model.hideOutput()
    model.setIntParam("parallel/maxnthreads", 1)
    model.setIntParam("randomization/randomseedshift", 0)
    model.setIntParam("randomization/permutationseed", 0)
    model.setRealParam("limits/gap", 0.0)
    model.setRealParam("limits/absgap", 0.0)
    model.setRealParam("numerics/feastol", _SCIP_NUMERICAL_TOLERANCE)
    if time_limit_seconds is not None:
        model.setRealParam("limits/time", time_limit_seconds)
    choices = {
        (context, state): model.addVar(vtype="B")
        for context in unary
        for state in range(domain_size)
    }
    for context in unary:
        model.addCons(quicksum(choices[(context, state)] for state in range(domain_size)) == 1.0)
    for response in forbidden_responses:
        if len(response) != len(unary):
            raise ValueError("forbidden response has the wrong number of contexts")
        model.addCons(
            quicksum(
                choices[(context, response[context])]
                for context in range(len(unary))
            )
            <= len(unary) - 1
        )
    objective: Any = constant + quicksum(
        unary[context][state] * choices[(context, state)]
        for context in unary
        for state in range(domain_size)
    )
    decompositions = {
        edge: _off_diagonal_additive_decomposition(table, domain_size)
        for edge, table in pairwise.items()
    }
    compact = all(value is not None for value in decompositions.values())
    if compact:
        for (left, right), decomposition in decompositions.items():
            if decomposition is None:
                raise RuntimeError("validated compact decomposition disappeared")
            row, column, diagonal = decomposition
            objective += quicksum(
                row[state] * choices[(left, state)] + column[state] * choices[(right, state)]
                for state in range(domain_size)
            )
            same = {state: model.addVar(lb=0.0, ub=1.0) for state in range(domain_size)}
            same_mass = quicksum(same.values())
            for state in range(domain_size):
                model.addCons(same[state] <= choices[(left, state)])
                model.addCons(same[state] <= choices[(right, state)])
                model.addCons(
                    choices[(left, state)] + choices[(right, state)] - 2.0 * same[state]
                    <= 1.0 - same_mass
                )
                if diagonal[state] > _SCIP_NUMERICAL_TOLERANCE:
                    objective += diagonal[state] * same[state]
        backend = "compact_same_label"
    else:
        for (left, right), table in pairwise.items():
            transport = {
                (left_state, right_state): model.addVar(lb=0.0, ub=1.0)
                for left_state in range(domain_size)
                for right_state in range(domain_size)
            }
            for left_state in range(domain_size):
                model.addCons(
                    quicksum(
                        transport[(left_state, right_state)] for right_state in range(domain_size)
                    )
                    == choices[(left, left_state)]
                )
            for right_state in range(domain_size - 1):
                model.addCons(
                    quicksum(
                        transport[(left_state, right_state)] for left_state in range(domain_size)
                    )
                    == choices[(right, right_state)]
                )
            objective += quicksum(
                table[left_state * domain_size + right_state] * transport[(left_state, right_state)]
                for left_state in range(domain_size)
                for right_state in range(domain_size)
            )
        backend = "transport"
    if forbidden_responses:
        backend += "_excluding_existing"
    model.setObjective(objective, "minimize")
    model.optimize()
    status = str(model.getStatus())
    response: tuple[int, ...] | None = None
    value = float("inf")
    if model.getNSols() > 0:
        solution = model.getBestSol()
        value = float(model.getSolObjVal(solution))
        response = tuple(
            next(
                state
                for state in range(domain_size)
                if model.getSolVal(solution, choices[(context, state)]) > 0.5
            )
            for context in range(len(unary))
        )
    return _PairwiseMapOptimization(
        value=value,
        response=response,
        status=status,
        solve_seconds=time.perf_counter() - started,
        variables=model.getNVars(),
        constraints=model.getNConss(),
        backend=backend,
    )


class _ResponsePricer(Pricer):
    """Exact MAP pricing over response values for every cyclic local block."""

    def __init__(self, blocks: list[_PricingBlock], domain_sizes: Mapping[int, int]) -> None:
        super().__init__()
        self.blocks = blocks
        self.domain_sizes = domain_sizes
        self.generated_columns = 0
        self.deadline: float | None = None
        self.min_sum_calls = 0
        self.max_min_sum_width = 0
        self.scip_fallback_calls = 0
        self.closed = not blocks
        self.timed_out = False
        self.rounds: list[_PricingRoundState] = []

    def begin_solve(self, *, deadline: float | None) -> None:
        """Reset solve-local proof state before SCIP enters pricing."""

        self.deadline = deadline
        self.closed = not self.blocks
        self.timed_out = False
        self.rounds = []

    def pricerinit(self) -> None:
        for block in self.blocks:
            block.normalization = self.model.getTransformedCons(block.normalization)
            block.marginals = {
                key: self.model.getTransformedCons(constraint)
                for key, constraint in block.marginals.items()
            }
            block.kernels = {
                key: self.model.getTransformedCons(constraint)
                for key, constraint in block.kernels.items()
            }

    def _add_response_column(
        self,
        block: _PricingBlock,
        response: tuple[int, ...],
    ) -> None:
        variable = self.model.addVar(
            name=f"lambda_{block.node}_{block.block_id}_{len(block.columns)}",
            lb=0.0,
            obj=(
                -_response_objective_value(block.direct_objective, response)
                if block.direct_objective is not None
                else 0.0
            ),
            pricedVar=True,
        )
        self.model.addConsCoeff(block.normalization, variable, 1.0)
        for (context, state), constraint in block.marginals.items():
            self.model.addConsCoeff(
                constraint,
                variable,
                float(response[context] == state),
            )
        for (left, right, left_state, right_state), constraint in block.kernels.items():
            self.model.addConsCoeff(
                constraint,
                variable,
                -float(response[left] == left_state and response[right] == right_state),
            )
        block.columns.append((response, variable))
        self.generated_columns += 1

    def _price_block(self, block: _PricingBlock, *, farkas: bool) -> bool | None:
        dual = self.model.getDualfarkasLinear if farkas else self.model.getDualsolLinear
        domain_size = self.domain_sizes[block.node]
        constant = -dual(block.normalization)
        unary_values = {
            context: [-dual(block.marginals[(context, state)]) for state in range(domain_size)]
            for context in range(len(block.contexts))
        }
        pairwise_values: dict[tuple[int, int], list[float]] = {}
        if block.direct_objective is None:
            kernel_duals = {key: dual(constraint) for key, constraint in block.kernels.items()}
            for (
                left,
                right,
                left_state,
                right_state,
            ), coefficient in kernel_duals.items():
                if abs(coefficient) <= _SCIP_NUMERICAL_TOLERANCE:
                    continue
                table = pairwise_values.setdefault(
                    (left, right),
                    [0.0] * domain_size**2,
                )
                table[left_state * domain_size + right_state] += coefficient
        else:
            # SCIP internally minimizes a transformed maximization problem, so
            # a direct original objective coefficient enters reduced-cost
            # pricing with the opposite sign.
            for (
                left,
                right,
                left_state,
                right_state,
            ), coefficient in block.direct_objective.items():
                transformed = -coefficient
                if abs(transformed) <= _SCIP_NUMERICAL_TOLERANCE:
                    continue
                if left == right:
                    if left_state == right_state:
                        unary_values[left][left_state] += transformed
                    continue
                if left > right:
                    left, right = right, left
                    left_state, right_state = right_state, left_state
                table = pairwise_values.setdefault(
                    (left, right),
                    [0.0] * domain_size**2,
                )
                table[left_state * domain_size + right_state] += transformed
        unary = {context: tuple(values) for context, values in unary_values.items()}
        pairwise = {edge: tuple(values) for edge, values in pairwise_values.items()}
        remaining: float | None = None
        if self.deadline is not None:
            remaining = self.deadline - time.perf_counter()
            if remaining <= 0.0:
                self.timed_out = True
                self.model.interruptSolve()
                return None
        pricing = _exact_pairwise_map(
            unary,
            pairwise,
            domain_size=domain_size,
            constant=constant,
            time_limit_seconds=remaining,
        )
        if pricing.backend.startswith("min_sum_width_"):
            self.min_sum_calls += 1
            self.max_min_sum_width = max(
                self.max_min_sum_width,
                int(pricing.backend.rsplit("_", 1)[1]),
            )
        else:
            self.scip_fallback_calls += 1
        if pricing.status != "optimal":
            if pricing.status == "timelimit":
                self.timed_out = True
                self.model.interruptSolve()
                return None
            raise RuntimeError("response pricing MIP did not solve to global optimality")
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            self.timed_out = True
            self.model.interruptSolve()
            return None
        if not self.model.isLT(pricing.value, -_SCIP_NUMERICAL_TOLERANCE):
            return False
        if pricing.response is None:
            raise RuntimeError("optimal pricing returned no deterministic response")
        existing = frozenset(response for response, _ in block.columns)
        if pricing.response in existing:
            remaining = None
            if self.deadline is not None:
                remaining = self.deadline - time.perf_counter()
                if remaining <= 0.0:
                    self.timed_out = True
                    self.model.interruptSolve()
                    return None
            pricing = _exact_pairwise_map(
                unary,
                pairwise,
                domain_size=domain_size,
                constant=constant,
                time_limit_seconds=remaining,
                forbidden_responses=existing,
            )
            self.scip_fallback_calls += 1
            if pricing.status == "infeasible":
                return False
            if pricing.status == "timelimit":
                self.timed_out = True
                self.model.interruptSolve()
                return None
            if pricing.status != "optimal":
                raise RuntimeError(
                    "response repricing excluding existing columns did not solve "
                    "to global optimality"
                )
            if not self.model.isLT(pricing.value, -_SCIP_NUMERICAL_TOLERANCE):
                return False
            if pricing.response is None or pricing.response in existing:
                raise RuntimeError("exact response repricing failed to return an unseen column")
        self._add_response_column(block, pricing.response)
        return True

    def _price(self, *, farkas: bool) -> dict[str, Any]:
        round_state = _PricingRoundState(farkas=farkas)
        self.rounds.append(round_state)
        if not farkas:
            self.closed = False
        generated_before = self.generated_columns
        for block in self.blocks:
            generated = self._price_block(block, farkas=farkas)
            if generated is None:
                round_state.generated_columns = self.generated_columns - generated_before
                result: dict[str, Any] = {"result": SCIP_RESULT.DIDNOTRUN}
                if not farkas:
                    result["stopearly"] = True
                return result
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            self.timed_out = True
            self.model.interruptSolve()
            round_state.generated_columns = self.generated_columns - generated_before
            result = {"result": SCIP_RESULT.DIDNOTRUN}
            if not farkas:
                result["stopearly"] = True
            return result
        round_state.completed = True
        round_state.generated_columns = self.generated_columns - generated_before
        if not farkas:
            self.closed = round_state.generated_columns == 0
        return {"result": SCIP_RESULT.SUCCESS}

    def pricerredcost(self) -> dict[str, Any]:
        return self._price(farkas=False)

    def pricerfarkas(self) -> dict[str, Any]:
        return self._price(farkas=True)


def _topological_order(world: WorldSpec) -> tuple[int, ...]:
    indegree = [len(world.parents.get(node, ())) for node in range(len(world.variables))]
    children: list[list[int]] = [[] for _ in world.variables]
    for parent, child in world.edges:
        children[parent].append(child)
    ready = sorted(node for node, degree in enumerate(indegree) if degree == 0)
    order: list[int] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if len(order) != len(world.variables):
        raise ValueError("WorldSpec graph must be acyclic")
    return tuple(order)


def _descendants(world: WorldSpec, source: int) -> frozenset[int]:
    children: list[list[int]] = [[] for _ in world.variables]
    for parent, child in world.edges:
        children[parent].append(child)
    seen: set[int] = set()
    stack = [source]
    while stack:
        current = stack.pop()
        for child in children[current]:
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return frozenset(seen)


def _ancestors(world: WorldSpec, target: int) -> frozenset[int]:
    seen: set[int] = set()
    stack = [target]
    while stack:
        current = stack.pop()
        for parent in world.parents[current]:
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return frozenset(seen)


def _row_index(world: WorldSpec, node: int, context: tuple[int, ...]) -> int:
    index = 0
    for parent, state in zip(world.parents[node], context, strict=True):
        index = index * world.domains[parent] + state
    return index


def _active_contexts(
    world: WorldSpec,
    node: int,
    treatment: int,
    baseline_value: int,
    treatment_value: int,
) -> tuple[tuple[int, ...], ...]:
    parents = world.parents[node]
    contexts = product(*(range(world.domains[parent]) for parent in parents))
    if treatment not in parents:
        return tuple(contexts)
    treatment_position = parents.index(treatment)
    return tuple(
        context
        for context in contexts
        if context[treatment_position] in {baseline_value, treatment_value}
    )


def _token_domain(world: WorldSpec, token: tuple[int, int]) -> int:
    return world.domains[token[0]]


def _token_states(
    world: WorldSpec,
    token: tuple[int, int],
    outcome: int,
    outcome_state: int | None,
    outcome_events: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
) -> tuple[int, ...]:
    """Return states compatible with the selected two-world outcome event."""

    if outcome_events is None:
        if outcome_state is None:
            raise ValueError("outcome_state is required for a transition event")
        outcome_events = (
            tuple(state for state in range(world.domains[outcome]) if state != outcome_state),
            (outcome_state,),
        )
    if token == (outcome, 0):
        return outcome_events[0]
    if token == (outcome, 1):
        return outcome_events[1]
    return tuple(range(_token_domain(world, token)))


def _edges_form_forest(node_count: int, edges: tuple[tuple[int, int], ...]) -> bool:
    representatives = list(range(node_count))

    def root(item: int) -> int:
        while representatives[item] != item:
            representatives[item] = representatives[representatives[item]]
            item = representatives[item]
        return item

    for left, right in edges:
        left_root = root(left)
        right_root = root(right)
        if left_root == right_root:
            return False
        representatives[left_root] = right_root
    return True


def _context_components(
    node_count: int,
    edges: tuple[tuple[int, int], ...],
) -> tuple[_ContextComponent, ...]:
    """Split the queried context graph without adding cross-component constraints."""

    neighbors: dict[int, set[int]] = {index: set() for index in range(node_count)}
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    components: list[_ContextComponent] = []
    seen: set[int] = set()
    for start in range(node_count):
        if start in seen or not neighbors[start]:
            continue
        stack = [start]
        vertices: set[int] = set()
        while stack:
            current = stack.pop()
            if current in vertices:
                continue
            vertices.add(current)
            seen.add(current)
            stack.extend(neighbors[current] - vertices)
        indices = tuple(sorted(vertices))
        component_edges = tuple(
            edge for edge in edges if edge[0] in vertices and edge[1] in vertices
        )
        components.append(
            _ContextComponent(
                indices=indices,
                edges=component_edges,
                forest=_edges_form_forest(node_count, component_edges),
            )
        )
    return tuple(components)


def _factor_upper(factor: _SymbolicFactor, projected: tuple[int, ...]) -> float:
    values = factor.upper_bounds
    return float(values(projected) if callable(values) else values[projected])


def _factor_initial(factor: _SymbolicFactor, projected: tuple[int, ...]) -> float:
    values = factor.initial_values
    return float(values(projected) if callable(values) else values[projected])


def _message_upper_bound(raw_upper: float, *, probability_bound: bool) -> float:
    """Return a valid bound for one conditional event-probability message.

    ``raw_upper`` is the legacy entrywise interval bound.  Every message made
    by the reverse-topological contraction is a conditional probability of the
    terminal event, so one is also a valid upper bound.  Keeping the legacy
    branch here gives tests an exact comparison path without duplicating the
    contraction or solver.
    """

    nonnegative_upper = max(0.0, raw_upper)
    return min(1.0, nonnegative_upper) if probability_bound else nonnegative_upper


def _resolved_terminal_events(
    world: WorldSpec,
    outcome: int,
    outcome_state: int | None,
    outcome_events: tuple[tuple[int, ...], tuple[int, ...]] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if outcome_events is not None:
        return outcome_events
    if outcome_state is None:
        raise ValueError("outcome_state is required for a transition event")
    return (
        tuple(
            state
            for state in range(world.domains[outcome])
            if state != outcome_state
        ),
        (outcome_state,),
    )


def _coarsen_terminal_event_outcome(
    world: WorldSpec,
    outcome: int,
    outcome_events: tuple[tuple[int, ...], tuple[int, ...]],
) -> tuple[WorldSpec, tuple[tuple[int, ...], tuple[int, ...]]] | None:
    """Quotient terminal states by the two event-membership bits.

    The two-world objective can observe an outcome state only through whether
    it belongs to the left and right terminal events.  Aggregating states with
    the same two membership bits is exact: every original response mechanism
    projects to the quotient, and every quotient response mechanism can be
    refined within each group to recover every original CPT row.
    """

    left_event, right_event = map(frozenset, outcome_events)
    signatures: list[tuple[bool, bool]] = []
    groups: list[list[int]] = []
    for state in range(world.domains[outcome]):
        signature = (state in left_event, state in right_event)
        if signature not in signatures:
            signatures.append(signature)
            groups.append([])
        groups[signatures.index(signature)].append(state)
    if len(groups) < 2 or len(groups) >= world.domains[outcome]:
        return None

    cpt = dict(world.cpt)
    cpt[outcome] = tuple(
        tuple(sum(row[state] for state in group) for group in groups)
        for row in world.cpt[outcome]
    )
    domains = list(world.domains)
    domains[outcome] = len(groups)
    state_names = list(world.state_names)
    state_names[outcome] = tuple(
        "{" + ",".join(world.state_names[outcome][state] for state in group) + "}"
        for group in groups
    )
    quotient = WorldSpec(
        family=world.family,
        topology=f"{world.topology}|terminal-event-quotient",
        variables=world.variables,
        domains=tuple(domains),
        state_names=tuple(state_names),
        edges=world.edges,
        parents=world.parents,
        cpt=cpt,
    )
    quotient_events = (
        tuple(index for index, signature in enumerate(signatures) if signature[0]),
        tuple(index for index, signature in enumerate(signatures) if signature[1]),
    )
    return quotient, quotient_events


def _terminal_event_endpoint_is_jointly_attainable(
    world: WorldSpec,
    treatment: int,
    outcome: int,
    outcome_events: tuple[tuple[int, ...], tuple[int, ...]],
    endpoint: str,
    *,
    baseline_value: int,
    treatment_value: int,
) -> bool:
    """Return whether every pointwise terminal endpoint has one joint owner.

    A direct treatment parent separates the factual and counterfactual context
    sets, so the usual lower or upper Frechet construction can be shared by
    every queried context pair. Without that separation, lower endpoints are
    still jointly attainable for disjoint terminal events: put every left
    event at the bottom of one common uniform coordinate and every right event
    at the top. Upper endpoints are jointly attainable for identical events by
    nesting all event indicators on the same uniform coordinate. Residual
    categorical states can be refined with additional exogenous coordinates,
    so neither construction restricts the supplied CPT rows.
    """

    if endpoint not in {"lower", "upper"}:
        raise ValueError("terminal endpoint must be lower or upper")
    if (
        treatment in world.parents[outcome]
        and baseline_value != treatment_value
    ):
        return True
    left_event, right_event = map(frozenset, outcome_events)
    if endpoint == "lower":
        return left_event.isdisjoint(right_event)
    return left_event == right_event


def _terminal_lower_is_constant_zero(
    world: WorldSpec,
    outcome: int,
    outcome_events: tuple[tuple[int, ...], tuple[int, ...]],
) -> bool:
    """Certify that the jointly attainable terminal lower cost is zero."""

    left_event, right_event = map(frozenset, outcome_events)
    if not left_event.isdisjoint(right_event):
        return False
    left_maximum = max(
        sum(float(row[state]) for state in left_event)
        for row in world.cpt[outcome]
    )
    right_maximum = max(
        sum(float(row[state]) for state in right_event)
        for row in world.cpt[outcome]
    )
    return left_maximum + right_maximum <= 1.0


def _eliminate_factor_tokens(
    world: WorldSpec,
    model: Model,
    factors: list[_SymbolicFactor],
    tokens: tuple[tuple[int, int], ...],
    local_factor: _SymbolicFactor,
    *,
    outcome: int,
    outcome_state: int | None,
    outcome_events: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
    auxiliary_values: list[tuple[Any, float]],
    probability_message_bounds: bool = True,
) -> list[_SymbolicFactor]:
    """Contract one local mechanism and all downstream messages in one step."""

    involving = [factor for factor in factors if any(token in factor.scope for token in tokens)]
    involving.append(local_factor)
    untouched = [factor for factor in factors if all(token not in factor.scope for token in tokens)]
    union_scope = tuple(
        sorted({item for factor in involving for item in factor.scope if item not in tokens})
    )
    full_scope = (*union_scope, *tokens)
    projection_positions = tuple(
        tuple(full_scope.index(item) for item in factor.scope) for factor in involving
    )
    values: dict[tuple[int, ...], Any] = {}
    upper_bounds: dict[tuple[int, ...], float] = {}
    initial_values: dict[tuple[int, ...], float] = {}
    for assignment in product(
        *(
            _token_states(world, item, outcome, outcome_state, outcome_events)
            for item in union_scope
        )
    ):
        terms: list[Any] = []
        upper = 0.0
        initial = 0.0
        for eliminated_assignment in product(
            *(
                _token_states(world, item, outcome, outcome_state, outcome_events)
                for item in tokens
            )
        ):
            full_assignment = (*assignment, *eliminated_assignment)
            term: Any = 1.0
            term_upper = 1.0
            term_initial = 1.0
            for factor, positions in zip(involving, projection_positions, strict=True):
                projected = tuple(full_assignment[position] for position in positions)
                factor_values = factor.values
                term *= (
                    factor_values(projected)
                    if callable(factor_values)
                    else factor_values[projected]
                )
                term_upper *= _factor_upper(factor, projected)
                term_initial *= _factor_initial(factor, projected)
            terms.append(term)
            upper += term_upper
            initial += term_initial
        declared_upper = _message_upper_bound(
            upper,
            probability_bound=probability_message_bounds,
        )
        if all(isinstance(term, (int, float)) for term in terms):
            value: Any = sum(float(term) for term in terms)
        else:
            value = model.addVar(
                name=f"ve_{len(auxiliary_values)}",
                lb=0.0,
                ub=declared_upper,
            )
            model.addCons(value == quicksum(terms))
            auxiliary_values.append((value, initial))
        values[assignment] = value
        upper_bounds[assignment] = declared_upper
        initial_values[assignment] = initial
    untouched.append(_SymbolicFactor(union_scope, values, upper_bounds, initial_values))
    return untouched


class _SparseResponseModel:
    def __init__(
        self,
        world: WorldSpec,
        treatment: int,
        outcome: int,
        *,
        baseline_value: int,
        treatment_value: int,
        outcome_state: int | None,
        sense: str,
        target_outer_bounds: tuple[float, float],
        outcome_events: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
        probability_message_bounds: bool = True,
        on_demand_response_columns: bool = True,
        terminal_event_endpoint: str | None = None,
    ) -> None:
        self.world = world
        self.treatment = treatment
        self.outcome = outcome
        self.baseline_value = baseline_value
        self.treatment_value = treatment_value
        self.outcome_state = outcome_state
        self.outcome_events = outcome_events
        self.sense = sense
        self.target_outer_bounds = target_outer_bounds
        self.probability_message_bounds = probability_message_bounds
        self.on_demand_response_columns = on_demand_response_columns
        if terminal_event_endpoint not in {None, "lower", "upper"}:
            raise ValueError("terminal_event_endpoint must be lower, upper, or None")
        resolved_terminal_events = _resolved_terminal_events(
            world,
            outcome,
            outcome_state,
            outcome_events,
        )
        if terminal_event_endpoint is not None and not (
            _terminal_event_endpoint_is_jointly_attainable(
                world,
                treatment,
                outcome,
                resolved_terminal_events,
                terminal_event_endpoint,
                baseline_value=baseline_value,
                treatment_value=treatment_value,
            )
        ):
            raise ValueError("terminal event endpoint has no joint response certificate")
        self.terminal_event_endpoint = terminal_event_endpoint
        self.model = Model(f"cpt-world-counterfactual-{sense}")
        self.model.hideOutput()
        self.model.setPresolve(SCIP_PARAMSETTING.OFF)
        self.model.setIntParam("parallel/maxnthreads", 1)
        self.model.setIntParam("randomization/randomseedshift", 0)
        self.model.setIntParam("randomization/permutationseed", 0)
        self.model.setRealParam("limits/gap", 0.0)
        self.model.setRealParam("limits/absgap", 0.0)
        self.model.setRealParam("numerics/feastol", _SCIP_NUMERICAL_TOLERANCE)

        ancestors = _ancestors(world, outcome) | {outcome}
        affected_set = _descendants(world, treatment) & ancestors
        order = _topological_order(world)
        self.affected = tuple(node for node in order if node in affected_set)
        self.mechanism_affected = tuple(
            node
            for node in self.affected
            if terminal_event_endpoint is None or node != outcome
        )
        self.shared = tuple(
            node
            for node in order
            if node in ancestors and node not in affected_set and node != treatment
        )
        self.contexts: dict[int, tuple[tuple[int, ...], ...]] = {}
        self.forest_edges: dict[int, tuple[tuple[int, int], ...]] = {}
        self.context_components: dict[int, tuple[_ContextComponent, ...]] = {}
        self.forest_edge_set: set[tuple[int, int, int]] = set()
        self.pricing_blocks: list[_PricingBlock] = []
        self.dynamic_pricing_blocks: list[_PricingBlock] = []
        self.pricing_block_by_edge: dict[tuple[int, int, int], _PricingBlock] = {}
        self.kernel_cache: dict[tuple[int, int, int, int, int], Any] = {}
        self.initial_slots: dict[int, tuple[tuple[float, tuple[int, ...]], ...]] = {}
        self.auxiliary_values: list[tuple[Any, float]] = []
        self._prepare_context_graphs()
        self._build_response_couplings()
        objective, self.initial_target = self._build_twin_probability()
        self.target = self.model.addVar(
            name="counterfactual_target",
            lb=target_outer_bounds[0],
            ub=target_outer_bounds[1],
        )
        self.model.addCons(self.target == objective, name="counterfactual_definition")
        self.model.setObjective(self.target, sense)
        self.sense = sense
        self.pricer = _ResponsePricer(
            self.dynamic_pricing_blocks,
            {block.node: self.world.domains[block.node] for block in self.pricing_blocks},
        )
        if self.dynamic_pricing_blocks:
            self.model.includePricer(
                self.pricer,
                "CPTWorldResponsePricer",
                "exact on-demand deterministic response columns",
            )
        self.original_pricing_state = {
            block.block_id: (
                block.normalization,
                dict(block.marginals),
                dict(block.kernels),
                tuple(block.columns),
            )
            for block in self.pricing_blocks
        }
        self._add_initial_completion()
        self.last_certification = "exact"
        self.last_endpoint_error = 0.0

    def _enable_static_presolve(self) -> None:
        """Enable owner presolve only after every response column is explicit."""

        if not self.dynamic_pricing_blocks:
            self.model.setPresolve(SCIP_PARAMSETTING.DEFAULT)

    def _required_context_edges(self, node: int) -> tuple[tuple[int, int], ...]:
        contexts = self.contexts[node]
        context_indices = {context: index for index, context in enumerate(contexts)}
        parent_tokens: list[tuple[int, int]] = []
        for parent in self.world.parents[node]:
            if parent == self.treatment:
                continue
            if parent in self.affected:
                parent_tokens.extend(((parent, 0), (parent, 1)))
            else:
                parent_tokens.append((parent, -1))
        edges: set[tuple[int, int]] = set()
        for assignment in product(
            *(range(_token_domain(self.world, token)) for token in parent_tokens)
        ):
            token_values = dict(zip(parent_tokens, assignment, strict=True))
            left_context: list[int] = []
            right_context: list[int] = []
            for parent in self.world.parents[node]:
                if parent == self.treatment:
                    left_context.append(self.baseline_value)
                    right_context.append(self.treatment_value)
                elif parent in self.affected:
                    left_context.append(token_values[(parent, 0)])
                    right_context.append(token_values[(parent, 1)])
                else:
                    state = token_values[(parent, -1)]
                    left_context.append(state)
                    right_context.append(state)
            left_index = context_indices[tuple(left_context)]
            right_index = context_indices[tuple(right_context)]
            if left_index != right_index:
                edges.add(tuple(sorted((left_index, right_index))))
        return tuple(sorted(edges))

    def _prepare_context_graphs(self) -> None:
        for node in self.mechanism_affected:
            self.contexts[node] = _active_contexts(
                self.world,
                node,
                self.treatment,
                self.baseline_value,
                self.treatment_value,
            )
            edges = self._required_context_edges(node)
            self.forest_edges[node] = edges
            self.context_components[node] = _context_components(len(self.contexts[node]), edges)

    def _build_response_couplings(self) -> None:
        for node in self.mechanism_affected:
            contexts = self.contexts[node]
            domain_size = self.world.domains[node]
            for component in self.context_components[node]:
                if component.forest:
                    for left_index, right_index in component.edges:
                        self.forest_edge_set.add((node, left_index, right_index))
                        left_row = self.world.cpt[node][
                            _row_index(self.world, node, contexts[left_index])
                        ]
                        right_row = self.world.cpt[node][
                            _row_index(self.world, node, contexts[right_index])
                        ]
                        entries: dict[tuple[int, int], Any] = {}
                        for left_state in range(domain_size):
                            for right_state in range(domain_size):
                                key = (
                                    node,
                                    left_index,
                                    right_index,
                                    left_state,
                                    right_state,
                                )
                                entry = self.model.addVar(
                                    name=(
                                        f"k_{node}_{left_index}_{right_index}_"
                                        f"{left_state}_{right_state}"
                                    ),
                                    lb=0.0,
                                    ub=float(min(left_row[left_state], right_row[right_state])),
                                )
                                self.kernel_cache[key] = entry
                                entries[(left_state, right_state)] = entry
                        for left_state in range(domain_size):
                            self.model.addCons(
                                quicksum(
                                    entries[(left_state, right_state)]
                                    for right_state in range(domain_size)
                                )
                                == float(left_row[left_state])
                            )
                        for right_state in range(domain_size):
                            self.model.addCons(
                                quicksum(
                                    entries[(left_state, right_state)]
                                    for left_state in range(domain_size)
                                )
                                == float(right_row[right_state])
                            )
                    continue

                component_contexts = tuple(contexts[index] for index in component.indices)
                rank = 1 + len(component_contexts) * (domain_size - 1)
                block_id = len(self.pricing_blocks)
                slots = self._comonotone_slots(node, component_contexts, rank)
                initial_weights: dict[tuple[int, ...], float] = {}
                for weight, response in slots:
                    if weight > _SCIP_NUMERICAL_TOLERANCE:
                        initial_weights[response] = initial_weights.get(response, 0.0) + weight
                response_count = domain_size ** len(component_contexts)
                dynamic = self.on_demand_response_columns or (
                    len(component_contexts) > _LEGACY_MAX_EXPLICIT_RESPONSE_CONTEXTS
                    or response_count > _LEGACY_MAX_EXPLICIT_RESPONSE_COLUMNS
                )
                responses = (
                    tuple(initial_weights)
                    if dynamic
                    else tuple(product(range(domain_size), repeat=len(component_contexts)))
                )
                columns = [
                    (
                        response,
                        self.model.addVar(
                            name=f"lambda_{node}_{block_id}_{index}",
                            lb=0.0,
                        ),
                    )
                    for index, response in enumerate(responses)
                ]
                normalization = self.model.addCons(
                    quicksum(variable for _, variable in columns) == 1.0,
                    separate=False,
                    modifiable=dynamic,
                )
                marginals: dict[tuple[int, int], Any] = {}
                for context_index, context in enumerate(component_contexts):
                    cpt_row = self.world.cpt[node][_row_index(self.world, node, context)]
                    for state, probability in enumerate(cpt_row):
                        constraint = self.model.addCons(
                            quicksum(
                                variable
                                for response, variable in columns
                                if response[context_index] == state
                            )
                            == float(probability),
                            separate=False,
                            modifiable=dynamic,
                        )
                        marginals[(context_index, state)] = constraint
                block = _PricingBlock(
                    block_id=block_id,
                    node=node,
                    contexts=component_contexts,
                    columns=columns,
                    initial_weights=initial_weights,
                    normalization=normalization,
                    marginals=marginals,
                    kernels={},
                    dynamic=dynamic,
                )
                self.pricing_blocks.append(block)
                if dynamic:
                    self.dynamic_pricing_blocks.append(block)
                for left_index, right_index in component.edges:
                    self.pricing_block_by_edge[(node, left_index, right_index)] = block
                self.initial_slots[block_id] = slots

    def _comonotone_slots(
        self,
        node: int,
        contexts: tuple[tuple[int, ...], ...],
        slot_count: int,
    ) -> tuple[tuple[float, tuple[int, ...]], ...]:
        """Construct one sparse feasible response coupling from common uniforms."""

        rows = tuple(
            tuple(float(value) for value in self.world.cpt[node][_row_index(self.world, node, c)])
            for c in contexts
        )
        support = [
            (weight, response)
            for response, weight in _comonotone_response_weights(rows).items()
        ]
        padding = [(0.0, (0,) * len(contexts))] * (slot_count - len(support))
        slots = [*support, *padding]
        domain_size = self.world.domains[node]
        slots.sort(
            key=lambda item: sum(
                state * domain_size**context for context, state in enumerate(item[1])
            )
        )
        return tuple(slots)

    def _kernel_entry(
        self,
        node: int,
        left_context: tuple[int, ...],
        right_context: tuple[int, ...],
        left_state: int,
        right_state: int,
    ) -> Any:
        if left_context == right_context:
            if left_state != right_state:
                return 0.0
            row = self.world.cpt[node][_row_index(self.world, node, left_context)]
            return float(row[left_state])
        contexts = self.contexts[node]
        left_index = contexts.index(left_context)
        right_index = contexts.index(right_context)
        if left_index > right_index:
            left_index, right_index = right_index, left_index
            left_state, right_state = right_state, left_state
        edge_key = (node, left_index, right_index)
        if edge_key in self.forest_edge_set:
            key = (node, left_index, right_index, left_state, right_state)
            return self.kernel_cache[key]
        block = self.pricing_block_by_edge.get(edge_key)
        if block is not None:
            cache_key = (node, left_index, right_index, left_state, right_state)
            if cache_key in self.kernel_cache:
                return self.kernel_cache[cache_key]
            kernel = self.model.addVar(
                name=f"k_{node}_{left_index}_{right_index}_{left_state}_{right_state}",
                lb=0.0,
                ub=float(
                    min(
                        self.world.cpt[node][_row_index(self.world, node, contexts[left_index])][
                            left_state
                        ],
                        self.world.cpt[node][_row_index(self.world, node, contexts[right_index])][
                            right_state
                        ],
                    )
                ),
            )
            local_left = block.contexts.index(contexts[left_index])
            local_right = block.contexts.index(contexts[right_index])
            constraint = self.model.addCons(
                kernel
                - quicksum(
                    variable
                    for response, variable in block.columns
                    if response[local_left] == left_state and response[local_right] == right_state
                )
                == 0.0,
                separate=False,
                modifiable=block.dynamic,
            )
            block.kernels[(local_left, local_right, left_state, right_state)] = constraint
            self.kernel_cache[cache_key] = kernel
            return kernel
        raise RuntimeError("affected node has no exact response-coupling owner")

    def _shared_factor(self, node: int) -> _SymbolicFactor:
        parents = self.world.parents[node]
        scope = tuple((parent, -1) for parent in parents) + ((node, -1),)

        def value(assignment: tuple[int, ...]) -> float:
            context = assignment[:-1]
            state = assignment[-1]
            return float(self.world.cpt[node][_row_index(self.world, node, context)][state])

        return _SymbolicFactor(scope, value, value, value)

    def _affected_factor(self, node: int) -> _SymbolicFactor:
        parent_tokens: list[tuple[int, int]] = []
        for parent in self.world.parents[node]:
            if parent == self.treatment:
                continue
            if parent in self.affected:
                parent_tokens.extend(((parent, 0), (parent, 1)))
            else:
                parent_tokens.append((parent, -1))
        scope = (*parent_tokens, (node, 0), (node, 1))
        parent_specs = self.world.parents[node]

        def decode(
            assignment: tuple[int, ...],
        ) -> tuple[tuple[int, ...], tuple[int, ...], int, int]:
            token_values = dict(zip(scope, assignment, strict=True))
            left_context: list[int] = []
            right_context: list[int] = []
            for parent in parent_specs:
                if parent == self.treatment:
                    left_context.append(self.baseline_value)
                    right_context.append(self.treatment_value)
                elif parent in self.affected:
                    left_context.append(token_values[(parent, 0)])
                    right_context.append(token_values[(parent, 1)])
                else:
                    state = token_values[(parent, -1)]
                    left_context.append(state)
                    right_context.append(state)
            left_context_tuple = tuple(left_context)
            right_context_tuple = tuple(right_context)
            left_state = token_values[(node, 0)]
            right_state = token_values[(node, 1)]
            return left_context_tuple, right_context_tuple, left_state, right_state

        def value(assignment: tuple[int, ...]) -> Any:
            left_context, right_context, left_state, right_state = decode(assignment)
            return self._kernel_entry(
                node,
                left_context,
                right_context,
                left_state,
                right_state,
            )

        def upper(assignment: tuple[int, ...]) -> float:
            left_context, right_context, left_state, right_state = decode(assignment)
            left_probability = self.world.cpt[node][_row_index(self.world, node, left_context)][
                left_state
            ]
            right_probability = self.world.cpt[node][_row_index(self.world, node, right_context)][
                right_state
            ]
            return float(min(left_probability, right_probability))

        def initial(assignment: tuple[int, ...]) -> float:
            left_context, right_context, left_state, right_state = decode(assignment)
            return self._initial_kernel_value(
                node,
                left_context,
                right_context,
                left_state,
                right_state,
            )

        return _SymbolicFactor(scope, value, upper, initial)

    def _build_twin_probability(self) -> tuple[Any, float]:
        factors = (
            [self._terminal_event_factor()]
            if self.terminal_event_endpoint is not None
            else []
        )
        relevant = set(self.shared) | set(self.mechanism_affected)
        for node in reversed(_topological_order(self.world)):
            if node not in relevant:
                continue
            if node in self.affected:
                tokens = ((node, 0), (node, 1))
                local_factor = self._affected_factor(node)
            else:
                tokens = ((node, -1),)
                local_factor = self._shared_factor(node)
            factors = _eliminate_factor_tokens(
                self.world,
                self.model,
                factors,
                tokens,
                local_factor,
                outcome=self.outcome,
                outcome_state=self.outcome_state,
                outcome_events=self.outcome_events,
                auxiliary_values=self.auxiliary_values,
                probability_message_bounds=self.probability_message_bounds,
            )
        expression: Any = 1.0
        initial = 1.0
        for factor in factors:
            factor_values = factor.values
            expression *= factor_values(()) if callable(factor_values) else factor_values[()]
            initial *= _factor_initial(factor, ())
        return expression, initial

    def _terminal_event_factor(self) -> _SymbolicFactor:
        """Return a pointwise terminal cost with one certified joint owner."""

        if self.terminal_event_endpoint is None:
            raise RuntimeError("terminal event factor requested on the default path")
        outcome_events = _resolved_terminal_events(
            self.world,
            self.outcome,
            self.outcome_state,
            self.outcome_events,
        )
        parent_tokens: list[tuple[int, int]] = []
        for parent in self.world.parents[self.outcome]:
            if parent == self.treatment:
                continue
            if parent in self.affected:
                parent_tokens.extend(((parent, 0), (parent, 1)))
            else:
                parent_tokens.append((parent, -1))
        scope = tuple(parent_tokens)

        def value(assignment: tuple[int, ...]) -> float:
            token_values = dict(zip(scope, assignment, strict=True))
            left_context: list[int] = []
            right_context: list[int] = []
            for parent in self.world.parents[self.outcome]:
                if parent == self.treatment:
                    left_context.append(self.baseline_value)
                    right_context.append(self.treatment_value)
                elif parent in self.affected:
                    left_context.append(token_values[(parent, 0)])
                    right_context.append(token_values[(parent, 1)])
                else:
                    state = token_values[(parent, -1)]
                    left_context.append(state)
                    right_context.append(state)
            left_row = self.world.cpt[self.outcome][
                _row_index(self.world, self.outcome, tuple(left_context))
            ]
            right_row = self.world.cpt[self.outcome][
                _row_index(self.world, self.outcome, tuple(right_context))
            ]
            left_probability = sum(float(left_row[state]) for state in outcome_events[0])
            right_probability = sum(
                float(right_row[state]) for state in outcome_events[1]
            )
            if self.terminal_event_endpoint == "lower":
                return max(0.0, left_probability + right_probability - 1.0)
            return min(left_probability, right_probability)

        return _SymbolicFactor(scope, value, value, value)

    def _initial_kernel_value(
        self,
        node: int,
        left_context: tuple[int, ...],
        right_context: tuple[int, ...],
        left_state: int,
        right_state: int,
    ) -> float:
        if left_context == right_context:
            if left_state != right_state:
                return 0.0
            return float(
                self.world.cpt[node][_row_index(self.world, node, left_context)][left_state]
            )
        contexts = self.contexts[node]
        left_index = contexts.index(left_context)
        right_index = contexts.index(right_context)
        if left_index > right_index:
            left_index, right_index = right_index, left_index
            left_state, right_state = right_state, left_state
        edge_key = (node, left_index, right_index)
        if edge_key in self.forest_edge_set:
            left_row = tuple(
                float(value)
                for value in self.world.cpt[node][
                    _row_index(self.world, node, contexts[left_index])
                ]
            )
            right_row = tuple(
                float(value)
                for value in self.world.cpt[node][
                    _row_index(self.world, node, contexts[right_index])
                ]
            )
            left_start = sum(left_row[:left_state])
            left_end = left_start + left_row[left_state]
            right_start = sum(right_row[:right_state])
            right_end = right_start + right_row[right_state]
            return max(0.0, min(left_end, right_end) - max(left_start, right_start))
        block = self.pricing_block_by_edge[edge_key]
        local_left = block.contexts.index(contexts[left_index])
        local_right = block.contexts.index(contexts[right_index])
        return sum(
            weight
            for weight, response in self.initial_slots[block.block_id]
            if response[local_left] == left_state and response[local_right] == right_state
        )

    def _add_initial_completion(self) -> None:
        solution = self.model.createSol()
        for block in self.pricing_blocks:
            for response, variable in block.columns:
                self.model.setSolVal(
                    solution,
                    variable,
                    block.initial_weights.get(response, 0.0),
                )
        for cache_key, kernel in self.kernel_cache.items():
            node, left_index, right_index, left_state, right_state = cache_key
            left_context = self.contexts[node][left_index]
            right_context = self.contexts[node][right_index]
            kernel_value = self._initial_kernel_value(
                node,
                left_context,
                right_context,
                left_state,
                right_state,
            )
            self.model.setSolVal(solution, kernel, kernel_value)
        for auxiliary, value in self.auxiliary_values:
            self.model.setSolVal(solution, auxiliary, value)
        self.model.setSolVal(solution, self.target, self.initial_target)
        if not self.model.addSol(solution):
            raise RuntimeError("failed to add the canonical feasible counterfactual completion")

    def restart_with_objective(self, sense: str) -> None:
        """Reuse the original arithmetic circuit for the opposite endpoint."""

        retained_responses = {
            block.block_id: tuple(response for response, _ in block.columns)
            for block in self.pricing_blocks
        }
        self.model.freeTransform()
        for block in self.pricing_blocks:
            normalization, marginals, kernels, initial_columns = self.original_pricing_state[
                block.block_id
            ]
            block.normalization = normalization
            block.marginals = dict(marginals)
            block.kernels = dict(kernels)
            block.columns = list(initial_columns)
            existing = {response for response, _ in block.columns}
            for response in retained_responses[block.block_id]:
                if response in existing:
                    continue
                variable = self.model.addVar(
                    name=(f"lambda_{block.node}_{block.block_id}_retained_{len(block.columns)}"),
                    lb=0.0,
                )
                self.model.addConsCoeff(block.normalization, variable, 1.0)
                for (context, state), constraint in block.marginals.items():
                    self.model.addConsCoeff(
                        constraint,
                        variable,
                        float(response[context] == state),
                    )
                for (left, right, left_state, right_state), constraint in block.kernels.items():
                    self.model.addConsCoeff(
                        constraint,
                        variable,
                        -float(response[left] == left_state and response[right] == right_state),
                    )
                block.columns.append((response, variable))
                existing.add(response)
        self.model.setObjective(self.target, sense)
        self.sense = sense

    def optimize(
        self,
        *,
        time_limit_seconds: float | None,
        accepted_absolute_gap: float = 0.0,
    ) -> tuple[float, float]:
        if not np.isfinite(accepted_absolute_gap) or accepted_absolute_gap < 0.0:
            raise ValueError("accepted absolute gap must be finite and nonnegative")
        deadline: float | None = None
        if time_limit_seconds is not None:
            self.model.setRealParam("limits/time", time_limit_seconds)
            deadline = time.perf_counter() + time_limit_seconds
        self.pricer.begin_solve(deadline=deadline)
        started = time.perf_counter()
        self.model.optimize()
        elapsed = time.perf_counter() - started
        scip_status = str(self.model.getStatus())
        pricing_closed = self.pricer.closed and not self.pricer.timed_out
        primal_bound = float(self.model.getPrimalbound())
        dual_bound = float(self.model.getDualbound())
        endpoint_error = abs(primal_bound - dual_bound)
        numerical_closure = scip_status == "timelimit" and (
            _global_bounds_numerically_closed(primal_bound, dual_bound)
        )
        certified_exact = pricing_closed and (
            scip_status == "optimal" or numerical_closure
        )
        certified_epsilon = (
            not certified_exact
            and accepted_absolute_gap > 0.0
            and pricing_closed
            and scip_status in {"gaplimit", "timelimit"}
            and np.isfinite(endpoint_error)
            and endpoint_error
            <= accepted_absolute_gap + 10.0 * _SCIP_NUMERICAL_TOLERANCE
        )
        if not certified_exact and not certified_epsilon:
            raise RuntimeError(
                "SCIP did not certify an optimal counterfactual bound: "
                f"status={scip_status}, pricing_closed={self.pricer.closed}, "
                f"pricing_timed_out={self.pricer.timed_out}, "
                f"sense={self.sense}, primal={primal_bound}, dual={dual_bound}"
            )
        self.last_certification = "epsilon_sharp" if certified_epsilon else "exact"
        self.last_endpoint_error = endpoint_error if certified_epsilon else 0.0
        # For epsilon-sharp termination the dual bound is the safe outer
        # endpoint: it lies below the true minimum or above the true maximum.
        endpoint = dual_bound if certified_epsilon else primal_bound
        return endpoint, elapsed


def _direct_treatment_terminal_bounds(
    world: WorldSpec,
    treatment: int,
    outcome: int,
    *,
    treatment_value: int,
    baseline_value: int,
    outcome_state: int | None,
    outcome_events: tuple[tuple[int, ...], tuple[int, ...]],
    target_outer_bounds: tuple[float, float],
    time_limit_seconds: float | None,
    accepted_absolute_gap: float = 0.0,
) -> CounterfactualBoundsResult | None:
    """Eliminate a direct-treatment terminal mechanism and keep exact owners."""

    if (
        treatment not in world.parents[outcome]
        or baseline_value == treatment_value
    ):
        return None
    build_started = time.perf_counter()
    lower_model = _SparseResponseModel(
        world,
        treatment,
        outcome,
        baseline_value=baseline_value,
        treatment_value=treatment_value,
        outcome_state=outcome_state,
        sense="minimize",
        target_outer_bounds=target_outer_bounds,
        outcome_events=outcome_events,
        terminal_event_endpoint="lower",
    )
    lower_build_seconds = time.perf_counter() - build_started
    lower, lower_seconds = lower_model.optimize(
        time_limit_seconds=time_limit_seconds,
        accepted_absolute_gap=accepted_absolute_gap,
    )
    lower_error = lower_model.last_endpoint_error
    lower_certification = lower_model.last_certification

    build_started = time.perf_counter()
    upper_model = _SparseResponseModel(
        world,
        treatment,
        outcome,
        baseline_value=baseline_value,
        treatment_value=treatment_value,
        outcome_state=outcome_state,
        sense="maximize",
        target_outer_bounds=target_outer_bounds,
        outcome_events=outcome_events,
        terminal_event_endpoint="upper",
    )
    upper_build_seconds = time.perf_counter() - build_started
    upper, upper_seconds = upper_model.optimize(
        time_limit_seconds=time_limit_seconds,
        accepted_absolute_gap=accepted_absolute_gap,
    )
    return CounterfactualBoundsResult(
        lower=lower,
        upper=upper,
        build_seconds=lower_build_seconds + upper_build_seconds,
        solve_seconds=lower_seconds + upper_seconds,
        affected_nodes=len(lower_model.affected),
        pair_kernel_entries=max(
            len(lower_model.kernel_cache), len(upper_model.kernel_cache)
        ),
        generated_columns=(
            lower_model.pricer.generated_columns
            + upper_model.pricer.generated_columns
        ),
        response_blocks=max(
            len(lower_model.pricing_blocks), len(upper_model.pricing_blocks)
        ),
        dynamic_response_blocks=max(
            len(lower_model.dynamic_pricing_blocks),
            len(upper_model.dynamic_pricing_blocks),
        ),
        max_response_contexts=max(
            (
                len(block.contexts)
                for model in (lower_model, upper_model)
                for block in model.pricing_blocks
            ),
            default=0,
        ),
        auxiliary_variables=max(
            len(lower_model.auxiliary_values),
            len(upper_model.auxiliary_values),
        ),
        certification=(
            "epsilon_sharp"
            if "epsilon_sharp"
            in {lower_certification, upper_model.last_certification}
            else "exact"
        ),
        endpoint_error=max(lower_error, upper_model.last_endpoint_error),
        backend="direct_treatment_terminal_elimination",
    )


def _root_separator(world: WorldSpec, treatment: int, outcome: int) -> int | None:
    """Find one shared root that separates every affected response context."""

    ancestors = _ancestors(world, outcome) | {outcome}
    affected = _descendants(world, treatment) & ancestors
    shared = tuple(
        node
        for node in ancestors
        if node not in affected and node != treatment
    )
    if len(shared) != 1:
        return None
    separator = shared[0]
    if world.parents[separator]:
        return None
    if any(separator not in world.parents[node] for node in affected):
        return None
    return separator


def _fix_root_separator(
    world: WorldSpec,
    separator: int,
    state: int,
) -> WorldSpec:
    """Slice every child CPT at one fixed root state and remove its arrows."""

    parents: dict[int, tuple[int, ...]] = {}
    cpt: dict[int, tuple[tuple[Any, ...], ...]] = {}
    for node in range(len(world.variables)):
        old_parents = world.parents[node]
        if separator not in old_parents:
            parents[node] = old_parents
            cpt[node] = world.cpt[node]
            continue
        new_parents = tuple(parent for parent in old_parents if parent != separator)
        parents[node] = new_parents
        rows: list[tuple[Any, ...]] = []
        for context in product(
            *(range(world.domains[parent]) for parent in new_parents)
        ):
            values = iter(context)
            old_context = tuple(
                state if parent == separator else next(values)
                for parent in old_parents
            )
            rows.append(world.cpt[node][_row_index(world, node, old_context)])
        cpt[node] = tuple(rows)
    return WorldSpec(
        family=world.family,
        topology=f"{world.topology}|root-{separator}={state}",
        variables=world.variables,
        domains=world.domains,
        state_names=world.state_names,
        edges=tuple(edge for edge in world.edges if edge[0] != separator),
        parents=parents,
        cpt=cpt,
    )


def _root_separator_bounds(
    world: WorldSpec,
    treatment: int,
    outcome: int,
    *,
    treatment_value: int,
    baseline_value: int,
    outcome_state: int | None,
    outcome_events: tuple[tuple[int, ...], tuple[int, ...]],
    time_limit_seconds: float | None,
    accepted_absolute_gap: float = 0.0,
) -> CounterfactualBoundsResult | None:
    """Condition on a shared root and solve its independent response strata."""

    separator = _root_separator(world, treatment, outcome)
    if separator is None:
        return None
    from .query_truth import interventional_probability

    results: list[tuple[float, CounterfactualBoundsResult]] = []
    for state, probability in enumerate(world.cpt[separator][0]):
        weight = float(probability)
        if weight <= 0.0:
            continue
        stratum = _fix_root_separator(world, separator, state)
        left_probability = sum(
            float(
                interventional_probability(
                    stratum,
                    {treatment: baseline_value},
                    outcome,
                    event_state,
                )
            )
            for event_state in outcome_events[0]
        )
        right_probability = sum(
            float(
                interventional_probability(
                    stratum,
                    {treatment: treatment_value},
                    outcome,
                    event_state,
                )
            )
            for event_state in outcome_events[1]
        )
        result = _solve_sparse_two_world_event_bounds(
            stratum,
            treatment,
            outcome,
            treatment_value=treatment_value,
            baseline_value=baseline_value,
            outcome_state=outcome_state,
            outcome_events=outcome_events,
            target_outer_bounds=(
                max(0.0, left_probability + right_probability - 1.0),
                min(left_probability, right_probability),
            ),
            time_limit_seconds=time_limit_seconds,
            accepted_absolute_gap=accepted_absolute_gap,
        )
        results.append((weight, result))
    if not results:
        raise ValueError("root separator has no positive-probability state")
    return CounterfactualBoundsResult(
        lower=sum(weight * result.lower for weight, result in results),
        upper=sum(weight * result.upper for weight, result in results),
        build_seconds=sum(result.build_seconds for _, result in results),
        solve_seconds=sum(result.solve_seconds for _, result in results),
        affected_nodes=max(result.affected_nodes for _, result in results),
        pair_kernel_entries=sum(
            result.pair_kernel_entries for _, result in results
        ),
        generated_columns=sum(result.generated_columns for _, result in results),
        response_blocks=sum(result.response_blocks for _, result in results),
        dynamic_response_blocks=sum(
            result.dynamic_response_blocks for _, result in results
        ),
        max_response_contexts=max(
            result.max_response_contexts for _, result in results
        ),
        auxiliary_variables=sum(
            result.auxiliary_variables for _, result in results
        ),
        certification=(
            "epsilon_sharp"
            if any(result.certification == "epsilon_sharp" for _, result in results)
            else "exact"
        ),
        endpoint_error=sum(
            weight * result.endpoint_error for weight, result in results
        ),
        backend="shared_root_separator_decomposition",
    )


def _partially_attainable_terminal_bounds(
    world: WorldSpec,
    treatment: int,
    outcome: int,
    *,
    treatment_value: int,
    baseline_value: int,
    outcome_state: int | None,
    outcome_events: tuple[tuple[int, ...], tuple[int, ...]],
    target_outer_bounds: tuple[float, float],
    time_limit_seconds: float | None,
    on_demand_response_columns: bool = True,
    accepted_absolute_gap: float = 0.0,
) -> CounterfactualBoundsResult | None:
    """Solve endpoints separately when a terminal response is removable."""

    lower_terminal = _terminal_event_endpoint_is_jointly_attainable(
        world,
        treatment,
        outcome,
        outcome_events,
        "lower",
        baseline_value=baseline_value,
        treatment_value=treatment_value,
    )
    upper_terminal = _terminal_event_endpoint_is_jointly_attainable(
        world,
        treatment,
        outcome,
        outcome_events,
        "upper",
        baseline_value=baseline_value,
        treatment_value=treatment_value,
    )
    if not lower_terminal and not upper_terminal:
        return None

    models: list[_SparseResponseModel] = []
    layered_results: list[CounterfactualBoundsResult] = []
    build_seconds = 0.0
    solve_seconds = 0.0
    generated_columns = 0
    endpoint_errors: list[float] = []
    certifications: list[str] = []

    if lower_terminal and _terminal_lower_is_constant_zero(
        world,
        outcome,
        outcome_events,
    ):
        lower = 0.0
    else:
        layered_lower = (
            _two_mediator_joint_bounds(
                world,
                treatment,
                outcome,
                baseline_value=baseline_value,
                treatment_value=treatment_value,
                outcome_events=outcome_events,
                time_limit_seconds=time_limit_seconds,
                endpoint_only="lower",
            )
            if lower_terminal
            else None
        )
        if layered_lower is not None:
            lower = layered_lower.lower
            build_seconds += layered_lower.build_seconds
            solve_seconds += layered_lower.solve_seconds
            generated_columns += layered_lower.generated_columns
            layered_results.append(layered_lower)
            endpoint_errors.append(layered_lower.endpoint_error)
            certifications.append(layered_lower.certification)
        else:
            started = time.perf_counter()
            lower_model = _SparseResponseModel(
                world,
                treatment,
                outcome,
                baseline_value=baseline_value,
                treatment_value=treatment_value,
                outcome_state=outcome_state,
                outcome_events=outcome_events,
                sense="minimize",
                target_outer_bounds=target_outer_bounds,
                terminal_event_endpoint="lower" if lower_terminal else None,
                on_demand_response_columns=on_demand_response_columns,
            )
            if not on_demand_response_columns:
                lower_model._enable_static_presolve()
            build_seconds += time.perf_counter() - started
            lower, elapsed = lower_model.optimize(
                time_limit_seconds=time_limit_seconds,
                accepted_absolute_gap=accepted_absolute_gap,
            )
            solve_seconds += elapsed
            generated_columns += lower_model.pricer.generated_columns
            models.append(lower_model)
            endpoint_errors.append(lower_model.last_endpoint_error)
            certifications.append(lower_model.last_certification)

    layered_upper = (
        _two_mediator_joint_bounds(
            world,
            treatment,
            outcome,
            baseline_value=baseline_value,
            treatment_value=treatment_value,
            outcome_events=outcome_events,
            time_limit_seconds=time_limit_seconds,
            endpoint_only="upper",
        )
        if upper_terminal
        else None
    )
    if layered_upper is not None:
        upper = layered_upper.upper
        build_seconds += layered_upper.build_seconds
        solve_seconds += layered_upper.solve_seconds
        generated_columns += layered_upper.generated_columns
        layered_results.append(layered_upper)
        endpoint_errors.append(layered_upper.endpoint_error)
        certifications.append(layered_upper.certification)
    else:
        started = time.perf_counter()
        upper_model = _SparseResponseModel(
            world,
            treatment,
            outcome,
            baseline_value=baseline_value,
            treatment_value=treatment_value,
            outcome_state=outcome_state,
            outcome_events=outcome_events,
            sense="maximize",
            target_outer_bounds=target_outer_bounds,
            terminal_event_endpoint="upper" if upper_terminal else None,
            on_demand_response_columns=on_demand_response_columns,
        )
        if not on_demand_response_columns:
            upper_model._enable_static_presolve()
        build_seconds += time.perf_counter() - started
        upper, elapsed = upper_model.optimize(
            time_limit_seconds=time_limit_seconds,
            accepted_absolute_gap=accepted_absolute_gap,
        )
        solve_seconds += elapsed
        generated_columns += upper_model.pricer.generated_columns
        models.append(upper_model)
        endpoint_errors.append(upper_model.last_endpoint_error)
        certifications.append(upper_model.last_certification)

    return CounterfactualBoundsResult(
        lower=lower,
        upper=upper,
        build_seconds=build_seconds,
        solve_seconds=solve_seconds,
        affected_nodes=max(
            [len(model.affected) for model in models]
            + [result.affected_nodes for result in layered_results],
            default=0,
        ),
        pair_kernel_entries=max(
            [len(model.kernel_cache) for model in models]
            + [result.pair_kernel_entries for result in layered_results],
            default=0,
        ),
        generated_columns=generated_columns,
        response_blocks=max(
            [len(model.pricing_blocks) for model in models]
            + [result.response_blocks for result in layered_results],
            default=0,
        ),
        dynamic_response_blocks=max(
            [len(model.dynamic_pricing_blocks) for model in models]
            + [result.dynamic_response_blocks for result in layered_results],
            default=0,
        ),
        max_response_contexts=max(
            [
                len(block.contexts)
                for model in models
                for block in model.pricing_blocks
            ]
            + [result.max_response_contexts for result in layered_results],
            default=0,
        ),
        auxiliary_variables=max(
            [len(model.auxiliary_values) for model in models]
            + [result.auxiliary_variables for result in layered_results],
            default=0,
        ),
        certification=(
            "epsilon_sharp" if "epsilon_sharp" in certifications else "exact"
        ),
        endpoint_error=max(endpoint_errors, default=0.0),
        backend=(
            "terminal_event_endpoint_decomposition+layered_endpoint"
            if layered_results
            else "terminal_event_endpoint_decomposition"
        ),
    )


def _solve_sparse_two_world_event_bounds(
    world: WorldSpec,
    treatment: int,
    outcome: int,
    *,
    treatment_value: int,
    baseline_value: int,
    outcome_state: int | None,
    outcome_events: tuple[tuple[int, ...], tuple[int, ...]] | None,
    target_outer_bounds: tuple[float, float],
    time_limit_seconds: float | None = None,
    prefer_static_response_columns: bool = False,
    accepted_absolute_gap: float = 0.0,
) -> CounterfactualBoundsResult:
    resolved_events = outcome_events
    if resolved_events is None:
        if outcome_state is None:
            raise ValueError("outcome_state is required for a transition event")
        resolved_events = (
            tuple(
                state
                for state in range(world.domains[outcome])
                if state != outcome_state
            ),
            (outcome_state,),
        )
    one_mediator = _one_mediator_joint_bounds(
        world,
        treatment,
        outcome,
        baseline_value=baseline_value,
        treatment_value=treatment_value,
        outcome_events=resolved_events,
        time_limit_seconds=time_limit_seconds,
    )
    if one_mediator is not None:
        return one_mediator
    two_mediator = _two_mediator_joint_bounds(
        world,
        treatment,
        outcome,
        baseline_value=baseline_value,
        treatment_value=treatment_value,
        outcome_events=resolved_events,
        time_limit_seconds=time_limit_seconds,
    )
    if two_mediator is not None:
        return two_mediator

    terminal = _direct_treatment_terminal_bounds(
        world,
        treatment,
        outcome,
        baseline_value=baseline_value,
        treatment_value=treatment_value,
        outcome_state=outcome_state,
        outcome_events=resolved_events,
        target_outer_bounds=target_outer_bounds,
        time_limit_seconds=time_limit_seconds,
        accepted_absolute_gap=accepted_absolute_gap,
    )
    if terminal is not None:
        return terminal

    root_separator = _root_separator_bounds(
        world,
        treatment,
        outcome,
        baseline_value=baseline_value,
        treatment_value=treatment_value,
        outcome_state=outcome_state,
        outcome_events=resolved_events,
        time_limit_seconds=time_limit_seconds,
        accepted_absolute_gap=accepted_absolute_gap,
    )
    if root_separator is not None:
        return root_separator

    quotient = _coarsen_terminal_event_outcome(
        world,
        outcome,
        resolved_events,
    )
    if quotient is not None:
        quotient_world, quotient_events = quotient
        return _solve_sparse_two_world_event_bounds(
            quotient_world,
            treatment,
            outcome,
            treatment_value=treatment_value,
            baseline_value=baseline_value,
            outcome_state=None,
            outcome_events=quotient_events,
            target_outer_bounds=target_outer_bounds,
            time_limit_seconds=time_limit_seconds,
            prefer_static_response_columns=True,
            accepted_absolute_gap=accepted_absolute_gap,
        )

    partial_terminal = _partially_attainable_terminal_bounds(
        world,
        treatment,
        outcome,
        baseline_value=baseline_value,
        treatment_value=treatment_value,
        outcome_state=outcome_state,
        outcome_events=resolved_events,
        target_outer_bounds=target_outer_bounds,
        time_limit_seconds=time_limit_seconds,
        on_demand_response_columns=not prefer_static_response_columns,
        accepted_absolute_gap=accepted_absolute_gap,
    )
    if partial_terminal is not None:
        return partial_terminal

    build_started = time.perf_counter()
    lower_model = _SparseResponseModel(
        world,
        treatment,
        outcome,
        baseline_value=baseline_value,
        treatment_value=treatment_value,
        outcome_state=outcome_state,
        sense="minimize",
        target_outer_bounds=target_outer_bounds,
        outcome_events=outcome_events,
        on_demand_response_columns=not prefer_static_response_columns,
    )
    if prefer_static_response_columns:
        lower_model._enable_static_presolve()
    lower_build_seconds = time.perf_counter() - build_started
    lower, lower_seconds = lower_model.optimize(
        time_limit_seconds=time_limit_seconds,
        accepted_absolute_gap=accepted_absolute_gap,
    )
    lower_error = lower_model.last_endpoint_error
    lower_certification = lower_model.last_certification
    restart_started = time.perf_counter()
    lower_model.restart_with_objective("maximize")
    restart_seconds = time.perf_counter() - restart_started
    upper, upper_seconds = lower_model.optimize(
        time_limit_seconds=time_limit_seconds,
        accepted_absolute_gap=accepted_absolute_gap,
    )
    return CounterfactualBoundsResult(
        lower=lower,
        upper=upper,
        build_seconds=lower_build_seconds + restart_seconds,
        solve_seconds=lower_seconds + upper_seconds,
        affected_nodes=len(lower_model.affected),
        pair_kernel_entries=len(lower_model.kernel_cache),
        generated_columns=lower_model.pricer.generated_columns,
        response_blocks=len(lower_model.pricing_blocks),
        dynamic_response_blocks=len(lower_model.dynamic_pricing_blocks),
        max_response_contexts=max(
            (len(block.contexts) for block in lower_model.pricing_blocks),
            default=0,
        ),
        auxiliary_variables=len(lower_model.auxiliary_values),
        certification=(
            "epsilon_sharp"
            if "epsilon_sharp"
            in {lower_certification, lower_model.last_certification}
            else "exact"
        ),
        endpoint_error=max(lower_error, lower_model.last_endpoint_error),
    )


def sparse_counterfactual_transition_bounds(
    world: WorldSpec,
    treatment: int,
    outcome: int,
    *,
    treatment_value: int,
    baseline_value: int,
    outcome_state: int,
    time_limit_seconds: float | None = None,
) -> CounterfactualBoundsResult:
    """Globally solve both sharp transition endpoints using sparse columns.

    A time limit is fail-closed: if either endpoint is not globally optimal,
    the solver raises instead of returning a partial interval.
    """

    if time_limit_seconds is not None and time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive")
    from .query_truth import interventional_frechet_transition_outer_bounds

    outer_lower, outer_upper = interventional_frechet_transition_outer_bounds(
        world,
        treatment,
        outcome,
        treatment_value=treatment_value,
        baseline_value=baseline_value,
        outcome_state=outcome_state,
    )
    return _solve_sparse_two_world_event_bounds(
        world,
        treatment,
        outcome,
        treatment_value=treatment_value,
        baseline_value=baseline_value,
        outcome_state=outcome_state,
        outcome_events=None,
        target_outer_bounds=(float(outer_lower), float(outer_upper)),
        time_limit_seconds=time_limit_seconds,
    )


def sparse_individual_counterfactual_probability_bounds(
    world: WorldSpec,
    treatment: int,
    outcome: int,
    *,
    factual_value: int,
    counterfactual_value: int,
    factual_outcome_state: int,
    target_outcome_state: int,
    time_limit_seconds: float | None = None,
    conditional_endpoint_tolerance: float = 0.0,
) -> CounterfactualBoundsResult:
    """Return exact or epsilon-sharp bounds for one individual prediction.

    The target is

    ``P(Y(counterfactual_value)=target_outcome_state |
         Y(factual_value)=factual_outcome_state)``.

    The conditioning event is an observed outcome under an assigned factual
    treatment.  Its probability is fixed by the CPT-World, so optimizing the
    conditional query is exactly equivalent to optimizing the two-world joint
    numerator and dividing both endpoints by that fixed positive mass.
    """

    if time_limit_seconds is not None and time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive")
    if (
        not np.isfinite(conditional_endpoint_tolerance)
        or conditional_endpoint_tolerance < 0.0
    ):
        raise ValueError(
            "conditional endpoint tolerance must be finite and nonnegative"
        )
    from .query_truth import interventional_probability

    factual_probability = float(
        interventional_probability(
            world,
            {treatment: factual_value},
            outcome,
            factual_outcome_state,
        )
    )
    if factual_probability <= 0.0:
        raise ValueError("the factual outcome has zero probability in the CPT-World")
    counterfactual_probability = float(
        interventional_probability(
            world,
            {treatment: counterfactual_value},
            outcome,
            target_outcome_state,
        )
    )
    joint_outer_bounds = (
        max(0.0, factual_probability + counterfactual_probability - 1.0),
        min(factual_probability, counterfactual_probability),
    )
    joint = _solve_sparse_two_world_event_bounds(
        world,
        treatment,
        outcome,
        treatment_value=counterfactual_value,
        baseline_value=factual_value,
        outcome_state=None,
        outcome_events=((factual_outcome_state,), (target_outcome_state,)),
        target_outer_bounds=joint_outer_bounds,
        time_limit_seconds=time_limit_seconds,
        accepted_absolute_gap=(
            conditional_endpoint_tolerance * factual_probability
        ),
    )
    return CounterfactualBoundsResult(
        lower=joint.lower / factual_probability,
        upper=joint.upper / factual_probability,
        build_seconds=joint.build_seconds,
        solve_seconds=joint.solve_seconds,
        affected_nodes=joint.affected_nodes,
        pair_kernel_entries=joint.pair_kernel_entries,
        generated_columns=joint.generated_columns,
        response_blocks=joint.response_blocks,
        dynamic_response_blocks=joint.dynamic_response_blocks,
        max_response_contexts=joint.max_response_contexts,
        auxiliary_variables=joint.auxiliary_variables,
        certification=joint.certification,
        endpoint_error=joint.endpoint_error / factual_probability,
        backend=joint.backend,
    )
