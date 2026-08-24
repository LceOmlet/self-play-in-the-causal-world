"""Exact sparse-response solver for Markovian counterfactual bounds.

Only context pairs that occur in the pruned twin world enter the model.
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


@dataclass(frozen=True, slots=True)
class CounterfactualBoundsResult:
    """Exact sharp endpoints plus non-semantic performance statistics."""

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
) -> _PairwiseMapOptimization:
    """Solve the categorical pairwise MAP used by response pricing exactly."""

    started = time.perf_counter()
    min_sum = _exact_pairwise_min_sum(
        unary,
        pairwise,
        domain_size=domain_size,
        constant=constant,
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
        if pricing.response in {response for response, _ in block.columns}:
            raise RuntimeError("pricing returned a strictly improving existing column")
        self._add_response_column(block, pricing.response)
        return True

    def _price(self, *, farkas: bool) -> dict[str, Any]:
        round_state = _PricingRoundState(farkas=farkas)
        self.rounds.append(round_state)
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
    outcome_state: int,
) -> tuple[int, ...]:
    """Return the states compatible with the terminal transition event."""

    if token == (outcome, 0):
        return tuple(state for state in range(world.domains[outcome]) if state != outcome_state)
    if token == (outcome, 1):
        return (outcome_state,)
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


def _eliminate_factor_tokens(
    world: WorldSpec,
    model: Model,
    factors: list[_SymbolicFactor],
    tokens: tuple[tuple[int, int], ...],
    local_factor: _SymbolicFactor,
    *,
    outcome: int,
    outcome_state: int,
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
        *(_token_states(world, item, outcome, outcome_state) for item in union_scope)
    ):
        terms: list[Any] = []
        upper = 0.0
        initial = 0.0
        for eliminated_assignment in product(
            *(_token_states(world, item, outcome, outcome_state) for item in tokens)
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
        outcome_state: int,
        sense: str,
        target_outer_bounds: tuple[float, float],
        probability_message_bounds: bool = True,
        on_demand_response_columns: bool = True,
    ) -> None:
        self.world = world
        self.treatment = treatment
        self.outcome = outcome
        self.baseline_value = baseline_value
        self.treatment_value = treatment_value
        self.outcome_state = outcome_state
        self.sense = sense
        self.target_outer_bounds = target_outer_bounds
        self.probability_message_bounds = probability_message_bounds
        self.on_demand_response_columns = on_demand_response_columns
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
        for node in self.affected:
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
        for node in self.affected:
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
        breakpoints = {0.0, 1.0}
        for row in rows:
            cumulative = 0.0
            for probability in row[:-1]:
                cumulative += probability
                breakpoints.add(min(1.0, max(0.0, cumulative)))
        ordered = sorted(breakpoints)
        support: list[tuple[float, tuple[int, ...]]] = []
        for left, right in pairwise(ordered):
            weight = right - left
            if weight <= _SCIP_NUMERICAL_TOLERANCE:
                continue
            midpoint = (left + right) / 2.0
            response: list[int] = []
            for row in rows:
                cumulative = 0.0
                selected = len(row) - 1
                for state, probability in enumerate(row):
                    cumulative += probability
                    if midpoint < cumulative:
                        selected = state
                        break
                response.append(selected)
            support.append((weight, tuple(response)))
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
        factors: list[_SymbolicFactor] = []
        relevant = set(self.shared) | set(self.affected)
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
    ) -> tuple[float, float]:
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
        certified_optimal = scip_status == "optimal" and pricing_closed
        if not certified_optimal:
            raise RuntimeError(
                "SCIP did not certify an optimal counterfactual bound: "
                f"status={scip_status}, pricing_closed={self.pricer.closed}, "
                f"pricing_timed_out={self.pricer.timed_out}"
            )
        return float(self.model.getPrimalbound()), elapsed


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
    """Globally solve both sharp endpoints using implicit sparse columns.

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
    target_outer_bounds = (float(outer_lower), float(outer_upper))

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
    )
    lower_build_seconds = time.perf_counter() - build_started
    lower, lower_seconds = lower_model.optimize(time_limit_seconds=time_limit_seconds)
    restart_started = time.perf_counter()
    lower_model.restart_with_objective("maximize")
    restart_seconds = time.perf_counter() - restart_started
    upper, upper_seconds = lower_model.optimize(time_limit_seconds=time_limit_seconds)
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
    )
