"""Exact query truth owners for generic finite discrete WorldSpec worlds.

Boundary: this module computes hidden-world truth for the query types whose
registry owner status is ``implemented``. It never renders prompts, parses
model answers, or scores them.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from math import fsum, isfinite, prod
from typing import Any

from .registry import OWNER_STATUS_IMPLEMENTED, query_truth_owner_status
from .world_space import Probability, WorldSpec, _backdoor_separated_structure

_PROBABILITY_TOLERANCE = 1e-12
INDIVIDUAL_COUNTERFACTUAL_ENDPOINT_TOLERANCE = 2e-3


@dataclass(frozen=True, slots=True)
class CounterfactualIntervalCertificate:
    """Certified interval plus its maximum endpoint approximation error."""

    lower: Probability
    upper: Probability
    certification: str
    endpoint_error: float


def _uses_exact_probabilities(world: WorldSpec) -> bool:
    return all(
        isinstance(probability, Fraction)
        for rows in world.cpt.values()
        for row in rows
        for probability in row
    )


def _zero(world: WorldSpec) -> Probability:
    return Fraction(0) if _uses_exact_probabilities(world) else 0.0


def _one(world: WorldSpec) -> Probability:
    return Fraction(1) if _uses_exact_probabilities(world) else 1.0


def _probability_sum(values: tuple[Probability, ...], *, exact: bool) -> Probability:
    if exact:
        return sum(values, start=Fraction(0))
    return fsum(float(value) for value in values)


def _require_normalized(values: tuple[Probability, ...], *, exact: bool) -> None:
    total = _probability_sum(values, exact=exact)
    if exact:
        if total != 1:
            raise RuntimeError("internal error: distribution is not normalized")
        return
    if not isfinite(float(total)) or abs(float(total) - 1.0) > _PROBABILITY_TOLERANCE:
        raise RuntimeError("internal error: distribution is not normalized")


def _node_index(world: WorldSpec, value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("node must be a variable name, not a bool")
    if isinstance(value, int):
        if 0 <= value < len(world.variables):
            return value
        raise ValueError(f"node index {value} out of range")
    name = str(value)
    if name in world.variables:
        return world.variables.index(name)
    raise ValueError(f"unknown world variable {value!r}")


def _parent_row_index(world: WorldSpec, node: int, values: tuple[int, ...]) -> int:
    row_index = 0
    for parent in world.parents.get(node, ()):
        row_index = row_index * world.domains[parent] + values[parent]
    return row_index


def _validate_interventions(
    world: WorldSpec,
    interventions: Mapping[int, int],
) -> dict[int, int]:
    if not isinstance(world, WorldSpec):
        raise TypeError("world must be a WorldSpec")
    fixed: dict[int, int] = {}
    for node, state in dict(interventions).items():
        if isinstance(node, bool) or not isinstance(node, int):
            raise TypeError("intervention node must be an integer")
        if isinstance(state, bool) or not isinstance(state, int):
            raise TypeError("intervention state must be an integer")
        if not 0 <= node < len(world.variables):
            raise ValueError("intervention node out of range")
        if not 0 <= state < world.domains[node]:
            raise ValueError("intervention state out of range")
        fixed[node] = state
    return fixed


@dataclass(frozen=True, slots=True)
class _Factor:
    variables: tuple[int, ...]
    values: tuple[Probability, ...]


def _factor_index(
    world: WorldSpec,
    variables: tuple[int, ...],
    assignment: tuple[int, ...],
) -> int:
    index = 0
    for variable, state in zip(variables, assignment, strict=True):
        index = index * world.domains[variable] + state
    return index


def _factor_value(
    world: WorldSpec,
    factor: _Factor,
    variables: tuple[int, ...],
    assignment: tuple[int, ...],
) -> Probability:
    positions = {variable: position for position, variable in enumerate(variables)}
    projected = tuple(assignment[positions[variable]] for variable in factor.variables)
    return factor.values[_factor_index(world, factor.variables, projected)]


def _multiply_factors(world: WorldSpec, factors: tuple[_Factor, ...]) -> _Factor:
    if not factors:
        return _Factor((), (_one(world),))
    exact = _uses_exact_probabilities(world)
    variables = tuple(sorted({variable for factor in factors for variable in factor.variables}))
    ranges = tuple(range(world.domains[variable]) for variable in variables)
    values: list[Probability] = []
    for assignment in product(*ranges):
        value: Probability = Fraction(1) if exact else 1.0
        for factor in factors:
            value *= _factor_value(world, factor, variables, assignment)
        values.append(value)
    return _Factor(variables, tuple(values))


def _sum_out(world: WorldSpec, factor: _Factor, variable: int) -> _Factor:
    remaining = tuple(item for item in factor.variables if item != variable)
    positions = {item: position for position, item in enumerate(remaining)}
    exact = _uses_exact_probabilities(world)
    values: list[Probability] = []
    for assignment in product(*(range(world.domains[item]) for item in remaining)):
        terms: list[Probability] = []
        for state in range(world.domains[variable]):
            original = tuple(
                state if item == variable else assignment[positions[item]]
                for item in factor.variables
            )
            terms.append(factor.values[_factor_index(world, factor.variables, original)])
        values.append(_probability_sum(tuple(terms), exact=exact))
    return _Factor(remaining, tuple(values))


def _elimination_key(
    world: WorldSpec,
    factors: tuple[_Factor, ...],
    variable: int,
) -> tuple[int, int, int, int]:
    neighbors = sorted(
        {
            item
            for factor in factors
            if variable in factor.variables
            for item in factor.variables
            if item != variable
        }
    )
    existing_edges = {
        tuple(sorted((left, right)))
        for factor in factors
        for left, right in combinations(factor.variables, 2)
    }
    fill = sum(
        tuple(sorted((left, right))) not in existing_edges
        for left, right in combinations(neighbors, 2)
    )
    factor_size = world.domains[variable] * prod(world.domains[item] for item in neighbors)
    return fill, factor_size, len(neighbors), variable


def _interventional_factors(
    world: WorldSpec,
    interventions: Mapping[int, int],
) -> tuple[_Factor, ...]:
    fixed = _validate_interventions(world, interventions)
    factors: list[_Factor] = []
    exact = _uses_exact_probabilities(world)
    for node in range(len(world.variables)):
        if node in fixed:
            factors.append(
                _Factor(
                    (node,),
                    tuple(
                        (Fraction(1) if exact else 1.0)
                        if state == fixed[node]
                        else (Fraction(0) if exact else 0.0)
                        for state in range(world.domains[node])
                    ),
                )
            )
            continue
        variables = (*world.parents.get(node, ()), node)
        values = tuple(probability for row in world.cpt[node] for probability in row)
        expected = prod(world.domains[variable] for variable in variables)
        if len(values) != expected:
            raise RuntimeError("internal error: CPT factor shape is inconsistent")
        factors.append(_Factor(variables, values))
    return tuple(factors)


def _variable_elimination_distribution(
    world: WorldSpec,
    interventions: Mapping[int, int],
    measure: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], Probability], ...]:
    factors = _interventional_factors(world, interventions)
    hidden = set(range(len(world.variables))) - set(measure)
    while hidden:
        variable = min(hidden, key=lambda item: _elimination_key(world, factors, item))
        involving = tuple(factor for factor in factors if variable in factor.variables)
        untouched = tuple(factor for factor in factors if variable not in factor.variables)
        factors = (*untouched, _sum_out(world, _multiply_factors(world, involving), variable))
        hidden.remove(variable)
    result = _multiply_factors(world, factors)
    assignments = tuple(product(*(range(world.domains[node]) for node in measure)))
    probabilities = tuple(
        _factor_value(world, result, measure, assignment) for assignment in assignments
    )
    _require_normalized(probabilities, exact=_uses_exact_probabilities(world))
    return tuple(zip(assignments, probabilities, strict=True))


def _topological_order(world: WorldSpec) -> tuple[int, ...]:
    indegree = [len(world.parents.get(node, ())) for node in range(len(world.variables))]
    children: list[list[int]] = [[] for _ in world.variables]
    for parent, child in world.edges:
        children[parent].append(child)
    ready = [node for node, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(order) != len(world.variables):
        raise ValueError("WorldSpec graph must be acyclic")
    return tuple(order)


def sample_worldspec_assignment(
    world: WorldSpec,
    interventions: Mapping[int, int],
    uniforms: tuple[Probability, ...],
) -> tuple[int, ...]:
    """Sample one assignment ancestrally from the hard-do law."""

    fixed = _validate_interventions(world, interventions)
    if len(uniforms) != len(world.variables):
        raise ValueError("uniforms must contain one draw per WorldSpec node")
    if any(
        isinstance(draw, bool)
        or not isinstance(draw, (Fraction, float, int))
        or not isfinite(float(draw))
        or not 0 <= draw < 1
        for draw in uniforms
    ):
        raise ValueError("uniform draws must be finite numbers in [0, 1)")
    exact = _uses_exact_probabilities(world)
    values = [0] * len(world.variables)
    for node in _topological_order(world):
        if node in fixed:
            values[node] = fixed[node]
            continue
        row_index = _parent_row_index(world, node, tuple(values))
        row = world.cpt[node][row_index]
        cumulative: Probability = Fraction(0) if exact else 0.0
        draw: Probability = Fraction(uniforms[node]) if exact else float(uniforms[node])
        selected = len(row) - 1
        for state, probability in enumerate(row):
            cumulative += probability
            if draw < cumulative:
                selected = state
                break
        values[node] = selected
    return tuple(values)


def worldspec_interventional_distribution(
    world: WorldSpec,
    interventions: Mapping[int, int],
) -> tuple[tuple[tuple[int, ...], Probability], ...]:
    """Return the full-joint law under a finite hard-do assignment.

    This is the executable probability-law owner for generic ``WorldSpec``
    worlds.  Query truth and the interactive sampler both consume this same
    canonical assignment order instead of reimplementing CPT traversal.
    """

    fixed = _validate_interventions(world, interventions)

    exact = _uses_exact_probabilities(world)
    distribution: list[tuple[tuple[int, ...], Probability]] = []
    for values in product(*(range(size) for size in world.domains)):
        if any(values[node] != state for node, state in fixed.items()):
            probability: Probability = Fraction(0) if exact else 0.0
        else:
            probability = Fraction(1) if exact else 1.0
            for node in range(len(world.variables)):
                if node in fixed:
                    continue
                row_index = _parent_row_index(world, node, values)
                probability *= world.cpt[node][row_index][values[node]]
        distribution.append((values, probability))
    _require_normalized(
        tuple(probability for _, probability in distribution),
        exact=exact,
    )
    return tuple(distribution)


def worldspec_projected_interventional_distribution(
    world: WorldSpec,
    interventions: Mapping[int, int],
    measure: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], Probability], ...]:
    """Return a selected-measure law under a finite hard intervention.

    The assignment order is the Cartesian order of ``measure``. Exact variable
    elimination uses the same CPT factors and hard-do replacement semantics as
    the full-joint reference owner above.
    """

    if not measure:
        raise ValueError("measure must not be empty")
    if len(set(measure)) != len(measure):
        raise ValueError("measure variables must not contain duplicates")
    if any(
        isinstance(node, bool) or not isinstance(node, int) or not 0 <= node < len(world.variables)
        for node in measure
    ):
        raise ValueError("measure node out of range")
    return _variable_elimination_distribution(world, interventions, measure)


def interventional_joint_probability(
    world: WorldSpec,
    interventions: Mapping[int, int],
    targets: Mapping[int, int],
) -> Probability:
    """Return the probability of ``targets`` under hard-do ``interventions``."""

    target_values: dict[int, int] = {}
    for node, state in dict(targets).items():
        if isinstance(node, bool) or not isinstance(node, int):
            raise TypeError("target node must be an integer")
        if isinstance(state, bool) or not isinstance(state, int):
            raise TypeError("target state must be an integer")
        if not 0 <= node < len(world.variables):
            raise ValueError("target node out of range")
        if not 0 <= state < world.domains[node]:
            raise ValueError("target state out of range")
        target_values[node] = state

    if not target_values:
        return _one(world)
    measure = tuple(sorted(target_values))
    assignment = tuple(target_values[node] for node in measure)
    law = worldspec_projected_interventional_distribution(world, interventions, measure)
    return dict(law)[assignment]


def interventional_probability(
    world: WorldSpec,
    interventions: Mapping[int, int],
    outcome: int,
    outcome_state: int,
) -> Probability:
    """Return P(outcome=outcome_state) under a hard-do assignment."""

    return interventional_joint_probability(
        world,
        interventions,
        {outcome: outcome_state},
    )


def ate_effect(
    world: WorldSpec,
    treatment: object,
    outcome: object,
    *,
    treatment_value: int = 1,
    baseline_value: int = 0,
    outcome_state: int = 1,
) -> Probability:
    """Return E[outcome | do(treatment=treatment_value)] minus baseline.

    The default states implement the binary ATE rendered as state_1 vs state_0.
    For multi-state worlds the caller must pass explicit state indices.
    """

    treatment_node = _node_index(world, treatment)
    outcome_node = _node_index(world, outcome)
    if treatment_node == outcome_node:
        raise ValueError("treatment and outcome must be different variables")
    do_one = interventional_probability(
        world,
        {treatment_node: treatment_value},
        outcome_node,
        outcome_state,
    )
    do_zero = interventional_probability(
        world,
        {treatment_node: baseline_value},
        outcome_node,
        outcome_state,
    )
    return do_one - do_zero


def counterfactual_transition_bounds(
    world: WorldSpec,
    treatment: object,
    outcome: object,
    *,
    treatment_value: int = 1,
    baseline_value: int = 0,
    outcome_state: int = 1,
) -> tuple[Probability, Probability]:
    """Return sharp Markovian bounds for a target-state transition.

    The target event is

    ``outcome(treatment_value) = outcome_state`` and
    ``outcome(baseline_value) != outcome_state``.

    Bounds range over all finite deterministic response-function completions
    whose node mechanisms are mutually independent and whose response
    marginals equal the WorldSpec CPT.  No single hidden SCM is selected.

    A direct-only causal family has an exact row-wise transport formula.  Every
    other legal WorldSpec uses the certified sparse-response solver.  The
    explicit response-vertex enumeration remains separately exposed only as a
    small-instance reference oracle for cross-checks.
    """

    treatment_node = _node_index(world, treatment)
    outcome_node = _node_index(world, outcome)
    if treatment_node == outcome_node:
        raise ValueError("treatment and outcome must be different variables")
    if treatment_value == baseline_value:
        raise ValueError("treatment and baseline values must differ")
    if not 0 <= treatment_value < world.domains[treatment_node]:
        raise ValueError("treatment value out of range")
    if not 0 <= baseline_value < world.domains[treatment_node]:
        raise ValueError("baseline value out of range")
    if not 0 <= outcome_state < world.domains[outcome_node]:
        raise ValueError("outcome state out of range")

    descendants = _descendants(world, treatment_node)
    if outcome_node not in descendants:
        zero = _zero(world)
        return zero, zero

    other_outcome_parents = tuple(
        parent for parent in world.parents[outcome_node] if parent != treatment_node
    )
    direct_only = (treatment_node, outcome_node) in world.edges and all(
        parent not in descendants for parent in other_outcome_parents
    )
    if direct_only:
        return _direct_only_counterfactual_transition_bounds(
            world,
            treatment_node,
            outcome_node,
            treatment_value=treatment_value,
            baseline_value=baseline_value,
            outcome_state=outcome_state,
        )
    from .counterfactual_solver import sparse_counterfactual_transition_bounds

    result = sparse_counterfactual_transition_bounds(
        world,
        treatment_node,
        outcome_node,
        treatment_value=treatment_value,
        baseline_value=baseline_value,
        outcome_state=outcome_state,
    )
    return result.lower, result.upper


def _individual_counterfactual_probability_certificate(
    world: WorldSpec,
    treatment: object,
    outcome: object,
    *,
    factual_value: int,
    counterfactual_value: int,
    factual_outcome_state: int,
    target_outcome_state: int,
    time_limit_seconds: float | None = None,
    endpoint_tolerance: float = 0.0,
) -> CounterfactualIntervalCertificate:
    """Return exact or epsilon-sharp bounds for an individual query.

    The query is

    ``P(Y(counterfactual_value)=target_outcome_state |
         Y(factual_value)=factual_outcome_state)``.

    The factual treatment is assigned, so its observed outcome is a clean
    potential-outcome event.  The interval ranges over the same full set of
    Markovian finite response mechanisms used by the population transition
    owner; no hidden SCM or response prior is selected.
    """

    treatment_node = _node_index(world, treatment)
    outcome_node = _node_index(world, outcome)
    if treatment_node == outcome_node:
        raise ValueError("treatment and outcome must be different variables")
    if factual_value == counterfactual_value:
        raise ValueError("factual and counterfactual treatment values must differ")
    if not 0 <= factual_value < world.domains[treatment_node]:
        raise ValueError("factual treatment value out of range")
    if not 0 <= counterfactual_value < world.domains[treatment_node]:
        raise ValueError("counterfactual treatment value out of range")
    if not 0 <= factual_outcome_state < world.domains[outcome_node]:
        raise ValueError("factual outcome state out of range")
    if not 0 <= target_outcome_state < world.domains[outcome_node]:
        raise ValueError("target outcome state out of range")
    if time_limit_seconds is not None and time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive")
    if not isfinite(endpoint_tolerance) or endpoint_tolerance < 0.0:
        raise ValueError("endpoint_tolerance must be finite and nonnegative")

    factual_probability = interventional_probability(
        world,
        {treatment_node: factual_value},
        outcome_node,
        factual_outcome_state,
    )
    if factual_probability <= 0:
        raise ValueError("the factual outcome has zero probability in the CPT-World")

    descendants = _descendants(world, treatment_node)
    if outcome_node not in descendants:
        one = _one(world)
        zero = _zero(world)
        lower, upper = (
            (one, one)
            if factual_outcome_state == target_outcome_state
            else (zero, zero)
        )
        return CounterfactualIntervalCertificate(lower, upper, "exact", 0.0)

    other_outcome_parents = tuple(
        parent for parent in world.parents[outcome_node] if parent != treatment_node
    )
    direct_only = (treatment_node, outcome_node) in world.edges and all(
        parent not in descendants for parent in other_outcome_parents
    )
    if direct_only:
        joint_lower, joint_upper = _direct_only_counterfactual_joint_bounds(
            world,
            treatment_node,
            outcome_node,
            treatment_value=counterfactual_value,
            baseline_value=factual_value,
            baseline_outcome_states=(factual_outcome_state,),
            treatment_outcome_states=(target_outcome_state,),
        )
        return CounterfactualIntervalCertificate(
            joint_lower / factual_probability,
            joint_upper / factual_probability,
            "exact",
            0.0,
        )

    from .counterfactual_solver import sparse_individual_counterfactual_probability_bounds

    result = sparse_individual_counterfactual_probability_bounds(
        world,
        treatment_node,
        outcome_node,
        factual_value=factual_value,
        counterfactual_value=counterfactual_value,
        factual_outcome_state=factual_outcome_state,
        target_outcome_state=target_outcome_state,
        time_limit_seconds=time_limit_seconds,
        conditional_endpoint_tolerance=endpoint_tolerance,
    )
    return CounterfactualIntervalCertificate(
        result.lower,
        result.upper,
        result.certification,
        result.endpoint_error,
    )


def individual_counterfactual_probability_bounds(
    world: WorldSpec,
    treatment: object,
    outcome: object,
    *,
    factual_value: int,
    counterfactual_value: int,
    factual_outcome_state: int,
    target_outcome_state: int,
    time_limit_seconds: float | None = None,
) -> tuple[Probability, Probability]:
    """Return sharp bounds, preserving the exact public calculation contract."""

    certificate = _individual_counterfactual_probability_certificate(
        world,
        treatment,
        outcome,
        factual_value=factual_value,
        counterfactual_value=counterfactual_value,
        factual_outcome_state=factual_outcome_state,
        target_outcome_state=target_outcome_state,
        time_limit_seconds=time_limit_seconds,
        endpoint_tolerance=0.0,
    )
    return certificate.lower, certificate.upper


def validate_individual_counterfactual_probability(
    world: WorldSpec,
    treatment: object,
    outcome: object,
    prediction: float,
    *,
    factual_value: int,
    counterfactual_value: int,
    factual_outcome_state: int,
    target_outcome_state: int,
    time_limit_seconds: float | None = None,
    numerical_tolerance: float = 1e-9,
    endpoint_tolerance: float = INDIVIDUAL_COUNTERFACTUAL_ENDPOINT_TOLERANCE,
) -> Mapping[str, Any]:
    """Validate one scalar against an exact or epsilon-sharp certificate.

    Exact completion returns the sharp interval.  Epsilon completion returns a
    safe outer interval whose two endpoints are each within ``endpoint_tolerance``
    of the sharp endpoints.  A larger unresolved gap still fails closed.
    """

    if not isfinite(prediction) or not 0.0 <= prediction <= 1.0:
        raise ValueError("prediction must be a finite probability in [0, 1]")
    if not isfinite(numerical_tolerance) or numerical_tolerance < 0.0:
        raise ValueError("numerical_tolerance must be finite and nonnegative")
    if not isfinite(endpoint_tolerance) or endpoint_tolerance < 0.0:
        raise ValueError("endpoint_tolerance must be finite and nonnegative")
    arguments = {
        "factual_value": factual_value,
        "counterfactual_value": counterfactual_value,
        "factual_outcome_state": factual_outcome_state,
        "target_outcome_state": target_outcome_state,
    }
    outer_lower, outer_upper = individual_counterfactual_frechet_outer_bounds(
        world,
        treatment,
        outcome,
        **arguments,
    )
    try:
        certificate = _individual_counterfactual_probability_certificate(
            world,
            treatment,
            outcome,
            time_limit_seconds=time_limit_seconds,
            endpoint_tolerance=endpoint_tolerance,
            **arguments,
        )
    except RuntimeError as error:
        outside = (
            prediction < float(outer_lower) - numerical_tolerance
            or prediction > float(outer_upper) + numerical_tolerance
        )
        return {
            "compatible": False if outside else None,
            "status": "rejected_by_frechet_outer" if outside else "unresolved_timeout",
            "interval_source": "frechet_outer",
            "lower": outer_lower,
            "upper": outer_upper,
            "numerical_tolerance": numerical_tolerance,
            "solver_error": str(error),
        }
    lower = certificate.lower
    upper = certificate.upper
    distance = max(float(lower) - prediction, 0.0, prediction - float(upper))
    return {
        "compatible": distance <= numerical_tolerance,
        "status": certificate.certification,
        "interval_source": (
            "exact_markovian"
            if certificate.certification == "exact"
            else "epsilon_sharp_markovian_outer"
        ),
        "lower": lower,
        "upper": upper,
        "endpoint_error": certificate.endpoint_error,
        "endpoint_tolerance": endpoint_tolerance,
        "distance_to_interval": distance,
        "numerical_tolerance": numerical_tolerance,
    }


def interventional_frechet_transition_outer_bounds(
    world: WorldSpec,
    treatment: object,
    outcome: object,
    *,
    treatment_value: int = 1,
    baseline_value: int = 0,
    outcome_state: int = 1,
) -> tuple[Probability, Probability]:
    """Return the marginal-only Frechet outer bounds for diagnostics.

    These bounds deliberately ignore the cross-node mechanism-independence
    restrictions of a Markovian CPT-World.  They are therefore a useful outer
    check, not the generic task truth owner.
    """

    treatment_node = _node_index(world, treatment)
    outcome_node = _node_index(world, outcome)
    treated_probability = interventional_probability(
        world,
        {treatment_node: treatment_value},
        outcome_node,
        outcome_state,
    )
    baseline_probability = interventional_probability(
        world,
        {treatment_node: baseline_value},
        outcome_node,
        outcome_state,
    )
    lower = max(_zero(world), treated_probability - baseline_probability)
    upper = min(treated_probability, _one(world) - baseline_probability)
    return lower, upper


def individual_counterfactual_frechet_outer_bounds(
    world: WorldSpec,
    treatment: object,
    outcome: object,
    *,
    factual_value: int,
    counterfactual_value: int,
    factual_outcome_state: int,
    target_outcome_state: int,
) -> tuple[Probability, Probability]:
    """Return the endpoint-marginal outer interval for one individual query."""

    treatment_node = _node_index(world, treatment)
    outcome_node = _node_index(world, outcome)
    if treatment_node == outcome_node:
        raise ValueError("treatment and outcome must be different variables")
    if factual_value == counterfactual_value:
        raise ValueError("factual and counterfactual treatment values must differ")
    if not 0 <= factual_value < world.domains[treatment_node]:
        raise ValueError("factual treatment value out of range")
    if not 0 <= counterfactual_value < world.domains[treatment_node]:
        raise ValueError("counterfactual treatment value out of range")
    if not 0 <= factual_outcome_state < world.domains[outcome_node]:
        raise ValueError("factual outcome state out of range")
    if not 0 <= target_outcome_state < world.domains[outcome_node]:
        raise ValueError("target outcome state out of range")
    factual_probability = interventional_probability(
        world,
        {treatment_node: factual_value},
        outcome_node,
        factual_outcome_state,
    )
    if factual_probability <= 0:
        raise ValueError("the factual outcome has zero probability in the CPT-World")
    counterfactual_probability = interventional_probability(
        world,
        {treatment_node: counterfactual_value},
        outcome_node,
        target_outcome_state,
    )
    joint_lower = max(
        _zero(world),
        factual_probability + counterfactual_probability - _one(world),
    )
    joint_upper = min(factual_probability, counterfactual_probability)
    return joint_lower / factual_probability, joint_upper / factual_probability


def _direct_only_counterfactual_transition_bounds(
    world: WorldSpec,
    treatment_node: int,
    outcome_node: int,
    *,
    treatment_value: int,
    baseline_value: int,
    outcome_state: int,
) -> tuple[Probability, Probability]:
    """Exact row-wise bounds when treatment reaches outcome only directly."""

    return _direct_only_counterfactual_joint_bounds(
        world,
        treatment_node,
        outcome_node,
        treatment_value=treatment_value,
        baseline_value=baseline_value,
        baseline_outcome_states=tuple(
            state for state in range(world.domains[outcome_node]) if state != outcome_state
        ),
        treatment_outcome_states=(outcome_state,),
    )


def _direct_only_counterfactual_joint_bounds(
    world: WorldSpec,
    treatment_node: int,
    outcome_node: int,
    *,
    treatment_value: int,
    baseline_value: int,
    baseline_outcome_states: tuple[int, ...],
    treatment_outcome_states: tuple[int, ...],
) -> tuple[Probability, Probability]:
    """Exact row-wise joint bounds for two endpoint outcome events."""

    other_parents = tuple(
        parent for parent in world.parents[outcome_node] if parent != treatment_node
    )
    if other_parents:
        parent_law = worldspec_projected_interventional_distribution(
            world,
            {treatment_node: baseline_value},
            other_parents,
        )
    else:
        parent_law = (((), _one(world)),)
    exact = _uses_exact_probabilities(world)
    lower_terms: list[Probability] = []
    upper_terms: list[Probability] = []
    for assignment, parent_probability in parent_law:
        shared_values = dict(zip(other_parents, assignment, strict=True))

        def outcome_probability(
            treatment_state: int,
            outcome_states: tuple[int, ...],
            current_shared: Mapping[int, int] = shared_values,
        ) -> Probability:
            row_index = 0
            for parent in world.parents[outcome_node]:
                state = treatment_state if parent == treatment_node else current_shared[parent]
                row_index = row_index * world.domains[parent] + state
            row = world.cpt[outcome_node][row_index]
            return _probability_sum(
                tuple(row[state] for state in outcome_states),
                exact=exact,
            )

        treated = outcome_probability(treatment_value, treatment_outcome_states)
        baseline = outcome_probability(baseline_value, baseline_outcome_states)
        lower_terms.append(parent_probability * max(_zero(world), treated + baseline - _one(world)))
        upper_terms.append(parent_probability * min(treated, baseline))
    return (
        _probability_sum(tuple(lower_terms), exact=exact),
        _probability_sum(tuple(upper_terms), exact=exact),
    )


def _ancestors(world: WorldSpec, node: int) -> frozenset[int]:
    seen: set[int] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        for parent in world.parents[current]:
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return frozenset(seen)


def _solve_square_system(
    matrix: tuple[tuple[Probability, ...], ...],
    target: tuple[Probability, ...],
    *,
    exact: bool,
) -> tuple[Probability, ...] | None:
    """Solve one candidate response-polytope basis by elimination."""

    size = len(target)
    augmented = [list(row) + [target[index]] for index, row in enumerate(matrix)]
    zero: Probability = Fraction(0) if exact else 0.0
    for column in range(size):
        candidates = range(column, size)
        if exact:
            pivot = next((row for row in candidates if augmented[row][column] != 0), None)
        else:
            pivot = max(candidates, key=lambda row: abs(float(augmented[row][column])))
            if abs(float(augmented[pivot][column])) <= _PROBABILITY_TOLERANCE:
                return None
        if pivot is None:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            multiplier = augmented[row][column]
            if multiplier == zero:
                continue
            augmented[row] = [
                value - multiplier * pivot_entry
                for value, pivot_entry in zip(augmented[row], augmented[column], strict=True)
            ]
    solution = tuple(row[-1] for row in augmented)
    if exact:
        return solution if all(value >= 0 for value in solution) else None
    if any(float(value) < -1e-10 or not isfinite(float(value)) for value in solution):
        return None
    cleaned = tuple(max(0.0, float(value)) for value in solution)
    for row, expected in zip(matrix, target, strict=True):
        actual = fsum(
            float(coefficient) * value for coefficient, value in zip(row, cleaned, strict=True)
        )
        if abs(actual - float(expected)) > 1e-9 + 1e-9 * abs(float(expected)):
            return None
    return cleaned


def _response_coupling_vertices(
    marginals: tuple[tuple[Probability, ...], ...],
) -> tuple[tuple[tuple[tuple[int, ...], Probability], ...], ...]:
    """Enumerate vertices of one finite multi-context response coupling."""

    if not marginals:
        raise ValueError("response coupling needs at least one parent context")
    domain_size = len(marginals[0])
    if domain_size < 1 or any(len(row) != domain_size for row in marginals):
        raise ValueError("response marginals have inconsistent domains")
    exact = all(isinstance(value, Fraction) for row in marginals for value in row)
    response_types = tuple(product(range(domain_size), repeat=len(marginals)))
    constraint_count = 1 + len(marginals) * (domain_size - 1)
    rows: list[tuple[Probability, ...]] = [tuple(_one_value(exact) for _ in response_types)]
    target: list[Probability] = [_one_value(exact)]
    for context, marginal in enumerate(marginals):
        for state in range(domain_size - 1):
            rows.append(
                tuple(
                    _one_value(exact) if response[context] == state else _zero_value(exact)
                    for response in response_types
                )
            )
            target.append(marginal[state])

    vertices: list[tuple[tuple[tuple[int, ...], Probability], ...]] = []
    seen_supports: set[tuple[int, ...]] = set()
    for basis in combinations(range(len(response_types)), constraint_count):
        matrix = tuple(tuple(row[column] for column in basis) for row in rows)
        solution = _solve_square_system(matrix, tuple(target), exact=exact)
        if solution is None:
            continue
        support = tuple(
            column
            for column, value in zip(basis, solution, strict=True)
            if value != 0 and (exact or float(value) > _PROBABILITY_TOLERANCE)
        )
        if support in seen_supports:
            continue
        seen_supports.add(support)
        vertex = tuple(
            (response_types[column], value)
            for column, value in zip(basis, solution, strict=True)
            if value != 0 and (exact or float(value) > _PROBABILITY_TOLERANCE)
        )
        vertices.append(vertex)
    if not vertices:
        raise RuntimeError("response-coupling polytope has no enumerated vertex")
    return tuple(vertices)


def _zero_value(exact: bool) -> Probability:
    return Fraction(0) if exact else 0.0


def _one_value(exact: bool) -> Probability:
    return Fraction(1) if exact else 1.0


def _active_parent_contexts(
    world: WorldSpec,
    node: int,
    treatment_node: int,
    baseline_value: int,
    treatment_value: int,
) -> tuple[tuple[int, ...], ...]:
    parents = world.parents[node]
    contexts = product(*(range(world.domains[parent]) for parent in parents))
    if treatment_node not in parents:
        return tuple(contexts)
    treatment_position = parents.index(treatment_node)
    return tuple(
        context
        for context in contexts
        if context[treatment_position] in {baseline_value, treatment_value}
    )


def _response_pair_kernel(
    vertex: tuple[tuple[tuple[int, ...], Probability], ...],
    contexts: tuple[tuple[int, ...], ...],
    baseline_context: tuple[int, ...],
    treated_context: tuple[int, ...],
    domain_size: int,
    *,
    exact: bool,
) -> tuple[tuple[tuple[int, int], Probability], ...]:
    baseline_index = contexts.index(baseline_context)
    treated_index = contexts.index(treated_context)
    masses = {
        (left, right): _zero_value(exact)
        for left in range(domain_size)
        for right in range(domain_size)
    }
    for response, probability in vertex:
        pair = (response[baseline_index], response[treated_index])
        masses[pair] += probability
    return tuple((pair, probability) for pair, probability in masses.items() if probability != 0)


def _counterfactual_probability_for_vertices(
    world: WorldSpec,
    treatment_node: int,
    outcome_node: int,
    affected: tuple[int, ...],
    shared: tuple[int, ...],
    contexts_by_node: Mapping[int, tuple[tuple[int, ...], ...]],
    vertices_by_node: Mapping[int, tuple[tuple[tuple[int, ...], Probability], ...]],
    *,
    treatment_value: int,
    baseline_value: int,
    baseline_outcome_states: tuple[int, ...],
    treatment_outcome_states: tuple[int, ...],
) -> Probability:
    exact = _uses_exact_probabilities(world)
    if shared:
        shared_law = worldspec_projected_interventional_distribution(
            world,
            {treatment_node: baseline_value},
            shared,
        )
    else:
        shared_law = (((), _one(world)),)
    total = _zero(world)
    for shared_assignment, shared_probability in shared_law:
        initial_left = {treatment_node: baseline_value}
        initial_right = {treatment_node: treatment_value}
        for node, state in zip(shared, shared_assignment, strict=True):
            initial_left[node] = state
            initial_right[node] = state
        frontier: list[tuple[dict[int, int], dict[int, int], Probability]] = [
            (initial_left, initial_right, shared_probability)
        ]
        for node in affected:
            expanded: list[tuple[dict[int, int], dict[int, int], Probability]] = []
            parents = world.parents[node]
            contexts = contexts_by_node[node]
            vertex = vertices_by_node[node]
            for left_values, right_values, mass in frontier:
                left_context = tuple(left_values[parent] for parent in parents)
                right_context = tuple(right_values[parent] for parent in parents)
                kernel = _response_pair_kernel(
                    vertex,
                    contexts,
                    left_context,
                    right_context,
                    world.domains[node],
                    exact=exact,
                )
                for (left_state, right_state), probability in kernel:
                    next_left = {**left_values, node: left_state}
                    next_right = {**right_values, node: right_state}
                    expanded.append((next_left, next_right, mass * probability))
            frontier = expanded
        total += _probability_sum(
            tuple(
                mass
                for left_values, right_values, mass in frontier
                if right_values[outcome_node] in treatment_outcome_states
                and left_values[outcome_node] in baseline_outcome_states
            ),
            exact=exact,
        )
    return total


def _reference_counterfactual_joint_bounds(
    world: WorldSpec,
    treatment_node: int,
    outcome_node: int,
    *,
    treatment_value: int,
    baseline_value: int,
    baseline_outcome_states: tuple[int, ...],
    treatment_outcome_states: tuple[int, ...],
) -> tuple[Probability, Probability]:
    """Enumerate the exact two-world joint event on small instances.

    The runtime is exponential in the local response-coupling dimensions and
    in the number of affected nodes.  This is the independent reference path
    used to cross-check the sparse production owner.
    """

    descendants = _descendants(world, treatment_node)
    if outcome_node not in descendants:
        raise ValueError("the reference joint owner requires a causal path")
    ancestors = _ancestors(world, outcome_node) | {outcome_node}
    affected_set = descendants & ancestors
    affected = tuple(node for node in _topological_order(world) if node in affected_set)
    shared = tuple(
        node
        for node in _topological_order(world)
        if node in ancestors and node not in affected_set and node != treatment_node
    )

    contexts_by_node: dict[int, tuple[tuple[int, ...], ...]] = {}
    vertex_sets: list[tuple[tuple[tuple[tuple[int, ...], Probability], ...], ...]] = []
    for node in affected:
        contexts = _active_parent_contexts(
            world,
            node,
            treatment_node,
            baseline_value,
            treatment_value,
        )
        contexts_by_node[node] = contexts
        marginals = tuple(
            world.cpt[node][
                sum(
                    state
                    * prod(world.domains[parent] for parent in world.parents[node][index + 1 :])
                    for index, (parent, state) in enumerate(
                        zip(world.parents[node], context, strict=True)
                    )
                )
            ]
            for context in contexts
        )
        vertex_sets.append(_response_coupling_vertices(marginals))

    lower: Probability | None = None
    upper: Probability | None = None
    for selected_vertices in product(*vertex_sets):
        probability = _counterfactual_probability_for_vertices(
            world,
            treatment_node,
            outcome_node,
            affected,
            shared,
            contexts_by_node,
            dict(zip(affected, selected_vertices, strict=True)),
            treatment_value=treatment_value,
            baseline_value=baseline_value,
            baseline_outcome_states=baseline_outcome_states,
            treatment_outcome_states=treatment_outcome_states,
        )
        lower = probability if lower is None else min(lower, probability)
        upper = probability if upper is None else max(upper, probability)
    if lower is None or upper is None:
        raise RuntimeError("counterfactual response enumeration produced no completion")
    return lower, upper


def reference_counterfactual_transition_bounds(
    world: WorldSpec,
    treatment: object,
    outcome: object,
    *,
    treatment_value: int = 1,
    baseline_value: int = 0,
    outcome_state: int = 1,
) -> tuple[Probability, Probability]:
    """Exact finite response-function oracle for transition cross-checks."""

    treatment_node = _node_index(world, treatment)
    outcome_node = _node_index(world, outcome)
    if outcome_node not in _descendants(world, treatment_node):
        zero = _zero(world)
        return zero, zero
    return _reference_counterfactual_joint_bounds(
        world,
        treatment_node,
        outcome_node,
        treatment_value=treatment_value,
        baseline_value=baseline_value,
        baseline_outcome_states=tuple(
            state for state in range(world.domains[outcome_node]) if state != outcome_state
        ),
        treatment_outcome_states=(outcome_state,),
    )


def reference_individual_counterfactual_probability_bounds(
    world: WorldSpec,
    treatment: object,
    outcome: object,
    *,
    factual_value: int,
    counterfactual_value: int,
    factual_outcome_state: int,
    target_outcome_state: int,
) -> tuple[Probability, Probability]:
    """Explicit small-instance oracle for an individual probability interval."""

    treatment_node = _node_index(world, treatment)
    outcome_node = _node_index(world, outcome)
    factual_probability = interventional_probability(
        world,
        {treatment_node: factual_value},
        outcome_node,
        factual_outcome_state,
    )
    if factual_probability <= 0:
        raise ValueError("the factual outcome has zero probability in the CPT-World")
    if outcome_node not in _descendants(world, treatment_node):
        one = _one(world)
        zero = _zero(world)
        return (one, one) if factual_outcome_state == target_outcome_state else (zero, zero)
    joint_lower, joint_upper = _reference_counterfactual_joint_bounds(
        world,
        treatment_node,
        outcome_node,
        treatment_value=counterfactual_value,
        baseline_value=factual_value,
        baseline_outcome_states=(factual_outcome_state,),
        treatment_outcome_states=(target_outcome_state,),
    )
    return joint_lower / factual_probability, joint_upper / factual_probability


def best_intervention_states(
    world: WorldSpec,
    outcome: object,
    objective: str,
    decision_target: object,
    *,
    outcome_state: int = 1,
) -> tuple[tuple[int, ...], Probability]:
    """Return every zero-regret state and its exact outcome probability.

    ``decision_target`` is deliberately independent of the variables that may
    be manipulated during the experiment.
    """

    outcome_node = _node_index(world, outcome)
    target_node = _node_index(world, decision_target)
    if target_node == outcome_node:
        raise ValueError("decision target must differ from the outcome")
    if objective not in {"minimize", "maximize"}:
        raise ValueError("objective must be minimize or maximize")
    probabilities = tuple(
        interventional_probability(
            world,
            {target_node: state},
            outcome_node,
            outcome_state,
        )
        for state in range(world.domains[target_node])
    )
    if not probabilities:
        raise ValueError("decision target has no candidate state")
    best_probability = min(probabilities) if objective == "minimize" else max(probabilities)
    return (
        tuple(
            state
            for state, probability in enumerate(probabilities)
            if probability == best_probability
        ),
        best_probability,
    )


def best_intervention_truth(
    world: WorldSpec,
    outcome: object,
    objective: str,
    decision_target: object,
    *,
    outcome_state: int = 1,
) -> tuple[str, int, Probability]:
    """Return the canonical deployment intervention.

    Every state returned by :func:`best_intervention_states` has zero regret.
    This compatibility entry point selects the first such state in domain
    order for deterministic truth serialization.
    """

    target_node = _node_index(world, decision_target)
    states, best_probability = best_intervention_states(
        world,
        outcome,
        objective,
        target_node,
        outcome_state=outcome_state,
    )
    best_state = states[0]
    best_name = world.variables[target_node]
    return best_name, best_state, best_probability


def _descendants(world: WorldSpec, node: int) -> frozenset[int]:
    seen: set[int] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        for parent, child in world.edges:
            if parent == current and child not in seen:
                seen.add(child)
                stack.append(child)
    return frozenset(seen)


def _backdoor_separated(
    world: WorldSpec,
    treatment: int,
    outcome: int,
    condition: frozenset[int],
) -> bool:
    """Test the back-door criterion without enumerating simple paths.

    Remove arrows leaving the treatment, restrict the resulting DAG to the
    ancestors of the query and conditioning nodes, moralize that ancestral
    graph, and test undirected separation after deleting the conditioning
    nodes. This is equivalent to d-separation in the back-door graph while
    remaining polynomial in the graph size for each candidate set.
    """

    return _backdoor_separated_structure(
        len(world.variables),
        world.edges,
        treatment,
        outcome,
        condition,
    )


def backdoor_adjustment_sets(
    world: WorldSpec, treatment: object, outcome: object
) -> tuple[tuple[str, ...], ...]:
    """Return every inclusion-minimal valid back-door adjustment set.

    A set is valid when it contains no descendant of treatment and blocks every
    back-door path from treatment to outcome. This is the standard back-door
    criterion for the total effect in a DAG.
    """

    treatment_node = _node_index(world, treatment)
    outcome_node = _node_index(world, outcome)
    if treatment_node == outcome_node:
        raise ValueError("treatment and outcome must be different variables")
    treatment_descendants = _descendants(world, treatment_node)
    allowed = [
        node
        for node in range(len(world.variables))
        if node not in {treatment_node, outcome_node} and node not in treatment_descendants
    ]
    valid: list[frozenset[int]] = []
    for size in range(len(allowed) + 1):
        for subset in combinations(allowed, size):
            condition = frozenset(subset)
            if any(other < condition for other in valid):
                continue
            if _backdoor_separated(world, treatment_node, outcome_node, condition):
                valid.append(condition)
    valid.sort(key=lambda candidate: (len(candidate), tuple(sorted(candidate))))
    return tuple(tuple(world.variables[node] for node in sorted(candidate)) for candidate in valid)


def collider_bias_effect(
    world: WorldSpec,
    treatment: object,
    outcome: object,
    collider: object,
    *,
    treatment_value: int = 1,
    baseline_value: int = 0,
    outcome_state: int = 1,
    condition_state: int = 1,
) -> Probability:
    """Return a collider-conditioned do contrast for ATE diagnostics.

    The estimand is
    P(outcome=outcome_state | do(treatment=treatment_value), collider=condition_state)
    minus the same quantity under do(treatment=baseline_value).
    """

    treatment_node = _node_index(world, treatment)
    outcome_node = _node_index(world, outcome)
    collider_node = _node_index(world, collider)
    if len({treatment_node, outcome_node, collider_node}) != 3:
        raise ValueError("treatment, outcome, and collider must be distinct")

    def conditional_effect(value: int) -> Probability:
        joint = interventional_joint_probability(
            world,
            {treatment_node: value},
            {collider_node: condition_state, outcome_node: outcome_state},
        )
        collider_mass = interventional_joint_probability(
            world,
            {treatment_node: value},
            {collider_node: condition_state},
        )
        if collider_mass == 0:
            raise ValueError("conditioning event has zero probability")
        return joint / collider_mass

    return conditional_effect(treatment_value) - conditional_effect(baseline_value)


def _directed_paths(world: WorldSpec, source: int, target: int) -> tuple[tuple[int, ...], ...]:
    paths: list[tuple[int, ...]] = []
    path = [source]
    seen = {source}

    def visit(current: int) -> None:
        if current == target:
            paths.append(tuple(path))
            return
        children = [child for parent, child in world.edges if parent == current]
        for child in sorted(children):
            if child in seen:
                continue
            seen.add(child)
            path.append(child)
            visit(child)
            path.pop()
            seen.remove(child)

    visit(source)
    return tuple(paths)


def mediator_set_truth(
    world: WorldSpec, treatment: object, outcome: object
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Return mediators and consecutive path edges for treatment->outcome.

    Mediators are every observed variable on at least one directed path from
    treatment to outcome. The order contains exactly the directed edges that
    appear consecutively on at least one such path.
    """

    treatment_node = _node_index(world, treatment)
    outcome_node = _node_index(world, outcome)
    paths = _directed_paths(world, treatment_node, outcome_node)
    if not paths:
        raise ValueError("treatment and outcome have no directed path")
    mediator_nodes: set[int] = set()
    consecutive_edges: set[tuple[int, int]] = set()
    for path in paths:
        mediator_nodes.update(path[1:-1])
        consecutive_edges.update(zip(path, path[1:], strict=False))
    mediators = tuple(world.variables[node] for node in sorted(mediator_nodes))
    order = tuple(
        (world.variables[source], world.variables[target])
        for source, target in sorted(consecutive_edges)
    )
    return mediators, order


def _state_index_for_node(world: WorldSpec, node: int, value: object, default: int = 1) -> int:
    if isinstance(value, bool):
        raise TypeError("state value must not be bool")
    if isinstance(value, int):
        return value
    text = str(value)
    if text.startswith("state_"):
        return int(text.removeprefix("state_"))
    if text.isdigit():
        return int(text)
    if text in world.state_names[node]:
        return world.state_names[node].index(text)
    return default


def _resolve_seed_node(world: WorldSpec, seed: Mapping[str, Any], value: object) -> int:
    if isinstance(value, str) and value not in world.variables:
        visible_schema = seed.get("visible_schema")
        if isinstance(visible_schema, Mapping):
            label_map = visible_schema.get("variable_labels")
            if isinstance(label_map, Mapping):
                inverse = {str(visible): str(internal) for internal, visible in label_map.items()}
                if value in inverse:
                    return _node_index(world, inverse[value])
    return _node_index(world, value)


def compute_query_truth(
    world: WorldSpec,
    seed: Mapping[str, Any],
    *,
    counterfactual_endpoint_time_limit_seconds: float | None = None,
) -> Mapping[str, Any]:
    """Compute hidden truth for a seed whose query owner is implemented."""

    query = seed.get("query")
    if not isinstance(query, Mapping):
        raise ValueError("seed must contain a query mapping")
    query_type = str(query.get("type"))
    status = query_truth_owner_status(query_type)
    if status != OWNER_STATUS_IMPLEMENTED:
        raise NotImplementedError(
            f"query truth owner for {query_type} is {status}, not implemented"
        )

    if query_type == "ate":
        treatment = query.get("treatment")
        outcome = query.get("outcome")
        if treatment is None or outcome is None:
            raise ValueError("ate query requires treatment and outcome")
        treatment_node = _resolve_seed_node(world, seed, treatment)
        outcome_node = _resolve_seed_node(world, seed, outcome)
        return {
            "type": "ate",
            "effect": ate_effect(
                world,
                treatment_node,
                outcome_node,
                treatment_value=_state_index_for_node(
                    world, treatment_node, query.get("treatment_value", 1)
                ),
                baseline_value=_state_index_for_node(
                    world, treatment_node, query.get("baseline_value", 0), default=0
                ),
                outcome_state=_state_index_for_node(
                    world, outcome_node, query.get("outcome_state", 1)
                ),
            ),
        }
    if query_type == "individual_counterfactual_probability":
        treatment = query.get("treatment")
        outcome = query.get("outcome")
        if treatment is None or outcome is None:
            raise ValueError(
                "individual_counterfactual_probability query requires treatment and outcome"
            )
        treatment_node = _resolve_seed_node(world, seed, treatment)
        outcome_node = _resolve_seed_node(world, seed, outcome)
        required_fields = (
            "factual_value",
            "counterfactual_value",
            "factual_outcome_state",
            "outcome_state",
        )
        missing = tuple(field for field in required_fields if field not in query)
        if missing:
            raise ValueError(
                "individual counterfactual query missing fields: " + ", ".join(missing)
            )
        certificate = _individual_counterfactual_probability_certificate(
            world,
            treatment_node,
            outcome_node,
            factual_value=_state_index_for_node(
                world, treatment_node, query["factual_value"], default=0
            ),
            counterfactual_value=_state_index_for_node(
                world, treatment_node, query["counterfactual_value"]
            ),
            factual_outcome_state=_state_index_for_node(
                world, outcome_node, query["factual_outcome_state"], default=0
            ),
            target_outcome_state=_state_index_for_node(world, outcome_node, query["outcome_state"]),
            time_limit_seconds=counterfactual_endpoint_time_limit_seconds,
            endpoint_tolerance=INDIVIDUAL_COUNTERFACTUAL_ENDPOINT_TOLERANCE,
        )
        return {
            "type": "individual_counterfactual_probability",
            "lower": certificate.lower,
            "upper": certificate.upper,
            "certification": certificate.certification,
            "interval_source": (
                "exact_markovian"
                if certificate.certification == "exact"
                else "epsilon_sharp_markovian_outer"
            ),
            "endpoint_error": certificate.endpoint_error,
            "endpoint_tolerance": INDIVIDUAL_COUNTERFACTUAL_ENDPOINT_TOLERANCE,
        }
    if query_type == "backadj_minimal_sets":
        treatment = query.get("treatment")
        outcome = query.get("outcome")
        if treatment is None or outcome is None:
            raise ValueError("backadj_minimal_sets query requires treatment and outcome")
        return {
            "type": "backadj_minimal_sets",
            "adjustment_sets": backdoor_adjustment_sets(
                world,
                _resolve_seed_node(world, seed, treatment),
                _resolve_seed_node(world, seed, outcome),
            ),
        }
    if query_type == "mediator_set":
        treatment = query.get("treatment")
        outcome = query.get("outcome")
        if treatment is None or outcome is None:
            raise ValueError("mediator_set query requires treatment and outcome")
        mediators, order = mediator_set_truth(
            world,
            _resolve_seed_node(world, seed, treatment),
            _resolve_seed_node(world, seed, outcome),
        )
        return {
            "type": "mediator_set",
            "mediators": mediators,
            "order": order,
        }
    if query_type == "best_intervention":
        outcome = query.get("outcome")
        decision_target = query.get("decision_target")
        if outcome is None or decision_target is None:
            raise ValueError("best_intervention query requires outcome and decision_target")
        outcome_node = _resolve_seed_node(world, seed, outcome)
        target, state, probability = best_intervention_truth(
            world,
            outcome_node,
            str(query.get("objective", "minimize")),
            _resolve_seed_node(world, seed, decision_target),
            outcome_state=_state_index_for_node(
                world,
                outcome_node,
                query.get("target_state", query.get("outcome_state", 1)),
            ),
        )
        return {
            "type": "best_intervention",
            "target": target,
            "value": state,
            "probability": probability,
        }
    raise NotImplementedError(
        f"query truth owner for {query_type} is implemented by a legacy owner, "
        "not by the generic WorldSpec owner"
    )
