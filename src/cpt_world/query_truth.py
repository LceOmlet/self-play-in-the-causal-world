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
from math import prod
from typing import Any

from .registry import (
    OWNER_STATUS_IMPLEMENTED,
    query_truth_owner_status,
)
from .world_space import WorldSpec


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
    values: tuple[Fraction, ...]


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
) -> Fraction:
    positions = {variable: position for position, variable in enumerate(variables)}
    projected = tuple(assignment[positions[variable]] for variable in factor.variables)
    return factor.values[_factor_index(world, factor.variables, projected)]


def _multiply_factors(world: WorldSpec, factors: tuple[_Factor, ...]) -> _Factor:
    if not factors:
        return _Factor((), (Fraction(1),))
    variables = tuple(sorted({variable for factor in factors for variable in factor.variables}))
    ranges = tuple(range(world.domains[variable]) for variable in variables)
    values: list[Fraction] = []
    for assignment in product(*ranges):
        value = Fraction(1)
        for factor in factors:
            value *= _factor_value(world, factor, variables, assignment)
        values.append(value)
    return _Factor(variables, tuple(values))


def _sum_out(world: WorldSpec, factor: _Factor, variable: int) -> _Factor:
    remaining = tuple(item for item in factor.variables if item != variable)
    positions = {item: position for position, item in enumerate(remaining)}
    values: list[Fraction] = []
    for assignment in product(*(range(world.domains[item]) for item in remaining)):
        total = Fraction(0)
        for state in range(world.domains[variable]):
            original = tuple(
                state if item == variable else assignment[positions[item]]
                for item in factor.variables
            )
            total += factor.values[_factor_index(world, factor.variables, original)]
        values.append(total)
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
    for node in range(len(world.variables)):
        if node in fixed:
            factors.append(
                _Factor(
                    (node,),
                    tuple(
                        Fraction(1) if state == fixed[node] else Fraction(0)
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
) -> tuple[tuple[tuple[int, ...], Fraction], ...]:
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
    if sum(probabilities, start=Fraction(0)) != 1:
        raise RuntimeError("internal error: eliminated distribution is not normalized")
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
    uniforms: tuple[Fraction, ...],
) -> tuple[int, ...]:
    """Sample one assignment ancestrally from the exact hard-do law."""

    fixed = _validate_interventions(world, interventions)
    if len(uniforms) != len(world.variables):
        raise ValueError("uniforms must contain one draw per WorldSpec node")
    if any(not isinstance(draw, Fraction) or not 0 <= draw < 1 for draw in uniforms):
        raise ValueError("uniform draws must be Fractions in [0, 1)")
    values = [0] * len(world.variables)
    for node in _topological_order(world):
        if node in fixed:
            values[node] = fixed[node]
            continue
        row_index = _parent_row_index(world, node, tuple(values))
        row = world.cpt[node][row_index]
        cumulative = Fraction(0)
        selected = len(row) - 1
        for state, probability in enumerate(row):
            cumulative += probability
            if uniforms[node] < cumulative:
                selected = state
                break
        values[node] = selected
    return tuple(values)


def worldspec_interventional_distribution(
    world: WorldSpec,
    interventions: Mapping[int, int],
) -> tuple[tuple[tuple[int, ...], Fraction], ...]:
    """Return the exact full-joint law under a finite hard-do assignment.

    This is the executable probability-law owner for generic ``WorldSpec``
    worlds.  Query truth and the interactive sampler both consume this same
    canonical assignment order instead of reimplementing CPT traversal.
    """

    fixed = _validate_interventions(world, interventions)

    distribution: list[tuple[tuple[int, ...], Fraction]] = []
    total = Fraction(0)
    for values in product(*(range(size) for size in world.domains)):
        if any(values[node] != state for node, state in fixed.items()):
            probability = Fraction(0)
        else:
            probability = Fraction(1)
            for node in range(len(world.variables)):
                if node in fixed:
                    continue
                row_index = _parent_row_index(world, node, values)
                probability *= world.cpt[node][row_index][values[node]]
        distribution.append((values, probability))
        total += probability
    if total != 1:
        raise RuntimeError("internal error: interventional distribution is not normalized")
    return tuple(distribution)


def worldspec_projected_interventional_distribution(
    world: WorldSpec,
    interventions: Mapping[int, int],
    measure: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], Fraction], ...]:
    """Return an exact selected-measure law under a finite hard intervention.

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
) -> Fraction:
    """Return the exact probability of ``targets`` under hard-do ``interventions``."""

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
        return Fraction(1)
    measure = tuple(sorted(target_values))
    assignment = tuple(target_values[node] for node in measure)
    law = worldspec_projected_interventional_distribution(world, interventions, measure)
    return dict(law)[assignment]


def interventional_probability(
    world: WorldSpec,
    interventions: Mapping[int, int],
    outcome: int,
    outcome_state: int,
) -> Fraction:
    """Return exact P(outcome=outcome_state) under a hard-do assignment."""

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
) -> Fraction:
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
) -> tuple[Fraction, Fraction]:
    """Return sharp bounds for a cross-world target-state transition.

    The target event is

    ``outcome(treatment_value) = outcome_state`` and
    ``outcome(baseline_value) != outcome_state``.

    A CPT-World fixes the two interventional event marginals but deliberately
    leaves their cross-world coupling unspecified.  The returned Frechet
    bounds therefore range over every coupling of those two Bernoulli events;
    no generated or hidden SCM is selected.
    """

    treatment_node = _node_index(world, treatment)
    outcome_node = _node_index(world, outcome)
    if treatment_node == outcome_node:
        raise ValueError("treatment and outcome must be different variables")
    if treatment_value == baseline_value:
        raise ValueError("treatment and baseline values must differ")
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
    lower = max(Fraction(0), treated_probability - baseline_probability)
    upper = min(treated_probability, Fraction(1) - baseline_probability)
    return lower, upper


def best_intervention_states(
    world: WorldSpec,
    outcome: object,
    objective: str,
    decision_target: object,
    *,
    outcome_state: int = 1,
) -> tuple[tuple[int, ...], Fraction]:
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
) -> tuple[str, int, Fraction]:
    """Return the canonical exact deployment intervention.

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


def _undirected_neighbors(world: WorldSpec, node: int) -> set[int]:
    return {
        child if parent == node else parent
        for parent, child in world.edges
        if parent == node or child == node
    }


def _backdoor_paths(world: WorldSpec, treatment: int, outcome: int) -> tuple[tuple[int, ...], ...]:
    """Return all simple back-door paths from treatment to outcome.

    A back-door path is an undirected path whose first step enters treatment
    through one of its parents.
    """

    paths: list[tuple[int, ...]] = []

    def visit(current: int, path: list[int], seen: set[int]) -> None:
        if current == outcome:
            paths.append(tuple(path))
            return
        for neighbor in sorted(_undirected_neighbors(world, current)):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            path.append(neighbor)
            visit(neighbor, path, seen)
            path.pop()
            seen.remove(neighbor)

    for start in world.parents.get(treatment, ()):
        visit(start, [treatment, start], {treatment, start})
    return tuple(paths)


def _path_is_open(world: WorldSpec, path: tuple[int, ...], condition: frozenset[int]) -> bool:
    edge_set = set(world.edges)
    descendants = {node: _descendants(world, node) for node in range(len(world.variables))}
    for index in range(1, len(path) - 1):
        node = path[index]
        previous = path[index - 1]
        following = path[index + 1]
        collider = (previous, node) in edge_set and (following, node) in edge_set
        if collider:
            if node not in condition and not (descendants[node] & condition):
                return False
        elif node in condition:
            return False
    return True


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
    paths = _backdoor_paths(world, treatment_node, outcome_node)
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
            if all(not _path_is_open(world, path, condition) for path in paths):
                valid.append(condition)
    minimal = [candidate for candidate in valid if not any(other < candidate for other in valid)]
    minimal.sort(key=lambda candidate: (len(candidate), tuple(sorted(candidate))))
    return tuple(
        tuple(world.variables[node] for node in sorted(candidate)) for candidate in minimal
    )


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
) -> Fraction:
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

    def conditional_effect(value: int) -> Fraction:
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


def compute_query_truth(world: WorldSpec, seed: Mapping[str, Any]) -> Mapping[str, Any]:
    """Compute exact truth for a seed whose query owner is implemented."""

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
        outcome_node = _resolve_seed_node(world, seed, outcome)
        return {
            "type": "ate",
            "effect": ate_effect(
                world,
                _resolve_seed_node(world, seed, treatment),
                outcome_node,
                outcome_state=_state_index_for_node(
                    world, outcome_node, query.get("outcome_state", 1)
                ),
            ),
        }
    if query_type == "counterfactual_transition_bounds":
        treatment = query.get("treatment")
        outcome = query.get("outcome")
        if treatment is None or outcome is None:
            raise ValueError(
                "counterfactual_transition_bounds query requires treatment and outcome"
            )
        treatment_node = _resolve_seed_node(world, seed, treatment)
        outcome_node = _resolve_seed_node(world, seed, outcome)
        lower, upper = counterfactual_transition_bounds(
            world,
            treatment_node,
            outcome_node,
            treatment_value=_state_index_for_node(
                world, treatment_node, query.get("treatment_value", 1)
            ),
            baseline_value=_state_index_for_node(
                world, treatment_node, query.get("baseline_value", 0), default=0
            ),
            outcome_state=_state_index_for_node(world, outcome_node, query.get("outcome_state", 1)),
        )
        return {
            "type": "counterfactual_transition_bounds",
            "lower": lower,
            "upper": upper,
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
