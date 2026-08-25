"""Probe one exact terminal-separator upper bound on fixed hard seeds.

The prototype stays outside the production solver. It solves a transport LP
over the twin states of the outcome's parents. Every exact Markovian SCM
induces a feasible transport, so the resulting optimum is a valid redundant
upper bound; adding it cannot change the exact counterfactual semantics.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

from probe_counterfactual_solver import _query_indices
from pyscipopt import Model, quicksum

from cpt_world import WorldGrammar, iter_sampled_seeds, sample_task_world
from cpt_world.counterfactual_solver import (
    _SCIP_NUMERICAL_TOLERANCE,
    _fix_root_separator,
    _row_index,
    _SparseResponseModel,
)
from cpt_world.query_truth import (
    interventional_probability,
    worldspec_projected_interventional_distribution,
)


@dataclass(frozen=True, slots=True)
class SeparatorBound:
    upper: float
    variables: int
    constraints: int
    solve_seconds: float


@dataclass(frozen=True, slots=True)
class RootProbe:
    seed: int
    scope: str
    augmented: bool
    outer_upper: float
    separator_raw_upper: float
    separator_safe_upper: float
    separator_ratio: float
    primal_bound: float
    dual_bound: float
    gap: float
    nodes: int
    variables: int
    constraints: int
    generated_columns: int
    pricing_closed: bool
    build_seconds: float
    solve_seconds: float
    separator_lp_seconds: float


def _terminal_context(
    owner: _SparseResponseModel,
    parent_nodes: tuple[int, ...],
    parent_assignment: tuple[int, ...],
    treatment_state: int,
) -> tuple[int, ...]:
    values = dict(zip(parent_nodes, parent_assignment, strict=True))
    return tuple(
        treatment_state if parent == owner.treatment else values[parent]
        for parent in owner.world.parents[owner.outcome]
    )


def _terminal_separator_upper_bound(owner: _SparseResponseModel) -> SeparatorBound:
    """Maximize pointwise terminal bounds over all legal parent transports."""

    parent_nodes = tuple(
        parent
        for parent in owner.world.parents[owner.outcome]
        if parent != owner.treatment
    )
    if not parent_nodes:
        raise ValueError("the outcome has no non-treatment separator parent")
    if owner.outcome_events is None:
        raise ValueError("the prototype expects explicit terminal events")

    left_law = worldspec_projected_interventional_distribution(
        owner.world,
        {owner.treatment: owner.baseline_value},
        parent_nodes,
    )
    right_law = worldspec_projected_interventional_distribution(
        owner.world,
        {owner.treatment: owner.treatment_value},
        parent_nodes,
    )
    shared_positions = tuple(
        index
        for index, parent in enumerate(parent_nodes)
        if parent not in owner.affected
    )
    compatible_pairs = tuple(
        (left_assignment, right_assignment)
        for left_assignment, _ in left_law
        for right_assignment, _ in right_law
        if all(
            left_assignment[position] == right_assignment[position]
            for position in shared_positions
        )
    )
    if not compatible_pairs:
        raise RuntimeError("terminal separator has no compatible twin assignment")

    model = Model("counterfactual-terminal-separator-prototype")
    model.hideOutput()
    model.setIntParam("parallel/maxnthreads", 1)
    model.setIntParam("randomization/randomseedshift", 0)
    model.setIntParam("randomization/permutationseed", 0)
    model.setRealParam("limits/gap", 0.0)
    model.setRealParam("limits/absgap", 0.0)
    transport = {
        pair: model.addVar(lb=0.0, ub=1.0)
        for pair in compatible_pairs
    }
    for left_assignment, probability in left_law:
        model.addCons(
            quicksum(
                variable
                for (current_left, _), variable in transport.items()
                if current_left == left_assignment
            )
            == float(probability)
        )
    for right_assignment, probability in right_law:
        model.addCons(
            quicksum(
                variable
                for (_, current_right), variable in transport.items()
                if current_right == right_assignment
            )
            == float(probability)
        )

    left_event, right_event = owner.outcome_events

    def pair_cost(pair: tuple[tuple[int, ...], tuple[int, ...]]) -> float:
        left_assignment, right_assignment = pair
        left_context = _terminal_context(
            owner,
            parent_nodes,
            left_assignment,
            owner.baseline_value,
        )
        right_context = _terminal_context(
            owner,
            parent_nodes,
            right_assignment,
            owner.treatment_value,
        )
        left_row = owner.world.cpt[owner.outcome][
            _row_index(owner.world, owner.outcome, left_context)
        ]
        right_row = owner.world.cpt[owner.outcome][
            _row_index(owner.world, owner.outcome, right_context)
        ]
        left_probability = sum(float(left_row[state]) for state in left_event)
        right_probability = sum(float(right_row[state]) for state in right_event)
        return min(left_probability, right_probability)

    model.setObjective(
        quicksum(pair_cost(pair) * variable for pair, variable in transport.items()),
        "maximize",
    )
    started = time.perf_counter()
    model.optimize()
    elapsed = time.perf_counter() - started
    if str(model.getStatus()) != "optimal":
        raise RuntimeError("terminal-separator transport did not solve exactly")
    return SeparatorBound(
        upper=float(model.getPrimalbound()),
        variables=model.getNVars(),
        constraints=model.getNConss(),
        solve_seconds=elapsed,
    )


def _instance(seed_index: int) -> tuple[Any, tuple[int, ...], str]:
    grammar = WorldGrammar()
    seed = iter_sampled_seeds(
        grammar,
        start_seed=seed_index,
        count=1,
        query_types=("individual_counterfactual_probability",),
    )[0]
    world = sample_task_world(
        grammar,
        seed_index,
        "individual_counterfactual_probability",
    )
    query = _query_indices(world, seed)
    if seed_index == 0:
        # Production already decomposes seed 0 over root 5. Its first stratum
        # is the first remaining exact upper-endpoint bottleneck.
        world = _fix_root_separator(world, 5, 0)
        return world, query, "root_5_state_0"
    return world, query, "full_world"


def _run_one(
    seed_index: int,
    *,
    augmented: bool,
    time_limit_seconds: float,
) -> RootProbe:
    world, query, scope = _instance(seed_index)
    treatment, outcome, factual, counterfactual, factual_outcome, target_outcome = query
    outcome_events = ((factual_outcome,), (target_outcome,))
    left_probability = float(
        interventional_probability(world, {treatment: factual}, outcome, factual_outcome)
    )
    right_probability = float(
        interventional_probability(
            world,
            {treatment: counterfactual},
            outcome,
            target_outcome,
        )
    )
    outer = (
        max(0.0, left_probability + right_probability - 1.0),
        min(left_probability, right_probability),
    )
    build_started = time.perf_counter()
    owner = _SparseResponseModel(
        world,
        treatment,
        outcome,
        baseline_value=factual,
        treatment_value=counterfactual,
        outcome_state=None,
        outcome_events=outcome_events,
        sense="maximize",
        target_outer_bounds=outer,
    )
    separator = _terminal_separator_upper_bound(owner)
    separator_safe_upper = min(
        outer[1],
        separator.upper + 10.0 * _SCIP_NUMERICAL_TOLERANCE,
    )
    if owner.initial_target > separator_safe_upper:
        raise RuntimeError(
            "separator bound excluded the canonical exact completion: "
            f"{owner.initial_target} > {separator_safe_upper}"
        )
    if separator.upper > outer[1] + 1e-9:
        raise RuntimeError("separator upper bound exceeded the Frechet outer bound")
    if augmented:
        owner.model.chgVarUb(owner.target, separator_safe_upper)
    build_seconds = time.perf_counter() - build_started

    owner.model.setLongintParam("limits/nodes", 1)
    solve_started = time.perf_counter()
    try:
        owner.optimize(time_limit_seconds=time_limit_seconds)
    except RuntimeError:
        pass
    solve_seconds = time.perf_counter() - solve_started
    return RootProbe(
        seed=seed_index,
        scope=scope,
        augmented=augmented,
        outer_upper=outer[1],
        separator_raw_upper=separator.upper,
        separator_safe_upper=separator_safe_upper,
        separator_ratio=separator_safe_upper / outer[1] if outer[1] > 0.0 else 1.0,
        primal_bound=float(owner.model.getPrimalbound()),
        dual_bound=float(owner.model.getDualbound()),
        gap=float(owner.model.getGap()),
        nodes=int(owner.model.getNNodes()),
        variables=int(owner.model.getNVars()),
        constraints=int(owner.model.getNConss()),
        generated_columns=owner.pricer.generated_columns,
        pricing_closed=owner.pricer.closed and not owner.pricer.timed_out,
        build_seconds=build_seconds,
        solve_seconds=solve_seconds,
        separator_lp_seconds=separator.solve_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 6))
    args = parser.parse_args()
    if args.seconds <= 0.0:
        raise ValueError("--seconds must be positive")
    for seed_index in args.seeds:
        for augmented in (False, True):
            result = _run_one(
                seed_index,
                augmented=augmented,
                time_limit_seconds=args.seconds,
            )
            print(json.dumps(asdict(result), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
