"""Aggregate shadow profile for single-variable stratified Frechet bounds.

For every pre-treatment variable that is shared by the two intervention
worlds, condition the two endpoint marginals on that variable and apply the
Frechet inequalities within each state stratum.  The script reports only
distribution-level aggregates and never emits seed or variable identities.
"""

from __future__ import annotations

import json
import statistics

from cpt_world import (
    WorldGrammar,
    iter_sampled_seeds,
    sample_task_world,
    worldspec_projected_interventional_distribution,
)
from cpt_world import counterfactual_solver as solver


DISTRIBUTION_START_SEED = 10_000
DISTRIBUTION_COUNT = 30
CONDITIONAL_ENDPOINT_TOLERANCE = 2e-3


def _query_indices(world: object, seed: dict[str, object]) -> tuple[int, ...]:
    query = seed["query"]
    visible_schema = seed["visible_schema"]
    assert isinstance(query, dict) and isinstance(visible_schema, dict)
    labels = visible_schema["variable_labels"]
    assert isinstance(labels, dict)
    visible_to_internal = {visible: internal for internal, visible in labels.items()}
    treatment = world.variables.index(visible_to_internal[query["treatment"]])
    outcome = world.variables.index(visible_to_internal[query["outcome"]])
    return (
        treatment,
        outcome,
        int(str(query["factual_value"]).removeprefix("state_")),
        int(str(query["counterfactual_value"]).removeprefix("state_")),
        int(str(query["factual_outcome_state"]).removeprefix("state_")),
        int(str(query["outcome_state"]).removeprefix("state_")),
    )


def _event_mass_by_stratum(
    world: object,
    *,
    treatment: int,
    treatment_value: int,
    outcome: int,
    outcome_state: int,
    stratum: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    law = worldspec_projected_interventional_distribution(
        world,
        {treatment: treatment_value},
        (stratum, outcome),
    )
    stratum_mass = [0.0] * world.domains[stratum]
    event_mass = [0.0] * world.domains[stratum]
    for (stratum_state, current_outcome), probability in law:
        mass = float(probability)
        stratum_mass[stratum_state] += mass
        if current_outcome == outcome_state:
            event_mass[stratum_state] += mass
    return tuple(stratum_mass), tuple(event_mass)


def _best_single_shared_stratification(
    world: object,
    *,
    treatment: int,
    outcome: int,
    factual_value: int,
    counterfactual_value: int,
    factual_outcome_state: int,
    target_outcome_state: int,
) -> tuple[float, float, float, float]:
    factual_law = worldspec_projected_interventional_distribution(
        world, {treatment: factual_value}, (outcome,)
    )
    counterfactual_law = worldspec_projected_interventional_distribution(
        world, {treatment: counterfactual_value}, (outcome,)
    )
    factual_probability = float(dict(factual_law)[(factual_outcome_state,)])
    counterfactual_probability = float(
        dict(counterfactual_law)[(target_outcome_state,)]
    )
    old_lower = max(0.0, factual_probability + counterfactual_probability - 1.0)
    old_upper = min(factual_probability, counterfactual_probability)
    best_lower, best_upper = old_lower, old_upper

    shared_nodes = set(range(len(world.variables))) - set(
        solver._descendants(world, treatment)
    ) - {treatment}
    for stratum in shared_nodes:
        factual_stratum, factual_event = _event_mass_by_stratum(
            world,
            treatment=treatment,
            treatment_value=factual_value,
            outcome=outcome,
            outcome_state=factual_outcome_state,
            stratum=stratum,
        )
        counterfactual_stratum, counterfactual_event = _event_mass_by_stratum(
            world,
            treatment=treatment,
            treatment_value=counterfactual_value,
            outcome=outcome,
            outcome_state=target_outcome_state,
            stratum=stratum,
        )
        if any(
            abs(left - right) > 1e-8
            for left, right in zip(
                factual_stratum, counterfactual_stratum, strict=True
            )
        ):
            raise RuntimeError("a selected stratum is not shared across interventions")
        lower = sum(
            max(0.0, left + right - mass)
            for mass, left, right in zip(
                factual_stratum,
                factual_event,
                counterfactual_event,
                strict=True,
            )
        )
        upper = sum(
            min(left, right)
            for left, right in zip(
                factual_event, counterfactual_event, strict=True
            )
        )
        best_lower = max(best_lower, lower)
        best_upper = min(best_upper, upper)
    return old_lower, old_upper, best_lower, best_upper


def _bucket(value: float) -> str:
    if value <= 0.0:
        return "none"
    if value <= 0.01:
        return "le_0.01"
    if value <= 0.1:
        return "le_0.1"
    if value <= 0.25:
        return "le_0.25"
    if value <= 0.5:
        return "le_0.5"
    return "gt_0.5"


def main() -> None:
    grammar = WorldGrammar()
    old_widths: list[float] = []
    new_widths: list[float] = []
    relative_shrink: dict[str, int] = {}
    strict = 0
    tolerance_closed = 0
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
            grammar, sample_index, "individual_counterfactual_probability"
        )
        treatment, outcome, factual, counterfactual, factual_y, target_y = (
            _query_indices(world, seed)
        )
        old_lower, old_upper, new_lower, new_upper = (
            _best_single_shared_stratification(
                world,
                treatment=treatment,
                outcome=outcome,
                factual_value=factual,
                counterfactual_value=counterfactual,
                factual_outcome_state=factual_y,
                target_outcome_state=target_y,
            )
        )
        old_width = max(0.0, old_upper - old_lower)
        new_width = max(0.0, new_upper - new_lower)
        old_widths.append(old_width)
        new_widths.append(new_width)
        shrink = 0.0 if old_width == 0.0 else (old_width - new_width) / old_width
        bucket = _bucket(shrink)
        relative_shrink[bucket] = relative_shrink.get(bucket, 0) + 1
        strict += int(new_width < old_width - 1e-12)
        factual_probability = float(
            dict(
                worldspec_projected_interventional_distribution(
                    world, {treatment: factual}, (outcome,)
                )
            )[(factual_y,)]
        )
        tolerance_closed += int(
            new_width <= CONDITIONAL_ENDPOINT_TOLERANCE * factual_probability
        )

    print(
        json.dumps(
            {
                "cohort_size": DISTRIBUTION_COUNT,
                "strictly_tightened": strict,
                "relative_shrink": dict(sorted(relative_shrink.items())),
                "old_width_median": statistics.median(old_widths),
                "new_width_median": statistics.median(new_widths),
                "old_width_mean": statistics.fmean(old_widths),
                "new_width_mean": statistics.fmean(new_widths),
                "outer_width_within_endpoint_tolerance": tolerance_closed,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
