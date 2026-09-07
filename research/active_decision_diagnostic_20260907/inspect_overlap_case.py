"""Inspect the largest-regret failure with a valid set, after answers were frozen."""

import itertools
import json
import pickle
from pathlib import Path

from cpt_world.query_truth import (
    compute_query_truth,
    worldspec_projected_interventional_distribution,
)

ROOT = Path("/home/chen/runs/active-decision-diagnostic-20260907")
path = Path("/home/chen/runs/environment-validation-20260907/cohort-b/task-68.pkl")
with path.open("rb") as f:
    row, world, seed, _ = pickle.load(f)
truth = compute_query_truth(world, seed)
x = world.variables.index(truth["target"])
labels = seed["visible_schema"]["variable_labels"]
by_label = {label: world.variables.index(name) for name, label in labels.items()}
y = by_label[seed["query"]["outcome"]]
parents = world.parents[x]
population = dict(worldspec_projected_interventional_distribution(world, {}, tuple(parents)))
cells = []
for states, cpt_row in zip(
    itertools.product(*(range(world.domains[z]) for z in parents)), world.cpt[x], strict=True
):
    probability = float(population[states])
    propensity = float(cpt_row[truth["value"]])
    cells.append(
        {
            "parent_states": list(states),
            "parent_probability": probability,
            "optimal_action_propensity": propensity,
            "expected_optimal_cell_count_in_27649_passive_units": 27649 * probability * propensity,
        }
    )
confounder = by_label["POF"]
experiment_coverage = []
arms = [("observe", {})]
arms += [
    (f"do({labels[name]}=state_{value})", {node: value})
    for node, name in enumerate(world.variables)
    if seed["manipulability"][name]
    for value in range(world.domains[node])
]
for name, intervention in arms:
    joint = dict(
        worldspec_projected_interventional_distribution(world, intervention, (x, confounder))
    )
    probability = float(joint[(truth["value"], 1)])
    # To learn Y's row at (X=2, POF=1), measure X,Y,POF, except that a do target
    # is known from its command and must not be measured or charged again.
    scalar_cost = len({x, y, confounder} - intervention.keys())
    experiment_coverage.append(
        {
            "arm": name,
            "critical_cell_probability": probability,
            "scalars_per_unit": scalar_cost,
            "critical_Y_samples_per_scalar": probability / scalar_cost,
        }
    )
best = max(experiment_coverage, key=lambda r: r["critical_Y_samples_per_scalar"])
result = {
    "scope": "Post-answer mechanism diagnosis of the largest observed valid-adjustment regret; "
    "not a newly constructed counterexample or an all-strategy risk bound.",
    "seed_id": seed["seed_id"],
    "x": labels[world.variables[x]],
    "y": labels[world.variables[y]],
    "x_parents": [labels[world.variables[z]] for z in parents],
    "y_parents": [labels[world.variables[z]] for z in world.parents[y]],
    "direct_x_to_y": (x, y) in world.edges,
    "candidate_probabilities": list(map(float, truth["candidate_probabilities"])),
    "optimal_action": truth["value"],
    "cells": cells,
    "parent_mass_with_expected_count_below_one": sum(
        c["parent_probability"]
        for c in cells
        if c["expected_optimal_cell_count_in_27649_passive_units"] < 1
    ),
    "critical_cell_coverage": {
        "scope": "Exact enumeration of every legal single intervention and observation for "
        "Y-row coverage at X=state_2, POF=state_1 per charged scalar. This uses the hidden model "
        "after commitment; it is not the globally optimal full-task policy or a learned design.",
        "arms": experiment_coverage,
        "best_arm": best,
        "best_to_observe_rate_ratio": best["critical_Y_samples_per_scalar"]
        / experiment_coverage[0]["critical_Y_samples_per_scalar"],
    },
}
(ROOT / "overlap-case.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
