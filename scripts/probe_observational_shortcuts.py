"""Measure how far observational shortcut targets are from causal task truth.

This is a fixed-seed diagnostic over the existing world-first sampler.  It
does not filter worlds, alter task targets, or define a new task family.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from cpt_world.query_truth import (
    compute_query_truth,
    interventional_probability,
    worldspec_projected_interventional_distribution,
)
from cpt_world.rewards import terminal_quality_reward
from cpt_world.world_space import (
    WorldGrammar,
    WorldSpec,
    _minimum_backdoor_adjustment_size,
    iter_sampled_seeds,
)

NUMERIC_QUERY_TYPES = (
    "ate",
    "individual_counterfactual_probability",
    "best_intervention",
)
SHORTCUT_SEPARATION_TOLERANCE = 1e-12


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Sequence[float]) -> Mapping[str, float]:
    return {
        "mean": statistics.fmean(values),
        "q10": _quantile(values, 0.10),
        "median": statistics.median(values),
        "q90": _quantile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def _state_index(value: object) -> int:
    text = str(value)
    if not text.startswith("state_"):
        raise ValueError(f"unsupported rendered state {value!r}")
    return int(text.removeprefix("state_"))


def _node_index(world: WorldSpec, seed: Mapping[str, Any], visible: object) -> int:
    labels = seed["visible_schema"]["variable_labels"]
    inverse = {str(label): str(name) for name, label in labels.items()}
    return world.variables.index(inverse[str(visible)])


def _task_pair(
    grammar: WorldGrammar,
    sample_index: int,
    query_type: str,
) -> tuple[WorldSpec, Mapping[str, Any]]:
    seed = iter_sampled_seeds(
        grammar,
        query_types=(query_type,),
        start_seed=sample_index,
        count=1,
    )[0]
    source = seed["world_source"]
    variables = tuple(source["variables"])
    variable_index = {name: index for index, name in enumerate(variables)}
    edges = tuple(
        sorted((variable_index[parent], variable_index[child]) for parent, child in source["edges"])
    )
    parents = {
        node: tuple(sorted(parent for parent, child in edges if child == node))
        for node in range(len(variables))
    }
    world = WorldSpec(
        family=str(source["family"]),
        topology=str(source["topology"]),
        variables=variables,
        domains=tuple(int(domain) for domain in source["domains"]),
        state_names=tuple(tuple(source["state_names"][name]) for name in variables),
        edges=edges,
        parents=parents,
        cpt={
            node: tuple(tuple(float(value) for value in row) for row in source["cpt"][name])
            for node, name in enumerate(variables)
        },
    )
    return world, seed


def _observational_conditional_outcomes(
    world: WorldSpec,
    conditioning_node: int,
    outcome_node: int,
) -> tuple[tuple[float, ...], ...]:
    law = worldspec_projected_interventional_distribution(
        world,
        {},
        (conditioning_node, outcome_node),
    )
    joint = [[0.0] * world.domains[outcome_node] for _ in range(world.domains[conditioning_node])]
    for (condition_value, outcome_value), probability in law:
        joint[condition_value][outcome_value] += float(probability)
    result: list[tuple[float, ...]] = []
    for row in joint:
        mass = math.fsum(row)
        if not mass > 0.0:
            raise RuntimeError("observational conditioning event has zero probability")
        result.append(tuple(value / mass for value in row))
    return tuple(result)


def _interventional_outcome(
    world: WorldSpec,
    treatment: int,
    treatment_state: int,
    outcome: int,
) -> tuple[float, ...]:
    law = worldspec_projected_interventional_distribution(
        world,
        {treatment: treatment_state},
        (outcome,),
    )
    probabilities = dict(law)
    return tuple(float(probabilities[(state,)]) for state in range(world.domains[outcome]))


def _total_variation(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(abs(a - b) for a, b in zip(left, right, strict=True)) / 2.0


def _parent_selection_shift(
    world: WorldSpec,
    treatment: int,
    treatment_state: int,
) -> float:
    parents = tuple(world.parents[treatment])
    if not parents:
        return 0.0
    parent_law = dict(worldspec_projected_interventional_distribution(world, {}, parents))
    joint_law = worldspec_projected_interventional_distribution(
        world,
        {},
        (*parents, treatment),
    )
    mass = math.fsum(
        float(probability)
        for assignment, probability in joint_law
        if assignment[-1] == treatment_state
    )
    if not mass > 0.0:
        raise RuntimeError("observational treatment state has zero probability")
    conditional = {
        assignment[:-1]: float(probability) / mass
        for assignment, probability in joint_law
        if assignment[-1] == treatment_state
    }
    return _total_variation(
        tuple(float(parent_law[assignment]) for assignment in parent_law),
        tuple(conditional[assignment] for assignment in parent_law),
    )


def _shortest_directed_path_length(world: WorldSpec, source: int, target: int) -> int:
    children: dict[int, tuple[int, ...]] = {
        node: tuple(child for parent, child in world.edges if parent == node)
        for node in range(len(world.variables))
    }
    queue = deque(((source, 0),))
    visited = {source}
    while queue:
        node, distance = queue.popleft()
        for child in children[node]:
            if child == target:
                return distance + 1
            if child not in visited:
                visited.add(child)
                queue.append((child, distance + 1))
    raise RuntimeError("query endpoints have no directed causal path")


def _shortest_open_backdoor_length(
    world: WorldSpec,
    treatment: int,
    outcome: int,
) -> int | None:
    backdoor_edges = frozenset(edge for edge in world.edges if edge[0] != treatment)
    neighbors = {node: set() for node in range(len(world.variables))}
    for source, target in backdoor_edges:
        neighbors[source].add(target)
        neighbors[target].add(source)
    queue = deque(((treatment, (treatment,)),))
    while queue:
        node, path = queue.popleft()
        if node == outcome:
            return len(path) - 1
        for neighbor in neighbors[node]:
            if neighbor in path:
                continue
            if len(path) >= 2:
                previous = path[-2]
                if (previous, node) in backdoor_edges and (neighbor, node) in backdoor_edges:
                    continue
            queue.append((neighbor, (*path, neighbor)))
    return None


def _direct_common_parent_count(world: WorldSpec, left: int, right: int) -> int:
    return len(set(world.parents[left]) & set(world.parents[right]))


def _minimum_adjustment_size_for_query(
    world: WorldSpec,
    seed: Mapping[str, Any],
    query: Mapping[str, Any],
) -> int:
    treatment = _node_index(
        world,
        seed,
        query.get("treatment", query.get("decision_target")),
    )
    outcome = _node_index(world, seed, query["outcome"])
    return _minimum_backdoor_adjustment_size(
        len(world.variables),
        world.edges,
        treatment,
        outcome,
    )


def _ate_record(world: WorldSpec, seed: Mapping[str, Any]) -> Mapping[str, Any]:
    query = seed["query"]
    treatment = _node_index(world, seed, query["treatment"])
    outcome = _node_index(world, seed, query["outcome"])
    baseline = _state_index(query["baseline_value"])
    treated = _state_index(query["treatment_value"])
    truth = tuple(float(value) for value in compute_query_truth(world, seed)["effect"])
    causal_by_state = tuple(
        _interventional_outcome(world, treatment, state, outcome)
        for state in range(world.domains[treatment])
    )
    observed_by_state = _observational_conditional_outcomes(world, treatment, outcome)
    observed_baseline = observed_by_state[baseline]
    observed_treated = observed_by_state[treated]
    shortcut = tuple(
        treated_probability - baseline_probability
        for treated_probability, baseline_probability in zip(
            observed_treated,
            observed_baseline,
            strict=True,
        )
    )
    l1_error = math.fsum(
        abs(shortcut_component - truth_component)
        for shortcut_component, truth_component in zip(shortcut, truth, strict=True)
    )
    endpoint_gaps = (
        _total_variation(observed_by_state[baseline], causal_by_state[baseline]),
        _total_variation(observed_by_state[treated], causal_by_state[treated]),
    )
    selection_shifts = (
        _parent_selection_shift(world, treatment, baseline),
        _parent_selection_shift(world, treatment, treated),
    )
    endpoint_gap_sum = math.fsum(endpoint_gaps)
    selection_shift_sum = math.fsum(selection_shifts)
    causal_effect_tv = _total_variation(
        causal_by_state[treated],
        causal_by_state[baseline],
    )
    observational_effect_tv = _total_variation(
        observed_by_state[treated],
        observed_by_state[baseline],
    )
    pair_gaps: list[float] = []
    pair_effects: list[float] = []
    for reference in range(world.domains[treatment]):
        for comparison in range(reference + 1, world.domains[treatment]):
            causal_effect = tuple(
                comparison_probability - reference_probability
                for comparison_probability, reference_probability in zip(
                    causal_by_state[comparison],
                    causal_by_state[reference],
                    strict=True,
                )
            )
            observational_effect = tuple(
                comparison_probability - reference_probability
                for comparison_probability, reference_probability in zip(
                    observed_by_state[comparison],
                    observed_by_state[reference],
                    strict=True,
                )
            )
            pair_effects.append(math.fsum(abs(value) for value in causal_effect) / 2.0)
            pair_gaps.append(
                math.fsum(
                    abs(observed - causal)
                    for observed, causal in zip(
                        observational_effect,
                        causal_effect,
                        strict=True,
                    )
                )
                / 2.0
            )
    total_variation_error = l1_error / 2.0
    maximum_pair_gap = max(pair_gaps)
    maximum_pair_effect = max(pair_effects)
    return {
        "sample_index": int(str(seed["seed_id"]).split("-")[1]),
        "minimum_adjustment_size": _minimum_adjustment_size_for_query(world, seed, query),
        "l1_error": l1_error,
        "total_variation_error": total_variation_error,
        "causal_effect_tv": causal_effect_tv,
        "observational_effect_tv": observational_effect_tv,
        "relative_error_to_causal_effect": (
            total_variation_error / causal_effect_tv if causal_effect_tv > 0.0 else None
        ),
        "baseline_endpoint_gap": endpoint_gaps[0],
        "treated_endpoint_gap": endpoint_gaps[1],
        "endpoint_gap_sum": endpoint_gap_sum,
        "contrast_retention": (
            total_variation_error / endpoint_gap_sum if endpoint_gap_sum > 0.0 else None
        ),
        "baseline_parent_selection_shift": selection_shifts[0],
        "treated_parent_selection_shift": selection_shifts[1],
        "parent_selection_shift_sum": selection_shift_sum,
        "outcome_transmission_retention": (
            endpoint_gap_sum / selection_shift_sum if selection_shift_sum > 0.0 else None
        ),
        "maximum_state_pair_gap": maximum_pair_gap,
        "selected_to_maximum_pair_gap": (
            total_variation_error / maximum_pair_gap if maximum_pair_gap > 0.0 else None
        ),
        "maximum_state_pair_causal_effect": maximum_pair_effect,
        "selected_to_maximum_pair_causal_effect": (
            causal_effect_tv / maximum_pair_effect if maximum_pair_effect > 0.0 else None
        ),
        "treatment_domain_size": world.domains[treatment],
        "outcome_domain_size": world.domains[outcome],
        "causal_path_length": _shortest_directed_path_length(world, treatment, outcome),
        "open_backdoor_length": _shortest_open_backdoor_length(
            world,
            treatment,
            outcome,
        ),
        "direct_common_parent_count": _direct_common_parent_count(
            world,
            treatment,
            outcome,
        ),
        "current_reward": float(
            terminal_quality_reward(
                {
                    "kind": "target_query",
                    "total_variation_error": total_variation_error,
                    "observational_shortcut_error": total_variation_error,
                }
            )
        ),
        "reward_separable": total_variation_error > SHORTCUT_SEPARATION_TOLERANCE,
        "exact_match": l1_error <= 1e-10,
    }


def _counterfactual_record(
    world: WorldSpec,
    seed: Mapping[str, Any],
    *,
    time_limit_seconds: float,
) -> Mapping[str, Any]:
    query = seed["query"]
    treatment = _node_index(world, seed, query["treatment"])
    outcome = _node_index(world, seed, query["outcome"])
    factual_value = _state_index(query["factual_value"])
    counterfactual_value = _state_index(query["counterfactual_value"])
    factual_outcome = _state_index(query["factual_outcome_state"])
    target_outcome = _state_index(query["outcome_state"])

    observational_laws = _observational_conditional_outcomes(world, treatment, outcome)
    factual_law = observational_laws[factual_value]
    counterfactual_law = observational_laws[counterfactual_value]
    factual_probability = factual_law[factual_outcome]
    counterfactual_probability = counterfactual_law[target_outcome]
    fake_lower = max(0.0, factual_probability + counterfactual_probability - 1.0) / (
        factual_probability
    )
    fake_upper = min(factual_probability, counterfactual_probability) / factual_probability

    truth = compute_query_truth(
        world,
        seed,
        counterfactual_endpoint_time_limit_seconds=time_limit_seconds,
    )
    truth_lower = float(truth["lower"])
    truth_upper = float(truth["upper"])
    endpoint_error = float(truth.get("endpoint_error", 0.0))
    certified_lower_range = (truth_lower, min(1.0, truth_lower + endpoint_error))
    certified_upper_range = (max(0.0, truth_upper - endpoint_error), truth_upper)

    def distance(value: float, interval: tuple[float, float]) -> float:
        return max(interval[0] - value, 0.0, value - interval[1])

    lower_error = distance(fake_lower, certified_lower_range)
    upper_error = distance(fake_upper, certified_upper_range)
    mean_error = (lower_error + upper_error) / 2.0
    return {
        "sample_index": int(str(seed["seed_id"]).split("-")[1]),
        "minimum_adjustment_size": _minimum_adjustment_size_for_query(world, seed, query),
        "certification": str(truth["certification"]),
        "true_lower": truth_lower,
        "true_upper": truth_upper,
        "fake_lower": fake_lower,
        "fake_upper": fake_upper,
        "lower_endpoint_error": lower_error,
        "upper_endpoint_error": upper_error,
        "mean_endpoint_error": mean_error,
        "current_reward": float(
            terminal_quality_reward(
                {
                    "kind": "counterfactual_roi",
                    "mean_absolute_endpoint_error": mean_error,
                    "observational_shortcut_error": mean_error,
                }
            )
        ),
        "reward_separable": mean_error > SHORTCUT_SEPARATION_TOLERANCE,
        "exact_match": lower_error <= 1e-10 and upper_error <= 1e-10,
    }


def _decision_record(world: WorldSpec, seed: Mapping[str, Any]) -> Mapping[str, Any]:
    query = seed["query"]
    decision = _node_index(world, seed, query["decision_target"])
    outcome = _node_index(world, seed, query["outcome"])
    outcome_state = _state_index(query["outcome_state"])
    objective = str(query["objective"])
    causal_probabilities = tuple(
        float(interventional_probability(world, {decision: state}, outcome, outcome_state))
        for state in range(world.domains[decision])
    )
    observational_probabilities = tuple(
        law[outcome_state] for law in _observational_conditional_outcomes(world, decision, outcome)
    )
    choose = min if objective == "minimize" else max
    causal_best_probability = choose(causal_probabilities)
    observational_best_probability = choose(observational_probabilities)
    causal_best_states = {
        state
        for state, probability in enumerate(causal_probabilities)
        if abs(probability - causal_best_probability) <= 1e-12
    }
    observational_best_states = tuple(
        state
        for state, probability in enumerate(observational_probabilities)
        if abs(probability - observational_best_probability) <= 1e-12
    )
    fake_state = observational_best_states[0]
    chosen_causal_probability = causal_probabilities[fake_state]
    probability_span = max(causal_probabilities) - min(causal_probabilities)
    regret = (
        causal_best_probability - chosen_causal_probability
        if objective == "maximize"
        else chosen_causal_probability - causal_best_probability
    )
    normalized_regret = 0.0 if probability_span <= 1e-15 else regret / probability_span
    target_linf_error = max(
        abs(observed - causal)
        for observed, causal in zip(
            observational_probabilities,
            causal_probabilities,
            strict=True,
        )
    )
    biases = tuple(
        observed - causal
        for observed, causal in zip(
            observational_probabilities,
            causal_probabilities,
            strict=True,
        )
    )
    bias_range = max(biases) - min(biases)
    profile_mae = math.fsum(abs(value) for value in biases) / len(biases)
    pairwise_gap_error = statistics.fmean(
        abs(biases[left] - biases[right])
        for left in range(len(biases))
        for right in range(left + 1, len(biases))
    )
    ordered_causal = sorted(set(causal_probabilities))
    if len(ordered_causal) == 1:
        best_second_gap = 0.0
    elif objective == "maximize":
        best_second_gap = ordered_causal[-1] - ordered_causal[-2]
    else:
        best_second_gap = ordered_causal[1] - ordered_causal[0]
    selection_shift_sum = math.fsum(
        _parent_selection_shift(world, decision, state) for state in range(world.domains[decision])
    )
    return {
        "sample_index": int(str(seed["seed_id"]).split("-")[1]),
        "minimum_adjustment_size": _minimum_adjustment_size_for_query(world, seed, query),
        "target_linf_error": target_linf_error,
        "profile_mae": profile_mae,
        "pairwise_gap_error": pairwise_gap_error,
        "bias_range": bias_range,
        "causal_probability_span": probability_span,
        "observational_probability_span": (
            max(observational_probabilities) - min(observational_probabilities)
        ),
        "causal_best_second_gap": best_second_gap,
        "bias_range_to_best_second_gap": (
            bias_range / best_second_gap if best_second_gap > 0.0 else None
        ),
        "parent_selection_shift_sum": selection_shift_sum,
        "outcome_transmission_retention": (
            math.fsum(abs(value) for value in biases) / selection_shift_sum
            if selection_shift_sum > 0.0
            else None
        ),
        "decision_domain_size": world.domains[decision],
        "causal_path_length": _shortest_directed_path_length(world, decision, outcome),
        "open_backdoor_length": _shortest_open_backdoor_length(world, decision, outcome),
        "direct_common_parent_count": _direct_common_parent_count(world, decision, outcome),
        "action_match": fake_state in causal_best_states,
        "same_action_but_different_profile": (
            fake_state in causal_best_states and target_linf_error > 1e-10
        ),
        "normalized_regret": normalized_regret,
        "current_reward": float(
            terminal_quality_reward(
                {
                    "kind": "decision",
                    "normalized_regret": normalized_regret,
                    "observational_shortcut_normalized_regret": normalized_regret,
                }
            )
        ),
        "reward_separable": regret > SHORTCUT_SEPARATION_TOLERANCE,
    }


def _backdoor_record(world: WorldSpec, seed: Mapping[str, Any]) -> Mapping[str, Any]:
    minimum_size = _minimum_adjustment_size_for_query(world, seed, seed["query"])
    exact_match = minimum_size == 0
    return {
        "sample_index": int(str(seed["seed_id"]).split("-")[1]),
        "minimum_adjustment_size": minimum_size,
        "current_reward": float(exact_match),
        "exact_match": exact_match,
    }


def _aggregate(
    ate: Sequence[Mapping[str, Any]],
    counterfactual: Sequence[Mapping[str, Any]],
    decision: Sequence[Mapping[str, Any]],
    backdoor: Sequence[Mapping[str, Any]],
    counterfactual_failures: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    def observed_adjustment_sizes(rows: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
        return tuple(sorted({int(row["minimum_adjustment_size"]) for row in rows}))

    return {
        "ate": {
            "count": len(ate),
            "do_vs_observational_tv_error": _summary(
                [float(row["total_variation_error"]) for row in ate]
            ),
            "observational_shortcut_reward": _summary(
                [float(row["current_reward"]) for row in ate]
            ),
            "reward_separable_rate": statistics.fmean(bool(row["reward_separable"]) for row in ate),
            "separable_shortcut_reward": (
                _summary([float(row["current_reward"]) for row in ate if row["reward_separable"]])
                if any(row["reward_separable"] for row in ate)
                else None
            ),
            "exact_match_rate": statistics.fmean(bool(row["exact_match"]) for row in ate),
            "by_minimum_adjustment_size": {
                str(size): {
                    "count": len(group),
                    "median_tv_error": statistics.median(
                        float(row["total_variation_error"]) for row in group
                    ),
                    "exact_match_rate": statistics.fmean(bool(row["exact_match"]) for row in group),
                }
                for size in observed_adjustment_sizes(ate)
                if (group := [row for row in ate if row["minimum_adjustment_size"] == size])
            },
        },
        "individual_counterfactual_probability": {
            "attempted": len(counterfactual) + len(counterfactual_failures),
            "scored": len(counterfactual),
            "unresolved": len(counterfactual_failures),
            "certification_counts": {
                certification: sum(row["certification"] == certification for row in counterfactual)
                for certification in sorted({row["certification"] for row in counterfactual})
            },
            "observational_shortcut_endpoint_error": (
                _summary([float(row["mean_endpoint_error"]) for row in counterfactual])
                if counterfactual
                else None
            ),
            "observational_shortcut_reward": (
                _summary([float(row["current_reward"]) for row in counterfactual])
                if counterfactual
                else None
            ),
            "reward_separable_rate": (
                statistics.fmean(bool(row["reward_separable"]) for row in counterfactual)
                if counterfactual
                else None
            ),
            "separable_shortcut_reward": (
                _summary(
                    [
                        float(row["current_reward"])
                        for row in counterfactual
                        if row["reward_separable"]
                    ]
                )
                if any(row["reward_separable"] for row in counterfactual)
                else None
            ),
            "exact_match_rate": (
                statistics.fmean(bool(row["exact_match"]) for row in counterfactual)
                if counterfactual
                else None
            ),
            "by_minimum_adjustment_size": {
                str(size): {
                    "count": len(group),
                    "median_endpoint_error": statistics.median(
                        float(row["mean_endpoint_error"]) for row in group
                    ),
                    "median_shortcut_reward": statistics.median(
                        float(row["current_reward"]) for row in group
                    ),
                }
                for size in observed_adjustment_sizes(counterfactual)
                if (
                    group := [
                        row for row in counterfactual if row["minimum_adjustment_size"] == size
                    ]
                )
            },
            "failures": list(counterfactual_failures),
        },
        "best_intervention": {
            "count": len(decision),
            "do_vs_observational_target_linf_error": _summary(
                [float(row["target_linf_error"]) for row in decision]
            ),
            "observational_action_match_rate": statistics.fmean(
                bool(row["action_match"]) for row in decision
            ),
            "observational_shortcut_reward": _summary(
                [float(row["current_reward"]) for row in decision]
            ),
            "reward_separable_rate": statistics.fmean(
                bool(row["reward_separable"]) for row in decision
            ),
            "separable_shortcut_reward": (
                _summary(
                    [float(row["current_reward"]) for row in decision if row["reward_separable"]]
                )
                if any(row["reward_separable"] for row in decision)
                else None
            ),
            "by_minimum_adjustment_size": {
                str(size): {
                    "count": len(group),
                    "action_match_rate": statistics.fmean(
                        bool(row["action_match"]) for row in group
                    ),
                    "median_target_linf_error": statistics.median(
                        float(row["target_linf_error"]) for row in group
                    ),
                    "median_shortcut_reward": statistics.median(
                        float(row["current_reward"]) for row in group
                    ),
                }
                for size in observed_adjustment_sizes(decision)
                if (group := [row for row in decision if row["minimum_adjustment_size"] == size])
            },
        },
        "backadj_minimal_sets": {
            "count": len(backdoor),
            "empty_set_exact_match_rate": statistics.fmean(
                bool(row["exact_match"]) for row in backdoor
            ),
            "empty_set_shortcut_reward": _summary(
                [float(row["current_reward"]) for row in backdoor]
            ),
            "minimum_adjustment_size_counts": {
                str(size): sum(row["minimum_adjustment_size"] == size for row in backdoor)
                for size in observed_adjustment_sizes(backdoor)
            },
        },
        "mediator_set": {
            "status": "not_applicable",
            "reason": (
                "an observational conditional law has no unique directed mediator-set-and-order "
                "terminal object"
            ),
        },
    }


def _cheap_record(job: tuple[int, str]) -> tuple[str, Mapping[str, Any]]:
    sample_index, query_type = job
    grammar = WorldGrammar()
    world, seed = _task_pair(grammar, sample_index, query_type)
    owners = {
        "ate": _ate_record,
        "best_intervention": _decision_record,
        "backadj_minimal_sets": _backdoor_record,
    }
    return query_type, owners[query_type](world, seed)


def run_probe(
    *,
    count: int,
    counterfactual_count: int,
    counterfactual_time_limit_seconds: float,
    workers: int,
) -> Mapping[str, Any]:
    grammar = WorldGrammar()
    ate: list[Mapping[str, Any]] = []
    counterfactual: list[Mapping[str, Any]] = []
    counterfactual_failures: list[Mapping[str, Any]] = []
    decision: list[Mapping[str, Any]] = []
    backdoor: list[Mapping[str, Any]] = []

    destinations = {
        "ate": ate,
        "best_intervention": decision,
        "backadj_minimal_sets": backdoor,
    }
    jobs = [
        (sample_index, query_type) for sample_index in range(count) for query_type in destinations
    ]
    if workers == 1:
        cheap_records = map(_cheap_record, jobs)
        for query_type, record in cheap_records:
            destinations[query_type].append(record)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for query_type, record in executor.map(_cheap_record, jobs, chunksize=4):
                destinations[query_type].append(record)

    for sample_index in range(counterfactual_count):
        world, seed = _task_pair(
            grammar,
            sample_index,
            "individual_counterfactual_probability",
        )
        try:
            counterfactual.append(
                _counterfactual_record(
                    world,
                    seed,
                    time_limit_seconds=counterfactual_time_limit_seconds,
                )
            )
        except RuntimeError as error:
            counterfactual_failures.append({"sample_index": sample_index, "error": str(error)})
        print(
            f"counterfactual {sample_index + 1}/{counterfactual_count}: "
            f"scored={len(counterfactual)} unresolved={len(counterfactual_failures)}",
            flush=True,
        )

    return {
        "contract": {
            "sampler": "main world-first sampler",
            "node_counts": list(grammar.node_counts),
            "fixed_sample_indices": list(range(count)),
            "counterfactual_sample_indices": list(range(counterfactual_count)),
            "counterfactual_endpoint_time_limit_seconds": (counterfactual_time_limit_seconds),
            "task_filtering": False,
            "cheap_task_workers": workers,
        },
        "summary": _aggregate(
            ate,
            counterfactual,
            decision,
            backdoor,
            counterfactual_failures,
        ),
        "records": {
            "ate": ate,
            "individual_counterfactual_probability": counterfactual,
            "best_intervention": decision,
            "backadj_minimal_sets": backdoor,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--counterfactual-count", type=int, default=30)
    parser.add_argument("--counterfactual-time-limit-seconds", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()
    if arguments.count <= 0 or arguments.counterfactual_count < 0:
        parser.error("count must be positive and counterfactual count must be nonnegative")
    if arguments.workers <= 0:
        parser.error("workers must be positive")
    if arguments.counterfactual_time_limit_seconds <= 0:
        parser.error("counterfactual time limit must be positive")

    result = run_probe(
        count=arguments.count,
        counterfactual_count=arguments.counterfactual_count,
        counterfactual_time_limit_seconds=arguments.counterfactual_time_limit_seconds,
        workers=arguments.workers,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    if not arguments.quiet:
        print(rendered)


if __name__ == "__main__":
    main()
