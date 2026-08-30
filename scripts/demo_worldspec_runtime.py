"""Print reproducible prompt -> intervention -> feedback -> terminal demos."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from cpt_world import (
    OutcomeTape,
    WorldGrammar,
    WorldSpec,
    WorldSpecEpisode,
    assemble_sampled_anchor_tasks,
    backdoor_adjustment_sets,
    compute_query_truth,
    iter_sampled_seeds,
)

_DEMO_SEEDS = {
    "ate": 0,
    "individual_counterfactual_probability": 1,
    "best_intervention": 0,
    "backadj_minimal_sets": 0,
    "mediator_set": 5,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _task(query_type: str) -> tuple[Mapping[str, Any], WorldSpec, str]:
    grammar = WorldGrammar()
    seed_number = _DEMO_SEEDS[query_type]
    (seed,) = iter_sampled_seeds(
        grammar,
        query_types=(query_type,),
        start_seed=seed_number,
        count=1,
    )
    seed_id = str(seed["seed_id"])
    proposal_index = int(seed_id.split("-", 2)[1])
    anchor_index = int(seed_id.rsplit("-a", 1)[1])
    ((world, regenerated_seed),) = assemble_sampled_anchor_tasks(
        grammar,
        proposal_index,
        query_type,
        anchor_index,
    )
    if regenerated_seed != seed:
        raise RuntimeError("demo task regeneration disagrees with the sampler owner")
    sampling_status = "current_admitted_sampler"
    return seed, world, sampling_status


def _experiment(seed: Mapping[str, Any], world: WorldSpec, measure_max: int) -> Mapping[str, Any]:
    labels = seed["visible_schema"]["variable_labels"]
    visible_query = seed["query"]
    target_name = next(
        name
        for name, allowed in seed["manipulability"].items()
        if allowed and labels[name] != visible_query["outcome"]
    )
    target_label = labels[target_name]
    inverse = {visible: internal for internal, visible in labels.items()}
    preferred = [visible_query.get("outcome"), visible_query.get("collider")]
    measure_labels: list[str] = []
    for label in preferred:
        if (
            isinstance(label, str)
            and label != target_label
            and seed["readable"][inverse[label]]
            and label not in measure_labels
            and len(measure_labels) < measure_max
        ):
            measure_labels.append(label)
    for name in world.variables:
        label = labels[name]
        if (
            seed["readable"][name]
            and name != target_name
            and label not in measure_labels
            and len(measure_labels) < measure_max
        ):
            measure_labels.append(label)
    return {
        "type": "intervene",
        "target": target_label,
        "value": "state_1",
        "measure": measure_labels,
        "batch_size": 8,
    }


def _answer(
    seed: Mapping[str, Any],
    world: WorldSpec,
    truth: Mapping[str, Any],
) -> Mapping[str, Any]:
    labels = seed["visible_schema"]["variable_labels"]
    query_type = seed["query"]["type"]
    if query_type == "ate":
        return {
            "type": "answer",
            "effect": {
                f"state_{state}": float(component)
                for state, component in enumerate(truth["effect"])
            },
        }
    if query_type == "individual_counterfactual_probability":
        return {
            "type": "answer",
            "lower": float(truth["lower"]),
            "upper": float(truth["upper"]),
        }
    if query_type == "best_intervention":
        return {
            "type": "answer",
            "value": f"state_{truth['value']}",
        }
    if query_type == "backadj_minimal_sets":
        inverse = {visible: internal for internal, visible in labels.items()}
        treatment = world.variables.index(inverse[str(seed["query"]["treatment"])])
        outcome = world.variables.index(inverse[str(seed["query"]["outcome"])])
        adjustment_set = backdoor_adjustment_sets(world, treatment, outcome)[0]
        return {
            "type": "answer",
            "adjustment_set": [
                labels[name] for name in adjustment_set
            ],
        }
    return {
        "type": "answer",
        "mediators": [labels[name] for name in truth["mediators"]],
        "order": [[labels[left], labels[right]] for left, right in truth["order"]],
    }


def run_demo(query_type: str) -> Mapping[str, Any]:
    seed, world, sampling_status = _task(query_type)
    episode = WorldSpecEpisode(
        world,
        seed,
        OutcomeTape(f"worldspec-demo-tape:{query_type}"),
    )
    prompt = episode.initial_messages()[1]["content"]
    if episode.measure_max is None:
        raise RuntimeError("sampled demo seed is missing observation bandwidth")
    intervention = _experiment(seed, world, episode.measure_max)
    batch_step = episode.step(json.dumps(intervention, separators=(",", ":")))
    truth = compute_query_truth(world, seed)
    terminal_answer = _answer(seed, world, truth)
    terminal = episode.step(json.dumps(terminal_answer, separators=(",", ":")))
    return {
        "query_type": query_type,
        "sampling_status": sampling_status,
        "seed_id": seed["seed_id"],
        "world_shape_sealed_from_model": {
            "nodes": len(world.variables),
            "domains": list(world.domains),
            "edges": len(world.edges),
        },
        "model_prompt": prompt,
        "model_intervention": intervention,
        "environment_feedback": batch_step.message,
        "model_terminal_answer": terminal_answer,
        "terminal_diagnostics": _jsonable(terminal.score),
        "terminal_reward": _jsonable(terminal.reward),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=tuple(_DEMO_SEEDS), default=None)
    args = parser.parse_args()
    modes = (args.mode,) if args.mode else tuple(_DEMO_SEEDS)
    print(json.dumps([run_demo(mode) for mode in modes], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
