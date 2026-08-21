"""Print reproducible prompt -> intervention -> feedback -> terminal demos."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from cpt_world import (
    HIDING_MODES,
    Budget,
    OutcomeTape,
    WorldGrammar,
    WorldSpec,
    WorldSpecEpisode,
    assemble_seed,
    compute_query_truth,
    legal_query_anchors,
    sample_task_world,
    sample_world,
)

_DEMO_SEEDS = {
    "ate": 0,
    "counterfactual_transition_bounds": 0,
    "best_intervention": 0,
    "backadj_minimal_sets": 0,
    "mediator_set": 5,
}
_NUMERICAL_MODES = frozenset({"ate", "counterfactual_transition_bounds", "best_intervention"})


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
    structural = sample_world(grammar, seed_number)
    anchors = legal_query_anchors(structural, query_type)[0]
    if query_type in _NUMERICAL_MODES:
        world = sample_task_world(grammar, seed_number, query_type, anchors)
    else:
        world = structural
    sampling_status = "current_admitted_sampler"
    task_head = (
        "target_query"
        if query_type in {"ate", "counterfactual_transition_bounds"}
        else ("decision" if query_type == "best_intervention" else "discovery")
    )
    seed = assemble_seed(
        world,
        tuple(sorted(HIDING_MODES)),
        query_type,
        task_head,
        anchors=anchors,
        seed_id=f"WORLDSPEC-DEMO-{query_type}-{seed_number}",
    )
    return seed, world, sampling_status


def _experiment(seed: Mapping[str, Any], world: WorldSpec) -> Mapping[str, Any]:
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
    measure_labels = [
        label
        for label in preferred
        if isinstance(label, str) and label != target_label and seed["readable"][inverse[label]]
    ]
    for name in world.variables:
        label = labels[name]
        if (
            seed["readable"][name]
            and name != target_name
            and label not in measure_labels
            and len(measure_labels) < 2
        ):
            measure_labels.append(label)
    return {
        "type": "intervene",
        "target": target_label,
        "value": "state_1",
        "measure": measure_labels,
        "batch_size": 8,
    }


def _answer(seed: Mapping[str, Any], truth: Mapping[str, Any]) -> Mapping[str, Any]:
    labels = seed["visible_schema"]["variable_labels"]
    query_type = seed["query"]["type"]
    if query_type == "ate":
        return {"type": "answer", "effect": float(truth["effect"])}
    if query_type == "counterfactual_transition_bounds":
        return {
            "type": "answer",
            "lower": float(truth["lower"]),
            "upper": float(truth["upper"]),
        }
    if query_type == "best_intervention":
        return {
            "type": "answer",
            "intervention": {
                "target": labels[truth["target"]],
                "value": f"state_{truth['value']}",
            },
        }
    if query_type == "backadj_minimal_sets":
        return {
            "type": "answer",
            "adjustment_sets": [
                [labels[name] for name in adjustment_set]
                for adjustment_set in truth["adjustment_sets"]
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
        budget=Budget(max_rounds=2, max_samples=16, batch_sizes=(4, 8)),
        measure_max=2,
    )
    prompt = episode.initial_messages()[1]["content"]
    intervention = _experiment(seed, world)
    batch_step = episode.step(json.dumps(intervention, separators=(",", ":")))
    truth = compute_query_truth(world, seed)
    terminal_answer = _answer(seed, truth)
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
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=tuple(_DEMO_SEEDS), default=None)
    args = parser.parse_args()
    modes = (args.mode,) if args.mode else tuple(_DEMO_SEEDS)
    print(json.dumps([run_demo(mode) for mode in modes], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
