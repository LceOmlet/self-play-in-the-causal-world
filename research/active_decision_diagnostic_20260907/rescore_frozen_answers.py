"""Diagnose cache drift and score already committed answers with one current truth owner."""

import argparse
import gzip
import hashlib
import json
import pickle
from collections import Counter
from fractions import Fraction
from pathlib import Path

from public_policy import parse_prompt, solve

from cpt_world.query_truth import compute_query_truth, nearest_backdoor_adjustment_set
from cpt_world.rewards import terminal_quality_reward
from cpt_world.task_scoring import score_terminal_answer


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(x) for x in value]
    if isinstance(value, Fraction):
        return float(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    baseline_rows = [
        json.loads(line)
        for line in (args.cohort_root / "finite-baselines.jsonl").read_text().splitlines()
    ]
    baselines = {
        (r["seed_id"], e["replicate"]): e
        for r in baseline_rows
        if r["query_type"] == "best_intervention"
        for e in r["records"]
        if e["full_budget"]
    }
    report = []
    drift = []
    for path in sorted(args.results.glob("*.public.json.gz")):
        name = path.name.removesuffix(".public.json.gz")
        cohort, task_index, replicate = (
            name.split("-task-")[0],
            name.split("-task-")[1].split("-r")[0],
            int(name.rsplit("-r", 1)[1]),
        )
        task = args.cohort_root / cohort / f"task-{task_index}.pkl"
        with task.open("rb") as f:
            row, world, seed, cached = pickle.load(f)
        public_bytes = gzip.decompress(path.read_bytes())
        public = json.loads(public_bytes)
        policy = public["policy"]
        replay = iter(public["transcript"])

        def act(command, replay=replay):
            event = next(replay)
            assert command == event["command"]
            return json.loads(event["feedback"].splitlines()[0])

        assert solve(public["prompt"], act) == policy
        assert next(replay, None) is None
        fresh = compute_query_truth(world, seed)
        drift.append(
            {
                "seed_id": seed["seed_id"],
                "replicate": replicate,
                "probability_drift": float(fresh["probability"] - cached["probability"]),
                "optimal_state_changed": fresh["value"] != cached["value"],
                "candidate_max_abs_drift": max(
                    abs(float(a - b))
                    for a, b in zip(
                        fresh["candidate_probabilities"],
                        cached["candidate_probabilities"],
                        strict=True,
                    )
                ),
            }
        )
        p = parse_prompt(public["prompt"])
        label_to_node = {
            label: world.variables.index(node)
            for node, label in seed["visible_schema"]["variable_labels"].items()
        }
        x, y = label_to_node[p["x"]], label_to_node[p["y"]]
        selected = [label_to_node[z] for z in policy["selected"]]
        distance, nearest = nearest_backdoor_adjustment_set(world, x, y, selected)
        old_passive = baselines[(seed["seed_id"], replicate)]
        answers = {
            "active_adjusted": policy["answer"],
            "same_samples_passive": policy["same_samples_passive_answer"],
            "full_budget_passive": old_passive["answer"],
            "constant_state_0": {"type": "answer", "value": "state_0"},
        }
        evaluated = {}
        for key, answer in answers.items():
            # No cached probability from a different numerical implementation.
            score = score_terminal_answer(json.dumps(answer), seed, world)
            evaluated[key] = {
                "answer": answer,
                "reward": float(terminal_quality_reward(score)),
                "diagnostic": score,
            }
        d = evaluated["active_adjusted"]["diagnostic"]
        report.append(
            {
                "seed_id": seed["seed_id"],
                "replicate": replicate,
                "public_file": path.name,
                "public_only_replay_passed": True,
                "public_transcript_sha256": hashlib.sha256(public_bytes).hexdigest(),
                "budget": p["budget"],
                "policy": policy,
                "strong_discordant": d["observational_shortcut_error"] > 0,
                "adjustment_valid": distance == 0,
                "adjustment_edit_distance": distance,
                "nearest_valid_set": nearest,
                "selected_nonancestors": [
                    z
                    for z, n in zip(policy["selected"], selected, strict=True)
                    if not world.path_exists(n, x)
                ],
                "missed_parents_of_x": [n for n in world.parents[x] if n not in selected],
                "evaluated": evaluated,
            }
        )
    assert len(report) == 100
    assert set(Counter(r["seed_id"] for r in report).values()) == {2}
    output = {
        "scope": "All answers fixed before these oracle diagnostics. Fresh current-world truth "
        "is used for every method, including the historical passive answers. "
        "No observations, policy choices, or task parameters were regenerated.",
        "cache_drift": drift,
        "episodes": report,
    }
    target = args.results.parent / "rescored-answers.json"
    target.write_text(json.dumps(clean(output), indent=2) + "\n")
    print(
        json.dumps(
            {
                "episodes": len(report),
                "failed_to_rescore": 0,
                "changed_optimal_states": sum(r["optimal_state_changed"] for r in drift),
                "max_cached_probability_drift": max(abs(r["probability_drift"]) for r in drift),
                "active_optimal": sum(
                    r["evaluated"]["active_adjusted"]["diagnostic"]["optimal_action"]
                    for r in report
                ),
                "passive_optimal": sum(
                    r["evaluated"]["full_budget_passive"]["diagnostic"]["optimal_action"]
                    for r in report
                ),
                "valid_adjustment": sum(r["adjustment_valid"] for r in report),
            }
        )
    )


if __name__ == "__main__":
    main()
