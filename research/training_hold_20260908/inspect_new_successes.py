"""Check all new successful strong-decision traces in actual updates 30--39."""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project / "src"))
    from cpt_world.query_truth import worldspec_projected_interventional_distribution

    audit_path = args.root / "training-signal-through39.json"
    trace_path = args.root / "new-strong-successes30-39.json"
    audit, traces = [json.loads(p.read_text()) for p in [audit_path, trace_path]]
    expected = {r["request_id"] for r in audit["successful_strong_trajectories"] if r["step"] > 29}
    assert len(expected) == 3 and {t["request_id"] for t in traces} == expected
    accepted = {
        (r["split"], r["index"]): r
        for line in (args.data / "accepted.jsonl").read_bytes().splitlines()
        if (r := json.loads(line))["query_type"] == "best_intervention"
    }
    records = []
    for trace in traces:
        assert hashlib.sha256(trace["output"].encode()).hexdigest() == trace["output_sha256"]
        path = args.data / "truth" / f"train-{trace['task_index']:04d}.pkl"
        assert sha(path) == accepted["train", trace["task_index"]]["world_file_sha256"]
        _, world, seed, truth = pickle.loads(path.read_bytes())
        labels = [seed["visible_schema"]["variable_labels"][v] for v in world.variables]
        x, y = [labels.index(seed["query"][k]) for k in ["decision_target", "outcome"]]
        state = int(seed["query"]["outcome_state"].removeprefix("state_"))
        assert seed["query"]["objective"] == "maximize" and truth["value"] == 0
        natural = trace["request_id"] == "803b597be9ec4e119f1958b5e7d3adef"
        selected = [
            e
            for e in trace["events"]
            if e["command"].get("type") == ("observe" if natural else "intervene")
            and (
                natural
                or (
                    e["command"].get("target") == ("IOV" if trace["task_index"] == 383 else "VER")
                    and (trace["task_index"] == 408 or e["command"].get("value") == "state_0")
                )
            )
        ]
        totals, successes = [0] * world.domains[x], [0] * world.domains[x]
        mixture = {}
        samples = 0
        for event in selected:
            feedback = json.JSONDecoder().raw_decode(event["feedback"])[0]
            h = feedback["batch"]["joint_histogram"]
            xi, yi = h["columns"].index(labels[x]), h["columns"].index(labels[y])
            count = feedback["batch"]["n"]
            assert count == sum(n for _, n in h["rows"])
            for assignment, n in h["rows"]:
                totals[assignment[xi]] += n
                successes[assignment[xi]] += n * (assignment[yi] == state)
            command = event["command"]
            intervention = (
                {}
                if natural
                else {labels.index(command["target"]): int(command["value"].removeprefix("state_"))}
            )
            law = worldspec_projected_interventional_distribution(world, intervention, (x, y))
            for key, probability in law:
                mixture[key] = mixture.get(key, 0.0) + count * float(probability)
            samples += count
        assert samples == sum(totals) and all(totals)
        population = [
            sum(p for (a, b), p in mixture.items() if a == action and b == state)
            / sum(p for (a, _), p in mixture.items() if a == action)
            for action in range(world.domains[x])
        ]
        empirical = [a / b for a, b in zip(successes, totals, strict=True)]
        if natural:
            assert successes == [13, 29, 1] and totals == [40, 84, 4]
            assert empirical.index(max(empirical)) == 1
            for literal in ["13/25", "15/39", "1/2"]:
                assert literal in trace["output"]
        records.append(
            {
                "step": trace["step"],
                "task_index": trace["task_index"],
                "request_id": trace["request_id"],
                "source_world_sha256": sha(path),
                "output_sha256": trace["output_sha256"],
                "sample_count": samples,
                "selected_commands": [e["command"] for e in selected],
                "actual_successes": successes,
                "actual_totals": totals,
                "actual_empirical_conditionals": empirical,
                "same_experiment_mixture_population_conditionals": population,
                "true_do_values": truth["candidate_probabilities"],
                "true_action": truth["value"],
                "decision_parents": [labels[n] for n in world.parents[x]],
                "outcome_parents": [labels[n] for n in world.parents[y]],
                "scope": "Post-answer accounting and population formula check; "
                "no hidden truth supplied to model.",
            }
        )
    report = {
        "script_sha256": sha(Path(__file__)),
        "audit_sha256": sha(audit_path),
        "traces_sha256": sha(trace_path),
        "all_new_successes_checked": 3,
        "records": records,
        "limits": "A correct outcome with an incorrect recorded calculation "
        "is not itself a DAPO bug. "
        "No inference about model learning or policy-wide causal reasoning from three successes.",
    }
    (args.root / "new-successes-accounting.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
