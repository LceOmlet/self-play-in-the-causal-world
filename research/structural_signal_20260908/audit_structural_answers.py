"""Classify every retained structural answer through39 against frozen graph truth.

This reads existing trajectories; it does not generate tasks, change rewards,
run a public solver or intervene in training.
"""

import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.project / "src"))
    from cpt_world.query_truth import worldspec_projected_interventional_distribution

    signal_path = args.root / "training-signal-through39.json"
    signal = json.loads(signal_path.read_text())
    data = args.root.parent / "training-submission-20260907/data"
    accepted = {
        (r["split"], r["index"]): r
        for line in (data / "accepted.jsonl").read_bytes().splitlines()
        if (r := json.loads(line))
    }
    snapshots = [Path(path).parent for path in signal["input_sha256"]]
    events = defaultdict(list)
    for snapshot in snapshots:
        progress = json.loads((snapshot / "progress.json").read_text())
        assert (
            sha(snapshot / "progress.json")
            == signal["input_sha256"][str(snapshot / "progress.json")]
        )
        for name, identity in progress["source_prefixes"].items():
            assert sha(snapshot / name) == identity["sha256"]
        for path in (snapshot / "environment").glob("*.jsonl"):
            for line in path.read_bytes().splitlines():
                event = json.loads(line)
                if event["event"] == "act":
                    events[event["trajectory_id"]].append(event)
    reader_path = args.project / "research/training_submission_20260907/capture_progress.py"
    spec = importlib.util.spec_from_file_location("existing_reader", reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    counts, records, worlds, success_cases = defaultdict(Counter), [], {}, []
    for group in signal["groups"]:
        family = group["family"]
        if family not in {"backadj_minimal_sets", "mediator_set"}:
            continue
        index = group["task_index"]
        path = data / "truth" / f"train-{index:04d}.pkl"
        assert sha(path) == accepted["train", index]["world_file_sha256"]
        _, world, seed, truth = pickle.loads(path.read_bytes())
        (args.output / path.name).write_bytes(path.read_bytes())
        labels = seed["visible_schema"]["variable_labels"]
        inverse = {label: world.variables.index(name) for name, label in labels.items()}
        x, y = seed["query"]["treatment"], seed["query"]["outcome"]
        available = set(labels.values()) - {x, y}
        worlds[index] = {
            "sha256": sha(path),
            "treatment": x,
            "outcome": y,
            "parents": {
                labels[world.variables[n]]: [labels[world.variables[p]] for p in world.parents[n]]
                for n in world.parents
            },
        }
        for row in group["rows"]:
            counter = counts[family]
            counter["retained"] += 1
            positive = row["official_scalar_advantage"] > 0
            counter["positive"] += positive
            if not row["completed"]:
                continue
            answer = row["answer"]
            prediction = set(
                answer["adjustment_set" if family == "backadj_minimal_sets" else "mediators"]
            )
            empty, all_variables = not prediction, prediction == available
            counter["empty"] += empty
            counter["empty_positive"] += empty and positive
            counter["empty_strict_correct"] += empty and row["strict_correct"]
            counter["all_nonanchors"] += all_variables
            counter["all_nonanchors_positive"] += all_variables and positive
            trace = events[row["request_id"]]
            measurement = []
            for event in trace:
                command = event["command"]
                if not isinstance(command, dict) or command.get("type") != "intervene":
                    continue
                feedback = json.JSONDecoder().raw_decode(event["feedback"])[0]
                if feedback["type"] == "batch_result" and x in command["measure"]:
                    measurement.append(command)
            record = {
                "step": group["step"],
                "task_index": index,
                "family": family,
                "request_id": row["request_id"],
                "answer": answer,
                "positive_advantage": positive,
                "strict_correct": row["strict_correct"],
                "empty": empty,
                "all_nonanchors": all_variables,
                "legal_interventions_measuring_treatment": measurement,
            }
            if family == "mediator_set":
                extra = prediction - {labels[v] for v in truth["mediators"]}
                ancestors = {z for z in extra if world.path_exists(inverse[z], inverse[x])}
                tested = {c["target"] for c in measurement} & ancestors
                counter["with_false_mediators"] += bool(extra)
                counter["with_ancestor_of_treatment_in_mediators"] += bool(ancestors)
                counter["no_intervention_measuring_treatment"] += not measurement
                counter["false_ancestor_without_testing_any_on_treatment"] += (
                    bool(ancestors) and not tested
                )
                record.update(
                    false_mediators=sorted(extra),
                    treatment_ancestors_in_answer=sorted(ancestors),
                    treatment_ancestors_tested=sorted(tested),
                )
            records.append(record)
            if row["strict_correct"] and (family == "mediator_set" or prediction):
                dump = next(
                    s / f"rollouts/{group['step']}.jsonl"
                    for s in snapshots
                    if (s / f"rollouts/{group['step']}.jsonl").exists()
                )
                outputs = [json.loads(line) for line in dump.read_bytes().splitlines()]
                matches = [
                    o
                    for o in outputs
                    if reader.commands_from_output(o["output"])
                    == [event["command"] for event in trace]
                ]
                assert len(matches) == 1
                case = {**record, "output": matches[0]["output"], "events": trace}
                if family == "backadj_minimal_sets":
                    null_checks = {}
                    for label in ["RQI", "PHE"]:
                        node = inverse[label]
                        assert not world.path_exists(node, inverse[x])
                        laws = [
                            dict(
                                worldspec_projected_interventional_distribution(
                                    world, {node: state}, (inverse[x],)
                                )
                            )
                            for state in range(world.domains[node])
                        ]
                        error = max(
                            abs(float(laws[0][k]) - float(law[k])) for law in laws for k in law
                        )
                        assert error < 1e-12
                        null_checks[label] = {
                            "not_ancestor_of_treatment": True,
                            "population_marginal_max_difference": error,
                            "laws": [
                                [float(law[(i,)]) for i in range(world.domains[inverse[x]])]
                                for law in laws
                            ],
                        }
                    case["null_effect_checks"] = null_checks
                success_cases.append(case)
    assert len(records) == 62 and len(success_cases) == 2
    summary = {
        "counts": {k: dict(v) for k, v in counts.items()},
        "records": records,
        "worlds": worlds,
        "signal_sha256": sha(signal_path),
        "script_sha256": sha(Path(__file__)),
        "source_sha256": {
            name: sha(args.project / "src/cpt_world" / name)
            for name in ["query_truth.py", "world_space.py", "task_scoring.py", "rewards.py"]
        },
        "scope": "All64 retained structural trajectories through39,62 completed. "
        "Descriptive answer patterns, not learned policy changes or an algorithm bug.",
    }
    for name, value in [
        ("summary.json", summary),
        ("correct-structural-cases.json", success_cases),
    ]:
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(summary["counts"]))


if __name__ == "__main__":
    main()
