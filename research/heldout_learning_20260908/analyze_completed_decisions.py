"""Audit all completed update25 decisions using their frozen events and worlds.

Hidden probabilities are used after the model's answers only to diagnose the
claimed causal formulas. This is neither a public solver nor a learning result.
"""

import argparse
import hashlib
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conditional(histogram, source, outcome, state, size):
    columns = histogram["columns"]
    xi, yi = columns.index(source), columns.index(outcome)
    totals, successes = [0] * size, [0] * size
    for assignment, count in histogram["rows"]:
        x = assignment[xi]
        totals[x] += count
        successes[x] += count * (assignment[yi] == state)
    return {
        "totals": totals,
        "successes": successes,
        "probabilities": [a / b if b else None for a, b in zip(successes, totals, strict=True)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project / "src"))
    from cpt_world.query_truth import (
        backdoor_adjustment_sets,
        interventional_probability,
        worldspec_interventional_distribution,
        worldspec_projected_interventional_distribution,
    )

    report_path = args.root / "update25-validation.json"
    report = json.loads(report_path.read_text())
    assert report["complete_official_pass"]
    assert sha(args.data / "validation.parquet") == report["dataset_sha256"]
    acceptance = json.loads((args.data / "acceptance.json").read_text())
    assert sha(args.data / "accepted.jsonl") == acceptance["labels_sha256"]
    admitted = {
        row["index"]: row
        for line in (args.data / "accepted.jsonl").read_text().splitlines()
        if (row := json.loads(line))["split"] == "validation"
    }
    snapshot = args.root / "update25-complete-snapshot"
    dump = snapshot / "validation/25.jsonl"
    assert sha(dump) == report["outputs_sha256"]
    outputs = {
        hashlib.sha256(item["output"].encode()).hexdigest(): item["output"]
        for line in dump.read_bytes().splitlines()
        if (item := json.loads(line))
    }
    traces = defaultdict(list)
    for name, fingerprint in report["event_prefixes"].items():
        file = snapshot / "environment" / name
        assert sha(file) == fingerprint["sha256"]
        for line in file.read_bytes().splitlines():
            event = json.loads(line)
            if event["event"] == "act":
                traces[event["trajectory_id"]].append(event)
    records = []
    for record in report["records"]:
        if record["query_type"] != "best_intervention" or not record["completed"]:
            continue
        index = record["index"]
        truth_file = args.data / "truth" / f"validation-{index:04}.pkl"
        assert sha(truth_file) == admitted[index]["world_file_sha256"]
        row, world, seed, truth = pickle.loads(truth_file.read_bytes())
        assert row["tape_key"] == record["tape_key"]
        labels = [seed["visible_schema"]["variable_labels"][v] for v in world.variables]
        query = seed["query"]
        source, outcome = query["decision_target"], query["outcome"]
        x, y = labels.index(source), labels.index(outcome)
        state = int(query["outcome_state"].removeprefix("state_"))
        assert query["objective"] == "minimize"
        trace = traces[record["request_id"]]
        assert trace[-1]["completed"] and trace[-1]["command"] == record["trajectory"]["answer"]
        observation_batches, interventions, natural_outcome_batches = [], [], []
        for event in trace[:-1]:
            feedback = json.loads(event["feedback"].splitlines()[0])
            if feedback["type"] != "batch_result":
                continue
            histogram = feedback["batch"]["joint_histogram"]
            command = event["command"]
            if command["type"] == "observe" and outcome in histogram["columns"]:
                yi = histogram["columns"].index(outcome)
                count = sum(n for values, n in histogram["rows"] if values[yi] == state)
                natural_outcome_batches.append(
                    {
                        "columns": histogram["columns"],
                        "successes": count,
                        "n": feedback["batch"]["n"],
                    }
                )
            if command["type"] == "observe" and {source, outcome} <= set(histogram["columns"]):
                result = conditional(histogram, source, outcome, state, world.domains[x])
                result["n"] = feedback["batch"]["n"]
                assert sum(result["totals"]) == result["n"]
                result["choice"] = min(
                    range(world.domains[x]), key=lambda j: result["probabilities"][j]
                )
                observation_batches.append(result)
            if command["type"] == "intervene" and outcome in histogram["columns"]:
                yi = histogram["columns"].index(outcome)
                successes = sum(n for assignment, n in histogram["rows"] if assignment[yi] == state)
                interventions.append(
                    {
                        "target": command["target"],
                        "state": command["value"],
                        "successes": successes,
                        "n": feedback["batch"]["n"],
                        "probability": successes / feedback["batch"]["n"],
                        "source_also_measured": source in histogram["columns"],
                    }
                )
        score = record["trajectory"]["terminal_score"]
        choice = int(trace[-1]["command"]["value"].removeprefix("state_"))
        assert list(truth["candidate_probabilities"]) == score["candidate_probabilities"]
        result = {
            "index": index,
            "tape_key": record["tape_key"],
            "query": query,
            "truth_file_sha256": sha(truth_file),
            "output_sha256": record["output_sha256"],
            "chosen": choice,
            "optimal": truth["value"],
            "regret": score["regret"],
            "population_observation_choice": score["observational_choice"],
            "population_observation": score["observational_shortcut"],
            "causal_probabilities": truth["candidate_probabilities"],
            "source_is_manipulable": seed["manipulability"][world.variables[x]],
            "source_parents": [labels[p] for p in world.parents[x]],
            "observation_batches": observation_batches,
            "interventions": interventions,
            "natural_outcome_batches": natural_outcome_batches,
        }
        if index == 18:
            output_text = outputs[record["output_sha256"]]
            formula_text = "P(TVQ=1 | do(SSD=s)) * P(SSD=s | LLM=0)"
            assert formula_text in output_text
            observed_sx = []
            observed_xz = []
            for event in trace[:-1]:
                feedback = json.loads(event["feedback"].splitlines()[0])
                if feedback["type"] != "batch_result" or event["command"]["type"] != "observe":
                    continue
                histogram = feedback["batch"]["joint_histogram"]
                if set(histogram["columns"]) == {source, "SSD"}:
                    observed_sx.append(histogram)
                if {source, outcome, "JET"} <= set(histogram["columns"]):
                    observed_xz.append(histogram)
            assert len(observed_sx) == len(observed_xz) == 1
            empirical_s_given_x = [
                conditional(observed_sx[0], source, "SSD", value, world.domains[x])["probabilities"]
                for value in range(3)
            ]
            empirical_response = {
                int(item["state"].removeprefix("state_")): item["probability"]
                for item in interventions
                if item["target"] == "SSD"
            }
            assert set(empirical_response) == {0, 1, 2}
            empirical_wrong = [
                sum(
                    empirical_response[value] * empirical_s_given_x[value][xx] for value in range(3)
                )
                for xx in range(world.domains[x])
            ]
            xz_hist = observed_xz[0]
            xz_counts = [[0] * 3 for _ in range(world.domains[x])]
            for values, count in xz_hist["rows"]:
                xz_counts[values[xz_hist["columns"].index(source)]][
                    values[xz_hist["columns"].index("JET")]
                ] += count
            s, z = labels.index("SSD"), labels.index("JET")
            assert world.path_exists(s, x) and not world.path_exists(x, s)
            assert world.path_exists(s, y) and world.parents[y] == (x, z)
            sx = dict(worldspec_projected_interventional_distribution(world, {}, (s, x)))
            px = [
                sum(p for (ss, xx), p in sx.items() if xx == value)
                for value in range(world.domains[x])
            ]
            response = [
                float(interventional_probability(world, {s: value}, y, state))
                for value in range(world.domains[s])
            ]
            wrong = [
                sum(response[ss] * sx[ss, value] / px[value] for ss in range(world.domains[s]))
                for value in range(world.domains[x])
            ]
            zx_y = dict(worldspec_projected_interventional_distribution(world, {}, (z, x, y)))
            pz = [
                sum(p for (zz, xx, yy), p in zx_y.items() if zz == value)
                for value in range(world.domains[z])
            ]
            corrected = []
            for value in range(world.domains[x]):
                corrected.append(
                    sum(
                        pz[zz]
                        * zx_y[zz, value, state]
                        / sum(zx_y[zz, value, yy] for yy in range(world.domains[y]))
                        for zz in range(world.domains[z])
                    )
                )
            valid_sets = backdoor_adjustment_sets(world, world.variables[x], world.variables[y])
            assert (world.variables[z],) in valid_sets
            assert math.prod(world.domains) < 100000
            reference = [
                float(
                    sum(
                        p
                        for assignment, p in worldspec_interventional_distribution(
                            world, {x: value}
                        )
                        if assignment[y] == state
                    )
                )
                for value in range(world.domains[x])
            ]
            assert max(abs(a - b) for a, b in zip(corrected, reference, strict=True)) < 1e-12
            assert (
                max(
                    abs(a - b)
                    for a, b in zip(reference, truth["candidate_probabilities"], strict=True)
                )
                < 1e-12
            )
            result["formula_audit"] = {
                "assumed_mediator": "SSD",
                "actual_role": "ancestor_of_source_and_outcome",
                "edges": [[labels[a], labels[b]] for a, b in world.edges],
                "model_formula_substring_verified": formula_text,
                "wrong_empirical_functional_from_recorded_counts": empirical_wrong,
                "observed_source_by_jet_cell_counts": xz_counts,
                "wrong_population_functional": wrong,
                "wrong_population_choice": min(range(len(wrong)), key=wrong.__getitem__),
                "valid_adjustment": ["JET"],
                "adjusted_population_values": corrected,
                "independent_full_joint_do_values": reference,
                "scope": "Hidden-world diagnosis after the recorded answer, "
                "not a public solver or a model result.",
            }
        records.append(result)
    assert [r["index"] for r in records] == [8, 13, 18, 23]
    output = {
        "script_sha256": sha(Path(__file__)),
        "validation_report_sha256": sha(report_path),
        "records": records,
        "scope": "All four completed update25 best-intervention tasks. Recorded finite "
        "observations, model choices and hidden-world formula diagnostics are kept distinct. "
        "No learning gain or population prevalence is inferred.",
    }
    with args.output.open("x") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"records": len(records), "formula_audit": records[2]["formula_audit"]}))


if __name__ == "__main__":
    main()
