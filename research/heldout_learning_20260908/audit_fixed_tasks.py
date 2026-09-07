"""Score fixed public constants and population-observation diagnostics offline.

Uses the production environment/scorer and original frozen validation tasks.
No actions, truths or altered rewards are passed to the training process.
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    sys.path.insert(0, str(args.project / "src"))
    import pyarrow.parquet as pq

    from cpt_world import CPTWorldEnvironment

    accepted = json.loads((args.data / "acceptance.json").read_text())
    labels_file = args.data / "accepted.jsonl"
    assert sha(labels_file) == accepted["labels_sha256"]
    frozen_labels = {
        record["index"]: record
        for line in labels_file.read_text().splitlines()
        if (record := json.loads(line))["split"] == "validation"
    }
    val_file = args.data / "validation.parquet"
    assert sha(val_file) == accepted["sha256"]["validation.parquet"]
    rows = pq.read_table(val_file).to_pylist()
    assert len(rows) == 25
    records = []
    for index, data in enumerate(rows):
        extra = data["extra_info"]
        row = json.loads(extra["tools_kwargs"]["act"]["create_kwargs"]["row_json"])
        path = args.data / "truth" / f"validation-{index:04}.pkl"
        assert sha(path) == frozen_labels[index]["world_file_sha256"]
        frozen_row, _, seed, truth = pickle.loads(path.read_bytes())
        assert frozen_row == row and row["tape_key"] == extra["tape_key"]
        labels = seed["visible_schema"]["variable_labels"]
        family = row["query_type"]

        def score(answer, task_row=row, task_index=index):
            environment = CPTWorldEnvironment()
            environment.reset(**task_row)
            environment.act(answer)
            assert environment.episode.completed, (task_index, answer)
            return {
                "answer": answer,
                "quality": environment.get_reward(),
                "metrics": clean(environment.episode.terminal_score),
            }

        if family == "ate":
            oracle = {
                "type": "answer",
                "effect": {f"state_{j}": float(v) for j, v in enumerate(truth["effect"])},
            }
            constants = {
                "zero_effect": {
                    "type": "answer",
                    "effect": {f"state_{j}": 0.0 for j in range(len(truth["effect"]))},
                }
            }
        elif family == "individual_counterfactual_probability":
            oracle = {
                "type": "answer",
                "lower": max(0, float(truth["lower"])),
                "upper": min(1, float(truth["upper"])),
            }
            constants = {
                "zero_interval": {"type": "answer", "lower": 0.0, "upper": 0.0},
                "ignorance_interval": {"type": "answer", "lower": 0.0, "upper": 1.0},
            }
        elif family == "best_intervention":
            oracle = {"type": "answer", "value": f"state_{truth['value']}"}
            constants = {"state_0": {"type": "answer", "value": "state_0"}}
        elif family == "mediator_set":
            oracle = {
                "type": "answer",
                "mediators": [labels[v] for v in truth["mediators"]],
                "order": [[labels[a], labels[b]] for a, b in truth["order"]],
            }
            constants = {"empty_structure": {"type": "answer", "mediators": [], "order": []}}
        elif family == "backadj_minimal_sets":
            oracle = None  # Frozen submission already checked a valid set.
            constants = {"empty_set": {"type": "answer", "adjustment_set": []}}
        else:
            raise AssertionError(family)
        baselines = {name: score(answer) for name, answer in constants.items()}
        if oracle is not None:
            assert abs(score(oracle)["quality"] - 1) < 1e-10
        first = next(iter(baselines.values()))
        if family in {"ate", "best_intervention"}:
            observation = first["metrics"]["observational_shortcut"]
            assert observation is not None
            if family == "ate":
                answer = {
                    "type": "answer",
                    "effect": {f"state_{j}": v for j, v in enumerate(observation)},
                }
            else:
                select = max if seed["query"]["objective"] == "maximize" else min
                chosen = select(range(len(observation)), key=lambda j: observation[j])
                answer = {"type": "answer", "value": f"state_{chosen}"}
            baselines["population_observation"] = score(answer)
        records.append(
            {
                "index": index,
                "tape_key": row["tape_key"],
                "query_type": family,
                "query": seed["query"],
                "truth_file_sha256": sha(path),
                "truth": clean(truth),
                "baselines": baselines,
            }
        )
    groups = defaultdict(list)
    for record in records:
        groups[record["query_type"]].append(record)
    summary = {}
    for family, items in groups.items():
        assert len(items) == 5
        summary[family] = {}
        for name in items[0]["baselines"]:
            results = [item["baselines"][name] for item in items]
            metrics = {}
            for key in [
                "total_variation_error",
                "mean_absolute_endpoint_error",
                "regret",
                "normalized_regret",
                "edit_distance",
                "valid_adjustment_set",
                "optimal_action",
                "mediators_exact_match",
                "order_exact_match",
            ]:
                if key in results[0]["metrics"]:
                    metrics[key] = [r["metrics"][key] for r in results]
            summary[family][name] = {
                "quality_mean": sum(r["quality"] for r in results) / len(results),
                "quality_values": [r["quality"] for r in results],
                "metrics": metrics,
            }
    report = {
        "validation_sha256": sha(val_file),
        "script_sha256": sha(Path(__file__)),
        "records": records,
        "summary": summary,
        "scope": "All original 25 fixed validation tasks, no selection by model result. "
        "Constants use the public schema only. Population-observation values use hidden "
        "world probabilities, not finite-budget public solutions or model performance. "
        "No claim about population prevalence or learning gains.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
