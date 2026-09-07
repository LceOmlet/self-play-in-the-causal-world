"""Summarize the existing diagnostic cohort; no resampling or training changes."""

import gzip
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return gzip.decompress((ROOT / (name + ".gz")).read_bytes()).decode("utf-8")


summary = json.loads(read("environment-summary.json"))
profiles = [
    json.loads(line)
    for name in ("cohort-a-profiles.jsonl", "cohort-b-profiles.jsonl")
    for line in read(name).splitlines()
]
assert len(profiles) == 250 == len({row["seed_id"] for row in profiles})
assert set(Counter(row["query_type"] for row in profiles).values()) == {50}
constants = {
    "ate": "constant_zero",
    "individual_counterfactual_probability": "constant_zero",
    "backadj_minimal_sets": "empty_set",
    "best_intervention": "constant_state_0",
    "mediator_set": "empty_structure",
}
families = {}
for family, baseline in constants.items():
    rows = [row for row in profiles if row["query_type"] == family]
    aggregate = summary["families"][family]
    constant = [row["scores"][baseline]["reward"] for row in rows]
    assert (
        abs(statistics.mean(constant) - aggregate["baselines"][baseline]["reward"]["mean"]) < 1e-12
    )
    item = {
        "worlds": len(rows),
        "constant_name": baseline,
        "constant_quality_mean": statistics.mean(constant),
        "constant_quality_at_least_0.9_count": sum(q >= 0.9 for q in constant),
        "constant_quality_at_least_0.95_count": sum(q >= 0.95 for q in constant),
        "finite_heuristic_quality_mean_ci": aggregate["finite_reward_mean_ci"],
        "finite_heuristic_paired_gain_over_constant_mean_ci": aggregate[
            "paired_reward_gain_over_" + baseline
        ],
        "finite_heuristic_task_metric_mean_ci": {
            key: value["world_mean_ci"] for key, value in aggregate["finite_metrics"].items()
        },
    }
    if "population_observational_shortcut" in aggregate["baselines"]:
        observation = [row["scores"]["population_observational_shortcut"]["reward"] for row in rows]
        item["population_observational_quality_mean"] = statistics.mean(observation)
        item["constant_quality_higher_than_population_observation_count"] = sum(
            a > b for a, b in zip(constant, observation, strict=True)
        )
    if family == "ate":
        item["effect_tv_below_0.01_count"] = sum(
            row["features"]["effect_tv"] < 0.01 for row in rows
        )
    if family == "backadj_minimal_sets":
        item["empty_set_valid_count"] = sum(
            row["scores"][baseline]["diagnostic"]["valid_adjustment_set"] for row in rows
        )
    if family == "best_intervention":
        item["population_observational_optimal_count"] = sum(
            row["scores"]["population_observational_shortcut"]["diagnostic"]["optimal_action"]
            for row in rows
        )
        item["constant_optimal_count"] = sum(
            row["scores"][baseline]["diagnostic"]["optimal_action"] for row in rows
        )
    families[family] = item

report = {
    "scope": "Existing 250-world diagnostic cohort, 50 per family; not an untouched evaluation set "
    "and not model learning evidence. Two finite-sample tapes per world are clustered "
    "in the archived confidence intervals.",
    "threshold_scope": "0.9 and 0.95 are descriptive score summaries, "
    "not new acceptance thresholds or reward changes.",
    "families": families,
    "input_sha256": {
        name: hashlib.sha256(read(name).encode("utf-8")).hexdigest()
        for name in (
            "environment-summary.json",
            "cohort-a-profiles.jsonl",
            "cohort-b-profiles.jsonl",
        )
    },
}
(ROOT / "task-value.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            family: {key: value for key, value in item.items() if "metric" not in key}
            for family, item in families.items()
        }
    )
)
