"""Paired, world-clustered comparison of frozen legal-experiment answers."""

import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
source = json.loads(gzip.decompress((ROOT / "rescored-answers.json.gz").read_bytes()))
episodes = source["episodes"]
assert len(episodes) == 100
methods = ("active_adjusted", "full_budget_passive", "same_samples_passive", "constant_state_0")


def values(row, method):
    scored = row["evaluated"][method]
    return [
        scored["reward"],
        float(scored["diagnostic"]["optimal_action"]),
        scored["diagnostic"]["regret"],
        scored["diagnostic"]["normalized_regret"],
    ]


groups = {}
for name, condition in (
    ("all", lambda r: True),
    ("strong_discordant", lambda r: r["strong_discordant"]),
    ("concordant", lambda r: not r["strong_discordant"]),
):
    rows = [row for row in episodes if condition(row)]
    worlds = sorted({row["seed_id"] for row in rows})
    cube = np.array(
        [
            [
                np.mean([values(row, method) for row in rows if row["seed_id"] == world], axis=0)
                for method in methods
            ]
            for world in worlds
        ]
    )
    rng = np.random.default_rng(20260907)
    indices = rng.integers(len(worlds), size=(20000, len(worlds)))
    gain = cube[:, 0] - cube[:, 1]
    bootstrap = gain[indices].mean(1)
    groups[name] = {
        "worlds": len(worlds),
        "episodes": len(rows),
        "method_mean": {
            method: dict(
                zip(
                    ("quality", "optimal_rate", "regret", "normalized_regret"),
                    cube[:, i].mean(0).tolist(),
                    strict=True,
                )
            )
            for i, method in enumerate(methods)
        },
        "active_minus_full_passive_paired_mean_ci": {
            field: [
                float(gain[:, i].mean()),
                *np.quantile(bootstrap[:, i], [0.025, 0.975]).tolist(),
            ]
            for i, field in enumerate(("quality", "optimal_rate", "regret", "normalized_regret"))
        },
        "valid_adjustment_episodes": sum(row["adjustment_valid"] for row in rows),
        "correct_with_valid_adjustment": sum(
            row["adjustment_valid"]
            and row["evaluated"]["active_adjusted"]["diagnostic"]["optimal_action"]
            for row in rows
        ),
        "wrong_with_valid_adjustment": sum(
            row["adjustment_valid"]
            and not row["evaluated"]["active_adjusted"]["diagnostic"]["optimal_action"]
            for row in rows
        ),
        "wrong_with_invalid_adjustment": sum(
            not row["adjustment_valid"]
            and not row["evaluated"]["active_adjusted"]["diagnostic"]["optimal_action"]
            for row in rows
        ),
    }

valid_wrong = []
for row in episodes:
    d = row["evaluated"]["active_adjusted"]["diagnostic"]
    if row["adjustment_valid"] and not d["optimal_action"]:
        optimum = d["optimal"]["value"]
        valid_wrong.append(
            {
                "seed_id": row["seed_id"],
                "replicate": row["replicate"],
                "regret": d["regret"],
                "quality": row["evaluated"]["active_adjusted"]["reward"],
                "selected_variables": len(row["policy"]["selected"]),
                "observed_strata": row["policy"]["observed_strata"],
                "samples": row["policy"]["observation_samples"],
                "true_best_missing_stratum_mass": row["policy"]["unobserved_stratum_mass"][optimum],
                "true_best_below_five_mass": row["policy"]["below_five_samples_stratum_mass"][
                    optimum
                ],
            }
        )
report = {
    "scope": "One predeclared public-only heuristic, 50 existing production worlds, two tapes "
    "each. Confidence intervals resample worlds, averaging their tapes; they are diagnostic "
    "cohort intervals, not model learning or a worst-case guarantee.",
    "groups": groups,
    "valid_adjustment_wrong_cases": valid_wrong,
    "cache_drift": {
        "max_probability_absolute_drift": max(
            abs(r["probability_drift"]) for r in source["cache_drift"]
        ),
        "changed_optimal_states": sum(r["optimal_state_changed"] for r in source["cache_drift"]),
    },
    "all_public_replays_passed": all(r["public_only_replay_passed"] for r in episodes),
}
(ROOT / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"groups": groups, "valid_wrong": valid_wrong}))
