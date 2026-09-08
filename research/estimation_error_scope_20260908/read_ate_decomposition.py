"""Read frozen ATE estimates; separate population-formula and finite-estimate error.

Run from any directory: python /path/to/read_ate_decomposition.py
Only Python's standard library is required. No model, task generation, scoring
solver, sampling, or original finite-program execution occurs in this reader.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROFILE_ROOT = ROOT.parent / "task_value_learning_20260907"
FIELDS = (
    "population_formula_bias_squared",
    "finite_estimate_residual_squared",
    "cross_term",
    "total_mse",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decomposition(estimate: list, observed: list, truth: list) -> dict:
    assert len(estimate) == len(observed) == len(truth)
    bias = [o - t for o, t in zip(observed, truth, strict=True)]
    residual = [h - o for h, o in zip(estimate, observed, strict=True)]
    result = {
        "population_formula_bias_squared": statistics.mean(b * b for b in bias),
        "finite_estimate_residual_squared": statistics.mean(e * e for e in residual),
        "cross_term": 2 * statistics.mean(b * e for b, e in zip(bias, residual, strict=True)),
        "total_mse": statistics.mean(
            (h - t) ** 2 for h, t in zip(estimate, truth, strict=True)
        ),
    }
    assert math.isclose(
        result["total_mse"], sum(result[field] for field in FIELDS[:3]), abs_tol=1e-12
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "summary.json")
    args = parser.parse_args()
    provenance_bytes = (ROOT / "provenance.json").read_bytes()
    provenance = json.loads(provenance_bytes)
    archive = (ROOT / "ate-finite-baselines.jsonl.gz").read_bytes()
    raw = gzip.decompress(archive)
    assert sha(archive) == provenance["selected_archive_sha256"]
    assert sha(raw) == provenance["selected_uncompressed_sha256"]
    assert sha((ROOT / provenance["finite_program_archive"]).read_bytes()) == provenance[
        "finite_program_sha256"
    ]
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == len({row["seed_id"] for row in rows}) == 50
    assert all(row["query_type"] == "ate" for row in rows)
    fingerprints = {}
    profiles = {}
    for name in ("cohort-a-profiles.jsonl.gz", "cohort-b-profiles.jsonl.gz"):
        compressed = (PROFILE_ROOT / name).read_bytes()
        content = gzip.decompress(compressed)
        fingerprints[name] = {
            "archive_sha256": sha(compressed),
            "uncompressed_sha256": sha(content),
        }
        for line in content.splitlines():
            row = json.loads(line)
            if row["query_type"] == "ate":
                assert row["seed_id"] not in profiles
                profiles[row["seed_id"]] = row
    assert set(profiles) == {row["seed_id"] for row in rows}
    worlds = []
    for row in rows:
        profile = profiles[row["seed_id"]]
        reference = profile["scores"]["population_observational_shortcut"]["diagnostic"]
        truth, observed = reference["truth"], reference["prediction"]
        assert observed == reference["observational_shortcut"]
        budget_samples = row["scalar_budget"] // 2
        expected_counts = {min(128, budget_samples), min(2048, budget_samples), budget_samples}
        assert len(row["records"]) == 2 * len(expected_counts)
        assert {
            (r["replicate"], r["sample_count"]) for r in row["records"]
        } == {(replicate, count) for replicate in (0, 1) for count in expected_counts}
        world = {"seed_id": row["seed_id"], "dimension": len(truth), "stages": {}}
        for stage in ("128", "2048", "full"):
            if stage == "full":
                selected = [r for r in row["records"] if r["full_budget"]]
            else:
                selected = [
                    r for r in row["records"]
                    if r["sample_count"] == min(int(stage), row["scalar_budget"] // 2)
                ]
            assert len(selected) == 2 and {r["replicate"] for r in selected} == {0, 1}
            assert selected[0]["sample_count"] == selected[1]["sample_count"]
            results = []
            for record in selected:
                assert record["policy"] == "passive_conditional_plugin"
                diagnostic = record["diagnostic"]
                assert diagnostic["truth"] == truth
                assert diagnostic["observational_shortcut"] == observed
                estimate = diagnostic["prediction"]
                result = decomposition(estimate, observed, truth)
                assert math.isclose(
                    result["total_mse"], diagnostic["squared_error"], abs_tol=1e-12
                )
                results.append(result)
            world["stages"][stage] = {
                "samples_per_tape": selected[0]["sample_count"],
                "tapes": 2,
                **{field: statistics.mean(r[field] for r in results) for field in FIELDS},
            }
        worlds.append(world)
    report = {
        "provenance": provenance,
        "provenance_sha256": sha(provenance_bytes),
        "reader_sha256": sha(Path(__file__).read_bytes()),
        "profile_inputs": fingerprints,
        "worlds": 50,
        "tapes_per_world": 2,
        "stages": ["128", "2048", "full"],
        "source_estimate_records": sum(len(row["records"]) for row in rows),
        "stage_references": 300,
        "stage_sample_definitions": {
            "128": "min(128, scalar_budget // 2)",
            "2048": "min(2048, scalar_budget // 2)",
            "full": "scalar_budget // 2",
        },
        "worlds_with_2048_stage_equal_to_full": sum(
            world["stages"]["2048"]["samples_per_tape"]
            == world["stages"]["full"]["samples_per_tape"]
            for world in worlds
        ),
        "aggregation": (
            "Each squared norm and inner product first averages over that world's "
            "ATE state dimension, then over its two tapes, then equally over 50 worlds. "
            "Stages are nested samples of each existing tape and are not independent runs."
        ),
        "identity": "MSE(h,truth) = MSE(o,truth) + MSE(h,o) + 2*mean((h-o)*(o-truth))",
        "stage_means": {
            stage: {
                field: statistics.mean(world["stages"][stage][field] for world in worlds)
                for field in FIELDS
            }
            for stage in ("128", "2048", "full")
        },
        "per_world": worlds,
        "scope": [
            "All 50 ATE worlds from the existing diagnostic cohort; no new worlds or samples.",
            "The named 2048 stage is capped by each world's budget. In three worlds "
            "it is already the full stage; 300 stage references reuse 294 source estimates.",
            "The finite program uses observational conditionals with fixed 0.5 smoothing.",
            "Population-formula bias measures the observational formula's mismatch with "
            "the causal ATE, not an error in the environment's scoring formula.",
            "Finite-estimate residual includes sampling and smoothing; it is not pure variance.",
            "The cross term is retained. Neither error term is presented as a fraction "
            "of total MSE or assumed to add without the cross term.",
            "This is a deterministic diagnostic program, not an LLM or evidence that "
            "LLM answers, rewards, or learning are dominated by either error source.",
            "No claim about the optimal achievable error of other legal experiment policies.",
        ],
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"source_verified": True, "stage_means": report["stage_means"]}))


if __name__ == "__main__":
    main()
