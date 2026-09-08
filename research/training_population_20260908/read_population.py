"""Recover all completed training groups before DAPO filtering from saved logs.

The currently frozen sources cover the first sampler epoch only. Each source
is bounded by its completed native-update frontier, excluding cancelled or
in-flight work beyond that frontier. Validation tapes are never training data.
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def read(path):
    return json.loads(Path(str(path).replace("\\", "/")).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(str(path).replace("\\", "/")).read_bytes()).hexdigest()


NUMERIC_FIELDS = {
    "ate": ("total_variation_error", "observational_shortcut_error"),
    "individual_counterfactual_probability": (
        "mean_absolute_endpoint_error",
        "observational_shortcut_error",
    ),
    "best_intervention": ("normalized_regret", "observational_shortcut_normalized_regret"),
}


def reference_comparison(trace):
    """Read owner errors; score references on the SAME existing task scale.

    This is offline reporting, not a reward or training implementation. The
    frozen production owner is separately replayed against these records.
    Missing answers have no numerical error and are not invented as zero.
    """
    family = trace["query_type"]
    if not trace["completed"] or family not in NUMERIC_FIELDS:
        return None
    score = trace["terminal_score"]
    assert score["reward_scalarization"] == "terminal-quality-v10"
    error_field, reference_field = NUMERIC_FIELDS[family]
    error, reference_error = score[error_field], score.get(reference_field)
    if reference_error is None:
        return None
    scale = reference_error + 1 / math.sqrt(2048)

    def quality(value):
        return scale / (scale + value)

    assert math.isclose(quality(error), trace["quality"], rel_tol=0, abs_tol=1e-12)
    result = {
        "model_error": error,
        "model_score": trace["quality"],
        "observational_error": reference_error,
        "observational_score": quality(reference_error),
    }
    if family == "best_intervention":
        # The optimum probability recovers the public objective direction;
        # it does not select the reference action. Action ranking uses ONLY
        # observational probabilities, with the owner's lowest-index tie rule.
        causal, observed = score["candidate_probabilities"], score["observational_shortcut"]
        optimum, span = score["optimal_probability"], score["probability_span"]
        assert len(causal) == len(observed) >= 2
        assert optimum in (min(causal), max(causal))
        maximize = optimum == max(causal)
        order = sorted(
            range(len(observed)), key=lambda i: ((-1 if maximize else 1) * observed[i], i)
        )
        if span > 0:
            assert order[0] == score["observational_choice"]
        second_error = abs(causal[order[1]] - optimum) / span if span else 0.0
        second_regret = abs(causal[order[1]] - optimum)
        result.update(
            model_regret=score["regret"],
            observational_regret=score["observational_shortcut_error"],
            observational_second_action=order[1],
            observational_second_error=second_error,
            observational_second_regret=second_regret,
            observational_second_score=quality(second_error),
            model_optimal=score["optimal_action"],
            observational_optimal=reference_error == 0,
            observational_second_optimal=second_regret == 0,
        )
    return result


def summarize_comparisons(groups, family):
    rows = [
        t["reference_comparison"]
        for g in groups
        if g["query_type"] == family
        for t in g["trajectories"]
        if t["reference_comparison"] is not None
    ]
    if not rows:
        return None
    fields = [k for k in rows[0] if not k.endswith("_action")]
    return {
        "paired_completed_trajectories": len(rows),
        "paired_groups": sum(
            g["query_type"] == family
            and any(t["reference_comparison"] is not None for t in g["trajectories"])
            for g in groups
        ),
        "means": {field: sum(r[field] for r in rows) / len(rows) for field in fields},
        "model_error_below_observational": sum(
            r["model_error"] < r["observational_error"] for r in rows
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve-manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, catalog = read(args.curve_manifest), read(args.catalog)
    assert catalog["sha256"] == "bcb84b4f1b915fd9d3f8f63dd116bfcca837f622a62402b135a3c8ff9fe4faf8"
    assert all(sha(p) == value for p, value in manifest["inputs"].items())
    assert all(sha(p) == value for p, value in manifest["native_progress_inputs"].items())
    sources = [read(p) for p in manifest["inputs"]]
    sources.sort(key=lambda s: min(r["step"] for r in s["retained_rollouts"]))
    metrics = {
        m["step"]: m for s in sources for m in s["official_metrics"] if "actor/grad_norm" in m
    }
    native = {d["completed_updates"]: d for d in map(read, manifest["native_progress_inputs"])}
    frontier = {0: 0}
    for step in manifest["updates"]:
        if step in metrics:
            frontier[step] = frontier[step - 1] + int(metrics[step]["train/num_gen_batches"])
            if "training/generated_batches_total" in metrics[step]:
                assert frontier[step] == metrics[step]["training/generated_batches_total"]
        else:
            frontier[step] = native[step]["generated_batches"]
        assert frontier[step] > frontier[step - 1]
    order = catalog["official_sampler_order"]
    assert max(frontier.values()) <= len(order), "This evidence reader covers the first epoch only"
    by_index = {r["index"]: r for r in catalog["rows"]}
    tape_catalog = {r["tape_key"]: r for r in catalog["rows"]}
    assert len(order) == len(by_index) == len(tape_catalog) == 500
    generation = {g + 1: by_index[index] for g, index in enumerate(order)}
    update_for_group = {
        g: step
        for step in manifest["updates"]
        for g in range(frontier[step - 1] + 1, frontier[step] + 1)
    }
    groups, exclusions = [], []
    for source in sources:
        retained = source["retained_rollouts"]
        first, last = min(r["step"] for r in retained), max(r["step"] for r in retained)
        selected = {
            generation[g]["tape_key"]: g for g in range(frontier[first - 1] + 1, frontier[last] + 1)
        }
        kept_tapes = {r["tape_key"]: r["step"] for r in retained}
        assert set(kept_tapes) <= set(selected)
        scores, traces = defaultdict(list), defaultdict(list)
        excluded = Counter()
        for score in source["raw_score_events"]:
            tape = score.get("tape_key")
            if tape not in selected:
                excluded[
                    "outside_completed_frontier" if tape in tape_catalog else "not_training_tape"
                ] += 1
                continue
            assert score["query_type"] == tape_catalog[tape]["query_type"]
            assert score["score"] == score["acc"] == score["raw_reward"]
            scores[tape].append(score)
        for trace in source["trajectories"]:
            if trace["tape_key"] in selected:
                traces[trace["tape_key"]].append(trace)
        assert set(scores) == set(traces) == set(selected)
        exclusions.append(
            {
                "run": source["run"],
                "first_update": first,
                "last_update": last,
                "excluded_owner_scores": dict(excluded),
            }
        )
        for tape, g in selected.items():
            assert len(scores[tape]) == len(traces[tape]) == 4, (g, tape)

            # The reward records lack request IDs. Verify group multisets, not
            # an invented individual match based on serialization order.
            def signature(record, quality):
                # The event reader leaves terminal quality absent on unfinished
                # traces; the actual environment gives those trajectories zero.
                value = record[quality]
                if not record["completed"]:
                    assert value in (None, 0)
                    value = 0
                return (
                    bool(record["completed"]),
                    value,
                    record["observations_used"],
                    record["queries_used"]
                    if "queries_used" in record
                    else sum(record["valid_experiments"].values()),
                )

            # raw_score.queries_used counts successful experiment batches.
            assert Counter(signature(s, "raw_reward") for s in scores[tape]) == Counter(
                signature(t, "quality") for t in traces[tape]
            ), (g, tape, "score/trajectory multiset mismatch")
            values = [s["raw_reward"] for s in scores[tape]]
            kept = tape in kept_tapes
            assert kept == (len(set(values)) > 1), (g, tape, "official acc filtering mismatch")
            if kept:
                assert kept_tapes[tape] == update_for_group[g]
                dumped = [r["raw_quality"] for r in retained if r["tape_key"] == tape]
                assert Counter(dumped) == Counter(values)
            status = (
                "retained"
                if kept
                else "filtered_all_one"
                if values[0] == 1
                else "filtered_all_zero"
                if values[0] == 0
                else "filtered_constant_partial"
            )
            groups.append(
                {
                    "generation": g,
                    "update": update_for_group[g],
                    **generation[g],
                    "retained": kept,
                    "status": status,
                    "qualities": values,
                    "completed": sum(bool(s["completed"]) for s in scores[tape]),
                    "trajectory_ids": [t["trajectory_id"] for t in traces[tape]],
                    "task_metrics": [t["task_metrics"] for t in traces[tape]],
                    "trajectories": [
                        {
                            "trajectory_id": t["trajectory_id"],
                            "completed": t["completed"],
                            "quality": t["quality"] if t["completed"] else 0.0,
                            "terminal_score": t["terminal_score"],
                            "reference_comparison": reference_comparison(t),
                        }
                        for t in traces[tape]
                    ],
                }
            )
    groups.sort(key=lambda g: g["generation"])
    assert [g["generation"] for g in groups] == list(range(1, max(frontier.values()) + 1))
    families = {}
    for family in sorted({g["query_type"] for g in groups}):
        rows = [g for g in groups if g["query_type"] == family]
        kept = [g for g in rows if g["retained"]]
        families[family] = {
            "generated_groups": len(rows),
            "retained_groups": len(kept),
            "all_mean_quality": sum(sum(g["qualities"]) for g in rows) / (4 * len(rows)),
            "retained_mean_quality": sum(sum(g["qualities"]) for g in kept) / (4 * len(kept)),
            "all_completed": sum(g["completed"] for g in rows),
            "retained_completed": sum(g["completed"] for g in kept),
            "filtered_status_counts": dict(Counter(g["status"] for g in rows if not g["retained"])),
            "reference_comparison": summarize_comparisons(groups, family),
        }
        strict_fields = {
            "backadj_minimal_sets": ("valid_adjustment_set",),
            "best_intervention": ("optimal_action",),
            "mediator_set": ("mediators_exact_match", "order_exact_match"),
        }.get(family)
        if strict_fields:
            for label, population in [("all", rows), ("retained", kept)]:
                families[family][label + "_strict_correct"] = sum(
                    m is not None and all(m[field] for field in strict_fields)
                    for g in population
                    for m in g["task_metrics"]
                )
    result = {
        "input_fingerprints": manifest["inputs"],
        "native_frontier_proofs": manifest["native_progress_inputs"],
        "catalog_sha256": sha(args.catalog),
        "reader_sha256": sha(Path(__file__)),
        "completed_updates": max(frontier),
        "generated_groups": len(groups),
        "all_training_trajectories": 4 * len(groups),
        "retained_training_trajectories": sum(g["retained"] for g in groups) * 4,
        "families": families,
        "excluded_records": exclusions,
        "groups": groups,
        "scope": "All fully scored training groups through completed native frontiers, including "
        "constant-acc filtered groups. "
        "Changing tasks and cumulative means are not paired learning gains. "
        "Only the first data epoch is covered; no new inference or training implementation.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ["groups", "input_fingerprints"]})
    )


if __name__ == "__main__":
    main()
