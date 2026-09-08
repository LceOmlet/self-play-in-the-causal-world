"""Audit observation-rank shortcuts on frozen tasks and legal sample streams.

This is a CPU environment diagnostic, never a training or reward implementation.
The rank rule reads only the public objective/domain and natural observations.
World/truth access is confined to the experiment owner and post-answer scoring.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import pickle
import sys
import tarfile
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def ranks(values, maximize):
    return sorted(range(len(values)), key=lambda i: -values[i] if maximize else values[i])


def json_default(value):
    if isinstance(value, Fraction):
        return float(value)
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(args.project / "src"), str(args.project)]
    import numpy as np

    from cpt_world import OutcomeTape, WorldSpecEpisode
    from cpt_world.query_truth import compute_query_truth
    from cpt_world.rewards import terminal_quality_reward
    from cpt_world.task_scoring import score_terminal_answer

    old = args.project / "research/active_decision_diagnostic_20260907"
    fingerprints = json.loads((old / "SHA256SUMS.json").read_text())
    inputs = {}
    for name in [
        "frozen-worlds.tar.gz",
        "evidence.tar.gz",
        "rescored-answers.json.gz",
        "public_policy.py",
    ]:
        inputs[name] = digest((old / name).read_bytes())
        assert inputs[name] == fingerprints[name]
    spec = importlib.util.spec_from_file_location(
        "existing_public_policy", old / "public_policy.py"
    )
    public_policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(public_policy)
    episodes = json.loads(gzip.decompress((old / "rescored-answers.json.gz").read_bytes()))[
        "episodes"
    ]
    assert len(episodes) == 100 and set(Counter(e["seed_id"] for e in episodes).values()) == {2}
    frozen = tarfile.open(old / "frozen-worlds.tar.gz")
    evidence = tarfile.open(old / "evidence.tar.gz")
    baseline_bytes = frozen.extractfile("finite-baselines.jsonl").read()
    baseline_rows = [json.loads(line) for line in baseline_bytes.splitlines()]
    historical = {
        (row["seed_id"], item["replicate"], item["sample_count"]): item
        for row in baseline_rows
        if row["query_type"] == "best_intervention"
        for item in row["records"]
    }
    world_cache, records = {}, []
    started = time.monotonic()
    with (args.output / "records.jsonl").open("x") as output:
        for entry in episodes:
            public_bytes = gzip.decompress(
                evidence.extractfile("results/" + entry["public_file"]).read()
            )
            assert digest(public_bytes) == entry["public_transcript_sha256"]
            public = json.loads(public_bytes)
            p = public_policy.parse_prompt(public["prompt"])
            assert "SAMPLED-" not in public["prompt"]
            name = entry["public_file"].removesuffix(".public.json.gz")
            cohort, suffix = name.split("-task-")
            index, replicate = suffix.split("-r")
            assert int(replicate) == entry["replicate"]
            world_name = f"{cohort}/task-{index}.pkl"
            if world_name not in world_cache:
                world_bytes = frozen.extractfile(world_name).read()
                row, world, seed, _ = pickle.loads(world_bytes)
                fresh = compute_query_truth(world, seed)
                world_cache[world_name] = (world, seed, fresh, digest(world_bytes))
            world, seed, truth, world_hash = world_cache[world_name]
            assert seed["seed_id"] == entry["seed_id"]
            assert p["x"] == seed["query"]["decision_target"] and p["y"] == seed["query"]["outcome"]
            assert p["maximize"] == (seed["query"]["objective"] == "maximize")
            assert p["budget"] == seed["observation_budget"]
            original = entry["evaluated"]["active_adjusted"]["diagnostic"]
            assert truth["value"] == original["optimal"]["value"]
            assert (
                max(
                    abs(a - b)
                    for a, b in zip(
                        truth["candidate_probabilities"],
                        original["candidate_probabilities"],
                        strict=True,
                    )
                )
                < 1e-12
            )

            def evaluate(answer, seed=seed, world=world):
                score = score_terminal_answer(json.dumps(answer), seed, world)
                return {
                    "answer": answer,
                    "quality": float(terminal_quality_reward(score)),
                    "metrics": score,
                }

            # Recompute the old marginal from the archived natural histograms.
            counts = np.zeros((p["domains"][p["x"]], p["domains"][p["y"]]))
            for event in public["transcript"]:
                if event["command"]["type"] != "observe":
                    continue
                feedback = json.loads(event["feedback"].splitlines()[0])
                histogram = feedback["batch"]["joint_histogram"]
                xi, yi = histogram["columns"].index(p["x"]), histogram["columns"].index(p["y"])
                for values, count in histogram["rows"]:
                    counts[values[xi], values[yi]] += count
            values = (counts[:, p["target"]] + 0.5) / (counts.sum(1) + 0.5 * counts.shape[1])
            assert np.allclose(
                values, entry["policy"]["passive_values_same_samples"], rtol=0, atol=1e-12
            )
            ordering = ranks(values, p["maximize"])
            assert entry["policy"]["same_samples_passive_answer"]["value"] == f"state_{ordering[0]}"
            same_second = evaluate({"type": "answer", "value": f"state_{ordering[1]}"})

            # Replay the original pure-observation stream and all its checkpoints.
            tape_key = f"finite-budget-audit:{seed['seed_id']}:{entry['replicate']}"
            episode = WorldSpecEpisode(world, seed, OutcomeTape(tape_key), terminal_truth=truth)
            counts = np.zeros_like(counts)
            consumed, checkpoints = 0, []
            for target_n in sorted(
                {min(128, p["budget"] // 2), min(2048, p["budget"] // 2), p["budget"] // 2}
            ):
                command = {
                    "type": "observe",
                    "measure": [p["x"], p["y"]],
                    "batch_size": target_n - consumed,
                }
                result = episode.step(json.dumps(command))
                assert result.kind == "batch"
                feedback = json.loads(result.message.splitlines()[0])
                histogram = feedback["batch"]["joint_histogram"]
                assert histogram["columns"] == [p["x"], p["y"]]
                for values, count in histogram["rows"]:
                    counts[tuple(values)] += count
                consumed = target_n
                values = (counts[:, p["target"]] + 0.5) / (counts.sum(1) + 0.5 * counts.shape[1])
                ordering = ranks(values, p["maximize"])
                first_answer = {"type": "answer", "value": f"state_{ordering[0]}"}
                prior = historical[(seed["seed_id"], entry["replicate"], consumed)]
                assert first_answer == prior["answer"]
                assert int(counts.sum()) == consumed and episode.observations_used == 2 * consumed
                checkpoints.append(
                    {
                        "sample_count": consumed,
                        "command": command,
                        "feedback": result.message,
                        "counts": counts.astype(int).tolist(),
                        "observational_values": values.tolist(),
                        "original_first_answer_replayed": first_answer,
                        "second_rank": evaluate(
                            {"type": "answer", "value": f"state_{ordering[1]}"}
                        ),
                    }
                )
            terminal = episode.step(json.dumps(checkpoints[-1]["second_rank"]["answer"]))
            assert abs(float(terminal.reward) - checkpoints[-1]["second_rank"]["quality"]) < 1e-12
            assert episode.observations_used <= p["budget"]
            record = {
                "seed_id": seed["seed_id"],
                "replicate": entry["replicate"],
                "world": world_name,
                "world_sha256": world_hash,
                "public_transcript_sha256": digest(public_bytes),
                "strong": entry["strong_discordant"],
                "domain_size": counts.shape[0],
                "public_query": p,
                "tape_key": tape_key,
                "budget": p["budget"],
                "active_correct": original["optimal_action"],
                "same_samples_first_correct": entry["evaluated"]["same_samples_passive"][
                    "diagnostic"
                ]["optimal_action"],
                "original_full_budget_first_correct": entry["evaluated"]["full_budget_passive"][
                    "diagnostic"
                ]["optimal_action"],
                "same_samples_second_rank": same_second,
                "pure_observation_checkpoints": checkpoints,
            }
            output.write(json.dumps(record, allow_nan=False, default=json_default) + "\n")
            output.flush()
            records.append(record)
            if len(records) % 10 == 0:
                print(
                    json.dumps({"completed": len(records), "seconds": time.monotonic() - started}),
                    flush=True,
                )
    frozen.close()
    evidence.close()

    # Existing submitted 100/5 best-intervention worlds: exact ranking diagnostic.
    accepted_path = args.training_data / "accepted.jsonl"
    accepted = [json.loads(line) for line in accepted_path.read_bytes().splitlines()]
    population = []
    for record in accepted:
        if record["query_type"] != "best_intervention":
            continue
        path = args.training_data / "truth" / f"{record['split']}-{record['index']:04d}.pkl"
        assert digest(path.read_bytes()) == record["world_file_sha256"]
        row, _, seed, truth = pickle.loads(path.read_bytes())
        assert row["tape_key"] == record["tape_key"]
        score = record["oracle_score"]
        assert list(truth["candidate_probabilities"]) == score["candidate_probabilities"]
        observed = score["observational_shortcut"]
        ordering = ranks(observed, seed["query"]["objective"] == "maximize")
        optimal = score["optimal"]["value"]
        assert ordering[0] == score["observational_choice"]
        population.append(
            {
                "split": record["split"],
                "index": record["index"],
                "strong": record["discordant"],
                "world_sha256": record["world_file_sha256"],
                "domain_size": len(observed),
                "public_objective": seed["query"]["objective"],
                "observational_values": observed,
                "causal_values": score["candidate_probabilities"],
                "optimal": optimal,
                "observational_order": ordering,
                "optimal_observational_rank": ordering.index(optimal) + 1,
                "observational_margin_to_causal_optimum": abs(
                    observed[optimal] - observed[ordering[0]]
                ),
                "causal_loss_of_observational_best": score["observational_shortcut_error"],
            }
        )
    assert len(population) == 105
    summaries = {}
    for label, subset in [
        ("all", records),
        ("strong", [r for r in records if r["strong"]]),
        ("concordant", [r for r in records if not r["strong"]]),
    ]:
        summaries[label] = {
            "episodes": len(subset),
            "worlds": len({r["seed_id"] for r in subset}),
            "active_correct": sum(r["active_correct"] for r in subset),
            "same_samples_first_correct": sum(r["same_samples_first_correct"] for r in subset),
            "same_samples_second_correct": sum(
                r["same_samples_second_rank"]["metrics"]["optimal_action"] for r in subset
            ),
            "pure_observation_first_correct": sum(
                r["original_full_budget_first_correct"] for r in subset
            ),
            "pure_observation_second_correct": sum(
                r["pure_observation_checkpoints"][-1]["second_rank"]["metrics"]["optimal_action"]
                for r in subset
            ),
        }
    summary = {
        "script_sha256": digest(Path(__file__).read_bytes()),
        "inputs_sha256": inputs,
        "accepted_tasks_sha256": digest(accepted_path.read_bytes()),
        "finite_baselines_sha256": digest(baseline_bytes),
        "records_sha256": digest((args.output / "records.jsonl").read_bytes()),
        "replayed_original_answers": sum(len(r["pure_observation_checkpoints"]) for r in records),
        "cohort_results": summaries,
        "submitted_population_rank_diagnostic": population,
        "scope": "Post-hoc diagnostic on previously analyzed cohorts, 50 worlds/two tapes. "
        "Pure observation replays all old checkpoints before selecting rank2 "
        "and submitting through production. No claim of unbiased new-heldout generalization, "
        "model learning, or an optimal causal policy.",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False, default=json_default) + "\n"
    )
    print(
        json.dumps(
            {
                "cohort_results": summaries,
                "original_answers_replayed": summary["replayed_original_answers"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
