"""Execute a fixed public-only policy on all frozen real decision tasks, then diagnose."""

import argparse
import gzip
import hashlib
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np
from public_policy import adjusted_values, parse_prompt, solve


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(x) for x in value]
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def check_reference():
    # Independent Simpson reversal table. Exact population standardization is
    # [.55, .45], whereas the unadjusted values are [.27, .73].
    counts = {
        (0, 0, 0): 720,
        (0, 1, 0): 180,
        (1, 0, 0): 90,
        (1, 1, 0): 10,
        (0, 0, 1): 10,
        (0, 1, 1): 90,
        (1, 0, 1): 180,
        (1, 1, 1): 720,
    }
    result = adjusted_values(counts, 2, 2, 1, alpha=0)
    assert np.allclose(
        result["values"], [float(Fraction(11, 20)), float(Fraction(9, 20))], rtol=0, atol=1e-15
    )
    assert np.allclose(result["passive_values_same_samples"], [0.27, 0.73], rtol=0, atol=1e-15)
    assert np.argmax(adjusted_values(counts, 2, 2, 1)["values"]) == 0
    # Relabel both action and stratum states: the conclusion must relabel too.
    swapped = {(1 - x, y, 1 - z): n for (x, y, z), n in counts.items()}
    assert np.argmax(adjusted_values(swapped, 2, 2, 1)["values"]) == 1


def worker(task):
    from cpt_world import OutcomeTape, WorldSpecEpisode
    from cpt_world.query_truth import nearest_backdoor_adjustment_set
    from cpt_world.task_scoring import score_terminal_answer

    path_string, replicate, directory_string = task
    path, directory = Path(path_string), Path(directory_string)
    started = time.monotonic()
    with path.open("rb") as f:
        row, world, seed, truth = pickle.load(f)
    tape_key = f"finite-budget-audit:{seed['seed_id']}:{replicate}"
    episode = WorldSpecEpisode(world, seed, OutcomeTape(tape_key), terminal_truth=truth)
    prompt = episode.initial_messages()[1]["content"]
    transcript = []

    def act(command):
        # Return only the public protocol text; never expose terminal truth.
        step = episode.step(json.dumps(command))
        transcript.append({"command": command, "feedback": step.message})
        assert step.kind == "batch", step.message
        return json.loads(step.message.splitlines()[0])

    policy = solve(prompt, act)
    # Lock the complete answer before accessing any diagnostic oracle.
    stem = f"{path.parent.name}-{path.stem}-r{replicate}"
    public = {"prompt": prompt, "tape_key": tape_key, "transcript": transcript, "policy": policy}
    public_bytes = json.dumps(public, allow_nan=False).encode()
    (directory / (stem + ".public.json.gz")).write_bytes(gzip.compress(public_bytes, mtime=0))
    terminal = episode.step(json.dumps(policy["answer"]))
    assert episode.completed
    assert episode.observations_used <= episode.budget.max_observations

    # Re-run with only saved public bytes: there is no episode/world/score input.
    replay = iter(json.loads(public_bytes)["transcript"])

    def replay_act(command):
        event = next(replay)
        assert command == event["command"]
        return json.loads(event["feedback"].splitlines()[0])

    assert solve(prompt, replay_act) == policy
    assert next(replay, None) is None

    p = parse_prompt(prompt)
    label_to_node = {
        label: world.variables.index(name)
        for name, label in seed["visible_schema"]["variable_labels"].items()
    }
    x, y = label_to_node[p["x"]], label_to_node[p["y"]]
    selected = [label_to_node[z] for z in policy["selected"]]
    distance, nearest = nearest_backdoor_adjustment_set(world, x, y, selected)
    same_passive = score_terminal_answer(
        json.dumps(policy["same_samples_passive_answer"]), seed, world, terminal_truth=truth
    )
    result = {
        "seed_id": seed["seed_id"],
        "replicate": replicate,
        "task_file": str(path),
        "query_type": row["query_type"],
        "budget": p["budget"],
        "observations_used": episode.observations_used,
        "tool_calls": len(transcript),
        "policy": policy,
        "reward": float(terminal.reward),
        "diagnostic": terminal.score,
        "same_samples_passive_score": same_passive,
        "adjustment_edit_distance": distance,
        "adjustment_valid": distance == 0,
        "nearest_valid_set": nearest,
        "selected_nonancestors": [
            z
            for z, node in zip(policy["selected"], selected, strict=True)
            if not world.path_exists(node, x)
        ],
        "missed_parents_of_x": [node for node in world.parents[x] if node not in selected],
        "public_only_replay_passed": True,
        "public_transcript_sha256": hashlib.sha256(public_bytes).hexdigest(),
        "seconds": time.monotonic() - started,
    }
    (directory / (stem + ".result.json")).write_text(json.dumps(clean(result), indent=2) + "\n")
    return clean(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--replicates", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    check_reference()
    tasks = []
    for cohort in ("cohort-a", "cohort-b"):
        for path in sorted(
            (args.cohort_root / cohort).glob("task-*.pkl"), key=lambda p: int(p.stem.split("-")[1])
        ):
            with path.open("rb") as f:
                row, _, _, _ = pickle.load(f)
            if row["query_type"] == "best_intervention":
                tasks.append(path)
    tasks = tasks[: args.limit]
    work = [
        (str(path), r, str(args.output))
        for path in tasks
        for r in range(args.replicates)
        if not (args.output / f"{path.parent.name}-{path.stem}-r{r}.result.json").exists()
    ]
    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "episodes_remaining": len(work),
                "reference_checks": "passed",
                "policy": "fixed-screen-adjust-v1",
            }
        ),
        flush=True,
    )
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, item): item for item in work}
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
            except Exception as error:
                failed += 1
                with (args.output / "errors.jsonl").open("a") as f:
                    f.write(json.dumps({"task": item, "error": repr(error)}) + "\n")
                print("FAILED", item[:2], repr(error), flush=True)
                continue
            print(
                json.dumps(
                    {
                        "seed_id": result["seed_id"],
                        "replicate": result["replicate"],
                        "quality": result["reward"],
                        "valid_adjustment": result["adjustment_valid"],
                        "optimal_action": result["diagnostic"]["optimal_action"],
                        "seconds": result["seconds"],
                    }
                ),
                flush=True,
            )
    if failed:
        raise RuntimeError(f"{failed} diagnostic episodes failed; inspect the preserved errors")


if __name__ == "__main__":
    main()
