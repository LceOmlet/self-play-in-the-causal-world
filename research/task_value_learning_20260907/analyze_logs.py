"""Read frozen training logs; distinguish legacy failures from the audited official path."""

import ast
import gzip
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_bytes(name):
    return gzip.decompress((ROOT / (name + ".gz")).read_bytes())


def read_text(name):
    return read_bytes(name).decode("utf-8")


records = []
for line in read_text("historical-train.log").replace("\r", "\n").splitlines():
    if line.startswith("{'loss':"):
        row = {key: float(value) for key, value in ast.literal_eval(line).items()}
        row["task"] = next(
            key.split("/")[1]
            for key in row
            if key.startswith("task/") and key.endswith("/reward_raw")
        )
        records.append(row)


def aggregate(rows):
    return {
        "groups": len(rows),
        "wall_hours_logged": sum(row["step_time"] for row in rows) / 3600,
        "zero_gradient_groups": sum(row["grad_norm"] == 0 for row in rows),
        "zero_gradient_wall_hours": sum(row["step_time"] for row in rows if row["grad_norm"] == 0)
        / 3600,
        "mean_completion_tokens": statistics.mean(row["completions/mean_length"] for row in rows),
        "mean_sequence_IS_ratio": statistics.mean(
            row["sampling/importance_sampling_ratio/mean"] for row in rows
        ),
    }


metric_line = next(
    line for line in read_text("official-train.log").splitlines() if "step:1 - " in line
)
metrics = {}
for piece in metric_line.split(" - "):
    if ":" not in piece:
        continue
    key, value = piece.split(":", 1)
    match = re.fullmatch(r"(?:np\.\w+\()?(-?[0-9.]+(?:[eE][+-]?[0-9]+)?)\)?", value.strip())
    if match:
        metrics[key] = float(match.group(1))
assert metrics["train/num_gen_batches"] == 1
rollouts = [json.loads(line) for line in read_text("official-rollouts.jsonl").splitlines()]
events = [json.loads(line) for line in read_text("official-environment.jsonl").splitlines()]
trajectories = {}
errors = Counter()
for event in events:
    if event["event"] != "act":
        continue
    item = trajectories.setdefault(
        event["trajectory_id"], {"actions": 0, "errors": 0, "experiments": 0, "answers": 0}
    )
    item["actions"] += 1
    item["answers"] += event["command"].get("type") == "answer"
    item["experiments"] += event["command"].get("type") in ("observe", "intervene")
    try:
        feedback = json.loads(event["feedback"].splitlines()[0])
    except (json.JSONDecodeError, IndexError):
        feedback = {}
    if feedback.get("type") == "protocol_error":
        item["errors"] += 1
        errors[feedback["error"]["message"]] += 1
    elif feedback.get("type") == "batch_result":
        item["accepted_experiments"] = item.get("accepted_experiments", 0) + 1
    elif not event["completed"]:
        raise ValueError("Unrecognized nonterminal feedback")
    item["observations_used"] = event.get("observations_used")
    if event["completed"]:
        item["answer"] = event["command"]
        item["terminal_quality"] = event["raw_reward"]
        item["terminal_score"] = event.get("terminal_score")
assert sum(item["actions"] for item in trajectories.values()) == 44
assert sum(item["errors"] for item in trajectories.values()) == 13
assert sum(item["accepted_experiments"] for item in trajectories.values()) == 27
assert all("terminal_score" in item for item in trajectories.values())
report = {
    "legacy": aggregate(records),
    "legacy_by_task": {
        task: aggregate([row for row in records if row["task"] == task])
        for task in sorted({row["task"] for row in records})
    },
    "official_audit": {
        "metrics": metrics,
        "generation_share": metrics["timing_s/gen"] / metrics["timing_s/step"],
        "actor_update_share": metrics["timing_s/update_actor"] / metrics["timing_s/step"],
        "profiler_overhead_present": True,
        "rollout_count": len(rollouts),
        "trajectories": trajectories,
        "protocol_errors": dict(errors),
    },
    "scope": [
        "Legacy checkpoints and trainer are disqualified; legacy metrics do not establish "
        "current official DAPO behavior.",
        "The official step used a whole-process Python observer "
        "and is not an uninstrumented speed benchmark.",
        "Four current trajectories share one ATE task; no claim about all five families "
        "or improvement after training.",
    ],
    "source_sha256": {
        name: hashlib.sha256(read_bytes(name)).hexdigest()
        for name in (
            "historical-train.log",
            "official-train.log",
            "official-rollouts.jsonl",
            "official-environment.jsonl",
        )
    },
}
(ROOT / "training-priority.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(
    json.dumps(
        {
            "legacy": report["legacy"],
            "official_generation_share": report["official_audit"]["generation_share"],
            "protocol_errors": dict(errors),
            "trajectories": [
                {k: v for k, v in item.items() if k != "terminal_score"}
                for item in trajectories.values()
            ],
        }
    )
)
