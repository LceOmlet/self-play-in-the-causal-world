"""Compare unchanged frozen tasks with the last accepted counterfactual owner."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root", type=Path, default=Path("/home/chen/runs/symbolic-contraction-20260907")
)
parser.add_argument("--variant", default="numeric-v1")
parser.add_argument(
    "--baseline",
    type=Path,
    default=Path("/home/chen/runs/kernel-root-cause-20260907/diagonal-transport"),
)
args = parser.parse_args()


def read(folder, index):
    path = folder / f"{index}.parent.json"
    if not path.exists():
        return None
    parent = json.loads(path.read_text())
    events = [
        json.loads(line) for line in (folder / f"{index}.events.jsonl").read_text().splitlines()
    ]
    result = next((event for event in reversed(events) if event["event"] == "result"), {})
    models = [
        event for event in events if event.get("name") == "__init__" and event["event"] == "exit"
    ]
    compilations = [
        event
        for event in events
        if event.get("name") == "_compile_numeric_terminal_factors" and event["event"] == "exit"
    ]
    return {
        "status": "timeout" if parent["hard_timeout"] else result.get("status", "no_result"),
        "truth": result.get("truth"),
        "seconds": parent["seconds"],
        "rss_kib": parent["rss_kib"],
        "models_built": len(models),
        "max_variables": max((event.get("vars", 0) for event in models), default=0),
        "max_auxiliaries": max((event.get("auxiliaries", 0) for event in models), default=0),
        "model_build_seconds": sum(event["seconds"] for event in models),
        "numeric_compilation_seconds": sum(event["seconds"] for event in compilations),
        "error": result.get("error"),
    }


records, regressions, mismatches, newly_closed = [], [], [], []
before_counts, after_counts = Counter(), Counter()
for index in range(74):
    after = read(args.root / args.variant, index)
    if after is None:
        continue
    before = read(args.baseline, index)
    assert before is not None, index
    before_counts[before["status"]] += 1
    after_counts[after["status"]] += 1
    row = {"index": index, "before": before, "after": after}
    if before["status"] == "ok":
        if after["status"] != "ok":
            regressions.append(index)
        else:
            a, b = before["truth"], after["truth"]
            difference = max(abs(a[key] - b[key]) for key in ("lower", "upper"))
            allowance = a["endpoint_error"] + b["endpoint_error"] + 1e-8
            row.update(endpoint_difference=difference, endpoint_allowance=allowance)
            if difference > allowance:
                mismatches.append(index)
    elif after["status"] == "ok":
        newly_closed.append(index)
    records.append(row)
report = {
    "complete": len(records) == 74,
    "base_commit": "95b84544f70aabdca74d73de3523e29e50549354",
    "compared": len(records),
    "before_counts": dict(before_counts),
    "after_counts": dict(after_counts),
    "regressions": regressions,
    "endpoint_mismatches": mismatches,
    "newly_closed": newly_closed,
    "candidate_source_sha256": hashlib.sha256(
        (args.root / "candidate/src/cpt_world/counterfactual_solver.py").read_bytes()
    ).hexdigest(),
    "inputs_sha256": {
        str(index): hashlib.sha256(
            (Path("/home/chen/runs/kernel-root-cause-20260907") / f"task-{index}.pkl").read_bytes()
        ).hexdigest()
        for index in range(74)
    },
    "records": records,
}
(args.root / f"{args.variant}-summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {key: value for key, value in report.items() if key not in {"records", "inputs_sha256"}},
        indent=2,
    )
)
