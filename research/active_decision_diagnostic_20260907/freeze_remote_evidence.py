"""Archive the fixed inputs and completed remote diagnostic, without rerunning it."""

import hashlib
import json
import subprocess
import tarfile
from importlib.metadata import version
from pathlib import Path

ROOT = Path("/home/chen/runs/active-decision-diagnostic-20260907")
COHORT = Path("/home/chen/runs/environment-validation-20260907")
PROJECT = Path("/home/chen/projects/self-play-in-the-causal-world-rewardv10-a92ab8e-20260831")
assert len(list((ROOT / "results").glob("*.public.json.gz"))) == 100
fingerprints = {
    "production_commit": subprocess.check_output(["git", "-C", str(PROJECT), "rev-parse", "HEAD"])
    .decode()
    .strip(),
    "numpy": version("numpy"),
    "transformers": version("transformers"),
    "source_sha256": {
        name: hashlib.sha256((PROJECT / "src/cpt_world" / name).read_bytes()).hexdigest()
        for name in (
            "query_truth.py",
            "world_space.py",
            "world_runtime.py",
            "world.py",
            "task_scoring.py",
            "rewards.py",
            "identification.py",
        )
    },
    "original_policy_sha256": hashlib.sha256((ROOT / "public_policy.py").read_bytes()).hexdigest(),
}
(ROOT / "runtime-fingerprints.json").write_text(json.dumps(fingerprints, indent=2) + "\n")
files = list((ROOT / "results").iterdir())
files += [
    ROOT / name
    for name in (
        "smoke.log",
        "full.log",
        "rescore.log",
        "context-initial-invalid.json",
        "context-measurement.json",
        "fresh-runner-check.json",
        "runtime-fingerprints.json",
    )
]
assert all(path.is_file() for path in files)
with tarfile.open(ROOT / "evidence.tar.gz", "w:gz") as archive:
    for path in sorted(files):
        archive.add(path, arcname=path.relative_to(ROOT).as_posix())
    for name, destination in (
        ("run_diagnostic.py", "initial_cached_runner.py"),
        ("public_policy.py", "public_policy.py"),
        ("verified_runner.py", "verified_runner.py"),
    ):
        archive.add(ROOT / name, arcname="original-code/" + destination)
tasks = set()
for path in (ROOT / "results").glob("*.public.json.gz"):
    cohort, remainder = path.name.split("-task-")
    task_index = remainder.split("-r")[0]
    tasks.add(COHORT / cohort / f"task-{task_index}.pkl")
assert len(tasks) == 50
with tarfile.open(ROOT / "frozen-worlds.tar.gz", "w:gz") as archive:
    for path in sorted(tasks):
        archive.add(path, arcname=path.relative_to(COHORT).as_posix())
    archive.add(COHORT / "finite-baselines.jsonl", arcname="finite-baselines.jsonl")
report = {
    name: {
        "bytes": (ROOT / name).stat().st_size,
        "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
    }
    for name in (
        "evidence.tar.gz",
        "frozen-worlds.tar.gz",
        "context-measurement.json",
        "runtime-fingerprints.json",
        "fresh-runner-check.json",
    )
}
(ROOT / "export-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
