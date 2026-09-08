"""Freeze actual training-signal and observation-rank evidence, read-only."""

import argparse
import hashlib
import io
import json
import os
import platform
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np
import psutil


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    project, root = args.project, args.root
    members = {}

    def add(path, name):
        assert path.is_file() and not path.is_symlink() and name not in members
        data = path.read_bytes()
        assert len(data) < 25_000_000
        members[name] = data

    def add_json(name, value):
        assert name not in members
        members[name] = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()

    snapshots = {
        "updates1-5": root.parent / "inference-efficiency-20260908/post-pause-snapshot",
        "updates6-15": root.parent / "dapo-progress-20260908/compiled-through15-snapshot",
        "updates16-25": root / "update25-complete-snapshot",
        "updates26-29": root / "training-signal-01-snapshot",
    }
    for label, folder in snapshots.items():
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                add(path, f"snapshots/{label}/{path.relative_to(folder).as_posix()}")
    for name in [
        "inspect_training_signal.py",
        "training-signal-inspection.log",
        "audit_training_signal.py",
        "audit_training_signal_v2.py",
        "training-signal-through29.json",
        "training-signal-through29-v2.json",
        "training-signal-through29.log",
        "training-signal-through29-v2.log",
        "training-signal-01-capture.log",
        "read_successful_decisions.py",
        "read_successful_decisions_v2.py",
        "read_successful_decisions_v3.py",
        "successful-strong-decisions-through29.json",
        "successful-strong-decisions-through29-v3.json",
        "successful-strong-decisions-through29.log",
        "successful-strong-decisions-through29-v2.log",
        "successful-strong-decisions-through29-v3.log",
        "audit_observation_rank.py",
        "audit_observation_rank_v2.py",
        "observation-rank-01.log",
        "observation-rank-02.log",
    ]:
        add(root / name, name)
    for folder in ["observation-rank-01", "observation-rank-02"]:
        for path in sorted((root / folder).iterdir()):
            if path.is_file():
                add(path, path.relative_to(root).as_posix())
    for name in [
        "process.json",
        "prepared.json",
        "resolved.yaml",
        "official-source-verification.json",
    ]:
        add(root / "resume-after-base-01" / name, "live-run/" + name)
    data_root = root.parent / "training-submission-20260907/data"
    for name in ["accepted.jsonl", "acceptance.json"]:
        add(data_root / name, "data/" + name)
    for index in [33, 93, 433]:
        add(data_root / f"truth/train-{index:04d}.pkl", f"data/truth/train-{index:04d}.pkl")
    for path in sorted((project / "src/cpt_world").rglob("*.py")):
        add(path, "source/" + path.relative_to(project).as_posix())
    for name in [
        "research/training_submission_20260907/capture_progress.py",
        "research/base_signal_diagnostic_20260907/analyze_events.py",
        "research/active_decision_diagnostic_20260907/runtime-fingerprints.json",
        "research/active_decision_diagnostic_20260907/SHA256SUMS.json",
    ]:
        add(project / name, "source/" + name)
    add(
        root.parent / "environment-validation-20260907/finite_budget_baselines.py",
        "source/original_finite_budget_baselines.py",
    )
    add(Path(__file__), "source/archive_evidence.py")
    training = json.loads(members["training-signal-through29-v2.json"])
    assert training["script_sha256"] == digest(members["audit_training_signal_v2.py"])
    assert training["completed_updates"] == 29 and training["generated_groups"] == 33
    rank = json.loads(members["observation-rank-02/summary.json"])
    assert rank["script_sha256"] == digest(members["audit_observation_rank_v2.py"])
    assert rank["records_sha256"] == digest(members["observation-rank-02/records.jsonl"])
    assert rank["replayed_original_answers"] == 300
    correct = json.loads(members["successful-strong-decisions-through29-v3.json"])
    assert {r["request_id"] for r in correct} == {
        r["request_id"] for r in training["successful_strong_trajectories"]
    }
    assert all(digest(r["output"].encode()) == r["output_sha256"] for r in correct)
    runtime = json.loads(
        members["source/research/active_decision_diagnostic_20260907/runtime-fingerprints.json"]
    )
    same_sources = {}
    for name, fingerprint in runtime["source_sha256"].items():
        actual = digest(members["source/src/cpt_world/" + name])
        assert actual == fingerprint
        same_sources[name] = actual
    assert np.__version__ == runtime["numpy"]
    add_json(
        "sampling-runtime-comparison.json",
        {
            "same_source_files": same_sources,
            "same_numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "scope": "Original diagnostic fingerprints match current "
            "world/tape/runtime/truth/scoring sources.",
        },
    )
    identity = json.loads(members["live-run/process.json"])
    process = psutil.Process(identity["pid"])
    assert abs(process.create_time() - identity["create_time"]) < 0.01
    assert process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    workers = [
        p for p in process.children(recursive=True) if p.name().startswith("ray::DAPOTaskRunner")
    ]
    assert len(workers) == 1
    worker = workers[0]
    source = Path(os.readlink(f"/proc/{worker.pid}/fd/1"))
    add(source, "live-run/latest-owned-task-runner.log")
    add_json(
        "live-run/status-at-archive.json",
        {
            "time": time.time(),
            "same_process_live": True,
            "process": identity,
            "worker": {
                "pid": worker.pid,
                "create_time": worker.create_time(),
                "stdout": str(source),
            },
            "latest_checkpoint_tracker": (
                root / "resume-after-base-01/checkpoints/latest_checkpointed_iteration.txt"
            )
            .read_text()
            .strip(),
            "project_head": subprocess.check_output(
                ["git", "-C", str(project), "rev-parse", "HEAD"], text=True
            ).strip(),
            "scope": "Later tracker is live status; trajectory signal analysis stops at update29.",
        },
    )
    archive = root / "training-signal-and-rank-evidence.tar.gz"
    manifest = root / "training-signal-and-rank-evidence.sha256.json"
    assert not archive.exists() and not manifest.exists()
    with archive.open("xb") as handle, tarfile.open(fileobj=handle, mode="w:gz") as tar:
        for name, data in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
    metadata = {
        "archive_sha256": digest(archive.read_bytes()),
        "archive_bytes": archive.stat().st_size,
        "members": {name: digest(data) for name, data in sorted(members.items())},
        "scope": "All 29 completed groups, all 4 successful strong decisions, "
        "and 100 pure-observation rank replays. "
        "Original cohorts are referenced by existing repository archive hashes.",
    }
    with manifest.open("x") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {k: v for k, v in metadata.items() if k != "members"} | {"member_count": len(members)}
        )
    )


if __name__ == "__main__":
    main()
