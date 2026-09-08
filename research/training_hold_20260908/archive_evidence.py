"""Archive completed logs and explicit hold/resume evidence; exclude model tensors."""

import argparse
import ast
import hashlib
import io
import json
import os
import tarfile
import time
from pathlib import Path

import psutil


def digest(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    members = {}

    def add(path, name):
        assert path.is_file() and not path.is_symlink() and name not in members
        content = path.read_bytes()
        assert len(content) < 25_000_000
        members[name] = content

    snapshots = {
        "updates1-5": root.parent / "inference-efficiency-20260908/post-pause-snapshot",
        "updates6-15": root.parent / "dapo-progress-20260908/compiled-through15-snapshot",
        "updates16-25": root / "update25-complete-snapshot",
        "updates26-39": root / "user-hold39-snapshot",
    }
    for label, folder in snapshots.items():
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                add(path, f"snapshots/{label}/{path.relative_to(folder).as_posix()}")
    for name in ["preserve_and_stop.py", "resume_after_hold.py", "inspect_new_successes.py"]:
        executed, reviewed = root / name, root / "hold-review-sources" / name
        assert ast.dump(ast.parse(executed.read_text())) == ast.dump(
            ast.parse(reviewed.read_text())
        )
        add(executed, "executed-sources/" + name)
        add(reviewed, "reviewed-sources/" + name)
    for name in [
        "training-signal-through39.json",
        "training-signal-through39.log",
        "user-hold39-capture.log",
        "user-hold39-capture-v2.log",
        "new-strong-successes30-39.json",
        "new-successes-accounting.json",
    ]:
        add(root / name, name)
    hold = root / "user-hold-01"
    for name in ["before-stop.json", "stopped.json", "process.json", "exit.json"]:
        add(hold / name, "hold/" + name)
    add(hold / "global_step_39/dapo_progress.json", "native39/dapo_progress.json")
    add(hold / "global_step_39/data.pt", "native39/data.pt")
    add(hold / "global_step_39/actor/extra_state_world_size_1_rank_0.pt", "native39/extra.pt")
    data = root.parent / "training-submission-20260907/data"
    add(data / "accepted.jsonl", "data/accepted.jsonl")
    for index in [383, 408]:
        add(data / "truth" / f"train-{index:04d}.pkl", f"data/train-{index:04d}.pkl")
    run = root / "resume-after-user-hold-01"
    for name in [
        "checkpoint-acceptance.json",
        "prepared.json",
        "resolved.yaml",
        "configuration.stderr",
        "source-verification.log",
        "preflight-source-verification.json",
        "official-source-verification.json",
        "paused-source.json",
        "process.json",
        "supervisor.json",
        "train.log",
    ]:
        add(run / name, "resumed/" + name)
    identity = json.loads((run / "process.json").read_text())
    proc = psutil.Process(identity["pid"])
    assert abs(proc.create_time() - identity["create_time"]) < 0.01
    assert proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    runners = [
        p for p in proc.children(recursive=True) if p.name().startswith("ray::DAPOTaskRunner")
    ]
    assert len(runners) == 1
    stdout = Path(os.readlink(f"/proc/{runners[0].pid}/fd/1"))
    add(stdout, "resumed/taskrunner.stdout")
    tracker = run / "checkpoints/latest_checkpointed_iteration.txt"
    observed = {
        "time": time.time(),
        "main": identity,
        "taskrunner_pid": runners[0].pid,
        "taskrunner_create_time": runners[0].create_time(),
        "same_process_live": True,
        "published_checkpoint": int(tracker.read_text()) if tracker.exists() else None,
        "scope": "Live identity at archive time. Completed results frozen through39 only.",
    }
    members["resumed/live-observation.json"] = (json.dumps(observed, indent=2) + "\n").encode()
    for name in [
        "research/training_signal_20260908/audit_training_signal.py",
        "research/training_submission_20260907/capture_progress.py",
        "research/base_signal_diagnostic_20260907/analyze_events.py",
        "research/dapo_progress_20260908/resume_progress.py",
        "research/inference_efficiency_20260908/resume_runtime.py",
    ]:
        add(args.project / name, "existing-sources/" + name)
    add(Path(__file__), "archive_evidence.py")
    manifest = {
        name: {"bytes": len(value), "sha256": digest(value)} for name, value in members.items()
    }
    archive = root / "user-hold-and-through39-evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, value in members.items():
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(value), 0
            tar.addfile(info, io.BytesIO(value))
    with tarfile.open(archive) as tar:
        assert set(tar.getnames()) == set(manifest)
        assert all(
            digest(tar.extractfile(n).read()) == item["sha256"] for n, item in manifest.items()
        )
    result = {
        "archive_sha256": digest(archive.read_bytes()),
        "archive_bytes": archive.stat().st_size,
        "members": manifest,
        "live_observation": observed,
        "scope": "Native39 tensors remain on server, all tensor hashes recorded. "
        "Reviewed and executed sources have identical Python ASTs.",
    }
    (root / "user-hold-and-through39-evidence.sha256.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "members"} | {"member_count": len(manifest)}
        )
    )


if __name__ == "__main__":
    main()
