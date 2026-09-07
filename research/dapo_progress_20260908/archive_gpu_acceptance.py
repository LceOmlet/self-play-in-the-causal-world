"""Freeze real update-16 acceptance and curve inputs; never touch training."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import psutil


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    root, project = args.root.resolve(), args.project.resolve()
    run = root / "resume-progress-01"
    members = {}

    def add(path, name):
        assert path.is_file() and not path.is_symlink(), path
        assert name not in members
        data = path.read_bytes()
        assert len(data) < 25_000_000, path
        members[name] = data

    def add_tree(path, prefix):
        for item in sorted(path.rglob("*")):
            if item.is_file():
                add(item, f"{prefix}/{item.relative_to(path).as_posix()}")

    def add_json(name, value):
        assert name not in members
        members[name] = (json.dumps(value, indent=2) + "\n").encode()

    eager = root.parent / "inference-efficiency-20260908/post-pause-snapshot"
    snapshots = {
        "eager-1-5": eager,
        "compiled-6-15": root / "compiled-through15-snapshot",
        "progress-16-17": root / "progress-update16-snapshot",
    }
    for label, path in snapshots.items():
        add_tree(path, f"snapshots/{label}")
        add(path / "progress.json", f"curve-inputs/{label}.json")
    for name in [
        "prepare-progress-01.log",
        "pause-progress-01.log",
        "launch-progress-01.log",
        "compiled-through15-before-reader-fix.json",
        "compiled-through15-capture.log",
        "compiled-through15-reanalyze.log",
        "reader-v2-acceptance.json",
        "reader-v2-reanalyze.log",
        "progress-update16-capture.log",
    ]:
        add(root / name, f"switch-and-reader/{name}")
    for name in [
        "checkpoint-acceptance.json",
        "configuration.stderr",
        "prepared.json",
        "paused-source.json",
        "preflight-source-verification.json",
        "official-source-verification.json",
        "process.json",
        "resolved.yaml",
        "source-verification.log",
        "supervisor.json",
        "verified-update16/acceptance.json",
        "verified-update16/global_step_16/dapo_progress.json",
        "verified-update16/global_step_16/data.pt",
        "preserved/global_step_15/data.pt",
    ]:
        add(run / name, f"resume-progress-01/{name}")
    for name in [
        "research/dapo_progress_20260908/resume_progress.py",
        "research/inference_efficiency_20260908/resume_runtime.py",
        "scripts/run_official_dapo.sh",
        "scripts/verify_official_dapo.py",
        "configs/verl/upstream_sources_progress_v1.json",
        "patches/verl-dapo-progress-v1.patch",
        "patches/verl-recipe-mask-progress-v1.patch",
        "research/base_signal_diagnostic_20260907/analyze_events.py",
    ]:
        add(project / name, f"source/{name}")
    reader = root / "reader-v2/research/training_submission_20260907/capture_progress.py"
    add(reader, "source/research/training_submission_20260907/capture_progress.py")
    add(Path(__file__).resolve(), "source/archive_gpu_acceptance.py")
    prepared = json.loads(members["resume-progress-01/prepared.json"])
    assert prepared["controller_sha256"] == digest(
        members["source/research/dapo_progress_20260908/resume_progress.py"]
    )
    assert prepared["resolved_sha256"] == digest(members["resume-progress-01/resolved.yaml"])
    reader_acceptance = json.loads(members["switch-and-reader/reader-v2-acceptance.json"])
    assert reader_acceptance["reader_sha256"] == digest(reader.read_bytes())
    acceptance = json.loads(members["resume-progress-01/verified-update16/acceptance.json"])
    for name in ["data.pt", "dapo_progress.json"]:
        assert acceptance["files"][name] == digest(
            members[f"resume-progress-01/verified-update16/global_step_16/{name}"]
        )

    rows, metrics = [], {}
    for label in snapshots:
        data = json.loads(members[f"curve-inputs/{label}.json"])
        assert not data["nonfinite_metrics"]
        rows.extend(data["retained_rollouts"])
        for metric in data["official_metrics"]:
            if "actor/grad_norm" in metric:
                assert metric["step"] not in metrics
                metrics[metric["step"]] = metric
    assert sorted(metrics) == list(range(1, 18))
    assert len(rows) == 68 and len({r["request_id"] for r in rows}) == 68
    assert all(r["event_join_method"] != "unresolved" for r in rows)
    families = defaultdict(list)
    for row in rows:
        families[row["query_type"]].append(row)
    family_stats = {}
    strict_keys = {
        "best_intervention": ["optimal_action"],
        "backadj_minimal_sets": ["valid_adjustment_set"],
        "mediator_set": ["mediators_exact_match", "order_exact_match"],
    }
    for family, records in families.items():
        stats = {
            "retained": len(records),
            "completed": sum(r["completed"] for r in records),
            "raw_quality_mean": sum(r["raw_quality"] for r in records) / len(records),
        }
        if family in strict_keys:
            keys = strict_keys[family]
            for row in records:
                if row["completed"]:
                    assert row["task_metrics"] is not None
                    assert all(k in row["task_metrics"] for k in keys)
            stats["strict_success_count"] = sum(
                bool(r["completed"] and all(r["task_metrics"][k] for k in keys)) for r in records
            )
        family_stats[family] = stats
    add_json(
        "retained-statistics.json",
        {
            "updates": sorted(metrics),
            "families": family_stats,
            "incomplete_retained": [r for r in rows if not r["completed"]],
            "length_penalized_retained": [r for r in rows if r["official_overlong_reward"] != 0],
            "update16_official_metrics": metrics[16],
            "scope": (
                "Different training tasks, not paired learning gains. Incomplete retained "
                "outputs count as strict failures; unrelated in-flight outputs do not."
            ),
        },
    )
    identity = json.loads(members["resume-progress-01/process.json"])
    process = psutil.Process(identity["pid"])
    assert abs(process.create_time() - identity["create_time"]) < 0.01
    assert process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    tracker = run / "checkpoints/latest_checkpointed_iteration.txt"
    add_json(
        "live-status-at-archive.json",
        {
            "time": time.time(),
            "process": identity,
            "same_process_live": True,
            "latest_checkpoint_tracker": tracker.read_text().strip(),
            "project_head": subprocess.check_output(
                ["git", "-C", str(project), "rev-parse", "HEAD"], text=True
            ).strip(),
            "scope": (
                "Read-only live identity/tracker observation; "
                "detailed tensor acceptance is update16, curves stop at17."
            ),
        },
    )
    archive = root / "gpu-update16-and-curves-evidence.tar.gz"
    with archive.open("xb") as handle, tarfile.open(fileobj=handle, mode="w:gz") as tar:
        for name, data in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
    manifest = {
        "archive_sha256": digest(archive.read_bytes()),
        "archive_bytes": archive.stat().st_size,
        "members": {name: digest(data) for name, data in sorted(members.items())},
    }
    with (root / "gpu-update16-and-curves-evidence.sha256.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "archive_sha256": manifest["archive_sha256"],
                "archive_bytes": manifest["archive_bytes"],
                "members": len(members),
                "families": family_stats,
            }
        )
    )


if __name__ == "__main__":
    main()
