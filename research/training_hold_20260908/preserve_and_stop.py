"""Preserve a published native checkpoint and honor an explicit training hold.

Uses the already audited process-tree terminator. No trainer or resume launcher.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import psutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--created", type=float, required=True)
    args = parser.parse_args()
    proc = psutil.Process(args.pid)
    assert abs(proc.create_time() - args.created) < 0.01
    assert "dapo.main_dapo" in proc.cmdline()
    assert os.getpgid(proc.pid) == proc.pid
    saved = json.loads((args.run / "process.json").read_text())
    assert saved == {"pid": args.pid, "create_time": args.created}
    helper_path = args.project / "research/inference_efficiency_20260908/resume_runtime.py"
    assert sha(helper_path) == "e93f1cd212182733644492021a1f23553966bed7cbb364e5d51787644a12023f"
    spec = importlib.util.spec_from_file_location("verified_process_helper", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    tracker = args.run / "checkpoints/latest_checkpointed_iteration.txt"
    step = int(tracker.read_text())
    source = args.run / "checkpoints" / f"global_step_{step}"
    assert all(
        (source / name).is_file()
        for name in [
            "data.pt",
            "dapo_progress.json",
            "actor/model_world_size_1_rank_0.pt",
            "actor/optim_world_size_1_rank_0.pt",
            "actor/extra_state_world_size_1_rank_0.pt",
        ]
    )
    files = {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}
    args.output.mkdir(parents=True, exist_ok=False)
    preserved = args.output / source.name
    shutil.copytree(source, preserved)
    assert all(
        sha(source / name) == value == sha(preserved / name) for name, value in files.items()
    )
    assert int(tracker.read_text()) == step, "New checkpoint published: preserve it before stopping"
    children = proc.children(recursive=True)
    runner_paths = [
        Path(os.readlink(f"/proc/{p.pid}/fd/1"))
        for p in children
        if p.name().startswith("ray::DAPOTaskRunner")
    ]
    assert len(runner_paths) == 1
    write(
        args.output / "before-stop.json",
        {
            "time": time.time(),
            "run": str(args.run),
            "pid": args.pid,
            "create_time": args.created,
            "step": step,
            "checkpoint": str(preserved),
            "files": files,
            "script_sha256": sha(Path(__file__)),
            "helper_sha256": sha(helper_path),
            "taskrunner_stdout": str(runner_paths[0]),
            "reason": "User explicitly renewed: 训练长跑保持停止; "
            "stop the continuous verification run too.",
            "scope": "Published checkpoint preserved; "
            "any unfinished rollout/update is not claimed complete.",
        },
    )
    identities = helper.terminate_tree(proc)
    remaining = []
    for identity in identities:
        try:
            child = psutil.Process(identity["pid"])
            if abs(child.create_time() - identity["create_time"]) < 0.01:
                if child.status() != psutil.STATUS_ZOMBIE:
                    remaining.append(identity)
        except psutil.NoSuchProcess:
            pass
    assert not remaining, remaining
    shutil.copyfile(runner_paths[0], args.output / "taskrunner.stdout")
    for name in [
        "process.json",
        "supervisor.json",
        "prepared.json",
        "resolved.yaml",
        "official-source-verification.json",
        "train.log",
        "exit.json",
    ]:
        path = args.run / name
        if path.exists():
            shutil.copyfile(path, args.output / name)
    for name in ["rollouts", "environment"]:
        shutil.copytree(args.run / name, args.output / name)
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    result = {
        "time": time.time(),
        "stopped": True,
        "step": step,
        "processes": identities,
        "remaining_owned_processes": remaining,
        "gpu_compute_pids": gpu,
        "checkpoint": str(preserved),
        "taskrunner_stdout_sha256": sha(args.output / "taskrunner.stdout"),
        "scope": "Explicit user hold. No resumption scheduled; no learning claim.",
    }
    write(args.output / "stopped.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
