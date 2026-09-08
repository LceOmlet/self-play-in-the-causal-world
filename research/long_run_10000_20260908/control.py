"""Prepare/launch the accepted continuous DAPO chain to 10,000 actual updates.

Preparation is CPU-only and uses the accepted offline kernel/source migrator.
This controller never pauses an existing run, implements no training algorithm,
and has no wall-clock stop. Launch is a separate explicit action after the old
process tree has been stopped and its native checkpoint preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

TARGET_UPDATES = 10000
ACCEPTED_CONTROL = "research/continuous_training_restore_20260908/control.py"
ACCEPTED_CONTROL_SHA = "9682f145e1a1c5a9d0ffaa32391834d7a30e978eabe7b9bf7206190658b86b47"
SOURCE_MIGRATOR = "research/terminal_packing_20260908/migrate_stream_source.py"
RUNTIME_KEYS = (
    "CPT_WORLD_VERL_AUDIT_DIR",
    "CPT_WORLD_EXPECTED_SOURCE",
    "VERL_ROOT",
    "VERL_RECIPE_ROOT",
    "CPT_WORLD_DAPO_SOURCE_PROFILE",
    "PYTHONPATH",
    "PATH",
)


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def accepted(project):
    path = project / ACCEPTED_CONTROL
    if sha(path) != ACCEPTED_CONTROL_SHA:
        raise ValueError("Accepted process/configuration controller changed")
    return load_module(path, "_accepted_continuous_controller")


def overrides_for(old_plan, checkpoint, run, env):
    changes = {
        "trainer.total_training_steps": str(TARGET_UPDATES),
        "trainer.resume_from_path": str(checkpoint),
        "trainer.validation_data_dir": str(run / "validation"),
    }
    result = [
        item
        for item in old_plan["command"][2:]
        if item.lstrip("+").partition("=")[0] not in changes
        and not item.lstrip("+").startswith("ray_kwargs.ray_init.runtime_env.env_vars.")
    ]
    result.extend(f"{key}={value}" for key, value in changes.items())
    result.extend(
        f"++ray_kwargs.ray_init.runtime_env.env_vars.{key}={env[key]}" for key in RUNTIME_KEYS
    )
    return result


def audit_configuration(base, old, new, old_project, new_project):
    before, after = base.flatten(old), base.flatten(new)
    differences = {
        key: {"before": before.get(key), "after": after.get(key)}
        for key in before.keys() | after.keys()
        if before.get(key) != after.get(key)
    }
    allowed = {
        "trainer.total_training_steps",
        "data.train_files",
        "trainer.resume_from_path",
        "trainer.default_local_dir",
        "trainer.rollout_data_dir",
        "trainer.validation_data_dir",
        "ray_kwargs.ray_init.runtime_env.env_vars.CPT_WORLD_VERL_AUDIT_DIR",
        "ray_kwargs.ray_init.runtime_env.env_vars.CPT_WORLD_EXPECTED_SOURCE",
        "ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH",
    }
    for key, values in differences.items():
        prior = values["before"]
        if (
            isinstance(prior, str)
            and prior.startswith(str(old_project) + "/")
            and values["after"] == str(new_project) + prior[len(str(old_project)) :]
        ):
            allowed.add(key)
    if set(differences) - allowed:
        raise ValueError(f"Unrequested training configuration changes: {differences}")
    required = {
        "trainer.total_training_steps": TARGET_UPDATES,
        "trainer.resume_mode": "resume_path",
        "trainer.test_freq": -1,
        "trainer.val_before_train": False,
        "trainer.val_only": False,
        "data.shuffle": False,
        "data.dataloader_num_workers": 0,
        "data.train_batch_size": 1,
        "data.gen_batch_size": 1,
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 0,
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": 0.0,
        "actor_rollout_ref.actor.optim.lr": 1e-6,
    }
    for key, value in required.items():
        if after.get(key) != value:
            raise ValueError(f"Unexpected required setting {key}: {after.get(key)!r}")
    return differences


def inspect_scheduler(checkpoint, updates):
    import torch

    # Native FSDP extra state is small; no model/Adam tensor is deserialized here.
    extra = torch.load(
        checkpoint / "actor/extra_state_world_size_1_rank_0.pt",
        map_location="cpu",
        weights_only=False,
    )
    schedule = extra["lr_scheduler"]
    if (
        schedule.get("last_epoch") != updates
        or schedule.get("_step_count") != updates + 1
        or schedule.get("base_lrs") != [1e-6]
        or schedule.get("_last_lr") != [1e-6]
        or schedule.get("lr_lambdas") != [None]
        or not {"cpu", "numpy", "random", "cuda"} <= extra.get("rng", {}).keys()
    ):
        raise ValueError("Native scheduler/RNG state disagrees with the accepted continuation")
    return schedule


def prepare(args):
    base = accepted(args.project)
    old_plan = read(args.old_run / "prepared.json")
    saved = read(args.hold / "preserved.json")
    source = Path(saved["checkpoint"])
    if not all(sha(source / name) == value for name, value in saved["files"].items()):
        raise ValueError("Preserved native checkpoint changed")
    old_project = Path(old_plan["project"])
    progress = read(source / "dapo_progress.json")
    updates = progress["completed_updates"]
    if updates != saved["step"] or not 0 < updates < TARGET_UPDATES:
        raise ValueError("Continuation must retain a completed native update below 10,000")
    env = base.environment(args.project, args.run)
    cpu = env | {"CUDA_VISIBLE_DEVICES": "", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    os.environ.update(cpu)
    sys.path[:0] = [str(args.project / "src"), str(args.project), str(base.RECIPE), str(base.VERL)]
    migrator = load_module(args.project / SOURCE_MIGRATOR, "_accepted_kernel_source_migrator")
    migration = migrator.migrate(
        source,
        args.old_run / "stream.json",
        old_project,
        args.run,
        args.evidence,
    )
    checkpoint = Path(migration["destination_checkpoint"])
    schedule = inspect_scheduler(checkpoint, updates)
    overrides = overrides_for(old_plan, checkpoint, args.run, env)
    inspection = [
        sys.executable,
        "-m",
        "dapo.main_dapo",
        f"hydra.searchpath=[file://{base.VERL}/verl/trainer/config,file://{args.project}/configs/verl]",
        "+profiles@_global_=cpt_world_dapo",
        *overrides,
        "--cfg",
        "job",
        "--resolve",
    ]
    result = subprocess.run(inspection, cwd=args.project, env=cpu, capture_output=True, text=True)
    (args.run / "resolved.yaml").write_text(result.stdout, encoding="utf-8")
    (args.run / "configuration.stderr").write_text(result.stderr, encoding="utf-8")
    result.check_returncode()
    from omegaconf import OmegaConf
    from verl.utils.config import validate_config

    old_config = OmegaConf.load(args.old_run / "resolved.yaml")
    config = OmegaConf.create(result.stdout)
    differences = audit_configuration(
        base,
        OmegaConf.to_container(old_config, resolve=True),
        OmegaConf.to_container(config, resolve=True),
        old_project,
        args.project,
    )
    validate_config(config, use_reference_policy=False, use_critic=False)
    proof = subprocess.run(
        [
            sys.executable,
            str(args.project / "scripts/verify_official_dapo.py"),
            "--output",
            str(args.run / "preflight-source-verification.json"),
        ],
        cwd=args.project,
        env=cpu,
        capture_output=True,
        text=True,
    )
    (args.run / "source-verification.log").write_text(proof.stdout + proof.stderr, encoding="utf-8")
    proof.check_returncode()
    write(args.run / "checkpoint-acceptance.json", migration)
    write(
        args.run / "prepared.json",
        {
            "version": 1,
            "time": time.time(),
            "step": updates,
            "generated_batches": progress["generated_batches"],
            "target_completed_updates": TARGET_UPDATES,
            "remaining_updates_at_preparation": TARGET_UPDATES - updates,
            "checkpoint": str(checkpoint),
            "source_run": str(args.old_run),
            "hold": str(args.hold),
            "project": str(args.project),
            "wall_deadline": None,
            "previous_wall_deadline": old_plan.get("wall_deadline"),
            "scheduler": schedule,
            "differences": differences,
            "controller_sha256": sha(Path(__file__)),
            "resolved_sha256": sha(args.run / "resolved.yaml"),
            "stream_sha256": sha(args.run / "stream.json"),
            "checkpoint_acceptance_sha256": sha(args.run / "checkpoint-acceptance.json"),
            "data_sha256": sha(checkpoint / "data.pt"),
            "progress_sha256": sha(checkpoint / "dapo_progress.json"),
            "command": ["bash", str(args.project / "scripts/run_official_dapo.sh"), *overrides],
        },
    )
    print(
        json.dumps({"prepared": str(args.run), "step": updates, "target": TARGET_UPDATES}),
        flush=True,
    )


def checked_plan(args):
    plan = read(args.run / "prepared.json")
    if (
        args.project.resolve() != Path(plan["project"]).resolve()
        or plan.get("wall_deadline", "missing") is not None
        or plan.get("target_completed_updates") != TARGET_UPDATES
        or sha(Path(__file__)) != plan["controller_sha256"]
    ):
        raise ValueError("Prepared long-run identity/target changed")
    for name, key in (
        ("resolved.yaml", "resolved_sha256"),
        ("stream.json", "stream_sha256"),
        ("checkpoint-acceptance.json", "checkpoint_acceptance_sha256"),
    ):
        if sha(args.run / name) != plan[key]:
            raise ValueError(f"Prepared artifact changed: {name}")
    for name, key in (("data.pt", "data_sha256"), ("dapo_progress.json", "progress_sha256")):
        if sha(Path(plan["checkpoint"]) / name) != plan[key]:
            raise ValueError(f"Prepared native data/progress changed: {name}")
    return plan


def launch(args):
    base = accepted(args.project)
    plan = checked_plan(args)
    for identity in read(Path(plan["hold"]) / "stopped.json")["processes"]:
        try:
            member = psutil.Process(identity["pid"])
            if (
                abs(member.create_time() - identity["create_time"]) < 0.01
                and member.status() != psutil.STATUS_ZOMBIE
            ):
                raise ValueError("The previous training process tree is still alive")
        except psutil.NoSuchProcess:
            pass
    if (args.run / "supervisor.json").exists():
        raise ValueError("Never launch the same prepared invocation twice")
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        raise ValueError("Expected the old training GPU processes to have exited")
    with (args.run / "train.log").open("x") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "supervise",
                "--project",
                str(args.project),
                "--run",
                str(args.run),
            ],
            env=base.environment(args.project, args.run),
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write(args.run / "supervisor.json", {"pid": process.pid, "time": time.time()})
    print(json.dumps({"launched": str(args.run), "supervisor": process.pid}), flush=True)


def supervise(args):
    base = accepted(args.project)
    plan = checked_plan(args)
    process = subprocess.Popen(
        plan["command"], env=base.environment(args.project, args.run), start_new_session=True
    )
    write(
        args.run / "process.json",
        {"pid": process.pid, "create_time": psutil.Process(process.pid).create_time()},
    )
    code = process.wait()  # No deadline, watchdog, or smaller training horizon.
    tracker = args.run / "checkpoints/latest_checkpointed_iteration.txt"
    updates = int(tracker.read_text()) if tracker.is_file() else None
    reached = False
    if updates == TARGET_UPDATES:
        progress = read(args.run / "checkpoints" / f"global_step_{updates}" / "dapo_progress.json")
        reached = progress.get("completed_updates") == TARGET_UPDATES
    write(
        args.run / "exit.json",
        {
            "time": time.time(),
            "exit_code": code,
            "wall_limit_reached": False,
            "completed_updates": updates,
            "target_completed_updates": TARGET_UPDATES,
            "target_reached": reached,
        },
    )
    return code if code else (0 if reached else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "supervise"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--old-run", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--hold", type=Path)
    parser.add_argument("--evidence", type=Path)
    arguments = parser.parse_args()
    if arguments.action == "prepare" and any(
        value is None for value in (arguments.old_run, arguments.hold, arguments.evidence)
    ):
        parser.error("prepare requires --old-run, --hold and --evidence")
    sys.exit(globals()[arguments.action](arguments) or 0)
