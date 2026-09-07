"""Run matched official base validation between verified native training runs.

Reuses the accepted process/checkpoint controller. Only its source-run path is
selected explicitly; no generation, validation, loss or update is implemented.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path("/home/chen/projects/self-play-in-the-causal-world-rewardv10-a92ab8e-20260831")
CORE_PATH = PROJECT / "research/dapo_progress_20260908/resume_progress.py"
CORE_SHA = "175ae0a21e2a70bd07d5f7716c449c85425aca33f6ad7a692be5908d6d1d3711"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def core():
    assert sha(CORE_PATH) == CORE_SHA
    return module(CORE_PATH, "accepted_native_controller")


def same_live_process(identity):
    import psutil

    process = psutil.Process(identity["pid"])
    assert abs(process.create_time() - identity["create_time"]) < 0.01
    assert process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    return process


def gpu_pids():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    )
    return [int(value) for value in output.splitlines() if value.strip()]


def native_plan(run):
    plan = read(run / "prepared.json")
    assert plan["matched_evaluation_controller_sha256"] == sha(Path(__file__))
    assert plan["controller_sha256"] == CORE_SHA
    assert plan["resolved_sha256"] == sha(run / "resolved.yaml")
    return plan


def prepare_resume(args):
    assert args.source_run is not None
    accepted = core()
    identity = read(args.source_run / "process.json")
    same_live_process(identity)
    # This is an operational path selector, not a patched trainer or algorithm.
    accepted.CURRENT = args.source_run
    accepted.prepare(args.resume_run)
    plan = read(args.resume_run / "prepared.json")
    plan.update(
        source_run=str(args.source_run),
        source_process=identity,
        matched_evaluation_controller_sha256=sha(Path(__file__)),
    )
    write(args.resume_run / "prepared.json", plan)
    print(json.dumps({"resume_prepared": str(args.resume_run), "step": plan["step"]}))


def validation_complete(source_run, step=25):
    process = same_live_process(read(source_run / "process.json"))
    runners = [
        p for p in process.children(recursive=True) if p.name().startswith("ray::DAPOTaskRunner")
    ]
    assert len(runners) == 1
    stdout = Path(os.readlink(f"/proc/{runners[0].pid}/fd/1"))
    raw = stdout.read_bytes()
    reader = module(PROJECT / "research/training_submission_20260907/capture_progress.py", "reader")
    rows = reader.metrics_from_log(raw)
    metrics = [m for m in rows if m["step"] == step and "actor/grad_norm" in m]
    assert len(metrics) == 1 and any(k.startswith("val-") for k in metrics[0]), (
        "Keep the current validation running until its completed metrics are logged"
    )
    dump = source_run / "validation" / f"{step}.jsonl"
    records = [json.loads(line) for line in dump.read_bytes().splitlines()]
    assert len(records) == 25
    return {
        "step": step,
        "dump_sha256": sha(dump),
        "records": len(records),
        "taskrunner_pid": runners[0].pid,
        "stdout": str(stdout),
        "stdout_prefix_sha256": hashlib.sha256(raw).hexdigest(),
        "metrics": metrics[0],
    }


def pause_training(args):
    plan = native_plan(args.resume_run)
    source = Path(plan["source_run"])
    assert read(source / "process.json") == plan["source_process"]
    proof = validation_complete(source)
    accepted = core()
    accepted.CURRENT = source
    accepted.pause(args.resume_run)
    write(args.resume_run / "completed-validation-before-pause.json", proof)
    print(json.dumps({"paused_for_matched_base": True, "preserved_step": plan["step"]}))


def verify_base(args, *, recompose):
    from omegaconf import OmegaConf

    prepared = read(args.base_run / "prepared.json")
    accepted = core()
    accepted.check_project()
    assert prepared["controller_path"] == str(CORE_PATH)
    assert prepared["controller_sha256"] == CORE_SHA
    assert prepared["configuration_only_passed"]
    assert prepared["base_config_sha256"] == sha(args.base_run / "resolved.yaml")
    config = OmegaConf.load(args.base_run / "resolved.yaml")
    assert sha(Path(config.data.val_files)) == prepared["validation_sha256"]
    assert config.trainer.val_only and config.trainer.val_before_train
    assert config.trainer.resume_mode == "disable" and config.trainer.resume_from_path is None
    assert config.actor_rollout_ref.model.lora_adapter_path is None
    env = accepted.environment(args.base_run)
    if recompose:
        inspection = [
            sys.executable,
            "-m",
            "dapo.main_dapo",
            f"hydra.searchpath=[file://{accepted.VERL}/verl/trainer/config,"
            f"file://{PROJECT}/configs/verl]",
            "+profiles@_global_=cpt_world_dapo",
            *prepared["command"][2:],
            "--cfg",
            "job",
            "--resolve",
        ]
        result = subprocess.run(
            inspection,
            env=env | {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
            cwd=PROJECT,
            capture_output=True,
            text=True,
        )
        (args.base_run / "launch-configuration.stdout").write_text(result.stdout)
        (args.base_run / "launch-configuration.stderr").write_text(result.stderr)
        result.check_returncode()
        actual = OmegaConf.to_container(OmegaConf.create(result.stdout), resolve=True)
        assert actual == OmegaConf.to_container(config, resolve=True)
    return prepared, env


def inspect_base(args):
    prepared, _ = verify_base(args, recompose=True)
    print(
        json.dumps(
            {
                "matches_prepared_official_configuration": True,
                "validation_sha256": prepared["validation_sha256"],
                "gpu_compute_pids": gpu_pids(),
                "gpu_run_launched": False,
            }
        )
    )


def launch_base(args):
    plan = native_plan(args.resume_run)
    assert (args.resume_run / "paused-source.json").exists()
    assert (args.resume_run / "completed-validation-before-pause.json").exists()
    assert not (args.base_run / "launch.json").exists()
    prepared, env = verify_base(args, recompose=True)
    assert not gpu_pids(), "The matched base run requires the released, exclusive GPU"
    assert time.time() < plan["wall_deadline"]
    metadata = {
        "time": time.time(),
        "wall_deadline": plan["wall_deadline"],
        "command": prepared["command"],
        "prepared_sha256": sha(args.base_run / "prepared.json"),
        "controller_sha256": sha(Path(__file__)),
        "resume_run": str(args.resume_run),
    }
    write(args.base_run / "launch.json", metadata)
    with (args.base_run / "train.log").open("x") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "supervise-base",
                "--base-run",
                str(args.base_run),
                "--resume-run",
                str(args.resume_run),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write(args.base_run / "supervisor.json", {"pid": process.pid, "time": time.time()})
    print(json.dumps({"base_supervisor_pid": process.pid, "base_run": str(args.base_run)}))


def supervise_base(args):
    import psutil

    metadata = read(args.base_run / "launch.json")
    assert metadata["controller_sha256"] == sha(Path(__file__))
    assert metadata["prepared_sha256"] == sha(args.base_run / "prepared.json")
    _, env = verify_base(args, recompose=False)
    process = subprocess.Popen(metadata["command"], env=env, start_new_session=True)
    write(
        args.base_run / "process.json",
        {"pid": process.pid, "create_time": psutil.Process(process.pid).create_time()},
    )
    expired = False
    try:
        code = process.wait(timeout=max(1, metadata["wall_deadline"] - time.time()))
    except subprocess.TimeoutExpired:
        expired = True
        core().helper().terminate_tree(psutil.Process(process.pid))
        code = process.wait(timeout=5)
    acceptance_error = None
    if code == 0:
        try:
            dump = args.base_run / "validation/0.jsonl"
            assert len(dump.read_bytes().splitlines()) == 25
            assert not (args.base_run / "checkpoints/latest_checkpointed_iteration.txt").exists()
            reader = module(
                PROJECT / "research/training_submission_20260907/capture_progress.py",
                "completed_base_reader",
            )
            metrics = reader.metrics_from_log((args.base_run / "train.log").read_bytes())
            assert all("actor/grad_norm" not in m for m in metrics)
        except Exception as error:
            acceptance_error = repr(error)
    write(
        args.base_run / "exit.json",
        {
            "time": time.time(),
            "exit_code": code,
            "wall_limit_reached": expired,
            "complete_validation_only": code == 0 and acceptance_error is None,
            "acceptance_error": acceptance_error,
        },
    )
    return code if code != 0 else int(acceptance_error is not None)


def resume_training(args):
    native_plan(args.resume_run)
    assert (args.base_run / "exit.json").exists(), "Wait for the owned base invocation to finish"
    assert not gpu_pids()
    core().launch(args.resume_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = {
        "prepare-resume": prepare_resume,
        "pause-training": pause_training,
        "inspect-base": inspect_base,
        "launch-base": launch_base,
        "supervise-base": supervise_base,
        "resume-training": resume_training,
    }
    parser.add_argument("action", choices=actions)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--resume-run", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    arguments = parser.parse_args()
    sys.exit(actions[arguments.action](arguments) or 0)
