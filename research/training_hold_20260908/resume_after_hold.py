"""Resume an explicitly stopped verification using the existing official launcher.

The stopped-process precondition differs from the earlier live-migration tool.
No trainer loop, numerical update or sampling implementation is supplied here.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import psutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--hold", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    stopped = read(args.hold / "stopped.json")
    saved = read(args.hold / "before-stop.json")
    assert stopped["stopped"] and stopped["step"] == saved["step"]
    for identity in stopped["processes"]:
        try:
            p = psutil.Process(identity["pid"])
            assert (
                abs(p.create_time() - identity["create_time"]) >= 0.01
                or p.status() == psutil.STATUS_ZOMBIE
            )
        except psutil.NoSuchProcess:
            pass
    assert not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    source = Path(saved["checkpoint"])
    assert all(sha(source / name) == value for name, value in saved["files"].items())
    core_path = args.project / "research/dapo_progress_20260908/resume_progress.py"
    core_sha = "175ae0a21e2a70bd07d5f7716c449c85425aca33f6ad7a692be5908d6d1d3711"
    assert sha(core_path) == core_sha
    spec = importlib.util.spec_from_file_location("accepted_controller", core_path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    core.check_project()
    args.run.mkdir(parents=True, exist_ok=False)
    checkpoint = args.run / "preserved" / source.name
    shutil.copytree(source, checkpoint)
    assert all(sha(checkpoint / name) == value for name, value in saved["files"].items())
    old_run = Path(saved["run"])
    old_plan = read(old_run / "prepared.json")
    env = core.environment(args.run)
    cpu = env | {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    changes = {
        "trainer.resume_from_path": str(checkpoint),
        "trainer.validation_data_dir": str(args.run / "validation"),
        "++ray_kwargs.ray_init.runtime_env.env_vars.CPT_WORLD_VERL_AUDIT_DIR": str(
            args.run / "environment"
        ),
    }
    overrides = [changes.get(item.partition("=")[0], None) for item in old_plan["command"][2:]]
    overrides = [
        item if value is None else item.partition("=")[0] + "=" + value
        for item, value in zip(old_plan["command"][2:], overrides, strict=True)
    ]
    command = ["bash", str(args.project / "scripts/run_official_dapo.sh"), *overrides]
    inspection = [
        sys.executable,
        "-m",
        "dapo.main_dapo",
        f"hydra.searchpath=[file://{core.VERL}/verl/trainer/config,file://{args.project}/configs/verl]",
        "+profiles@_global_=cpt_world_dapo",
        *overrides,
        "--cfg",
        "job",
        "--resolve",
    ]
    result = subprocess.run(inspection, env=cpu, cwd=args.project, capture_output=True, text=True)
    (args.run / "resolved.yaml").write_text(result.stdout)
    (args.run / "configuration.stderr").write_text(result.stderr)
    result.check_returncode()
    from omegaconf import OmegaConf

    config = OmegaConf.create(result.stdout)
    old_config = OmegaConf.load(old_run / "resolved.yaml")

    def flatten(value, prefix=""):
        if isinstance(value, dict):
            return {
                key: item
                for name, child in value.items()
                for key, item in flatten(child, f"{prefix}.{name}" if prefix else name).items()
            }
        return {prefix: value}

    before, after = [flatten(OmegaConf.to_container(c, resolve=True)) for c in [old_config, config]]
    diff = {
        k: {"before": before.get(k), "after": after.get(k)}
        for k in before.keys() | after.keys()
        if before.get(k) != after.get(k)
    }
    assert set(diff) == {
        "trainer.resume_from_path",
        "trainer.default_local_dir",
        "trainer.rollout_data_dir",
        "trainer.validation_data_dir",
        "ray_kwargs.ray_init.runtime_env.env_vars.CPT_WORLD_VERL_AUDIT_DIR",
    }, diff
    assert config.trainer.total_training_steps == 500 and config.trainer.total_epochs == 20
    assert not config.actor_rollout_ref.rollout.enforce_eager
    proof = subprocess.run(
        [
            sys.executable,
            str(args.project / "scripts/verify_official_dapo.py"),
            "--output",
            str(args.run / "preflight-source-verification.json"),
        ],
        env=cpu,
        cwd=args.project,
        capture_output=True,
        text=True,
    )
    (args.run / "source-verification.log").write_text(proof.stdout + proof.stderr)
    proof.check_returncode()
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    sys.path[:0] = [str(core.RECIPE), str(core.VERL)]
    import torch
    from dapo.dapo_ray_trainer import RayDAPOTrainer
    from torch.distributed.tensor import DTensor
    from torchdata.stateful_dataloader import StatefulDataLoader
    from verl.trainer.ppo.utils import create_rl_sampler
    from verl.utils.config import validate_config

    validate_config(config, use_reference_policy=False, use_critic=False)
    torch.set_num_threads(1)

    def local(t):
        return t.to_local() if isinstance(t, DTensor) else t

    def load(name):
        return torch.load(checkpoint / name, map_location="cpu", weights_only=False)

    step = saved["step"]
    model = load("actor/model_world_size_1_rank_0.pt")
    assert len(model) == 496 and all(
        "lora_" in k and torch.isfinite(local(t)).all() for k, t in model.items()
    )
    states = [
        state for state in load("actor/optim_world_size_1_rank_0.pt")["state"].values() if state
    ]
    assert len(states) == 496 and all(float(local(s["step"])) == step for s in states)
    assert all(torch.isfinite(local(s[k])).all() for s in states for k in ["exp_avg", "exp_avg_sq"])
    assert load("actor/extra_state_world_size_1_rank_0.pt")["lr_scheduler"]["last_epoch"] == step
    progress = read(checkpoint / "dapo_progress.json")
    data = load("data.pt")
    assert (
        progress["completed_updates"] == step
        and progress["generated_batches"] == data["_num_yielded"]
    )

    def loader():
        rows = list(range(500))
        return StatefulDataLoader(
            rows,
            batch_size=1,
            num_workers=0,
            drop_last=True,
            sampler=create_rl_sampler(config.data, rows),
        )

    trainer = object.__new__(RayDAPOTrainer)
    trainer.config, trainer.global_steps, trainer.use_critic = config, 0, False
    trainer.actor_rollout_wg = Mock()
    trainer.train_dataloader = loader()
    trainer._load_checkpoint()
    assert trainer.global_steps == step and trainer.gen_steps == progress["generated_batches"]
    assert trainer.data_epoch == progress["data_epoch"] == 0
    actual_it, expected_it = iter(trainer.train_dataloader), iter(loader())
    for _ in range(data["_num_yielded"]):
        next(expected_it)
    actual, expected = [[int(next(it).item()) for _ in range(8)] for it in [actual_it, expected_it]]
    assert actual == expected and not torch.cuda.is_initialized()
    acceptance = {
        "step": step,
        "generated_batches": progress["generated_batches"],
        "finite_lora_tensors": 496,
        "finite_optimizer_states": 496,
        "finite_moments": 992,
        "expected_next_rows": expected,
        "restored_next_rows": actual,
        "files": saved["files"],
        "scope": "CPU native content and official data restore; only GPU RPC mocked. "
        "Actual GPU resume awaits launch.",
    }
    write(args.run / "checkpoint-acceptance.json", acceptance)
    write(
        args.run / "prepared.json",
        {
            "time": time.time(),
            "step": step,
            "checkpoint": str(checkpoint),
            "controller_sha256": core_sha,
            "preparer_sha256": sha(Path(__file__)),
            "resolved_sha256": sha(args.run / "resolved.yaml"),
            "differences": diff,
            "wall_deadline": old_plan["wall_deadline"],
            "command": command,
            "source_run": str(old_run),
            "explicit_resumption_authorized": True,
        },
    )
    write(args.run / "paused-source.json", stopped)
    # Restore visibility before the unchanged launcher constructs the child env.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    core.launch(args.run)
    print(
        json.dumps(
            {
                "launched_from_step": step,
                "generated_batches": progress["generated_batches"],
                "next_rows": actual,
                "run": str(args.run),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
