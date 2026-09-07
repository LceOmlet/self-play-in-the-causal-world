"""Resume the existing verification on the exact reviewed progress sources.

Process/configuration/checkpoint control only. The official DAPO module owns
generation and every training operation. This retains the verification's
original 500-update target and wall deadline; it is not final 10,000-step work.
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

PROJECT = Path("/home/chen/projects/self-play-in-the-causal-world-rewardv10-a92ab8e-20260831")
ROOT = Path("/home/chen/runs/dapo-progress-20260908")
CURRENT = Path("/home/chen/runs/inference-efficiency-20260908/resume-compiled-02")
VERL = Path("/home/chen/vendor/dapo-official-20260906/verl-progress-candidate-v2")
RECIPE = Path("/home/chen/vendor/dapo-official-20260906/verl-recipe-progress-candidate-v2")
PROFILE = "progress-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def helper():
    path = PROJECT / "research/inference_efficiency_20260908/resume_runtime.py"
    assert sha(path) == "e93f1cd212182733644492021a1f23553966bed7cbb364e5d51787644a12023f"
    spec = importlib.util.spec_from_file_location("resume_runtime_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_project():
    accepted = json.loads(
        Path("/home/chen/runs/training-submission-20260907/preflight-acceptance.json").read_text()
    )
    assert accepted["passed"]
    stage = ROOT / "provenance-v2-project"
    for name, digest in accepted["project_files"].items():
        if name == "scripts/verify_official_dapo.py":
            assert sha(PROJECT / name) == sha(stage / name)
        else:
            assert sha(PROJECT / name) == digest, name
    for name in [
        "configs/verl/upstream_sources_progress_v1.json",
        "patches/verl-dapo-progress-v1.patch",
        "patches/verl-recipe-mask-progress-v1.patch",
    ]:
        assert sha(PROJECT / name) == sha(stage / name), name


def environment(run):
    env = helper().original_controller().environment()
    env.update(
        VERL_ROOT=str(VERL),
        VERL_RECIPE_ROOT=str(RECIPE),
        CPT_WORLD_DAPO_SOURCE_PROFILE=PROFILE,
        CPT_WORLD_RUN_DIR=str(run),
        CPT_WORLD_VERL_AUDIT_DIR=str(run / "environment"),
        PYTHONPATH=":".join(map(str, [PROJECT / "src", PROJECT, RECIPE, VERL])),
        RAY_TMPDIR="/tmp/cpt-" + run.name,
    )
    return env


def identified_current():
    import psutil

    identity = json.loads((CURRENT / "process.json").read_text())
    process = psutil.Process(identity["pid"])
    assert abs(process.create_time() - identity["create_time"]) < 0.01
    assert process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    assert "dapo.main_dapo" in process.cmdline()
    return process


def prepare(run):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    check_project()
    identified_current()
    current_plan = json.loads((CURRENT / "prepared.json").read_text())
    step = int((CURRENT / "checkpoints/latest_checkpointed_iteration.txt").read_text())
    assert 5 <= step < 50, "This migration proves only a first-epoch legacy checkpoint"
    run.mkdir(parents=True, exist_ok=False)
    checkpoint = run / "preserved" / f"global_step_{step}"
    env = environment(run)
    cpu = env | {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    replaced = (
        "trainer.resume_from_path=",
        "trainer.validation_data_dir=",
        "trainer.experiment_name=",
        "++ray_kwargs.ray_init.runtime_env.env_vars.",
    )
    overrides = [x for x in current_plan["command"][2:] if not x.startswith(replaced)]
    overrides += [
        f"trainer.resume_from_path={checkpoint}",
        f"trainer.validation_data_dir={run}/validation",
        "trainer.experiment_name=verified-resume-progress",
    ]
    for key in [
        "CPT_WORLD_VERL_AUDIT_DIR",
        "CPT_WORLD_EXPECTED_SOURCE",
        "VERL_ROOT",
        "VERL_RECIPE_ROOT",
        "CPT_WORLD_DAPO_SOURCE_PROFILE",
        "PYTHONPATH",
        "PATH",
    ]:
        overrides.append(f"++ray_kwargs.ray_init.runtime_env.env_vars.{key}={env[key]}")
    inspection = [
        sys.executable,
        "-m",
        "dapo.main_dapo",
        f"hydra.searchpath=[file://{VERL}/verl/trainer/config,file://{PROJECT}/configs/verl]",
        "+profiles@_global_=cpt_world_dapo",
        *overrides,
        "--cfg",
        "job",
        "--resolve",
    ]
    result = subprocess.run(inspection, env=cpu, cwd=PROJECT, capture_output=True, text=True)
    (run / "resolved.yaml").write_text(result.stdout)
    (run / "configuration.stderr").write_text(result.stderr)
    result.check_returncode()
    from omegaconf import OmegaConf

    config = OmegaConf.create(result.stdout)
    old = OmegaConf.load(CURRENT / "resolved.yaml")
    spec = importlib.util.spec_from_file_location(
        "configuration_diff",
        PROJECT / "research/inference_efficiency_20260908/validate_candidates.py",
    )
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    before, after = [
        compare.flatten(OmegaConf.to_container(x, resolve=True)) for x in (old, config)
    ]
    diff = {
        k: {"before": before.get(k), "after": after.get(k)}
        for k in sorted(before.keys() | after.keys())
        if before.get(k) != after.get(k)
    }
    allowed = {
        "trainer.resume_from_path",
        "trainer.default_local_dir",
        "trainer.rollout_data_dir",
        "trainer.validation_data_dir",
        "trainer.experiment_name",
        "actor_rollout_ref.rollout.trace.experiment_name",
    }
    allowed.update(
        "ray_kwargs.ray_init.runtime_env.env_vars." + k
        for k in [
            "CPT_WORLD_VERL_AUDIT_DIR",
            "VERL_ROOT",
            "VERL_RECIPE_ROOT",
            "PYTHONPATH",
            "CPT_WORLD_DAPO_SOURCE_PROFILE",
        ]
    )
    assert set(diff) <= allowed, diff
    assert config.trainer.total_training_steps == 500 and config.trainer.total_epochs == 20
    sys.path[:0] = [str(RECIPE), str(VERL)]
    from verl.utils.config import validate_config

    validate_config(config, use_reference_policy=False, use_critic=False)
    proof = subprocess.run(
        [
            sys.executable,
            str(PROJECT / "scripts/verify_official_dapo.py"),
            "--output",
            str(run / "preflight-source-verification.json"),
        ],
        env=cpu,
        cwd=PROJECT,
        capture_output=True,
        text=True,
    )
    (run / "source-verification.log").write_text(proof.stdout + proof.stderr)
    proof.check_returncode()

    # Preserve a published native checkpoint before touching its live process.
    source = CURRENT / "checkpoints" / f"global_step_{step}"
    files = {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}
    shutil.copytree(source, checkpoint)
    assert all(sha(checkpoint / n) == digest for n, digest in files.items())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from dapo.dapo_ray_trainer import RayDAPOTrainer
    from torch.distributed.tensor import DTensor
    from torchdata.stateful_dataloader import StatefulDataLoader
    from verl.trainer.ppo.utils import create_rl_sampler

    assert (
        Path(RayDAPOTrainer.fit.__code__.co_filename).resolve()
        == (RECIPE / "dapo/dapo_ray_trainer.py").resolve()
    )

    def local(t):
        return t.to_local() if isinstance(t, DTensor) else t

    def load(name):
        return torch.load(checkpoint / name, map_location="cpu", weights_only=False)

    model = load("actor/model_world_size_1_rank_0.pt")
    assert len(model) == 496 and all(
        "lora_" in n and torch.isfinite(local(t)).all() for n, t in model.items()
    )
    states = [s for s in load("actor/optim_world_size_1_rank_0.pt")["state"].values() if s]
    assert len(states) == 496 and all(float(local(s["step"])) == step for s in states)
    assert all(torch.isfinite(local(s[k])).all() for s in states for k in ["exp_avg", "exp_avg_sq"])
    assert load("actor/extra_state_world_size_1_rank_0.pt")["lr_scheduler"]["last_epoch"] == step
    data = load("data.pt")
    assert step <= data["_num_yielded"] <= step * 10 < 500 and not data["_iterator_finished"]

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
    trainer.config = config
    trainer.global_steps = 0
    trainer.use_critic = False
    trainer.actor_rollout_wg = Mock()
    trainer.train_dataloader = loader()
    trainer._load_checkpoint()
    assert trainer.global_steps == step and trainer.gen_steps == data["_num_yielded"]
    assert trainer.data_epoch == 0
    iterator = iter(trainer.train_dataloader)
    actual = [int(next(iterator).item()) for _ in range(8)]
    iterator = iter(loader())
    for _ in range(data["_num_yielded"]):
        next(iterator)
    expected = [int(next(iterator).item()) for _ in range(8)]
    assert actual == expected and not torch.cuda.is_initialized()
    write(
        run / "checkpoint-acceptance.json",
        {
            "source": str(source),
            "checkpoint": str(checkpoint),
            "step": step,
            "files": files,
            "finite_lora_tensors": 496,
            "finite_optimizer_states": 496,
            "finite_optimizer_moments": 992,
            "generated_batches": data["_num_yielded"],
            "data_epoch": 0,
            "expected_next_rows": expected,
            "restored_next_rows": actual,
            "scope": "CPU native content and actual candidate data-load audit; "
            "GPU loading RPC mocked.",
        },
    )
    write(
        run / "prepared.json",
        {
            "time": time.time(),
            "step": step,
            "checkpoint": str(checkpoint),
            "controller_sha256": sha(__file__),
            "resolved_sha256": sha(run / "resolved.yaml"),
            "differences": diff,
            "wall_deadline": current_plan["wall_deadline"],
            "command": ["bash", str(PROJECT / "scripts/run_official_dapo.sh"), *overrides],
        },
    )
    print(json.dumps({"prepared": str(run), "step": step, "next_rows": actual}), flush=True)


def pause(run):
    check_project()
    plan = json.loads((run / "prepared.json").read_text())
    assert sha(__file__) == plan["controller_sha256"]
    assert sha(run / "resolved.yaml") == plan["resolved_sha256"]
    assert (
        int((CURRENT / "checkpoints/latest_checkpointed_iteration.txt").read_text()) == plan["step"]
    ), "A newer native update is available; prepare from that checkpoint before pausing"
    accepted = json.loads((run / "checkpoint-acceptance.json").read_text())
    assert all(sha(Path(accepted["checkpoint"]) / n) == h for n, h in accepted["files"].items())
    process = identified_current()
    assert not (run / "paused-source.json").exists()
    identities = helper().terminate_tree(process)
    write(
        run / "paused-source.json",
        {
            "time": time.time(),
            "processes": identities,
            "reason": "Resume preserved native state on verified official progress source profile.",
        },
    )


def launch(run):
    check_project()
    assert (run / "paused-source.json").exists() and not (run / "supervisor.json").exists()
    plan = json.loads((run / "prepared.json").read_text())
    assert sha(__file__) == plan["controller_sha256"] and time.time() < plan["wall_deadline"]
    assert sha(run / "resolved.yaml") == plan["resolved_sha256"]
    assert not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    with (run / "train.log").open("x") as output:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "supervise", "--run", str(run)],
            env=environment(run),
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write(run / "supervisor.json", {"pid": process.pid, "time": time.time()})
    print(json.dumps({"launched": str(run), "supervisor": process.pid}), flush=True)


def supervise(run):
    import psutil

    plan = json.loads((run / "prepared.json").read_text())
    process = subprocess.Popen(plan["command"], env=environment(run), start_new_session=True)
    write(
        run / "process.json",
        {"pid": process.pid, "create_time": psutil.Process(process.pid).create_time()},
    )
    expired = False
    try:
        code = process.wait(timeout=max(1, plan["wall_deadline"] - time.time()))
    except subprocess.TimeoutExpired:
        expired = True
        helper().terminate_tree(psutil.Process(process.pid))
        code = process.wait(timeout=5)
    write(
        run / "exit.json", {"time": time.time(), "exit_code": code, "wall_limit_reached": expired}
    )
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "pause", "launch", "supervise"])
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    sys.exit(globals()[args.action](args.run) or 0)
