"""Resolve a matched base evaluation through the actual official entry point.

CPU preparation only. This neither launches GPU work nor implements evaluation
or training. The official val_only path owns the eventual model evaluation.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    args.output.mkdir(parents=True, exist_ok=False)
    controller_path = args.project / "research/dapo_progress_20260908/resume_progress.py"
    assert (
        sha(controller_path) == "175ae0a21e2a70bd07d5f7716c449c85425aca33f6ad7a692be5908d6d1d3711"
    )
    controller = module(controller_path, "accepted_progress_controller")
    controller.check_project()
    env = controller.environment(args.output)
    cpu = env | {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    submitted = json.loads((args.training_run / "prepared.json").read_text())
    assert submitted["resolved_sha256"] == sha(args.training_run / "resolved.yaml")
    replacement = {
        "trainer.val_before_train": "true",
        "trainer.val_only": "true",
        "trainer.resume_mode": "disable",
        "trainer.resume_from_path": "null",
        "trainer.validation_data_dir": str(args.output / "validation"),
        "trainer.experiment_name": "matched-base-validation",
    }
    overrides = [
        item
        for item in submitted["command"][2:]
        if item.split("=", 1)[0] not in replacement
        and not item.startswith("++ray_kwargs.ray_init.runtime_env.env_vars.")
    ]
    overrides += [f"{key}={value}" for key, value in replacement.items()]
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
    invocation = [
        sys.executable,
        "-m",
        "dapo.main_dapo",
        f"hydra.searchpath=[file://{controller.VERL}/verl/trainer/config,"
        f"file://{args.project}/configs/verl]",
        "+profiles@_global_=cpt_world_dapo",
        *overrides,
    ]
    result = subprocess.run(
        invocation + ["--cfg", "job", "--resolve"],
        env=cpu,
        cwd=args.project,
        capture_output=True,
        text=True,
    )
    (args.output / "resolved.yaml").write_text(result.stdout)
    (args.output / "configuration.stderr").write_text(result.stderr)
    result.check_returncode()
    from omegaconf import OmegaConf

    config = OmegaConf.create(result.stdout)
    old = OmegaConf.load(args.training_run / "resolved.yaml")
    compare = module(
        args.project / "research/inference_efficiency_20260908/validate_candidates.py",
        "accepted_config_diff",
    )
    before, after = [
        compare.flatten(OmegaConf.to_container(c, resolve=True)) for c in [old, config]
    ]
    changes = {
        key: {"before": before.get(key), "after": after.get(key)}
        for key in sorted(before.keys() | after.keys())
        if before.get(key) != after.get(key)
    }
    allowed = set(replacement) | {
        "trainer.default_local_dir",
        "trainer.rollout_data_dir",
        "actor_rollout_ref.rollout.trace.experiment_name",
        "ray_kwargs.ray_init.runtime_env.env_vars.CPT_WORLD_VERL_AUDIT_DIR",
    }
    assert set(changes) <= allowed, changes
    assert config.trainer.val_only and config.trainer.val_before_train
    assert config.trainer.resume_mode == "disable" and config.trainer.resume_from_path is None
    assert config.actor_rollout_ref.model.lora_adapter_path is None
    assert config.actor_rollout_ref.rollout.val_kwargs.n == 1
    assert not config.actor_rollout_ref.rollout.val_kwargs.do_sample
    assert config.actor_rollout_ref.rollout.val_kwargs.temperature == 0
    assert not config.actor_rollout_ref.rollout.enforce_eager
    sys.path[:0] = [str(controller.RECIPE), str(controller.VERL)]
    from verl.utils.config import validate_config

    validate_config(config, use_reference_policy=False, use_critic=False)
    proof = subprocess.run(
        [
            sys.executable,
            str(args.project / "scripts/verify_official_dapo.py"),
            "--output",
            str(args.output / "source-verification.json"),
        ],
        env=cpu,
        cwd=args.project,
        capture_output=True,
        text=True,
    )
    (args.output / "source-verification.log").write_text(proof.stdout + proof.stderr)
    proof.check_returncode()
    report = {
        "configuration_only_passed": True,
        "training_run": str(args.training_run),
        "training_config_sha256": sha(args.training_run / "resolved.yaml"),
        "base_config_sha256": sha(args.output / "resolved.yaml"),
        "changes": changes,
        "command": ["bash", str(args.project / "scripts/run_official_dapo.sh"), *overrides],
        "controller_path": str(controller_path),
        "controller_sha256": sha(controller_path),
        "script_sha256": sha(Path(__file__)),
        "validation_sha256": sha(Path(config.data.val_files)),
        "scope": "Same fixed tasks, environment tapes, sampler and compiled runtime settings. "
        "No checkpoint or trained adapter loaded; official fresh LoRA initialization. "
        "Actual GPU baseline evaluation is pending. Source/data identity and exclusive GPU "
        "availability must be checked before launch.",
    }
    (args.output / "prepared.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"configuration_only_passed": True, "changes": changes}))


if __name__ == "__main__":
    main()
