"""Preserve and resume native DAPO state with continual task supply.

Only process, configuration and checkpoint orchestration lives here. The
pinned official launcher owns every model rollout and training update.
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

import psutil

VERL = Path("/home/chen/vendor/dapo-official-20260906/verl-progress-candidate-v2")
RECIPE = Path("/home/chen/vendor/dapo-official-20260906/verl-recipe-progress-candidate-v2")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def old_helper(project):
    path = project / "research/inference_efficiency_20260908/resume_runtime.py"
    assert sha(path) == "e93f1cd212182733644492021a1f23553966bed7cbb364e5d51787644a12023f"
    spec = importlib.util.spec_from_file_location("accepted_process_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment(project, run):
    env = old_helper(project).original_controller().environment()
    env.update(
        VERL_ROOT=str(VERL), VERL_RECIPE_ROOT=str(RECIPE),
        CPT_WORLD_DAPO_SOURCE_PROFILE="progress-v1", CPT_WORLD_PROJECT=str(project),
        CPT_WORLD_RUN_DIR=str(run), CPT_WORLD_TRAIN_DATA=str(run / "stream.json"),
        CPT_WORLD_VERL_AUDIT_DIR=str(run / "environment"),
        CPT_WORLD_EXPECTED_SOURCE=str(project / "src/cpt_world"),
        PYTHONPATH=":".join(map(str, [project / "src", project, RECIPE, VERL])),
        RAY_TMPDIR="/tmp/cpt-" + run.name,
    )
    return env


def pause(args):
    identity = read(args.old_run / "process.json")
    process = psutil.Process(identity["pid"])
    assert abs(process.create_time() - identity["create_time"]) < 0.01
    assert "dapo.main_dapo" in process.cmdline() and os.getpgid(process.pid) == process.pid
    args.hold.mkdir(parents=True, exist_ok=False)
    tracker = args.old_run / "checkpoints/latest_checkpointed_iteration.txt"
    step = int(tracker.read_text())
    source = args.old_run / "checkpoints" / f"global_step_{step}"
    for name in (
        "data.pt", "dapo_progress.json", "actor/model_world_size_1_rank_0.pt",
        "actor/optim_world_size_1_rank_0.pt", "actor/extra_state_world_size_1_rank_0.pt",
    ):
        assert (source / name).is_file(), name
    files = {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}
    preserved = args.hold / source.name
    shutil.copytree(source, preserved)
    assert all(sha(source / name) == value == sha(preserved / name) for name, value in files.items())
    write(args.hold / "initial-preservation.json", {
        "time": time.time(), "old_run": str(args.old_run), "identity": identity,
        "checkpoint": str(preserved), "step": step, "files": files,
        "reason": "User requires continual new tasks; migrate fixed-cohort data state.",
    })
    identities = old_helper(args.project).terminate_tree(process)
    remaining = []
    for item in identities:
        try:
            member = psutil.Process(item["pid"])
            if abs(member.create_time() - item["create_time"]) < 0.01 and member.status() != psutil.STATUS_ZOMBIE:
                remaining.append(item)
        except psutil.NoSuchProcess:
            pass
    assert not remaining
    # A complete checkpoint can be published between the initial copy and the
    # process termination. The stopped process tree can no longer advance it.
    final_step = int(tracker.read_text())
    assert final_step >= step
    if final_step != step:
        step = final_step
        source = args.old_run / "checkpoints" / f"global_step_{step}"
        files = {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}
        assert {"data.pt", "dapo_progress.json"} <= files.keys()
        preserved = args.hold / source.name
        shutil.copytree(source, preserved)
        assert all(sha(source / name) == value == sha(preserved / name) for name, value in files.items())
    write(args.hold / "preserved.json", {
        "time": time.time(), "old_run": str(args.old_run), "identity": identity,
        "checkpoint": str(preserved), "step": step, "files": files,
        "reason": "User requires continual new tasks; migrate fixed-cohort data state.",
    })
    write(args.hold / "stopped.json", {"time": time.time(), "step": step, "processes": identities})
    print(json.dumps({"preserved_step": step, "checkpoint": str(preserved)}), flush=True)


def flatten(value, prefix=""):
    if isinstance(value, dict):
        return {key: item for name, child in value.items()
                for key, item in flatten(child, f"{prefix}.{name}" if prefix else name).items()}
    return {prefix: value}


def prepare(args):
    saved = read(args.hold / "preserved.json")
    source = Path(saved["checkpoint"])
    assert all(sha(source / name) == value for name, value in saved["files"].items())
    old_plan = read(args.old_run / "prepared.json")
    progress = read(source / "dapo_progress.json")
    args.run.mkdir(parents=True, exist_ok=False)
    env = environment(args.project, args.run)
    os.environ.update(CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    sys.path[:0] = [str(args.project / "src"), str(args.project), str(RECIPE), str(VERL)]
    from cpt_world.identification import INTERACTION_SURFACE_VERSION
    from cpt_world.verl_streaming_dataset import current_source_fingerprints

    write(args.run / "stream.json", {
        "version": 1, "kind": "cpt_world_continuous", "environment_version": INTERACTION_SURFACE_VERSION,
        "start_seed": args.start_seed, "stream_start_index": progress["generated_batches"],
        "journal_dir": str(args.run / "task-journal"), "source_fingerprints": current_source_fingerprints(),
    })
    checkpoint = args.run / "preserved" / source.name
    changes = {
        "data.shuffle": "false", "trainer.resume_from_path": str(checkpoint),
        "trainer.validation_data_dir": str(args.run / "validation"),
        "trainer.test_freq": "-1", "trainer.val_before_train": "false",
    }
    overrides = [item for item in old_plan["command"][2:]
                 if item.partition("=")[0] not in changes
                 and not item.startswith("++ray_kwargs.ray_init.runtime_env.env_vars.")]
    overrides.extend(f"{key}={value}" for key, value in changes.items())
    for key in ("CPT_WORLD_VERL_AUDIT_DIR", "CPT_WORLD_EXPECTED_SOURCE", "VERL_ROOT", "VERL_RECIPE_ROOT",
                "CPT_WORLD_DAPO_SOURCE_PROFILE", "PYTHONPATH", "PATH"):
        overrides.append(f"++ray_kwargs.ray_init.runtime_env.env_vars.{key}={env[key]}")
    cpu = env | {"CUDA_VISIBLE_DEVICES": "", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    inspect = [sys.executable, "-m", "dapo.main_dapo",
               f"hydra.searchpath=[file://{VERL}/verl/trainer/config,file://{args.project}/configs/verl]",
               "+profiles@_global_=cpt_world_dapo", *overrides, "--cfg", "job", "--resolve"]
    result = subprocess.run(inspect, cwd=args.project, env=cpu, capture_output=True, text=True)
    (args.run / "resolved.yaml").write_text(result.stdout)
    (args.run / "configuration.stderr").write_text(result.stderr)
    result.check_returncode()
    from omegaconf import OmegaConf
    from verl.utils.config import validate_config

    config = OmegaConf.create(result.stdout)
    old_config = OmegaConf.load(args.old_run / "resolved.yaml")
    before, after = [flatten(OmegaConf.to_container(c, resolve=True)) for c in (old_config, config)]
    diff = {key: {"before": before.get(key), "after": after.get(key)}
            for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    allowed = {
        "data.train_files", "data.shuffle", "data.custom_cls.path", "data.custom_cls.name",
        "trainer.resume_from_path", "trainer.default_local_dir", "trainer.rollout_data_dir",
        "trainer.validation_data_dir",
    }
    allowed.update("ray_kwargs.ray_init.runtime_env.env_vars." + key
                   for key in ("CPT_WORLD_VERL_AUDIT_DIR", "CPT_WORLD_EXPECTED_SOURCE", "PYTHONPATH"))
    # A CPU preflight may use an isolated checkout of this same project. Only
    # equivalent project-relative configuration paths may follow that checkout.
    prior_project = str(Path(old_plan["command"][1]).parents[1])
    for key, values in diff.items():
        if (isinstance(values["before"], str) and isinstance(values["after"], str)
                and values["before"].startswith(prior_project + "/")
                and values["after"] == str(args.project) + values["before"][len(prior_project):]):
            allowed.add(key)
    assert set(diff) <= allowed, diff
    assert config.trainer.total_training_steps == old_config.trainer.total_training_steps
    assert config.trainer.total_epochs == old_config.trainer.total_epochs
    assert config.trainer.resume_mode == "resume_path"
    assert config.trainer.test_freq == -1 and not config.trainer.val_before_train
    assert not config.data.shuffle and not config.actor_rollout_ref.rollout.enforce_eager
    validate_config(config, use_reference_policy=False, use_critic=False)
    # The dedicated migration program copies native weights/state unchanged and
    # validates the new data cursor through the actual official load hook.
    from cpt_world.verl_streaming_dataset import CPTWorldStreamingDataset
    from scripts.migrate_dapo_continuous_data import migrate

    dataset = CPTWorldStreamingDataset(
        data_files=[str(args.run / "stream.json")], tokenizer=None, config=config.data,
    )
    migration = migrate(source, checkpoint, args.run / "stream.json", dataset)
    write(args.run / "checkpoint-acceptance.json", migration)
    proof = subprocess.run([sys.executable, str(args.project / "scripts/verify_official_dapo.py"),
                            "--output", str(args.run / "preflight-source-verification.json")],
                           env=cpu, cwd=args.project, capture_output=True, text=True)
    (args.run / "source-verification.log").write_text(proof.stdout + proof.stderr)
    proof.check_returncode()
    write(args.run / "prepared.json", {
        "time": time.time(), "step": saved["step"], "generated_batches": progress["generated_batches"],
        "checkpoint": str(checkpoint), "source_run": str(args.old_run), "hold": str(args.hold),
        "controller_sha256": sha(Path(__file__)), "resolved_sha256": sha(args.run / "resolved.yaml"),
        "differences": diff, "wall_deadline": old_plan["wall_deadline"],
        "command": ["bash", str(args.project / "scripts/run_official_dapo.sh"), *overrides],
        "project": str(args.project),
    })
    print(json.dumps({"prepared": str(args.run), "step": saved["step"]}), flush=True)


def launch(args):
    plan = read(args.run / "prepared.json")
    assert args.project.resolve() == Path(plan["project"]).resolve()
    assert sha(Path(__file__)) == plan["controller_sha256"]
    assert sha(args.run / "resolved.yaml") == plan["resolved_sha256"]
    for identity in read(Path(plan["hold"]) / "stopped.json")["processes"]:
        try:
            process = psutil.Process(identity["pid"])
            assert abs(process.create_time() - identity["create_time"]) >= 0.01 or process.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass
    assert time.time() < plan["wall_deadline"]
    assert not (args.run / "supervisor.json").exists()
    assert not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    with (args.run / "train.log").open("x") as stream:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "supervise", "--project", str(args.project),
             "--run", str(args.run)], env=environment(args.project, args.run), stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    write(args.run / "supervisor.json", {"pid": process.pid, "time": time.time()})
    print(json.dumps({"launched": str(args.run), "supervisor": process.pid}), flush=True)


def supervise(args):
    plan = read(args.run / "prepared.json")
    process = subprocess.Popen(plan["command"], env=environment(args.project, args.run), start_new_session=True)
    write(args.run / "process.json", {"pid": process.pid, "create_time": psutil.Process(process.pid).create_time()})
    expired = False
    try:
        code = process.wait(timeout=max(1, plan["wall_deadline"] - time.time()))
    except subprocess.TimeoutExpired:
        expired = True
        old_helper(args.project).terminate_tree(psutil.Process(process.pid))
        code = process.wait(timeout=5)
    write(args.run / "exit.json", {"time": time.time(), "exit_code": code, "wall_limit_reached": expired})
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pause", "prepare", "launch", "supervise"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--old-run", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--hold", type=Path)
    parser.add_argument("--start-seed", type=int, default=4000000)
    args = parser.parse_args()
    sys.exit(globals()[args.action](args) or 0)
