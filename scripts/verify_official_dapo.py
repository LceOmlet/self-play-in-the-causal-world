#!/usr/bin/env python3
"""Verify official training-source provenance; contains no training algorithm."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

RUNTIME_PATCH_PATHS = {
    "verl": {
        "verl/tools/schemas.py",
        "verl/experimental/agent_loop/tool_agent_loop.py",
    },
    "verl-recipe": {"dapo/dapo_ray_trainer.py"},
}

# The recipe exception admits exactly the upstream-style action-mask guard,
# not an arbitrary replacement of this trainer's algorithm code.
RECIPE_ACTION_MASK_PATCH_SHA256 = "a4541de892529cc29d8c03819c207c1786e9bb91b51a7d4a50896886075406e7"


def verify_repositories(manifest, roots, project):
    """Keep original hashes; permit only separately recorded runtime patches."""
    patched = {}
    patch_reports = []
    for patch in manifest.get("reviewed_runtime_patches", []):
        name = patch["repository"]
        definition = manifest["repositories"][name]
        if patch["base_commit"] != definition["commit"]:
            raise RuntimeError("Runtime patch has the wrong upstream base")
        patch_path = (project / patch["path"]).resolve()
        if not patch_path.is_relative_to(project.resolve()):
            raise RuntimeError("Runtime patch escaped the project")
        if hashlib.sha256(patch_path.read_bytes()).hexdigest() != patch["sha256"]:
            raise RuntimeError("Reviewed runtime patch content changed")
        if name == "verl-recipe" and patch["sha256"] != RECIPE_ACTION_MASK_PATCH_SHA256:
            raise RuntimeError(
                "Recipe patch must match the reviewed action-mask compatibility change"
            )
        declared = set(patch["files"])
        diff_paths = {
            line.removeprefix("+++ b/")
            for line in patch_path.read_text().splitlines()
            if line.startswith("+++ b/")
        }
        if diff_paths != declared or not declared <= RUNTIME_PATCH_PATHS.get(name, set()):
            raise RuntimeError("Runtime patch attempts to change undeclared or algorithm sources")
        for relative, hashes in patch["files"].items():
            if hashes["upstream_sha256"] != definition["files"][relative]:
                raise RuntimeError("Runtime patch replaced the original upstream fingerprint")
            key = (name, relative)
            if key in patched:
                raise RuntimeError("Overlapping runtime patches are not supported")
            patched[key] = hashes["patched_sha256"]
        patch_reports.append(patch)
    repositories = {}
    for name, definition in manifest["repositories"].items():
        root = roots[name]
        for relative, upstream_hash in definition["files"].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise RuntimeError(f"Official source escaped its root: {path}")
            expected = patched.get((name, relative), upstream_hash)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Official DAPO source differs from reviewed upstream: {path}")
        repositories[name] = {
            "root": str(root),
            "commit": definition["commit"],
            "verified_files": len(definition["files"]),
            "reviewed_runtime_patch_files": sum(key[0] == name for key in patched),
        }
    return repositories, patch_reports


def verify():
    project = Path(__file__).resolve().parents[1]
    manifest = json.loads((project / "configs/verl/upstream_sources.json").read_text())
    roots = {
        "verl": Path(os.environ["VERL_ROOT"]).resolve(),
        "verl-recipe": Path(os.environ["VERL_RECIPE_ROOT"]).resolve(),
    }
    repositories, patches = verify_repositories(manifest, roots, project)
    report = {
        "repositories": repositories,
        "reviewed_runtime_patches": patches,
        "loaded": {},
        "packages": {},
    }
    for module, repository in {
        "dapo.main_dapo": "verl-recipe",
        "dapo.dapo_ray_trainer": "verl-recipe",
        "verl.trainer.ppo.core_algos": "verl",
        "verl.workers.engine_workers": "verl",
        "verl.experimental.agent_loop.tool_agent_loop": "verl",
        "verl.experimental.reward_loop.reward_manager.dapo": "verl",
    }.items():
        loaded = importlib.import_module(module)
        source = Path(loaded.__file__).resolve()
        if not source.is_relative_to(roots[repository]):
            raise RuntimeError(f"Training imported a different implementation: {module}: {source}")
        report["loaded"][module] = str(source)
    if importlib.util.find_spec("trl") is not None:
        raise RuntimeError(
            "Use the independent official verl environment, which has no TRL installed"
        )
    for package in ["verl", "torch", "transformers", "vllm", "peft", "ray", "datasets"]:
        report["packages"][package] = importlib.metadata.version(package)
    import pyarrow.parquet as pq
    from cpt_world.identification import INTERACTION_SURFACE_VERSION

    report["environment_version"] = INTERACTION_SURFACE_VERSION
    report["project_sources"] = {
        str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((project / "src/cpt_world").glob("*.py"))
    }
    report["datasets"] = {}
    for role, variable in [("train", "CPT_WORLD_TRAIN_DATA"), ("validation", "CPT_WORLD_VAL_DATA")]:
        path = Path(os.environ[variable]).resolve()
        count = 0
        for batch in pq.ParquetFile(path).iter_batches(columns=["extra_info"]):
            for row in batch.to_pylist():
                if row["extra_info"].get("environment_version") != INTERACTION_SURFACE_VERSION:
                    raise RuntimeError(f"Regenerate obsolete {role} data: {path}")
                count += 1
        if count == 0:
            raise RuntimeError(f"Empty {role} data: {path}")
        report["datasets"][role] = {
            "path": str(path),
            "rows": count,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify()
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("OFFICIAL_DAPO_SOURCES=" + json.dumps(report, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
