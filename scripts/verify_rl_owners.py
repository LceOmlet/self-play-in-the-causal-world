"""Verify the loaded production owners; contains no training implementation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path


def require_rl_owners() -> dict:
    manifest_path = Path(__file__).resolve().parents[1] / "patches" / "rl-owner-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loaded = {}
    for package, expected in manifest["packages"].items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"Unverified RL dependency: {package}={actual}, expected {expected}")
    for module, expected in manifest["module_sha256"].items():
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"RL owner is missing: {module}")
        path = Path(spec.origin).resolve()
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Unverified RL owner source: {module} at {path}")
        loaded[module] = {"path": str(path), "sha256": actual}
    from trl import GRPOTrainer
    from trl.trainer.grpo_trainer import GRPOTrainer as Owner

    if GRPOTrainer is not Owner or Owner.__module__ != "trl.trainer.grpo_trainer":
        raise RuntimeError("Production training must use the owning TRL GRPOTrainer")
    return {"packages": manifest["packages"], "loaded": loaded}


if __name__ == "__main__":
    print(json.dumps(require_rl_owners(), indent=2))
