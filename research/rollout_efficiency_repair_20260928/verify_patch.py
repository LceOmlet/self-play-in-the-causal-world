"""Check the registered source diff and lifecycle without loading a model."""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager, nullcontext
from functools import wraps
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def methods(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GRPOTrainer")
    return {n.name: n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def verify_lifecycle(path: Path) -> list[str]:
    """Execute the actual context/decorator AST; no copied production logic."""
    tree = ast.parse(path.read_text(encoding="utf8"))
    decorator = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == "with_vllm_rollout_session")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VLLMGeneration")
    context = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "rollout_session")
    namespace = {"contextmanager": contextmanager, "nullcontext": nullcontext, "wraps": wraps}
    exec(compile(ast.Module(body=[decorator, context], type_ignores=[]), str(path), "exec"), namespace)
    engine_type = type("LifecycleFixture", (), {"rollout_session": namespace["rollout_session"]})

    def engine(enabled=True):
        obj = engine_type()
        obj.mode = "colocate"
        obj.enable_sleep_mode = True
        obj.rollout_residency = enabled
        obj._rollout_session_depth = 0
        obj.events = []
        obj.llm = SimpleNamespace(sleep=lambda **kwargs: obj.events.append(kwargs))
        return obj

    normal = engine()
    with normal.rollout_session():
        with normal.rollout_session():
            assert not normal.events
        assert not normal.events
    assert normal.events == [{"level": 2}] and normal._rollout_session_depth == 0
    assert normal._llm_weights_sleeping and normal._llm_kv_cache_sleeping
    exceptional = engine()
    try:
        with exceptional.rollout_session():
            raise ValueError("exercise cleanup")
    except ValueError:
        pass
    assert exceptional.events == [{"level": 2}] and exceptional._rollout_session_depth == 0
    disabled = engine(False)
    with disabled.rollout_session():
        pass
    assert not disabled.events
    trainer = SimpleNamespace(vllm_generation=engine())

    @namespace["with_vllm_rollout_session"]
    def generate(owner):
        assert not owner.vllm_generation.events
        return "official body placeholder"

    assert generate(trainer) == "official body placeholder"
    assert trainer.vllm_generation.events == [{"level": 2}]
    return ["nested normal return", "exceptional cleanup", "disabled mode", "decorator boundary"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--patched", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "source-contract.json").read_text())
    patch = here.parents[1] / "patches/trl-1.14.0-rollout-residency.patch"
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == manifest["patch_sha256"]
    assert (args.upstream / "VERSION").read_text().strip() == "1.14.0"
    assert (args.patched / "VERSION").read_text().strip() == manifest["version"]
    file_sets = [{p.relative_to(root) for p in (root / "trl").rglob("*.py")}
                 for root in [args.upstream, args.patched]]
    assert file_sets[0] == file_sets[1], "Unexpected added or removed Python files"
    changed = sorted(str(p).replace("\\", "/") for p in file_sets[0]
                     if (args.upstream / p).read_bytes() != (args.patched / p).read_bytes())
    assert changed == sorted(p for p in manifest["declared_changed_files"] if p.endswith(".py"))
    relative = "trl/trainer/grpo_trainer.py"
    before, after = methods(args.upstream / relative), methods(args.patched / relative)
    assert before.keys() == after.keys()
    checks = {}
    for name, original in before.items():
        if name == "__init__":
            continue
        candidate = after[name]
        if name == "_generate":
            assert ast.dump(candidate.decorator_list.pop()) == ast.dump(ast.Name(id="with_vllm_rollout_session", ctx=ast.Load()))
        checks[name] = ast.dump(original, include_attributes=False) == ast.dump(candidate, include_attributes=False)
    assert checks == manifest["trainer_method_body_checks"] and all(checks.values())
    print(json.dumps({"unchanged_trainer_methods": len(checks), "registered_changed_python_files": changed,
                      "lifecycle_cases": verify_lifecycle(args.patched / "trl/generation/vllm_generation.py"),
                      "scope": "CPU source/lifecycle check; not a new GPU, numerical or throughput measurement"}, indent=2))


if __name__ == "__main__":
    main()
