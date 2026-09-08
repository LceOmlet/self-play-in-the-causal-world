"""Keep PEFT's FP32 trainable adapters for FSDP2 in an isolated official tree.

The FSDP1 flat-parameter compatibility cast stays intact. No loss, optimizer,
training loop, rollout implementation, or live source tree is replaced.
"""

import argparse
import difflib
import hashlib
import json
import shutil
from pathlib import Path

RELATIVE = "verl/workers/engine/fsdp/transformer_impl.py"
UPSTREAM_SHA = "5fe61752d52607dff63c3c4c3d513b2503e2bc851eb1646bf90f68f7ebe57a9c"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.source.resolve() != args.candidate.resolve()
    assert not args.candidate.exists()
    assert sha(args.source / RELATIVE) == UPSTREAM_SHA
    before = (args.source / RELATIVE).read_text()
    old = """            # FSDP requires all params in a flat group to share dtype: cast a
            # fp32 adapter to the bf16 base dtype only when they actually differ.
            base_dtype = next((p.dtype for p in module.parameters() if not p.requires_grad), None)
            if base_dtype is not None:
"""
    new = """            # Only FSDP1 flat groups require the frozen base and adapters
            # to share dtype. FSDP2 keeps PEFT's FP32 trainable parameters
            # for optimizer updates and casts them for mixed-precision compute.
            base_dtype = next((p.dtype for p in module.parameters() if not p.requires_grad), None)
            if self.engine_config.strategy == "fsdp" and base_dtype is not None:
"""
    assert before.count(old) == 1
    after = before.replace(old, new)
    shutil.copytree(
        args.source, args.candidate, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    (args.candidate / RELATIVE).write_text(after)
    args.output.mkdir(parents=True, exist_ok=True)
    patch = args.output / "verl-fsdp2-lora-fp32-v1.patch"
    patch.write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile="a/" + RELATIVE,
                tofile="b/" + RELATIVE,
            )
        )
    )
    changed = []
    checked = 0
    for path in args.source.rglob("*"):
        if (
            not path.is_file()
            or ".git" in path.parts
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
        ):
            continue
        relative = path.relative_to(args.source)
        checked += 1
        if sha(path) != sha(args.candidate / relative):
            changed.append(str(relative))
    assert changed == [RELATIVE]
    result = {
        "source": str(args.source),
        "candidate": str(args.candidate),
        "upstream_sha256": UPSTREAM_SHA,
        "patched_sha256": sha(args.candidate / RELATIVE),
        "patch_sha256": sha(patch),
        "checked_files": checked,
        "changed_files": changed,
        "status": "candidate; not installed in live training",
    }
    (args.output / "candidate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
