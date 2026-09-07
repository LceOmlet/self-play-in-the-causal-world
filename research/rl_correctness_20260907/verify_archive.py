"""Verify archived evidence and reviewed patch bytes using only the standard library."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]


def main():
    manifest = json.loads((ROOT / "archive-manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    upstream = json.loads(
        (PROJECT / "configs/verl/upstream_sources.json").read_text(encoding="utf-8")
    )
    for patch in upstream["reviewed_runtime_patches"]:
        path = (PROJECT / patch["path"]).resolve()
        assert path.is_relative_to(PROJECT), patch["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == patch["sha256"], patch["path"]
    print(
        json.dumps(
            {
                "passed": True,
                "archived_files": len(manifest),
                "reviewed_patch_files": len(upstream["reviewed_runtime_patches"]),
            }
        )
    )


if __name__ == "__main__":
    main()
