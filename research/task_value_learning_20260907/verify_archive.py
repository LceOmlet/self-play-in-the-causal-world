"""Verify the two completed archives and the uncompressed evidence provenance."""

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
counts = {}
for name in ("task_value_learning_20260907", "decision_budget_20260907"):
    directory = ROOT.parent / name
    manifest = json.loads((directory / "SHA256SUMS.json").read_text(encoding="utf-8"))
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.json" and "__pycache__" not in path.parts
    }
    assert actual == set(manifest), (name, actual.symmetric_difference(manifest))
    for relative, digest in manifest.items():
        assert hashlib.sha256((directory / relative).read_bytes()).hexdigest() == digest, relative
    counts[name] = len(manifest)
provenance = json.loads((ROOT / "input-provenance.json").read_text(encoding="utf-8"))
for name, record in provenance.items():
    data = gzip.decompress((ROOT / (name + ".gz")).read_bytes())
    assert len(data) == record["uncompressed_bytes"], name
    assert hashlib.sha256(data).hexdigest() == record["uncompressed_sha256"], name
print(json.dumps({"passed": True, "archive_files": counts, "input_files": len(provenance)}))
