"""Check frozen evidence, answer provenance, legal charges, and complete cohort coverage."""

import gzip
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "SHA256SUMS.json").read_text(encoding="utf-8"))
actual = {path.name for path in ROOT.iterdir() if path.is_file() and path.name != "SHA256SUMS.json"}
assert actual == set(manifest)
for name, digest in manifest.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
with tarfile.open(ROOT / "frozen-worlds.tar.gz") as archive:
    assert sum(member.name.endswith(".pkl") for member in archive.getmembers()) == 50
    assert archive.getmember("finite-baselines.jsonl").isfile()
source = json.loads(gzip.decompress((ROOT / "rescored-answers.json.gz").read_bytes()))
episodes = source["episodes"]
assert len(episodes) == 100
assert len({row["seed_id"] for row in episodes}) == 50
assert set(Counter(row["seed_id"] for row in episodes).values()) == {2}
with tarfile.open(ROOT / "evidence.tar.gz") as archive:
    policy = archive.extractfile("original-code/public_policy.py").read()
    assert policy == (ROOT / "public_policy.py").read_bytes()
    original_runner = archive.extractfile("original-code/initial_cached_runner.py").read()
    assert original_runner == (ROOT / "initial_cached_runner.py").read_bytes()
    assert len(archive.extractfile("results/errors.jsonl").read().splitlines()) == 15
    for row in episodes:
        data = gzip.decompress(archive.extractfile("results/" + row["public_file"]).read())
        assert hashlib.sha256(data).hexdigest() == row["public_transcript_sha256"]
        public = json.loads(data)
        assert public["policy"] == row["policy"]
        assert public["policy"]["answer"] == row["evaluated"]["active_adjusted"]["answer"]
        remaining = row["budget"]
        for event in public["transcript"]:
            command = event["command"]
            feedback = json.loads(event["feedback"].splitlines()[0])
            assert command["type"] in {"observe", "intervene"}
            assert feedback["type"] == "batch_result"
            if command["type"] == "intervene":
                assert command["target"] not in command["measure"]
            remaining -= command["batch_size"] * len(command["measure"])
            assert remaining >= 0
            assert feedback["remaining_budget"] == remaining
        assert remaining == row["policy"]["remaining_budget"]
        assert row["public_only_replay_passed"]
context = json.loads((ROOT / "context-measurement.json").read_text(encoding="utf-8"))["episodes"]
assert len(context) == 100
assert {r["public_file"] for r in context} == {r["public_file"] for r in episodes}
assert sum(r["exceeds_32768"] for r in context) == 41
assert all(r["initial_tokens"] > 100 and r["total_tokens"] > r["initial_tokens"] for r in context)
assert all(r["exceeds_32768"] == (r["total_tokens"] > 32768) for r in context)
print(
    json.dumps(
        {
            "passed": True,
            "archive_files": len(manifest),
            "frozen_worlds": 50,
            "locked_answers_and_legal_charge_traces": 100,
            "original_policy_and_failed_runner_bytes_preserved": True,
            "initial_terminal_cache_errors_preserved": 15,
            "context_measurements": 100,
        }
    )
)
