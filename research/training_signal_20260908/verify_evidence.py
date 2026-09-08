"""Verify archived bytes, source identities, figures and all observation ranks."""

import hashlib
import json
import tarfile
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "training-signal-and-rank-evidence.sha256.json").read_text())
    archive = root / "training-signal-and-rank-evidence.tar.gz"
    assert digest(archive.read_bytes()) == manifest["archive_sha256"]
    assert archive.stat().st_size == manifest["archive_bytes"]
    with tarfile.open(archive) as handle:
        assert all(item.isfile() for item in handle.getmembers())
        members = {item.name: handle.extractfile(item).read() for item in handle.getmembers()}
    assert len(members) == 119 and set(members) == set(manifest["members"])
    assert all(digest(value) == manifest["members"][name] for name, value in members.items())
    successful = "successful-strong-decisions-through29"
    for local, captured in {
        "audit_training_signal.py": "audit_training_signal_v2.py",
        "audit_observation_rank.py": "audit_observation_rank_v2.py",
        "archive_evidence.py": "source/archive_evidence.py",
        "training-signal-through29.json": "training-signal-through29-v2.json",
        "observation-rank-summary.json": "observation-rank-02/summary.json",
        "observation-rank-records.jsonl": "observation-rank-02/records.jsonl",
        f"{successful}.json": f"{successful}-v3.json",
    }.items():
        assert (root / local).read_bytes() == members[captured], local
    for name in ["updates1-5", "updates6-15", "updates16-25", "updates26-29"]:
        assert (root / f"curve-inputs/{name}.json").read_bytes() == members[
            f"snapshots/{name}/progress.json"
        ]
    for filename, script in [
        ("manifest.json", root.parent / "dapo_progress_20260908/plot_trusted_progress.py"),
        ("rank-figure-manifest.json", root / "plot_rank_comparison.py"),
    ]:
        figure = json.loads((root / "figures" / filename).read_text())
        assert digest(script.read_bytes()) == figure["script_sha256"]
        for name, fingerprint in figure["files"].items():
            assert digest((root / "figures" / name).read_bytes()) == fingerprint
    rows = [
        json.loads(line)
        for line in (root / "observation-rank-records.jsonl").read_bytes().splitlines()
    ]
    assert len(rows) == 100 and len({row["seed_id"] for row in rows}) == 50
    checks = 0
    for row in rows:
        p = row["public_query"]
        nx, ny = p["domains"][p["x"]], p["domains"][p["y"]]
        counts = [[0] * ny for _ in range(nx)]
        for record in row["pure_observation_checkpoints"]:
            command = record["command"]
            feedback = json.loads(record["feedback"].splitlines()[0])
            histogram = feedback["batch"]["joint_histogram"]
            assert command["type"] == "observe" and histogram["columns"] == [p["x"], p["y"]]
            for values, count in histogram["rows"]:
                counts[values[0]][values[1]] += count
            assert counts == record["counts"]
            assert sum(map(sum, counts)) == record["sample_count"]
            values = [(c[p["target"]] + 0.5) / (sum(c) + 0.5 * ny) for c in counts]
            assert values == record["observational_values"]
            order = sorted(range(nx), key=lambda i: -values[i] if p["maximize"] else values[i])
            assert record["original_first_answer_replayed"]["value"] == f"state_{order[0]}"
            second = record["second_rank"]
            assert second["answer"]["value"] == f"state_{order[1]}"
            assert second["metrics"]["chosen"]["value"] == order[1]
            assert 2 * record["sample_count"] <= row["budget"]
            checks += 1
        assert 2 * row["pure_observation_checkpoints"][-1]["sample_count"] == row["budget"]
    assert checks == 300
    print(
        json.dumps(
            {
                "archive_members_verified": len(members),
                "histogram_and_rank_checks": checks,
                "curve_updates": 29,
                "figures_hash_verified": 9,
            }
        )
    )


if __name__ == "__main__":
    main()
