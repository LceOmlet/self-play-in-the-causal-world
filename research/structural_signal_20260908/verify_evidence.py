"""Verify frozen evidence bytes, answer joins and the reported RQI counts."""

import hashlib
import json
import tarfile
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    archive = root / "structural-answer-evidence.tar.gz"
    manifest = json.loads((root / "structural-answer-evidence.sha256.json").read_bytes())
    assert sha(archive.read_bytes()) == manifest["archive_sha256"]
    with tarfile.open(archive) as tar:
        assert set(tar.getnames()) == set(manifest["members"])
        contents = {n: tar.extractfile(n).read() for n in tar.getnames()}
    for name, identity in manifest["members"].items():
        assert len(contents[name]) == identity["bytes"]
        assert sha(contents[name]) == identity["sha256"]
    assert (
        contents["audit_structural_answers.py"]
        == (root / "audit_structural_answers.py").read_bytes()
    )
    old_root = root.parent / "training_hold_20260908"
    assert (
        sha((old_root / "user-hold-and-through39-evidence.tar.gz").read_bytes())
        == manifest["prior_raw_archive_sha256"]
    )
    assert (
        contents["training-signal-through39.json"]
        == (old_root / "training-signal-through39.json").read_bytes()
    )
    signal = json.loads(contents["training-signal-through39.json"])
    summary = json.loads(contents["summary.json"])
    assert summary["signal_sha256"] == sha(contents["training-signal-through39.json"])
    assert summary["script_sha256"] == sha(contents["audit_structural_answers.py"])
    rows = {
        row["request_id"]: row
        for group in signal["groups"]
        if group["family"] in {"mediator_set", "backadj_minimal_sets"}
        for row in group["rows"]
    }
    assert len(rows) == 64
    assert {r["request_id"] for r in summary["records"]} == {
        rid for rid, row in rows.items() if row["completed"]
    }
    for r in summary["records"]:
        old = rows[r["request_id"]]
        assert r["answer"] == old["answer"] and r["strict_correct"] == old["strict_correct"]
        assert r["positive_advantage"] == (old["official_scalar_advantage"] > 0)
    for index, world in summary["worlds"].items():
        assert sha(contents[f"train-{int(index):04d}.pkl"]) == world["sha256"]
    cases = json.loads(contents["correct-structural-cases.json"])
    assert {r["request_id"] for r in cases} == {
        r["request_id"]
        for r in summary["records"]
        if r["strict_correct"] and (r["family"] == "mediator_set" or not r["empty"])
    }
    backdoor = next(r for r in cases if r["family"] == "backadj_minimal_sets")
    counts = {}
    for event in backdoor["events"]:
        command = event["command"]
        if (
            not isinstance(command, dict)
            or command.get("target") != "RQI"
            or command.get("batch_size") != 50
            or command.get("measure") != ["OHT", "RFU"]
        ):
            continue
        feedback = json.JSONDecoder().raw_decode(event["feedback"])[0]
        if feedback["type"] != "batch_result" or "OHT" not in command["measure"]:
            continue
        histogram = feedback["batch"]["joint_histogram"]
        position = histogram["columns"].index("OHT")
        marginal = [0, 0]
        for states, n in histogram["rows"]:
            marginal[states[position]] += n
        assert command["value"] not in counts
        counts[command["value"]] = marginal
    assert counts == {"state_0": [13, 37], "state_1": [18, 32]}
    print(
        json.dumps(
            {
                "verified_members": len(contents),
                "structural_rows": len(rows),
                "completed": len(summary["records"]),
                "complete_success_cases": len(cases),
                "RQI_intervention_OHT_counts": counts,
            }
        )
    )


if __name__ == "__main__":
    main()
