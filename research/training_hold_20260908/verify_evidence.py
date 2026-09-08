"""Verify archived bytes, executed source equivalence and real transcript counts."""

import ast
import hashlib
import json
import tarfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    archive = root / "user-hold-and-through39-evidence.tar.gz"
    manifest = json.loads(
        (root / "user-hold-and-through39-evidence.sha256.json").read_text(encoding="utf-8")
    )
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
    with tarfile.open(archive) as tar:
        assert set(tar.getnames()) == set(manifest["members"])
        contents = {name: tar.extractfile(name).read() for name in tar.getnames()}
    for name, item in manifest["members"].items():
        assert len(contents[name]) == item["bytes"]
        assert hashlib.sha256(contents[name]).hexdigest() == item["sha256"]
    for name in ["preserve_and_stop.py", "resume_after_hold.py", "inspect_new_successes.py"]:
        assert ast.dump(ast.parse(contents["executed-sources/" + name])) == ast.dump(
            ast.parse((root / name).read_text(encoding="utf-8"))
        )
    for name in ["training-signal-through39.json", "new-successes-accounting.json"]:
        assert contents[name] == (root / name).read_bytes()
    signal = json.loads(contents["training-signal-through39.json"])
    rows = [r for g in signal["groups"] for r in g["rows"]]
    assert [g["step"] for g in signal["groups"]] == list(range(1, 40))
    assert len(rows) == 156 and sum(r["completed"] for r in rows) == 153
    assert signal["generated_groups"] == 45 and len(signal["filtered_groups"]) == 6
    traces = json.loads(contents["new-strong-successes30-39.json"])
    counts = json.loads(contents["new-successes-accounting.json"])
    expected = {r["request_id"] for r in signal["successful_strong_trajectories"] if r["step"] > 29}
    assert {t["request_id"] for t in traces} == expected
    for record in counts["records"]:
        trace = next(t for t in traces if t["request_id"] == record["request_id"])
        assert hashlib.sha256(trace["output"].encode()).hexdigest() == record["output_sha256"]
        selected = [e for e in trace["events"] if e["command"] in record["selected_commands"]]
        assert len(selected) == len(record["selected_commands"])
        successes, totals = [0] * len(record["actual_totals"]), [0] * len(record["actual_totals"])
        event_state = 3 if record["task_index"] == 383 else 0
        for event in selected:
            feedback = json.JSONDecoder().raw_decode(event["feedback"])[0]
            histogram = feedback["batch"]["joint_histogram"]
            assert histogram["columns"] == event["command"]["measure"]
            for (x, y), n in histogram["rows"]:
                totals[x] += n
                successes[x] += n * (y == event_state)
        assert totals == record["actual_totals"] and successes == record["actual_successes"]
    print(
        json.dumps(
            {
                "verified_members": len(contents),
                "completed_updates": 39,
                "retained_trajectories": len(rows),
                "new_successes_checked": len(traces),
                "executed_reviewed_ast_equal": True,
            }
        )
    )


if __name__ == "__main__":
    main()
