"""Check saved evidence bytes and the source revision they actually exercised."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
report = json.loads((ROOT / "admission-acceptance.json").read_text())
assert report["passed"]
assert report["base_commit"] == "aa86218d0280b8abdfd34bdc07788501cad17d10"
assert (
    hashlib.sha256((PROJECT / "src/cpt_world/world_space.py").read_bytes()).hexdigest()
    == report["world_space_sha256"]
)
for case in report["cases"]:
    assert case["entire_tasks_and_proposal_streams_equal"]
    for key in ("output_sha256", "proposal_sha256", "interventional_queries", "world_builds"):
        assert len({run[key] for run in case["runs"]}) == 1, (case["name"], key)
    old = [run for run in case["runs"] if run["label"] == "baseline"]
    new = [run for run in case["runs"] if run["label"] == "candidate"]
    assert len(old) == len(new) == 2
    assert case["observational_calls_removed"] > 0
    assert old[0]["observational_queries"] == old[1]["observational_queries"]
    assert new[0]["observational_queries"] == new[1]["observational_queries"]
    assert (
        old[0]["observational_queries"] - new[0]["observational_queries"]
        == case["observational_calls_removed"]
    )
assert "3 passed, 3344 subtests passed" in (ROOT / "independent-tests.log").read_text()
regression = (ROOT / "regression.log").read_text()
assert "failed" not in regression.lower() and "error" not in regression.lower()
assert "107 passed, 3349 subtests passed" in regression
count = 0
for line in (ROOT / "SHA256SUMS").read_text().splitlines():
    expected, name = line.split("  ", 1)
    path = (ROOT / name).resolve()
    assert path.is_relative_to(ROOT)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    count += 1
print(
    json.dumps(
        {
            "passed": True,
            "verified_files": count,
            "generation_cases": len(report["cases"]),
            "source_sha256": report["world_space_sha256"],
        }
    )
)
