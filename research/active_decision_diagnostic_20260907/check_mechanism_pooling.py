"""Post-answer test of exact CPT-implied pooling, using the same public samples."""

import gzip
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path

from public_policy import adjusted_values, parse_prompt

ROOT = Path(__file__).resolve().parent
mechanism = json.loads((ROOT / "overlap-case.json").read_text(encoding="utf-8"))
assert mechanism["x_parents"] == ["DVR", "POF"]
assert mechanism["y_parents"] == ["OHV", "POF"]
rows = []
with tarfile.open(ROOT / "evidence.tar.gz") as archive:
    for replicate in (0, 1):
        name = f"results/cohort-b-task-68-r{replicate}.public.json.gz"
        public = json.loads(gzip.decompress(archive.extractfile(name).read()))
        p = parse_prompt(public["prompt"])
        assert p["x"] == "OHV" and p["y"] == "MQD"
        pooled = defaultdict(int)
        original = defaultdict(int)
        for event in public["transcript"]:
            if event["command"]["type"] != "observe":
                continue
            histogram = json.loads(event["feedback"].splitlines()[0])["batch"]["joint_histogram"]
            assert histogram["columns"] == ["OHV", "MQD", "DVR", "POF"]
            for values, count in histogram["rows"]:
                original[tuple(values)] += count
                pooled[(values[0], values[1], values[3])] += count
        before = adjusted_values(original, p["domains"][p["x"]], p["domains"][p["y"]], p["target"])
        assert all(
            abs(a - b) < 1e-12
            for a, b in zip(before["values"], public["policy"]["values"], strict=True)
        )
        after = adjusted_values(pooled, p["domains"][p["x"]], p["domains"][p["y"]], p["target"])
        choose = max if p["maximize"] else min
        chosen = choose(range(len(after["values"])), key=after["values"].__getitem__)
        rows.append(
            {
                "replicate": replicate,
                "same_observation_units": sum(pooled.values()),
                "original_estimate": before,
                "pooled_estimate": after,
                "pooled_action": chosen,
                "true_optimal_action": mechanism["optimal_action"],
                "original_measured_joint_states": math.prod(
                    p["domains"][v] for v in ["OHV", "MQD", "DVR", "POF"]
                ),
                "pooled_measured_joint_states": math.prod(
                    p["domains"][v] for v in ["OHV", "MQD", "POF"]
                ),
            }
        )
report = {
    "scope": "This calculation uses the revealed true Y parents after answers were committed. "
    "It isolates redundant conditioning and is not a new public-only success rate. "
    "No new observations or changed CPTs are used.",
    "identity": "Y is independent of DVR given X and POF, because Pa(Y)={X,POF} and DVR is "
    "an ancestor of X. POF is a parent of X, hence is not its descendant; adjusting for POF "
    "alone blocks every backdoor into Y. Pooling DVR therefore preserves the target.",
    "cases": rows,
}
(ROOT / "mechanism-pooling.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report))
