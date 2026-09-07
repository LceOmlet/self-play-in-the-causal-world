"""Paired complete generation against pinned original acceptance/inference code."""

import ast
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import FunctionType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from cpt_world import query_truth as q  # noqa: E402
from cpt_world import world_space as w  # noqa: E402

BASE = "aa86218d0280b8abdfd34bdc07788501cad17d10"
original_source = subprocess.check_output(
    ["git", "-C", str(PROJECT), "show", BASE + ":src/cpt_world/world_space.py"]
).decode()
names = {"_best_intervention_observational_relation", "iter_sampled_seeds"}
definitions = [
    n for n in ast.parse(original_source).body if isinstance(n, ast.FunctionDef) and n.name in names
]
assert len(definitions) == len(names)
module = ast.Module(
    body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *definitions,
    ],
    type_ignores=[],
)
namespace = dict(vars(w))
exec(compile(ast.fix_missing_locations(module), "pinned_original_admission", "exec"), namespace)
baseline = {}
for name in names:
    compiled = namespace[name]
    function = FunctionType(compiled.__code__, vars(w), name, compiled.__defaults__)
    function.__kwdefaults__ = compiled.__kwdefaults__
    baseline[name] = function
candidate = {name: getattr(w, name) for name in names}
source_hash = hashlib.sha256(Path(w.__file__).read_bytes()).hexdigest()


def run(label, config):
    owners = baseline if label == "baseline" else candidate
    sample, infer = w.sample_task_world, q.worldspec_projected_interventional_distribution
    counts = {"observational_queries": 0, "interventional_queries": 0}
    proposals = []

    def sample_world(grammar, seed, query_type, *args, **kwargs):
        proposals.append((seed, query_type))
        return sample(grammar, seed, query_type, *args, **kwargs)

    def inference(world, interventions, *args, **kwargs):
        counts["interventional_queries" if interventions else "observational_queries"] += 1
        return infer(world, interventions, *args, **kwargs)

    started = time.perf_counter()
    with (
        patch.multiple(w, **owners),
        patch.object(w, "sample_task_world", sample_world),
        patch.object(q, "worldspec_projected_interventional_distribution", inference),
    ):
        result = w.iter_sampled_seeds(w.WorldGrammar(), **config)
    seconds = time.perf_counter() - started
    output = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    return (
        result,
        proposals,
        {
            "label": label,
            "seconds": seconds,
            **counts,
            "world_builds": len(proposals),
            "task_count": len(result),
            "output_sha256": hashlib.sha256(output).hexdigest(),
            "proposal_sha256": hashlib.sha256(json.dumps(proposals).encode()).hexdigest(),
            "seed_ids": [s["seed_id"] for s in result],
        },
    )


report = {"passed": False, "base_commit": BASE, "world_space_sha256": source_hash, "cases": []}
cases = [
    ("best-20", dict(query_types=("best_intervention",), count=20)),
    ("all-25", dict(query_types=tuple(w.TASK_FAMILY_QUERY_TYPES), count=5)),
    (
        "best-offset-10",
        dict(
            query_types=("best_intervention",),
            start_seed=20,
            count=10,
            best_intervention_balance_start=3,
        ),
    ),
]
for name, config in cases:
    case = {"name": name, "config": config, "runs": []}
    expected_tasks = expected_proposals = None
    for repeat, labels in enumerate((("baseline", "candidate"), ("candidate", "baseline"))):
        for label in labels:
            tasks, proposals, measurement = run(label, config)
            if expected_tasks is None:
                expected_tasks, expected_proposals = tasks, proposals
            assert tasks == expected_tasks, (name, repeat, label, "tasks differ")
            assert proposals == expected_proposals, (name, repeat, label, "proposal stream differs")
            measurement["repeat"] = repeat
            case["runs"].append(measurement)
            print(json.dumps({"case": name, **measurement}), flush=True)
    old = [r for r in case["runs"] if r["label"] == "baseline"]
    new = [r for r in case["runs"] if r["label"] == "candidate"]
    assert len({r["interventional_queries"] for r in case["runs"]}) == 1
    case["entire_tasks_and_proposal_streams_equal"] = True
    case["observational_calls_removed"] = (
        old[0]["observational_queries"] - new[0]["observational_queries"]
    )
    case["baseline_mean_seconds"] = sum(r["seconds"] for r in old) / len(old)
    case["candidate_mean_seconds"] = sum(r["seconds"] for r in new) / len(new)
    report["cases"].append(case)
    (ROOT / "admission-acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
assert hashlib.sha256(Path(w.__file__).read_bytes()).hexdigest() == source_hash
report["passed"] = True
report["scope"] = (
    "Fixed complete-generation cases, two runs per path in reversed order; no law change."
)
(ROOT / "admission-acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
