"""Compare the exact production role populations and complete task outputs."""

import ast
import hashlib
import json
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from cpt_world import world_space as w  # noqa: E402

BASE = "dd62d1d340fbdd162ac84d63b80b0941c2abb281"
source = subprocess.check_output(
    ["git", "-C", str(PROJECT), "show", BASE + ":src/cpt_world/world_space.py"]
).decode()
definition = next(
    node
    for node in ast.parse(source).body
    if isinstance(node, ast.FunctionDef) and node.name == "_minimum_backdoor_adjustment_size"
)
module = ast.Module(
    body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        definition,
    ],
    type_ignores=[],
)
namespace = dict(vars(w))
exec(compile(ast.fix_missing_locations(module), "pinned_subset_enumeration", "exec"), namespace)
baseline = namespace["_minimum_backdoor_adjustment_size"]
candidate = w._minimum_backdoor_adjustment_size
source_hash = hashlib.sha256(Path(w.__file__).read_bytes()).hexdigest()


@contextmanager
def owner(function):
    original = w.sample_task_world
    record = {"minimum_calls": 0, "world_proposals": []}

    def minimum(*args):
        record["minimum_calls"] += 1
        return function(*args)

    def sample(grammar, seed, query_type, *args, **kwargs):
        record["world_proposals"].append((seed, query_type))
        return original(grammar, seed, query_type, *args, **kwargs)

    with (
        patch.object(w, "_minimum_backdoor_adjustment_size", minimum),
        patch.object(w, "sample_task_world", sample),
    ):
        yield record


report = {"passed": False, "base_commit": BASE, "candidate_world_space_sha256": source_hash}
structures = [w._sample_structure(w.WorldGrammar(), random.Random(seed)) for seed in range(96)]
roles, costs = [], []
for function in (baseline, candidate):
    start = time.perf_counter()
    with owner(function):
        result = [
            w._sampled_role_assignments(s.node_count, s.edges, query_type, seed)
            for seed, s in enumerate(structures)
            for query_type in w.TASK_FAMILY_QUERY_TYPES
        ]
    costs.append(time.perf_counter() - start)
    roles.append(result)
assert roles[0] == roles[1]
report["role_populations"] = {
    "structures": len(structures),
    "family_structure_cases": len(roles[0]),
    "identical_ordered_role_lists": True,
    "old_seconds": costs[0],
    "new_seconds": costs[1],
    "speedup": costs[0] / costs[1],
}
print(json.dumps(report["role_populations"]), flush=True)

report["complete_generation"] = []
for query_types, count in ((("best_intervention",), 20), (tuple(w.TASK_FAMILY_QUERY_TYPES), 5)):
    results, costs, streams = [], [], []
    for function in (baseline, candidate):
        start = time.perf_counter()
        with owner(function) as record:
            result = w.iter_sampled_seeds(w.WorldGrammar(), query_types=query_types, count=count)
        costs.append(time.perf_counter() - start)
        results.append(result)
        streams.append(record)
    assert results[0] == results[1]
    assert streams[0] == streams[1]
    item = {
        "query_types": query_types,
        "count": len(results[0]),
        "old_seconds": costs[0],
        "new_seconds": costs[1],
        "speedup": costs[0] / costs[1],
        "entire_tasks_equal_including_cpt": True,
        "entire_proposal_sequence_equal": True,
        "minimum_calls": streams[0]["minimum_calls"],
        "world_builds": len(streams[0]["world_proposals"]),
        "seed_ids": [seed["seed_id"] for seed in results[0]],
        "serialized_output_sha256": hashlib.sha256(
            json.dumps(results[0], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    report["complete_generation"].append(item)
    print(json.dumps(item), flush=True)

assert source_hash == hashlib.sha256(Path(w.__file__).read_bytes()).hexdigest()
report["passed"] = True
report["scope"] = (
    "Paired fixed fixtures, one timing per path; no sampling-law or acceptance change."
)
(ROOT / "mincut-acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
