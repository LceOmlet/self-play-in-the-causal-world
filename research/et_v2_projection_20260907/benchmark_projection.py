"""Compare the real generator using the pinned old and candidate projection.

Only the offline comparison swaps a kernel owner. No training path is invoked.
"""

import ast
import copy
import gc
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from cpt_world import world_space as w  # noqa: E402

BASE_COMMIT = "9f8a343fca62febfb3f962b5e264d7f399f3f161"
source = subprocess.check_output(
    ["git", "-C", str(PROJECT), "show", BASE_COMMIT + ":src/cpt_world/world_space.py"]
).decode()
definition = next(
    node
    for node in ast.parse(source).body
    if isinstance(node, ast.FunctionDef) and node.name == "_parent_interaction_projection"
)
module = ast.Module(
    body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        definition,
    ],
    type_ignores=[],
)
namespace = dict(vars(w))
exec(compile(ast.fix_missing_locations(module), "pinned_baseline_projection", "exec"), namespace)
baseline = namespace["_parent_interaction_projection"]
candidate = w._parent_interaction_projection
candidate_source_hash = hashlib.sha256(Path(w.__file__).read_bytes()).hexdigest()


def maximum_error(left, right):
    assert len(left) == len(right)
    return max(
        abs(float(a) - float(b))
        for row_a, row_b in zip(left, right, strict=True)
        for a, b in zip(row_a, row_b, strict=True)
    )


def measure(function, *args):
    times = []
    for _ in range(3):
        start = time.perf_counter()
        result = function(*args)
        times.append(time.perf_counter() - start)
    gc.collect()
    tracemalloc.start()
    memory_result = function(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert result == memory_result
    return result, {
        "median_seconds": statistics.median(times),
        "times": times,
        "peak_python_bytes": peak,
    }


@contextmanager
def generator_owner(projection):
    original_build = w._build_world
    record = {"world_builds": 0, "rng_after_build": []}

    def build(structure, rng):
        result = original_build(structure, rng)
        record["world_builds"] += 1
        record["rng_after_build"].append(hashlib.sha256(repr(rng.getstate()).encode()).hexdigest())
        return result

    with (
        patch.object(w, "_parent_interaction_projection", projection),
        patch.object(w, "_build_world", build),
    ):
        yield record


report = {"base_commit": BASE_COMMIT, "projection_cases": [], "passed": False}
for domains, child, positions in (
    ((2,), 2, (0,)),
    ((5,), 5, (0,)),
    ((2, 3, 4), 3, (0, 2)),
    ((4,) * 4, 4, (0, 1, 2, 3)),
    ((5,) * 5, 5, (0, 1, 2, 3, 4)),
):
    rng = random.Random(101 + len(domains))
    table = tuple(
        tuple(rng.uniform(-1, 1) for _ in range(child)) for _ in range(math.prod(domains))
    )
    old, old_cost = measure(baseline, table, domains, positions)
    new, new_cost = measure(candidate, table, domains, positions)
    error = maximum_error(old, new)
    assert error < 1e-12
    report["projection_cases"].append(
        {
            "domains": domains,
            "child_domain": child,
            "positions": positions,
            "old": old_cost,
            "new": new_cost,
            "max_abs_error": error,
            "speedup": old_cost["median_seconds"] / new_cost["median_seconds"],
            "peak_allocation_ratio": old_cost["peak_python_bytes"] / new_cost["peak_python_bytes"],
        }
    )
    print(
        json.dumps({"projection_completed": domains, "result": report["projection_cases"][-1]}),
        flush=True,
    )

worlds, costs, streams = [], [], []
for owner in (baseline, candidate):
    start = time.perf_counter()
    with generator_owner(owner) as record:
        values = [w.sample_world(w.WorldGrammar(), seed) for seed in range(64)]
    worlds.append(values)
    costs.append(time.perf_counter() - start)
    streams.append(record)
assert streams[0] == streams[1]
max_cpt_error = 0.0
max_log_cpt_error = 0.0
for old, new in zip(*worlds, strict=True):
    assert old.edges == new.edges and old.domains == new.domains and old.parents == new.parents
    assert w.legal_world(old) and w.legal_world(new)
    max_cpt_error = max(
        max_cpt_error, *(maximum_error(old.cpt[node], new.cpt[node]) for node in old.cpt)
    )
    for node in old.cpt:
        for left, right in zip(old.cpt[node], new.cpt[node], strict=True):
            for a, b in zip(left, right, strict=True):
                assert a > 0 and b > 0
                max_log_cpt_error = max(max_log_cpt_error, abs(math.log(a) - math.log(b)))
assert max_cpt_error < 1e-12
assert max_log_cpt_error < 1e-10
report["worlds"] = {
    "count": 64,
    "old_seconds": costs[0],
    "new_seconds": costs[1],
    "speedup": costs[0] / costs[1],
    "max_cpt_error": max_cpt_error,
    "same_rng_after_every_world": True,
    "same_structures": True,
    "max_abs_log_cpt_error": max_log_cpt_error,
}
print(json.dumps({"worlds_completed": report["worlds"]}), flush=True)

tasks, costs, streams = [], [], []
for owner in (baseline, candidate):
    start = time.perf_counter()
    with generator_owner(owner) as record:
        values = w.iter_sampled_seeds(
            w.WorldGrammar(),
            query_types=(
                "ate",
                "individual_counterfactual_probability",
                "backadj_minimal_sets",
                "best_intervention",
                "mediator_set",
            ),
            start_seed=0,
            count=5,
        )
    tasks.append(values)
    costs.append(time.perf_counter() - start)
    streams.append(record)
assert streams[0] == streams[1]
max_task_cpt_error = 0.0
for old, new in zip(*tasks, strict=True):
    old, new = copy.deepcopy(old), copy.deepcopy(new)
    cpts = old["world_source"].pop("cpt"), new["world_source"].pop("cpt")
    assert old == new
    for name in cpts[0]:
        max_task_cpt_error = max(max_task_cpt_error, maximum_error(cpts[0][name], cpts[1][name]))
assert max_task_cpt_error < 1e-12
report["tasks"] = {
    "count": len(tasks[0]),
    "old_seconds": costs[0],
    "new_seconds": costs[1],
    "speedup": costs[0] / costs[1],
    "max_cpt_error": max_task_cpt_error,
    "world_builds": streams[0]["world_builds"],
    "same_rng_after_every_build": True,
    "same_selected_tasks_and_non_cpt_fields": True,
    "seed_ids": [seed["seed_id"] for seed in tasks[0]],
}
report["passed"] = True
assert hashlib.sha256(Path(w.__file__).read_bytes()).hexdigest() == candidate_source_hash
report["candidate_world_space_sha256"] = candidate_source_hash
report["scope"] = (
    "Paired fixed fixtures; Python allocation peaks, not whole-process RSS or training throughput."
)
(ROOT / "projection-acceptance.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report["tasks"]), flush=True)
