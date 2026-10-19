"""Verify frozen integrity and independent-process replay, never re-solve CF.

CPT normalization uses the native 1e-12 validity tolerance. All replay,
prompt, binary64 snapshot and native scoring comparisons are exact. CF truth
retains its original certification/endpoint_error; hashes do not recertify it.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from capsule import HERE, FrozenEnvironment, canonical, restore_episode, sha256, world_to_data
from cpt_world import TERMINAL_QUALITY_REWARD_VERSION
from cpt_world.query_truth import INDIVIDUAL_COUNTERFACTUAL_ENDPOINT_TOLERANCE
from cpt_world.task_scoring import score_terminal_answer
from cpt_world.rewards import terminal_quality_reward
from cpt_world.world_runtime import sample_worldspec_batch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def diagnostic_data(value):
    """Exact and hash-seed-independent encoding of native diagnostics."""
    if isinstance(value, Fraction):
        return {"fraction": [value.numerator, value.denominator]}
    if is_dataclass(value):
        return diagnostic_data(asdict(value))
    if isinstance(value, dict):
        return {str(k): diagnostic_data(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((diagnostic_data(v) for v in value), key=canonical)
    if isinstance(value, (list, tuple)):
        return [diagnostic_data(v) for v in value]
    return value


def read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def protocol_module(suite):
    spec = importlib.util.spec_from_file_location("frozen_restore_protocol", suite / "frozen/protocol/tool_mode.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_capsule(capsule, protocol):
    ep = restore_episode(capsule)
    require(world_to_data(ep.world) == capsule["world"], "world round-trip differs")
    require(capsule["max_graph_nodes"] == 16, "wrong scoring graph normalization")
    row = capsule["row"]
    require(row["task_id"] == capsule["task_id"], "row identity differs")
    require(row["query_type"] == capsule["family"] == ep.seed["query"]["type"], "family differs")
    require(row["tape_key"] == ep.tape.tape_key, "row tape differs")
    require(json.loads(row["terminal_truth_json"]) == capsule["terminal_truth"], "row truth differs")
    original = dict(row, prompt=list(ep.initial_messages()))
    require(canonical(protocol.prepare_row(original)) == canonical(row), "prepared prompt/row differs")
    components = dict(capsule, prompt=row["prompt"])
    for name, digest in capsule["components_sha256"].items():
        require(sha256(components[name]) == digest, f"component hash differs: {name}")
    truth = capsule["terminal_truth"]
    require(truth["type"] == capsule["family"], "terminal truth type differs")
    if capsule["family"] == "individual_counterfactual_probability":
        error = float(truth.get("endpoint_error", 0))
        require(0 <= error <= INDIVIDUAL_COUNTERFACTUAL_ENDPOINT_TOLERANCE,
                "native CF endpoint tolerance exceeded")
        lower, upper = float(truth["lower"]), float(truth["upper"])
        # The native solver can produce 1 + one binary64 ulp. Validate its
        # range with a machine-roundoff allowance; preserve values unchanged.
        require(-1e-12 <= lower <= upper + 1e-12 and upper <= 1 + 1e-12,
                "CF interval invalid beyond binary64 roundoff allowance")
    return ep


def safe_answer(capsule):
    """Fixed legal diagnostic answers, deliberately independent of truth values."""
    family, seed = capsule["family"], capsule["seed"]
    answer = {"type": "answer"}
    if family == "ate":
        ep = restore_episode(capsule)
        labels = seed["visible_schema"]["variable_labels"]
        outcome = seed["query"]["outcome"]
        internal = next((k for k, v in labels.items() if v == outcome), outcome)
        answer["effect"] = {f"state_{s}": 0.0 for s in range(ep.world.domains[ep.world.variables.index(internal)])}
    elif family == "individual_counterfactual_probability":
        answer.update(lower=0.0, upper=1.0)
    elif family == "best_intervention":
        answer["value"] = "state_0"
    elif family == "backadj_minimal_sets":
        answer["adjustment_set"] = []
    elif family == "mediator_set":
        answer.update(mediators=[], order=[])
    else:
        raise ValueError(f"unknown family {family}")
    return answer


def replay_capsule(capsule):
    env = FrozenEnvironment()
    env.reset(capsule=capsule, prompt=capsule["row"]["prompt"])
    ep = env.episode
    labels = ep.seed["visible_schema"]["variable_labels"]
    readable = [name for name in ep.world.variables if ep.seed["readable"][name]]
    target = next(name for name in ep.world.variables if ep.seed["manipulability"][name]
                  and any(other != name for other in readable))
    intervention_measures = [name for name in readable if name != target]
    commands = []
    # Repeat an arm with different measurement subsets: offsets belong to the
    # arm, not the requested projection. Passive offsets form a separate stream.
    for kind, names in (("observe", readable), ("intervene", intervention_measures)):
        for i in range(3):
            measured = names[:min(2, len(names), ep.measure_max or 2)] if i == 1 else names[-1:]
            command = {"type": kind, "measure": [labels[n] for n in measured], "batch_size": 3}
            if kind == "intervene":
                command.update(target=labels[target], value="state_0")
            commands.append(command)
    trace = []
    offsets = {"observe": 0, "intervene": 0}
    for command in commands:
        raw = canonical(command).decode()
        parsed = ep.parse_observation(raw) if command["type"] == "observe" else ep.parse_intervention(raw)
        expected = sample_worldspec_batch(ep.world, ep.tape, parsed, start_index=offsets[command["type"]])
        before = ep.remaining_budget
        feedback = env.act(command)
        require(ep.history[-1] == expected, "measurement subset changed arm random offset")
        require(before - ep.remaining_budget == 3 * len(command["measure"]), "budget charge differs")
        offsets[command["type"]] += 3
        trace.append((command, feedback, ep.remaining_budget, diagnostic_data(ep.history[-1])))
    answer = safe_answer(capsule)
    direct = score_terminal_answer(canonical(answer).decode(), ep.seed, ep.world,
                                   terminal_truth=ep.terminal_truth)
    expected_reward = terminal_quality_reward(direct, max_graph_nodes=16)
    acknowledgement = env.act(answer)
    require(ep.completed, "safe native terminal answer rejected")
    require(ep.terminal_score == direct and ep.terminal_reward == expected_reward,
            "native terminal diagnostics differ")
    require(env.get_reward() == float(expected_reward), "adapter reward differs")
    trace.append((answer, acknowledgement, diagnostic_data(direct), str(expected_reward), env.get_reward()))
    return sha256(trace)


def task_path(data, entry):
    path = (data / entry["file"]).resolve()
    require(path.is_relative_to(data.resolve()), "task path escapes data directory")
    return path


def verify_suite(suite, data, verify_replays=False):
    manifest = read_json(data / "manifest.json")
    require(manifest["format"] == "cpt-world-suite-v1", "unknown suite format")
    entries = manifest["tasks"]
    require(len(entries) == manifest["total"] == 500, "suite must contain 500 tasks")
    require(len({e["task_id"] for e in entries}) == 500, "duplicate task IDs")
    require(set(Counter(e["family"] for e in entries).values()) == {100}, "family counts differ")
    for family in {e["family"] for e in entries}:
        require({e["ordinal"] for e in entries if e["family"] == family} == set(range(100)),
                "family ordinal coverage differs")
    require(sha256(entries) == manifest["tasks_merkle_input_sha256"], "task manifest hash differs")
    owner = manifest["owner"]
    require(sha256(owner["files"]) == owner["sha256"], "owner manifest hash differs")
    for name, digest in owner["files"].items():
        source = (suite / name).resolve()
        require(source.is_relative_to(suite.resolve()), "owner path escapes suite")
        require(hashlib.sha256(source.read_bytes()).hexdigest() == digest, f"owner changed: {name}")
    require(owner["reward_version"] == TERMINAL_QUALITY_REWARD_VERSION, "reward version differs")
    protocol = protocol_module(suite)
    require(owner["protocol"] == protocol.PROTOCOL_VERSION, "protocol version differs")
    evaluation = read_json(data / "eval100.json")
    require(sha256(evaluation) == manifest["eval100_sha256"], "eval100 hash differs")
    ids = evaluation["task_ids"]
    require(len(ids) == len(set(ids)) == 100, "eval100 IDs differ")
    require(set(ids) <= {e["task_id"] for e in entries}, "eval100 has unknown tasks")
    grouped = evaluation["task_ids_by_family"]
    require([task for family in grouped for task in grouped[family]] == ids or
            set(task for family in grouped for task in grouped[family]) == set(ids),
            "eval100 grouped selection differs")
    family_by_id = {e["task_id"]: e["family"] for e in entries}
    for family, members in grouped.items():
        require(len(members) == len(set(members)) == 20 and
                all(family_by_id.get(member) == family for member in members),
                "eval100 grouped family membership differs")
    require(Counter(e["family"] for e in entries if e["task_id"] in ids) ==
            {f: 20 for f in evaluation["task_ids_by_family"]}, "eval100 family counts differ")
    for entry in entries:
        path = task_path(data, entry)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == entry["compressed_sha256"], "compressed hash differs")
        capsule = read_json(path)
        require(sha256(capsule) == entry["sha256"], f"capsule hash differs: {entry['task_id']}")
        require(all(capsule[k] == entry[k] for k in ("task_id", "family", "ordinal")), "task metadata differs")
        verify_capsule(capsule, protocol)
    result = {"status": "verified", "tasks": len(entries), "dataset_sha256": sha256(entries),
              "cpt_sum_tolerance": 1e-12, "replay_tolerance": 0,
              "cf_range_roundoff_allowance": 1e-12,
              "cf_endpoint_tolerance": INDIVIDUAL_COUNTERFACTUAL_ENDPOINT_TOLERANCE,
              "cf_recertified": False, "native_scoring": "cached native truth; fixed legal diagnostic answers"}
    versions = {}
    for name in owner.get("packages", {}):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result["runtime"] = {"python": sys.version, "packages": versions,
                         "matches_frozen_python": sys.version == owner.get("python"),
                         "matches_frozen_packages": versions == owner.get("packages", {})}
    result["verifier_sources_sha256"] = {
        name: hashlib.sha256((suite / name).read_bytes()).hexdigest()
        for name in ("capsule.py", "verify_suite.py", "tests_restore.py")}
    if verify_replays:
        command = [sys.executable, str(Path(__file__).resolve()), "--suite", str(suite), "--data", str(data), "--replay-worker"]
        first = subprocess.check_output(command, text=True)
        second = subprocess.check_output(command, text=True)
        require(first == second, "independent-process replay differs")
        result.update(cold_process_passes=2, replay_sha256=hashlib.sha256(first.encode()).hexdigest())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=HERE)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--verify-replays", action="store_true")
    parser.add_argument("--replay-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    suite = args.suite.resolve()
    require(suite == HERE, "run the verifier copied inside the requested suite")
    data = args.data.resolve() if args.data else suite / "data"
    if args.replay_worker:
        entries = read_json(data / "manifest.json")["tasks"]
        print(json.dumps({entry["task_id"]: replay_capsule(read_json(task_path(data, entry)))
                          for entry in entries}, sort_keys=True))
    else:
        print(json.dumps(verify_suite(suite, data, args.verify_replays), sort_keys=True))


if __name__ == "__main__":
    main()
