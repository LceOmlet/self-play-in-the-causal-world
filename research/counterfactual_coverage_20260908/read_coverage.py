"""Read frozen CF admission and DAG coverage; never solve or regenerate a world."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import tarfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAMILY = "individual_counterfactual_probability"
TRAIN_ARCHIVE = "research/training_submission_20260907/certified-data.tar.gz"
TRAIN_SHA = "4a8a1149e5907c96f102b8b63c62e1841e930ee49f65d6536f729df97fc73cd4"
FROZEN_ARCHIVE = "research/symbolic_contraction_20260907/inputs.tar.gz"
FROZEN_SHA = "313498a434a60b79d54ada0fe9eb44412fae86e9b0b21e5dc85ce046dc287ce5"
REPLAY = "research/symbolic_contraction_20260907/numeric-v1-summary.json"
REPLAY_SHA = "f779159e761d55f4797572676f20e5a812a6c7b1c8ebd9c8dd718bc3af38cc17"
WORLD_FIELDS = (
    "family", "topology", "variables", "domains", "state_names", "edges", "parents", "cpt"
)


def sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def checked_file(relative, expected):
    blob = (ROOT / relative).read_bytes()
    if sha256(blob) != expected:
        raise ValueError(f"Source fingerprint mismatch: {relative}")
    return blob


class WorldRecord:
    """Only the eight serialized WorldSpec fields; no production methods."""

    def __setstate__(self, state):
        if not isinstance(state, (list, tuple)) or len(state) != len(WORLD_FIELDS):
            raise ValueError("Unexpected WorldSpec pickle state")
        self.__dict__.update(zip(WORLD_FIELDS, state, strict=True))


class RecordReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("cpt_world.world_space", "WorldSpec"):
            return WorldRecord
        if (module, name) == ("fractions", "Fraction"):
            return Fraction
        raise pickle.UnpicklingError(f"Unsupported serialized global: {module}.{name}")


def member_bytes(archive, name):
    member = archive.getmember(name)
    if not member.isfile():
        raise ValueError(f"Expected regular archive member: {name}")
    with archive.extractfile(member) as stream:
        return stream.read()


def checked_world(archive, name, expected):
    blob = member_bytes(archive, name)
    if sha256(blob) != expected:
        raise ValueError(f"World fingerprint mismatch: {name}")
    return RecordReader(io.BytesIO(blob)).load()


def json_lines(blob):
    return [json.loads(line) for line in blob.decode("utf-8").splitlines() if line]


def graph_traits(world, seed):
    query = seed["query"]
    inverse = {v: k for k, v in seed["visible_schema"]["variable_labels"].items()}
    source, outcome = (
        world.variables.index(inverse.get(query[key], query[key]))
        for key in ("treatment", "outcome")
    )
    children = {node: [] for node in range(len(world.variables))}
    for parent, child in world.edges:
        children[parent].append(child)
    expected_parents = {
        child: tuple(sorted(parent for parent, target in world.edges if target == child))
        for child in children
    }
    if any(tuple(sorted(world.parents[node])) != parents for node, parents in expected_parents.items()):
        raise ValueError("Serialized parents disagree with serialized edges")

    def reachable(start, target):
        queue, seen = [start], set()
        while queue:
            node = queue.pop()
            if node == target:
                return True
            if node not in seen:
                seen.add(node)
                queue.extend(children[node])
        return False

    if source == outcome or not reachable(source, outcome):
        raise ValueError("The source must be a strict ancestor of the outcome")
    affected = {
        node for node in children
        if node != source and reachable(source, node) and reachable(node, outcome)
    }
    convergence = any(sum(parent in affected for parent in world.parents[node]) >= 2 for node in affected)
    return {
        "nodes": len(world.variables),
        "edges": len(world.edges),
        "affected_mechanisms": len(affected),
        "affected_convergence": convergence,
        "direct_source_outcome_edge": source in world.parents[outcome],
    }


def aggregate_graphs(records):
    return {
        "worlds": len(records),
        "affected_mechanism_counts": dict(sorted(Counter(r["affected_mechanisms"] for r in records).items())),
        "affected_convergence": sum(r["affected_convergence"] for r in records),
        "direct_source_outcome_edge": sum(r["direct_source_outcome_edge"] for r in records),
        "at_most_two_affected_mechanisms": sum(r["affected_mechanisms"] <= 2 for r in records),
    }


def production_coverage():
    with tarfile.open(fileobj=io.BytesIO(checked_file(TRAIN_ARCHIVE, TRAIN_SHA))) as archive:
        attempts = json_lines(member_bytes(archive, "cf-attempts.jsonl"))
        acceptance = json.loads(member_bytes(archive, "data/acceptance.json"))
        accepted_blob = member_bytes(archive, "data/accepted.jsonl")
        if sha256(accepted_blob) != acceptance["labels_sha256"]:
            raise ValueError("Accepted-row ledger fingerprint mismatch")
        dataset_hashes = acceptance["sha256"]
        for name, digest in dataset_hashes.items():
            if sha256(member_bytes(archive, f"data/{name}")) != digest:
                raise ValueError(f"Dataset fingerprint mismatch: {name}")
        accepted = json_lines(accepted_blob)
        if Counter(r["split"] for r in accepted) != Counter(acceptance["counts"]):
            raise ValueError("Accepted ledger split counts disagree with admission receipt")
        cf_rows = [row for row in accepted if row["query_type"] == FAMILY]
        if {r["seed_id"] for r in cf_rows} != {
            r["seed_id"] for r in attempts if r["status"] == "accepted"
        }:
            raise ValueError("CF accepted attempts do not match exported tasks")
        if len({r["seed_id"] for r in attempts}) != len(attempts):
            raise ValueError("Duplicate CF attempt identity")
        records = []
        for item in cf_rows:
            path = f"data/truth/{item['split']}-{item['index']:04}.pkl"
            row, world, seed, _truth = checked_world(archive, path, item["world_file_sha256"])
            if seed["seed_id"] != item["seed_id"] or row["tape_key"] != item["tape_key"]:
                raise ValueError("Accepted world identity mismatch")
            records.append({
                "split": item["split"], "index": item["index"], "seed_id": item["seed_id"],
                "world_member": path, "world_sha256": item["world_file_sha256"],
                **graph_traits(world, seed),
            })
        # The executed prepare.py uses start seeds 2,000,000 and 3,000,000.
        # Verify that every recorded attempt is owned by exactly one such range.
        prefixes = {"train": "SAMPLED-2", "validation": "SAMPLED-3"}
        if any(sum(r["seed_id"].startswith(p) for p in prefixes.values()) != 1 for r in attempts):
            raise ValueError("Unexpected CF split seed range")
        by_split = {}
        for split, prefix in prefixes.items():
            group = [r for r in attempts if r["seed_id"].startswith(prefix)]
            rejected = [r for r in group if r["status"] == "rejected"]
            graphs = [r for r in records if r["split"] == split]
            if sum(r["status"] == "accepted" for r in group) != len(graphs):
                raise ValueError("CF split admission mismatch")
            by_split[split] = {
                "attempts": len(group), "accepted": len(graphs), "rejected": len(rejected),
                "rejection_fraction_in_this_attempt_stream": len(rejected) / len(group),
                "rejection_error_types": dict(Counter(r["error_type"] for r in rejected)),
                "attempt_seconds": sum(r["seconds"] for r in group),
                "rejected_attempt_seconds": sum(r["seconds"] for r in rejected),
                "accepted_graphs": aggregate_graphs(graphs),
            }
        serialized_members = {m.name for m in archive.getmembers() if m.name.endswith(".pkl")}
        expected_members = {f"data/truth/{r['split']}-{r['index']:04}.pkl" for r in accepted}
        if serialized_members != expected_members:
            raise ValueError("Archive serialized members extend beyond accepted worlds")
        return {
            "splits": by_split, "accepted_worlds": records, "attempts": attempts,
            "dataset_sha256": dataset_hashes, "accepted_ledger_sha256": sha256(accepted_blob),
            "attempt_ledger_sha256": sha256(member_bytes(archive, "cf-attempts.jsonl")),
            "rejected_worlds_serialized": 0,
        }


def frozen_coverage():
    replay = json.loads(checked_file(REPLAY, REPLAY_SHA))
    if not replay["complete"] or replay["compared"] != 74:
        raise ValueError("Expected completed frozen 74-task replay")
    records = []
    with tarfile.open(fileobj=io.BytesIO(checked_file(FROZEN_ARCHIVE, FROZEN_SHA))) as archive:
        candidates = json.loads(member_bytes(archive, "candidates.json"))
        for item in replay["records"]:
            index = item["index"]
            digest = replay["inputs_sha256"][str(index)]
            path = f"task-{index}.pkl"
            world, seed = checked_world(archive, path, digest)
            if seed["seed_id"] != candidates[index]["seed_id"]:
                raise ValueError("Frozen world identity mismatch")
            records.append({
                "index": index, "seed_id": seed["seed_id"], "world_member": path,
                "world_sha256": digest, "status": item["before"]["status"],
                **graph_traits(world, seed),
            })
    classes = {
        "direct_source_outcome_edge": lambda r: r["direct_source_outcome_edge"],
        "no_direct_source_outcome_edge": lambda r: not r["direct_source_outcome_edge"],
        "at_most_two_affected_mechanisms": lambda r: r["affected_mechanisms"] <= 2,
        "at_least_three_affected_mechanisms": lambda r: r["affected_mechanisms"] >= 3,
        "affected_convergence": lambda r: r["affected_convergence"],
    }
    return {
        "replay_base_commit": replay["base_commit"],
        "status_counts": dict(Counter(r["status"] for r in records)),
        "by_class": {
            name: {"total": sum(test(r) for r in records),
                   "certified": sum(test(r) and r["status"] == "ok" for r in records)}
            for name, test in classes.items()
        },
        "certified_graphs": aggregate_graphs([r for r in records if r["status"] == "ok"]),
        "unresolved_graphs": aggregate_graphs([r for r in records if r["status"] != "ok"]),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("summary.json"))
    args = parser.parse_args()
    result = {
        "scope": "Existing production admission archive and completed frozen replay; no new solving or generation.",
        "definitions": {
            "affected_mechanisms": "Nodes strictly downstream of X and ancestral to Y, including Y and excluding X.",
            "affected_convergence": "An affected node has at least two parents that are also affected nodes.",
            "direct_source_outcome_edge": "The serialized DAG contains X -> Y.",
            "frozen_certified": "The existing replay's before.status equals ok; not a fresh solver result.",
        },
        "limitations": [
            "Production attempt fractions describe this exact prepared dataset, not the full generator law.",
            "Frozen 74 tasks are a development collection; their failure and conditional rates are not population estimates.",
            "Rejected production worlds are not serialized. Their structural rejection rates are unknown and were not reconstructed.",
            "Accepted task coverage is not evidence that adding harder tasks would improve learning or explain old 800-step failure.",
            "No production code, rewards, budgets, training or validation configuration is changed by this reader.",
        ],
        "source_sha256": {
            TRAIN_ARCHIVE: TRAIN_SHA, FROZEN_ARCHIVE: FROZEN_SHA, REPLAY: REPLAY_SHA,
            "reader": sha256(Path(__file__).read_bytes()),
            "worldspec_schema_source": sha256((ROOT / "src/cpt_world/world_space.py").read_bytes()),
        },
        "production": production_coverage(),
        "frozen": frozen_coverage(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"production_splits": result["production"]["splits"],
                      "frozen_classes": result["frozen"]["by_class"],
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
