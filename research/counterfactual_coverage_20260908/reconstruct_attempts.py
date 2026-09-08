"""Reconstruct recorded CF candidates using unchanged production generation.

No counterfactual solve, label substitution, or model inference is performed.
Every accepted reconstruction must equal its archived WorldSpec and seed.
"""

import argparse
import dataclasses
import hashlib
import io
import json
import pickle
import re
import subprocess
import sys
import tarfile
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

from read_coverage import (
    FAMILY, TRAIN_ARCHIVE, TRAIN_SHA, WORLD_FIELDS, RecordReader,
    aggregate_graphs, graph_traits, json_lines, member_bytes,
)

GENERATION_COMMIT = "3f83d21dbe925d8c3e37f2242d391e65e2302840"
START_ARCHIVE = "research/training_submission_20260907/training-start.tar.gz"
START_SHA = "6f551cbe9c64213d67857b7cb0b91f4b47229c9b14c3dd143155f478deb383a6"


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def canonical(value):
    if dataclasses.is_dataclass(value):
        return canonical(dataclasses.asdict(value))
    if isinstance(value, Fraction):
        return {"fraction": [value.numerator, value.denominator]}
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected-worlds", type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    tracked = subprocess.check_output(
        ["git", "-C", str(project), "ls-tree", "-r", "--name-only", GENERATION_COMMIT, "src"],
        text=True,
    ).splitlines()
    start_blob = (project / START_ARCHIVE).read_bytes()
    assert digest(start_blob) == START_SHA
    with tarfile.open(fileobj=io.BytesIO(start_blob)) as archive:
        preflight = json.loads(member_bytes(archive, "preflight-acceptance.json"))
    assert preflight["passed"]
    # Compare exact captured runtime bytes. Git stores normalized LF for some
    # source files whose actual generation-time working tree retained CRLF.
    source_sha = {}
    for relative in tracked:
        original = subprocess.check_output(
            ["git", "-C", str(project), "show", f"{GENERATION_COMMIT}:{relative}"]
        )
        actual = (project / relative).read_bytes()
        assert digest(actual) == preflight["project_files"][relative], relative
        assert actual.replace(b"\r\n", b"\n") == original, relative
        source_sha[relative] = digest(actual)
    sys.path.insert(0, str(project / "src"))
    from cpt_world.world_space import WorldGrammar, assemble_sampled_anchor_tasks

    archive_blob = (project / TRAIN_ARCHIVE).read_bytes()
    assert digest(archive_blob) == TRAIN_SHA
    started = time.monotonic()
    records = []
    rejected_blobs = []
    with tarfile.open(fileobj=io.BytesIO(archive_blob)) as archive:
        attempts_blob = member_bytes(archive, "cf-attempts.jsonl")
        attempts = json_lines(attempts_blob)
        accepted = {
            row["seed_id"]: row for row in json_lines(member_bytes(archive, "data/accepted.jsonl"))
            if row["query_type"] == FAMILY
        }
        for index, attempt in enumerate(attempts):
            identity = attempt["seed_id"]
            match = re.fullmatch(r"SAMPLED-(\d+)-individual_counterfactual_probability-target_query-a(\d+)", identity)
            assert match, identity
            sample_index, anchor_index = map(int, match.groups())
            before = time.monotonic()
            built = assemble_sampled_anchor_tasks(WorldGrammar(), sample_index, FAMILY, anchor_index)
            assert len(built) == 1
            world, seed = built[0]
            assert seed["seed_id"] == identity
            split = "train" if 2000000 <= sample_index < 3000000 else "validation"
            assert 2000000 <= sample_index < 4000000
            archived_equal = None
            if attempt["status"] == "accepted":
                item = accepted[identity]
                assert item["split"] == split
                blob = member_bytes(archive, f"data/truth/{split}-{item['index']:04}.pkl")
                assert digest(blob) == item["world_file_sha256"]
                _row, prior_world, prior_seed, _truth = RecordReader(io.BytesIO(blob)).load()
                assert all(getattr(world, field) == getattr(prior_world, field) for field in WORLD_FIELDS)
                assert seed == prior_seed, identity
                archived_equal = True
            else:
                assert identity not in accepted and attempt["status"] == "rejected"
                rejected_blob = pickle.dumps((world, seed), protocol=4)
                rejected_name = f"{split}/{identity}.pkl"
                rejected_blobs.append((rejected_name, rejected_blob))
            records.append({
                "seed_id": identity, "split": split, "status": attempt["status"],
                "error_type": attempt.get("error_type"), "accepted_archive_equal": archived_equal,
                "reconstruction_seconds": time.monotonic() - before,
                "world_and_seed_sha256": digest(json.dumps(canonical([world, seed]), sort_keys=True).encode()),
                **graph_traits(world, seed),
            })
            if attempt["status"] == "rejected":
                records[-1].update(
                    reconstructed_world_member=rejected_name,
                    reconstructed_world_sha256=digest(rejected_blob),
                )
            if (index + 1) % 20 == 0:
                print(json.dumps({"reconstructed": index + 1, "total": len(attempts)}), flush=True)
    by_split = {}
    predicates = {
        "at_most_two_affected_mechanisms": lambda row: row["affected_mechanisms"] <= 2,
        "at_least_three_affected_mechanisms": lambda row: row["affected_mechanisms"] >= 3,
        "affected_convergence": lambda row: row["affected_convergence"],
        "direct_source_outcome_edge": lambda row: row["direct_source_outcome_edge"],
        "no_direct_source_outcome_edge": lambda row: not row["direct_source_outcome_edge"],
    }
    for split in ("train", "validation"):
        selected = [row for row in records if row["split"] == split]
        by_split[split] = {
            "all": aggregate_graphs(selected),
            "accepted": aggregate_graphs([row for row in selected if row["status"] == "accepted"]),
            "rejected": aggregate_graphs([row for row in selected if row["status"] == "rejected"]),
            "by_class": {
                name: dict(Counter(row["status"] for row in selected if predicate(row)))
                for name, predicate in predicates.items()
            },
        }
    report = {
        "generation_commit": GENERATION_COMMIT, "source_sha256": source_sha,
        "archive_sha256": TRAIN_SHA, "generation_preflight_archive_sha256": START_SHA,
        "attempt_ledger_sha256": digest(attempts_blob),
        "reader_sha256": digest(Path(__file__).read_bytes()),
        "graph_reader_sha256": digest(Path(__file__).with_name("read_coverage.py").read_bytes()),
        "python": sys.version, "wall_seconds": time.monotonic() - started,
        "accepted_worlds_and_seeds_equal": sum(row["accepted_archive_equal"] is True for row in records),
        "splits": by_split, "records": records,
        "scope": [
            "Recorded candidate identities only; no new random task sample or CF solver call.",
            "Acceptance labels and failures are the original recorded outcomes, not rerun solver outcomes.",
            "Rejected worlds are deterministic reconstructions, not previously saved world files.",
            "Rates describe this candidate stream, not a population guarantee or effect on LLM learning.",
        ],
    }
    if args.rejected_worlds:
        # Preserve actual rejected candidate inputs for subsequent kernel work.
        # They have no certified truth labels and never enter training here.
        with tarfile.open(args.rejected_worlds, "x:gz") as archive:
            for name, blob in rejected_blobs:
                member = tarfile.TarInfo(name)
                member.size = len(blob)
                archive.addfile(member, io.BytesIO(blob))
        report["rejected_world_archive"] = {
            "file": args.rejected_worlds.name,
            "sha256": digest(args.rejected_worlds.read_bytes()),
            "worlds": len(rejected_blobs),
            "certified_labels": False,
        }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"accepted_archive_equal": report["accepted_worlds_and_seeds_equal"], "splits": by_split}), flush=True)


if __name__ == "__main__":
    main()
