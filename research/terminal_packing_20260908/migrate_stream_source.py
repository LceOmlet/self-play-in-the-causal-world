"""Prepare an offline, data-identity-only migration between certified kernels.

Requires a quiescent checkpoint/journal and acceptance evidence bound to the new
kernel SHA. Never controls training, generates a task, or loads model weights.
The old source, checkpoint and journal remain read-only. A failed preparation
leaves its output for inspection and never writes a successful receipt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

KERNEL = "src/cpt_world/counterfactual_solver.py"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def runtime():
    from cpt_world import verl_streaming_dataset as streaming
    from cpt_world.trl_environment import BalancedTrainingRowStream

    project = Path(streaming.__file__).resolve().parents[2]
    path = project / "scripts/migrate_dapo_continuous_data.py"
    spec = importlib.util.spec_from_file_location("_native_continuous_migration", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return streaming, BalancedTrainingRowStream, helper


def files_under(directory):
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Expected a real directory: {directory}")
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError(f"Unsupported link or special file: {path}")
        if path.is_file():
            raw = path.read_bytes()
            result[path.relative_to(directory).as_posix()] = {
                "sha256": digest(raw),
                "size": len(raw),
            }
    return result


def project_fingerprints(project):
    paths = sorted((project / "src/cpt_world").rglob("*.py"))
    paths.append(project / "scripts/prepare_verl_cpt_data.py")
    if any(path.is_symlink() for path in paths):
        raise ValueError("Source file symlinks are unsupported")
    return {path.relative_to(project).as_posix(): digest(path.read_bytes()) for path in paths}


@contextmanager
def exclusive_existing_journal(directory):
    if os.name != "posix":
        raise RuntimeError("This offline migration uses the server's POSIX journal lock")
    import fcntl

    lock = directory / ".lock"
    if lock.is_symlink() or not lock.is_file():
        raise ValueError("The source journal lock is missing or is a link")
    with lock.open("rb") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Source journal is busy; use a quiescent snapshot") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def inspect_journal(descriptor, streaming, stream_class):
    directory = Path(descriptor["journal_dir"])
    inventory = files_under(directory)
    expected_fixed = {".lock", "journal.json", "head.json"}
    if not expected_fixed <= inventory.keys():
        raise ValueError("Partial journal: metadata, head or lock is missing")
    row_names = sorted(name for name in inventory if name.startswith("rows/"))
    if set(inventory) != expected_fixed | set(row_names) or any(
        re.fullmatch(r"rows/\d{20}\.json", name) is None for name in row_names
    ):
        raise ValueError(
            "Partial/unknown journal artifact; no temporary or future row is discarded"
        )
    meta, head = read(directory / "journal.json"), read(directory / "head.json")
    descriptor_hash = digest(streaming._json_bytes(descriptor))
    if (
        meta.get("version") != 1
        or meta.get("descriptor") != descriptor
        or meta.get("descriptor_sha256") != descriptor_hash
        or not isinstance(meta.get("journal_id"), str)
    ):
        raise ValueError("Source journal metadata does not match its descriptor")
    start, end = descriptor["stream_start_index"], head.get("next_index")
    if head.get("version") != 1 or type(end) is not int or not start <= end < sys.maxsize:
        raise ValueError("Invalid source journal head")
    if row_names != [f"rows/{index:020d}.json" for index in range(start, end)]:
        raise ValueError(
            "Partial journal: missing row or complete row outside head; recover with old runtime"
        )
    previous_hash = None
    stream = stream_class(start_seed=descriptor["start_seed"])
    previous_cursor = stream.state_dict()
    records, hashes = [], {}
    for index, name in enumerate(row_names, start):
        record = read(directory / name)
        if (
            record.get("version") != 1
            or type(record.get("index")) is not int
            or record["index"] != index
            or record.get("journal_id") != meta["journal_id"]
            or record.get("descriptor_sha256") != descriptor_hash
            or record.get("previous_record_sha256") != previous_hash
        ):
            raise ValueError(f"Journal chain/identity mismatch at row {index}")
        for field in ("row", "converted_row"):
            if not isinstance(record.get(field), dict) or record.get(field + "_sha256") != digest(
                streaming._json_bytes(record[field])
            ):
                raise ValueError(f"Journal task content mismatch at row {index}")
        cursor = record.get("next_generator_state")
        stream.load_state_dict(cursor)
        if cursor["accepted_rows"] != index - start + 1:
            raise ValueError(f"Generator delivery cursor mismatch at row {index}")
        attempts = record.get("last_attempts")
        if (
            not isinstance(attempts, list)
            or not attempts
            or any(not isinstance(item, dict) for item in attempts)
            or [item.get("status") for item in attempts]
            != ["rejected"] * (len(attempts) - 1) + ["accepted"]
            or [item.get("sample_index") for item in attempts]
            != list(range(previous_cursor["next_sample_index"], cursor["next_sample_index"]))
        ):
            raise ValueError(f"Generator rejection/acceptance cursor mismatch at row {index}")
        previous_cursor = cursor
        previous_hash = inventory[name]["sha256"]
        records.append(record)
        hashes[index] = previous_hash
    if head.get("last_record_sha256") != previous_hash:
        raise ValueError("Journal head hash does not match the complete row chain")
    return inventory, meta, head, records, hashes


def validate_checkpoint(state, progress, checkpoint, descriptor, meta, head, hashes, helper):
    match = re.fullmatch(r"global_step_(\d+)", checkpoint.name)
    if (
        not match
        or not isinstance(progress, dict)
        or set(progress) != helper.PROGRESS_KEYS
        or any(type(value) is not int for value in progress.values())
    ):
        raise ValueError("Unsupported DAPO progress schema")
    updates, generated = int(match.group(1)), progress["generated_batches"]
    if (
        progress["version"] != 1
        or progress["completed_updates"] != updates
        or not 1 <= updates <= generated < sys.maxsize
        or progress["data_epoch"] != 0
        or progress["batches_per_epoch"] != sys.maxsize
        or not isinstance(state, dict)
        or set(state) != helper.STATE_KEYS
    ):
        raise ValueError("Checkpoint is not a continuous-stream native checkpoint")
    if (
        state["_num_yielded"] != generated
        or state["_sampler_iter_yielded"] != generated
        or state["_sampler_iter_state"] != {"samples_yielded": generated}
        or state["_iterator_finished"] is not False
        or any(
            state[key] is not None
            for key in (
                "_index_sampler_state",
                "_IterableDataset_len_called",
                "_shared_seed",
                "fetcher_state",
            )
        )
    ):
        raise ValueError("Unsupported or inconsistent native sequential sampler state")
    saved = state["dataset_state"]
    expected_keys = {
        "version",
        "kind",
        "journal_id",
        "descriptor_sha256",
        "next_index",
        "last_record_sha256",
    }
    if not isinstance(saved, dict) or set(saved) != expected_keys:
        raise ValueError("Checkpoint is missing its continuous journal identity")
    start, committed = descriptor["stream_start_index"], saved["next_index"]
    if (
        saved["version"] != 1
        or saved["kind"] != "cpt_world_continuous_journal"
        or saved["journal_id"] != meta["journal_id"]
        or saved["descriptor_sha256"] != meta["descriptor_sha256"]
        or type(committed) is not int
        or not start <= generated <= committed <= head["next_index"]
        or saved["last_record_sha256"] != (None if committed == start else hashes[committed - 1])
    ):
        raise ValueError("Checkpoint journal prefix disagrees with the complete source journal")
    return updates, generated, committed


def migrate(source_checkpoint, source_descriptor, old_project, destination_run, evidence_path):
    import torch

    streaming, stream_class, helper = runtime()
    source_checkpoint, source_descriptor, old_project, destination_run, evidence_path = (
        Path(path).resolve()
        for path in (
            source_checkpoint,
            source_descriptor,
            old_project,
            destination_run,
            evidence_path,
        )
    )
    if destination_run.exists() or not source_checkpoint.is_dir():
        raise ValueError("Source checkpoint must exist and destination run must not exist")
    original_descriptor_raw = source_descriptor.read_bytes()
    descriptor = json.loads(original_descriptor_raw)
    if (
        descriptor.get("version") != 1
        or descriptor.get("kind") != "cpt_world_continuous"
        or descriptor.get("environment_version") != streaming.INTERACTION_SURFACE_VERSION
        or any(
            type(descriptor.get(key)) is not int or descriptor[key] < 0
            for key in ("start_seed", "stream_start_index")
        )
    ):
        raise ValueError("Unsupported source continuous descriptor")
    directory = Path(descriptor["journal_dir"]).expanduser()
    if not directory.is_absolute():
        directory = source_descriptor.parent / directory
    descriptor["journal_dir"] = str(directory.resolve())
    directory = Path(descriptor["journal_dir"])
    for prior in (source_checkpoint, directory, old_project):
        if destination_run.is_relative_to(prior) or prior.is_relative_to(destination_run):
            raise ValueError("Source and destination trees must be disjoint")
    old_sources, new_sources = (
        project_fingerprints(old_project),
        streaming.current_source_fingerprints(),
    )
    if descriptor.get("source_fingerprints") != old_sources:
        raise ValueError("Old source tree no longer matches the journal descriptor")
    changed = {
        name
        for name in old_sources.keys() | new_sources.keys()
        if old_sources.get(name) != new_sources.get(name)
    }
    if changed != {KERNEL}:
        raise ValueError(f"Only the verified counterfactual kernel may change: {sorted(changed)}")
    evidence_raw = evidence_path.read_bytes()
    evidence = json.loads(evidence_raw)
    if (
        evidence.get("passed") is not True
        or evidence.get("candidate_sha256") != new_sources[KERNEL]
    ):
        raise ValueError("Acceptance evidence must pass and match the exact new kernel SHA")

    with exclusive_existing_journal(directory):
        inventory, meta, head, records, hashes = inspect_journal(
            descriptor, streaming, stream_class
        )
        state = torch.load(source_checkpoint / "data.pt", map_location="cpu", weights_only=False)
        progress = read(source_checkpoint / "dapo_progress.json")
        updates, generated, committed = validate_checkpoint(
            state, progress, source_checkpoint, descriptor, meta, head, hashes, helper
        )
        checkpoint_inventory = helper.source_inventory(source_checkpoint, updates)
        destination_run.mkdir(parents=True)
        archive = destination_run / "source-migration"
        archive.mkdir()
        destination_checkpoint = destination_run / "preserved" / source_checkpoint.name
        shutil.copytree(source_checkpoint, destination_checkpoint)
        shutil.copytree(directory, archive / "old-journal")
        (archive / "old-descriptor.json").write_bytes(original_descriptor_raw)
        (archive / "acceptance-evidence.json").write_bytes(evidence_raw)
        for name in ("data.pt", "dapo_progress.json"):
            shutil.copyfile(source_checkpoint / name, archive / name)

        new_descriptor = copy.deepcopy(descriptor)
        new_descriptor.update(
            journal_dir=str(destination_run / "task-journal"),
            source_fingerprints=new_sources,
            source_migration={
                "version": 1,
                "archive": str(archive),
                "inherited_until_index": head["next_index"],
                "inherited_range": {
                    "start": descriptor["stream_start_index"],
                    "end_exclusive": head["next_index"],
                    "owner": "old descriptor, preserved in old-descriptor.json",
                },
                "new_source_from_index": head["next_index"],
                "old_journal_id": meta["journal_id"],
                "old_descriptor_sha256": meta["descriptor_sha256"],
                "old_source_fingerprints": old_sources,
                "old_journal_manifest_sha256": digest(streaming._json_bytes(inventory)),
                "acceptance_evidence_sha256": digest(evidence_raw),
            },
        )
        new_descriptor_path = destination_run / "stream.json"
        new_descriptor_path.write_bytes(streaming._json_bytes(new_descriptor))
        new_hash, new_id = digest(streaming._json_bytes(new_descriptor)), uuid4().hex
        new_directory = Path(new_descriptor["journal_dir"])
        (new_directory / "rows").mkdir(parents=True)
        (new_directory / ".lock").touch()
        streaming._atomic_json(
            new_directory / "journal.json",
            {
                "version": 1,
                "descriptor": new_descriptor,
                "descriptor_sha256": new_hash,
                "journal_id": new_id,
            },
        )
        previous_hash, new_hashes = None, {}
        for record in records:
            rebound = dict(
                record,
                journal_id=new_id,
                descriptor_sha256=new_hash,
                previous_record_sha256=previous_hash,
            )
            index = record["index"]
            previous_hash = streaming._atomic_json(
                new_directory / "rows" / f"{index:020d}.json", rebound
            )
            new_hashes[index] = previous_hash
        streaming._atomic_json(
            new_directory / "head.json",
            {
                "version": 1,
                "next_index": head["next_index"],
                "last_record_sha256": previous_hash,
            },
        )
        new_state = copy.deepcopy(state)
        new_state["dataset_state"] = dict(
            state["dataset_state"],
            journal_id=new_id,
            descriptor_sha256=new_hash,
            last_record_sha256=(
                None if committed == descriptor["stream_start_index"] else new_hashes[committed - 1]
            ),
        )
        torch.save(new_state, destination_checkpoint / "data.pt")
        dataset = streaming.CPTWorldStreamingDataset(
            data_files=[str(new_descriptor_path)],
            tokenizer=None,
            config={
                "shuffle": False,
                "dataloader_num_workers": 0,
                "train_batch_size": 1,
                "gen_batch_size": 1,
                "filter_overlong_prompts": False,
            },
        )
        native_acceptance = helper.verify_official_restore(
            destination_checkpoint, dataset, updates, generated
        )
        inspect_journal(new_descriptor, streaming, stream_class)
        copied = helper.source_inventory(destination_checkpoint, updates)
        if {k: v for k, v in copied.items() if k != "data.pt"} != {
            k: v for k, v in checkpoint_inventory.items() if k != "data.pt"
        }:
            raise RuntimeError("Migration changed a native training artifact other than data.pt")
        if (
            helper.source_inventory(source_checkpoint, updates) != checkpoint_inventory
            or files_under(directory) != inventory
            or files_under(archive / "old-journal") != inventory
            or project_fingerprints(old_project) != old_sources
            or streaming.current_source_fingerprints() != new_sources
            or source_descriptor.read_bytes() != original_descriptor_raw
            or evidence_path.read_bytes() != evidence_raw
        ):
            raise RuntimeError("Source changed during migration; prepared output is not approved")
        for name in ("data.pt", "dapo_progress.json"):
            if digest((archive / name).read_bytes()) != checkpoint_inventory[name]["sha256"]:
                raise RuntimeError("Original native data/progress archive was not preserved")
        report = {
            "passed": True,
            "version": 1,
            "completed_updates": updates,
            "generated_batches": generated,
            "saved_journal_prefix": committed,
            "inherited_until_index": head["next_index"],
            "first_new_kernel_index": head["next_index"],
            "source_checkpoint": str(source_checkpoint),
            "source_descriptor": str(source_descriptor),
            "destination_checkpoint": str(destination_checkpoint),
            "destination_descriptor": str(new_descriptor_path),
            "checkpoint_inventory": checkpoint_inventory,
            "old_journal_inventory": inventory,
            "new_journal_inventory": files_under(new_directory),
            "changed_training_files": ["data.pt:dataset_state"],
            "native_acceptance": native_acceptance,
            "next_generator_state": records[-1]["next_generator_state"]
            if records
            else stream_class(start_seed=descriptor["start_seed"]).state_dict(),
            "migration_script_sha256": digest(Path(__file__).read_bytes()),
            "scope": "Offline journal identity rebinding; task contents and cursors retained. "
            "Official load exercised with GPU actor RPC mocked. No task generation or training.",
        }
        streaming._atomic_json(archive / "receipt.json", report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-checkpoint",
        "source-descriptor",
        "old-project",
        "destination-run",
        "evidence",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    report = migrate(
        args.source_checkpoint,
        args.source_descriptor,
        args.old_project,
        args.destination_run,
        args.evidence,
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "passed",
                    "completed_updates",
                    "generated_batches",
                    "first_new_kernel_index",
                    "destination_checkpoint",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
