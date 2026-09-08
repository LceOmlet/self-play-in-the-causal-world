#!/usr/bin/env python3
"""Explicit data-only migration from the finite tape to continuous CPT tasks.

Run against a complete, quiescent checkpoint. The source is read-only. This is
checkpoint preparation, not a trainer: model/Adam/scheduler/RNG bytes are copied
unchanged, while a native StatefulDataLoader supplies the replacement cursor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

STATE_KEYS = {
    "_index_sampler_state", "_sampler_iter_state", "_sampler_iter_yielded",
    "_num_yielded", "_IterableDataset_len_called", "_shared_seed", "fetcher_state",
    "dataset_state", "_iterator_finished",
}
PROGRESS_KEYS = {
    "version", "completed_updates", "generated_batches", "data_epoch", "batches_per_epoch",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_source_cursor(state: dict, progress: dict, checkpoint: Path) -> tuple[int, int]:
    """Accept only the installed torchdata 0.11 single-process random tape schema."""
    match = re.fullmatch(r"global_step_(\d+)", checkpoint.name)
    if not match or type(progress) is not dict or set(progress) != PROGRESS_KEYS:
        raise ValueError("Unsupported checkpoint name or DAPO progress schema")
    if any(type(progress[key]) is not int for key in PROGRESS_KEYS):
        raise ValueError("DAPO progress fields must be integers")
    updates = int(match.group(1))
    generated, epoch, length = (
        progress["generated_batches"], progress["data_epoch"], progress["batches_per_epoch"]
    )
    if (progress["version"] != 1 or progress["completed_updates"] != updates
            or updates < 1 or generated < updates or epoch < 0 or length < 1):
        raise ValueError("Invalid DAPO update/generation progress")
    if type(state) is not dict or set(state) != STATE_KEYS:
        raise ValueError("Unsupported StatefulDataLoader state schema")
    cursor = state["_num_yielded"]
    if (type(cursor) is not int or not 0 <= cursor <= length
            or generated != epoch * length + cursor
            or type(state["_iterator_finished"]) is not bool
            or (state["_iterator_finished"] and cursor != length)):
        raise ValueError("DAPO progress disagrees with source data cursor")
    for key in ("_index_sampler_state", "_IterableDataset_len_called", "_shared_seed",
                "fetcher_state", "dataset_state"):
        if state[key] is not None:
            raise ValueError(f"Unsupported finite tape state: {key}")
    sampler = state["_sampler_iter_state"]
    if (type(state["_sampler_iter_yielded"]) is not int
            or state["_sampler_iter_yielded"] != cursor or type(sampler) is not dict
            or set(sampler) != {"samples_yielded", "sampler_iter_state"}
            or type(sampler["samples_yielded"]) is not int
            or sampler["samples_yielded"] != cursor):
        raise ValueError("Unsupported random sampler cursor")
    random_state = sampler["sampler_iter_state"]
    if (type(random_state) is not dict or set(random_state) != {"yielded", "generator"}
            or type(random_state["yielded"]) is not int or random_state["yielded"] != cursor):
        raise ValueError("Unsupported random sampler iterator state")
    generator_state = random_state["generator"]
    if (not isinstance(generator_state, torch.Tensor) or generator_state.device.type != "cpu"
            or generator_state.dtype != torch.uint8 or generator_state.ndim != 1):
        raise ValueError("Unsupported random sampler RNG state")
    # Native validation also catches malformed byte sequences of otherwise valid type.
    torch.Generator().set_state(generator_state)
    return updates, generated


def source_inventory(source: Path, updates: int) -> dict[str, dict]:
    """Reject links/incomplete rank checkpoints; inventory every byte to be retained."""
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Checkpoint symlinks are unsupported: {path}")
    fsdp = read_json(source / "actor/fsdp_config.json")
    world_size = fsdp.get("world_size")
    if type(world_size) is not int or world_size < 1 or fsdp.get("FSDP_version") != 2:
        raise ValueError("Migration requires the verified native FSDP2 checkpoint format")
    for rank in range(world_size):
        suffix = f"world_size_{world_size}_rank_{rank}.pt"
        for stem in ("model", "optim", "extra_state"):
            path = source / "actor" / f"{stem}_{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Incomplete actor checkpoint: {path}")
        extra = torch.load(source / "actor" / f"extra_state_{suffix}",
                           map_location="cpu", weights_only=False)
        if (set(extra) != {"lr_scheduler", "rng"}
                or extra["lr_scheduler"].get("last_epoch") != updates
                or not {"cpu", "numpy", "random", "cuda"} <= set(extra["rng"])):
            raise ValueError("Missing RNG or scheduler state, or scheduler update mismatch")
    return {
        path.relative_to(source).as_posix(): {"size": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(source.rglob("*")) if path.is_file()
    }


class IndexOnlyDataset:
    """Exercise the native sampler without generating any past/new environment."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index

    def state_dict(self):
        return self.dataset.state_dict()

    def load_state_dict(self, state):
        return self.dataset.load_state_dict(state)


def native_loader(dataset) -> StatefulDataLoader:
    return StatefulDataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset),
                              num_workers=0, drop_last=True, collate_fn=lambda rows: rows[0])


def native_cursor(dataset, generated: int) -> dict:
    if len(dataset) != sys.maxsize or not 0 <= generated < len(dataset):
        raise ValueError("Continuous dataset must expose the sys.maxsize index address space")
    loader = native_loader(IndexOnlyDataset(dataset))
    iterator = iter(loader)
    for expected in range(generated):
        if next(iterator) != expected:
            raise AssertionError("Native sequential sampler changed index order")
    state = loader.state_dict()
    if (state["_num_yielded"] != generated or state["_iterator_finished"]
            or state["_sampler_iter_state"] != {"samples_yielded": generated}):
        raise ValueError("Unexpected installed StatefulDataLoader sequential schema")
    restored = native_loader(IndexOnlyDataset(dataset))
    restored.load_state_dict(state)
    if next(iter(restored)) != generated:
        raise AssertionError("Native loader failed to restore the next global stream index")
    return state


def verify_official_restore(checkpoint: Path, dataset, updates: int, generated: int) -> dict:
    """Run actual official base + DAPO load hooks; mock only the GPU actor RPC."""
    from unittest.mock import Mock

    from dapo.dapo_ray_trainer import RayDAPOTrainer
    from omegaconf import OmegaConf

    trainer = object.__new__(RayDAPOTrainer)
    trainer.config = OmegaConf.create({"trainer": {
        "resume_mode": "resume_path", "resume_from_path": str(checkpoint),
        "default_local_dir": str(checkpoint.parent), "default_hdfs_dir": None,
        "del_local_ckpt_after_load": False,
    }})
    trainer.global_steps = 0
    trainer.use_critic = False
    trainer.actor_rollout_wg = Mock()
    trainer.train_dataloader = native_loader(IndexOnlyDataset(dataset))
    trainer._load_checkpoint()
    trainer.actor_rollout_wg.load_checkpoint.assert_called_once_with(
        str(checkpoint / "actor"), del_local_after_load=False)
    next_index = next(iter(trainer.train_dataloader))
    if (trainer.global_steps != updates or trainer.gen_steps != generated
            or trainer.data_epoch != 0 or next_index != generated):
        raise AssertionError("Official DAPO restore did not preserve U/G and next stream index")
    return {"completed_updates": trainer.global_steps, "generated_batches": trainer.gen_steps,
            "data_epoch": trainer.data_epoch, "next_global_index": next_index,
            "recipe_source": str(Path(RayDAPOTrainer._load_checkpoint.__code__.co_filename)),
            "scope": "Actual official load hooks/native loader; only GPU actor RPC mocked; "
                     "index-only dataset preserves actual journal state without generating tasks."}


def migrate(source: Path, destination: Path, descriptor_path: Path, dataset) -> dict:
    from cpt_world.verl_streaming_dataset import validate_stream_descriptor

    source, destination, descriptor_path = (
        path.resolve() for path in (source, destination, descriptor_path)
    )
    if not source.is_dir() or destination.exists():
        raise ValueError("Source must exist and destination must not exist")
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and destination checkpoint trees must be disjoint")
    if destination.name != source.name:
        raise ValueError("Destination must retain the original global_step_U directory name")
    old_progress = read_json(source / "dapo_progress.json")
    old_state = torch.load(source / "data.pt", map_location="cpu", weights_only=False)
    updates, generated = validate_source_cursor(old_state, old_progress, source)
    descriptor = validate_stream_descriptor(descriptor_path)
    if descriptor.get("stream_start_index") != generated:
        raise ValueError("Descriptor stream_start_index must equal saved generated_batches G")
    if dataset._continuous_descriptor != descriptor:
        raise ValueError("Dataset was initialized from a different continuous descriptor")
    inventory = source_inventory(source, updates)
    if any(name.startswith("migration_provenance/") for name in inventory):
        raise ValueError("Source already contains a prior data migration")
    new_state = native_cursor(dataset, generated)
    new_progress = dict(old_progress, data_epoch=0, batches_per_epoch=sys.maxsize)
    destination.mkdir(parents=True, exist_ok=False)
    provenance = destination / "migration_provenance"
    provenance.mkdir()
    # Copy all original training bytes before writing the replacement data files.
    for name, record in inventory.items():
        target = (provenance / name if name in {"data.pt", "dapo_progress.json"}
                  else destination / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
        if target.stat().st_size != record["size"] or sha256(target) != record["sha256"]:
            raise RuntimeError(f"Checkpoint copy did not retain source bytes: {name}")
    torch.save(new_state, destination / "data.pt")
    write_json(destination / "dapo_progress.json", new_progress)
    shutil.copyfile(descriptor_path, provenance / "continuous_descriptor.json")
    acceptance = verify_official_restore(destination, dataset, updates, generated)
    # Fail if any source file changed while preparing, including a concurrent save.
    if inventory != source_inventory(source, updates):
        raise RuntimeError("Source checkpoint changed during migration; output is not approved")
    report = {
        "version": 1, "source": str(source), "destination": str(destination),
        "source_files": inventory, "old_progress": old_progress, "new_progress": new_progress,
        "changed_training_files": ["data.pt", "dapo_progress.json"],
        "descriptor_path": str(descriptor_path), "descriptor_sha256": sha256(descriptor_path),
        "native_acceptance": acceptance, "torch_version": torch.__version__,
        "migration_script_sha256": sha256(Path(__file__)),
        "new_data_sha256": sha256(destination / "data.pt"),
        "new_progress_sha256": sha256(destination / "dapo_progress.json"),
        "source_read_only": True,
    }
    write_json(provenance / "migration.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    args = parser.parse_args()
    from cpt_world.verl_streaming_dataset import CPTWorldStreamingDataset

    dataset = CPTWorldStreamingDataset(
        data_files=[str(args.descriptor)], tokenizer=None,
        config={"shuffle": False, "dataloader_num_workers": 0, "train_batch_size": 1,
                "gen_batch_size": 1, "filter_overlong_prompts": False})
    report = migrate(args.source, args.destination, args.descriptor, dataset)
    print(json.dumps(report["native_acceptance"], indent=2))


if __name__ == "__main__":
    main()
