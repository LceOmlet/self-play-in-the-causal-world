"""CPU-only native checkpoint migration; no inference or training implementation."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import RandomSampler

from cpt_world import verl_streaming_dataset as streaming
from scripts.migrate_dapo_continuous_data import (
    migrate,
    native_cursor,
    native_loader,
    sha256,
    validate_source_cursor,
    write_json,
)


class FixtureStream:
    """Deterministic environment-supplier fixture; the data and load paths stay native."""

    calls = []

    def __init__(self, start_seed):
        self.start_seed = start_seed
        self.accepted = 0
        self.last_attempts = []

    def __next__(self):
        self.calls.append(self.start_seed + self.accepted)
        row = {"seed": self.start_seed + self.accepted}
        self.accepted += 1
        return row

    def state_dict(self):
        return {"accepted_rows": self.accepted}

    def load_state_dict(self, state):
        self.accepted = state["accepted_rows"]


def fixture_convert(row, index):
    return {"prompt": [{"role": "user", "content": str(row["seed"])}],
            "extra_info": {"index": index}, "data_source": "fixture"}


class ContinuousMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source/global_step_3"
        self.destination = self.root / "destination/global_step_3"
        self.source.mkdir(parents=True)
        actor = self.source / "actor"
        actor.mkdir()
        write_json(actor / "fsdp_config.json", {"FSDP_version": 2, "world_size": 1})
        torch.save({"lora": torch.arange(8.)}, actor / "model_world_size_1_rank_0.pt")
        torch.save({"state": {0: {"step": torch.tensor(3.), "exp_avg": torch.arange(8.),
                                 "exp_avg_sq": torch.arange(8.)}}},
                   actor / "optim_world_size_1_rank_0.pt")
        torch.save({"lr_scheduler": {"last_epoch": 3},
                    "rng": {"cpu": torch.get_rng_state(), "numpy": None,
                            "random": None, "cuda": None}},
                   actor / "extra_state_world_size_1_rank_0.pt")
        rows = range(11)
        loader = StatefulDataLoader(rows, batch_size=1, num_workers=0, drop_last=True,
                                    sampler=RandomSampler(rows, generator=torch.Generator()
                                                          .manual_seed(42)))
        iterator = iter(loader)
        for _ in range(5):
            next(iterator)
        self.old_state = loader.state_dict()
        self.progress = {"version": 1, "completed_updates": 3, "generated_batches": 5,
                         "data_epoch": 0, "batches_per_epoch": 11}
        torch.save(self.old_state, self.source / "data.pt")
        write_json(self.source / "dapo_progress.json", self.progress)
        self.descriptor_path = self.root / "continuous.json"
        self.descriptor = {
            "version": 1, "kind": "cpt_world_continuous", "start_seed": 900,
            "stream_start_index": 5, "journal_dir": str(self.root / "journal"),
            "environment_version": streaming.INTERACTION_SURFACE_VERSION,
            "source_fingerprints": streaming.current_source_fingerprints(),
        }
        write_json(self.descriptor_path, self.descriptor)
        self.dataset = self.make_dataset()
        FixtureStream.calls = []

    def make_dataset(self):
        return streaming.CPTWorldStreamingDataset(
            data_files=[str(self.descriptor_path)], tokenizer=None,
            config={"shuffle": False, "dataloader_num_workers": 0, "train_batch_size": 1,
                    "gen_batch_size": 1, "filter_overlong_prompts": False})

    def test_full_migration_preserves_training_bytes_and_actual_adapter_next_index(self):
        original_hashes = {path.relative_to(self.source).as_posix(): sha256(path)
                           for path in self.source.rglob("*") if path.is_file()}
        with (patch.object(streaming, "_make_stream", side_effect=FixtureStream),
              patch.object(streaming, "_convert_row", side_effect=fixture_convert)):
            report = migrate(self.source, self.destination, self.descriptor_path, self.dataset)
            self.assertEqual(FixtureStream.calls, [], "Migration must not generate any task")
            self.assertEqual(report["native_acceptance"]["completed_updates"], 3)
            self.assertEqual(report["native_acceptance"]["generated_batches"], 5)
            self.assertEqual(report["new_progress"]["batches_per_epoch"], sys.maxsize)
            restored = native_loader(self.make_dataset())
            restored.load_state_dict(torch.load(self.destination / "data.pt", weights_only=False))
            first = next(iter(restored))
            self.assertEqual(first["index"], 5)
            self.assertEqual(first["raw_prompt"], [{"role": "user", "content": "900"}])
            self.assertEqual(FixtureStream.calls, [900])
            # Restore that same original checkpoint after the journal has advanced.
            again = native_loader(self.make_dataset())
            again.load_state_dict(torch.load(self.destination / "data.pt", weights_only=False))
            self.assertEqual(next(iter(again))["index"], 5)
            self.assertEqual(FixtureStream.calls, [900], "Committed row must replay without rerun")
        for name, expected in original_hashes.items():
            self.assertEqual(sha256(self.source / name), expected)
            target = (self.destination / "migration_provenance" / name
                      if name in {"data.pt", "dapo_progress.json"} else self.destination / name)
            self.assertEqual(sha256(target), expected)

    def test_nonzero_old_epoch_uses_global_generated_count(self):
        self.progress.update(generated_batches=27, data_epoch=2)
        write_json(self.source / "dapo_progress.json", self.progress)
        self.descriptor.update(stream_start_index=27, journal_dir=str(self.root / "journal27"))
        write_json(self.descriptor_path, self.descriptor)
        report = migrate(self.source, self.destination, self.descriptor_path, self.make_dataset())
        self.assertEqual(report["native_acceptance"]["next_global_index"], 27)
        self.assertEqual(report["new_progress"]["completed_updates"], 3)
        self.assertEqual(report["new_progress"]["generated_batches"], 27)
        self.assertEqual(report["new_progress"]["data_epoch"], 0)

    def test_reject_existing_destination_and_mismatched_descriptor(self):
        self.destination.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "destination must not exist"):
            migrate(self.source, self.destination, self.descriptor_path, self.dataset)
        destination = self.root / "other/global_step_3"
        self.descriptor["stream_start_index"] = 6
        write_json(self.descriptor_path, self.descriptor)
        with self.assertRaisesRegex(ValueError, "must equal saved"):
            migrate(self.source, destination, self.descriptor_path, self.dataset)
        self.assertFalse(destination.exists())

    def test_reject_missing_optimizer(self):
        optimizer = self.source / "actor/optim_world_size_1_rank_0.pt"
        optimizer.unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete actor checkpoint"):
            migrate(self.source, self.destination, self.descriptor_path, self.dataset)
        self.assertFalse(self.destination.exists())

    def test_reject_scheduler_mismatch(self):
        path = self.source / "actor/extra_state_world_size_1_rank_0.pt"
        extra = torch.load(path, weights_only=False)
        extra["lr_scheduler"]["last_epoch"] = 2
        torch.save(extra, path)
        with self.assertRaisesRegex(ValueError, "scheduler update mismatch"):
            migrate(self.source, self.destination, self.descriptor_path, self.dataset)
        self.assertFalse(self.destination.exists())

    def test_reject_unknown_and_inconsistent_old_data_schemas(self):
        for mutation in (
            lambda state: state.update(unknown_field=None),
            lambda state: state.update(_num_yielded=4),
            lambda state: state.update(_iterator_finished=True),
            lambda state: state.update(_sampler_iter_yielded=6),
            lambda state: state["_sampler_iter_state"]["sampler_iter_state"].update(yielded=4),
        ):
            state = copy.deepcopy(self.old_state)
            mutation(state)
            with self.assertRaises(ValueError):
                validate_source_cursor(state, self.progress, self.source)

    def test_native_cursor_never_calls_actual_adapter_getitem(self):
        with patch.object(streaming.CPTWorldStreamingDataset, "__getitem__",
                          side_effect=AssertionError("No task request allowed")):
            state = native_cursor(self.dataset, 5)
        self.assertEqual(state["_num_yielded"], 5)
        self.assertEqual(state["dataset_state"], self.dataset.state_dict())
        self.assertEqual(json.loads((self.root / "journal/head.json").read_text())["next_index"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
