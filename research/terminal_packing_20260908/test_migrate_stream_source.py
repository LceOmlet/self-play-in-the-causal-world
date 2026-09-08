"""CPU migration fixtures use the real journal, loader, and official load hook."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from cpt_world import verl_streaming_dataset as streaming
from cpt_world.registry import TASK_FAMILY_QUERY_TYPES
from cpt_world.trl_environment import BalancedTrainingRowStream


def load_migration():
    path = Path(__file__).with_name("migrate_stream_source.py")
    spec = importlib.util.spec_from_file_location("_stream_source_migration_tested", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MIGRATION = load_migration()
NATIVE = MIGRATION.runtime()[2]
CONFIG = {
    "shuffle": False,
    "dataloader_num_workers": 0,
    "train_batch_size": 1,
    "gen_batch_size": 1,
    "filter_overlong_prompts": False,
}


class FixtureStream:
    """No environment calls; cursors follow the real generator's five-family law."""

    calls = []
    owner = "old"

    def __init__(self, start_seed):
        self.real_cursor = BalancedTrainingRowStream(start_seed=start_seed)
        self.last_attempts = []

    def state_dict(self):
        return self.real_cursor.state_dict()

    def load_state_dict(self, state):
        self.real_cursor.load_state_dict(state)

    def __next__(self):
        state = self.state_dict()
        self.calls.append((self.owner, copy.deepcopy(state)))
        family = TASK_FAMILY_QUERY_TYPES[state["next_family_offset"]]
        start = state["next_sample_index"]
        reject = int(family == "individual_counterfactual_probability")
        self.last_attempts = [
            {
                "sample_index": index,
                "query_type": family,
                "seed_id": f"fixture-{index}",
                "status": "rejected" if index < start + reject else "accepted",
            }
            for index in range(start, start + reject + 1)
        ]
        count = state["accepted_rows"] + 1
        state.update(
            accepted_rows=count,
            next_sample_index=start + reject + 1,
            next_family_offset=count % 5,
            best_intervention_balance_slot=count // 5 + int(count % 5 > 3),
        )
        self.load_state_dict(state)
        return {
            "query_type": family,
            "tape_key": f"{self.owner}:{start + reject}",
            "terminal_truth_json": json.dumps({"owner": self.owner, "seed": start + reject}),
            "environment_version": streaming.INTERACTION_SURFACE_VERSION,
        }


def fixture_convert(row, index):
    return {
        "prompt": [{"role": "user", "content": row["tape_key"]}],
        "data_source": "fixture",
        "extra_info": {
            "index": index,
            "need_tools_kwargs": True,
            "tools_kwargs": {"act": {"create_kwargs": {"row_json": json.dumps(row)}}},
        },
    }


class SourceMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = Path(streaming.__file__).resolve().parents[2]
        self.old_project = self.root / "old-project"
        self.new_sources = streaming.current_source_fingerprints()
        for name in self.new_sources:
            target = self.old_project / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((self.project / name).read_bytes())
        kernel = self.old_project / MIGRATION.KERNEL
        kernel.write_bytes(kernel.read_bytes() + b"\n# Prior source fixture only.\n")
        self.old_sources = MIGRATION.project_fingerprints(self.old_project)
        self.descriptor_path = self.root / "old-stream.json"
        self.descriptor = {
            "version": 1,
            "kind": "cpt_world_continuous",
            "start_seed": 900,
            "stream_start_index": 5,
            "journal_dir": str(self.root / "old-journal"),
            "environment_version": streaming.INTERACTION_SURFACE_VERSION,
            "source_fingerprints": self.old_sources,
        }
        self.write(self.descriptor_path, self.descriptor)
        FixtureStream.owner, FixtureStream.calls = "old", []
        with (
            patch.object(streaming, "current_source_fingerprints", return_value=self.old_sources),
            patch.object(streaming, "_make_stream", side_effect=FixtureStream),
            patch.object(streaming, "_convert_row", side_effect=fixture_convert),
        ):
            dataset = self.dataset(self.descriptor_path)
            dataset[5]
            dataset[6]
            self.saved_state = NATIVE.native_cursor(dataset, 7)
            self.tail = [dataset[7], dataset[8]]
        self.source = self.root / "checkpoints/global_step_3"
        actor = self.source / "actor"
        actor.mkdir(parents=True)
        self.write(actor / "fsdp_config.json", {"FSDP_version": 2, "world_size": 1})
        torch.save({"lora": torch.arange(4.0)}, actor / "model_world_size_1_rank_0.pt")
        torch.save(
            {"state": {0: {"step": torch.tensor(3.0), "exp_avg": torch.arange(4.0)}}},
            actor / "optim_world_size_1_rank_0.pt",
        )
        torch.save(
            {
                "lr_scheduler": {"last_epoch": 3},
                "rng": {
                    "cpu": torch.get_rng_state(),
                    "numpy": None,
                    "random": None,
                    "cuda": None,
                },
            },
            actor / "extra_state_world_size_1_rank_0.pt",
        )
        self.progress = {
            "version": 1,
            "completed_updates": 3,
            "generated_batches": 7,
            "data_epoch": 0,
            "batches_per_epoch": sys.maxsize,
        }
        self.write(self.source / "dapo_progress.json", self.progress)
        torch.save(self.saved_state, self.source / "data.pt")
        previous = self.source / "migration_provenance"
        previous.mkdir()
        (previous / "earlier-data.bin").write_bytes(b"prior finite migration artifact")
        self.destination = self.root / "new-run"
        self.evidence = self.root / "acceptance.json"
        self.write(
            self.evidence,
            {
                "passed": True,
                "candidate_sha256": self.new_sources[MIGRATION.KERNEL],
                "scope": "Synthetic migration test; not a kernel acceptance claim",
            },
        )

    @staticmethod
    def write(path, value):
        path.write_bytes(streaming._json_bytes(value))

    @staticmethod
    def dataset(path):
        return streaming.CPTWorldStreamingDataset(
            data_files=[str(path)], tokenizer=None, config=CONFIG
        )

    def migrate(self):
        return MIGRATION.migrate(
            self.source, self.descriptor_path, self.old_project, self.destination, self.evidence
        )

    def test_rebind_replays_all_future_truth_then_uses_saved_generator_cursor(self):
        original = MIGRATION.files_under(self.root / "old-journal")
        checkpoint_files = NATIVE.source_inventory(self.source, 3)
        FixtureStream.calls = []
        with patch.object(
            streaming, "_make_stream", side_effect=AssertionError("No task generation")
        ):
            report = self.migrate()
        self.assertEqual(report["native_acceptance"]["next_global_index"], 7)
        self.assertEqual(report["first_new_kernel_index"], 9)
        self.assertEqual(report["saved_journal_prefix"], 7)
        new_state = torch.load(
            Path(report["destination_checkpoint"]) / "data.pt", weights_only=False
        )
        for key in self.saved_state:
            if key != "dataset_state":
                self.assertEqual(new_state[key], self.saved_state[key])
        self.assertEqual(new_state["dataset_state"]["next_index"], 7)
        descriptor = MIGRATION.read(report["destination_descriptor"])
        self.assertEqual(descriptor["stream_start_index"], 5)
        self.assertEqual(descriptor["start_seed"], 900)
        self.assertEqual(descriptor["source_migration"]["new_source_from_index"], 9)
        self.assertEqual(
            descriptor["source_migration"]["old_source_fingerprints"], self.old_sources
        )
        restored = NATIVE.native_loader(self.dataset(report["destination_descriptor"]))
        restored.load_state_dict(new_state)
        iterator = iter(restored)
        with patch.object(
            streaming, "_make_stream", side_effect=AssertionError("Must replay future row")
        ):
            for expected in self.tail:
                actual = next(iterator)
                self.assertEqual(actual["tools_kwargs"], expected["tools_kwargs"])
                self.assertEqual(actual["raw_prompt"], expected["raw_prompt"])
        FixtureStream.owner = "new"
        with (
            patch.object(streaming, "_make_stream", side_effect=FixtureStream),
            patch.object(streaming, "_convert_row", side_effect=fixture_convert),
        ):
            actual = next(iterator)
        self.assertEqual(actual["index"], 9)
        self.assertEqual(FixtureStream.calls, [("new", report["next_generator_state"])])
        self.assertEqual(report["next_generator_state"]["next_sample_index"], 905)
        self.assertEqual(report["next_generator_state"]["next_family_offset"], 4)
        self.assertEqual(report["next_generator_state"]["best_intervention_balance_slot"], 1)
        self.assertEqual(MIGRATION.files_under(self.root / "old-journal"), original)
        self.assertEqual(NATIVE.source_inventory(self.source, 3), checkpoint_files)
        self.assertEqual(
            MIGRATION.files_under(self.destination / "source-migration/old-journal"), original
        )
        self.assertFalse(torch.cuda.is_initialized())

    def test_reject_missing_future_row_before_writing_destination(self):
        (self.root / "old-journal/rows/00000000000000000008.json").unlink()
        with self.assertRaisesRegex(ValueError, "Partial journal"):
            self.migrate()
        self.assertFalse(self.destination.exists())

    def test_reject_head_lag_instead_of_discarding_published_future_row(self):
        path = self.root / "old-journal/head.json"
        head = MIGRATION.read(path)
        head["next_index"] -= 1
        self.write(path, head)
        with self.assertRaisesRegex(ValueError, "complete row outside head"):
            self.migrate()
        self.assertFalse(self.destination.exists())

    def test_reject_partial_temporary_artifact(self):
        (self.root / "old-journal/rows/.00000000000000000009.partial.tmp").write_bytes(b"{")
        with self.assertRaisesRegex(ValueError, "temporary or future row"):
            self.migrate()
        self.assertFalse(self.destination.exists())

    def test_reject_broken_hash_chain(self):
        path = self.root / "old-journal/rows/00000000000000000007.json"
        record = MIGRATION.read(path)
        record["previous_record_sha256"] = "0" * 64
        self.write(path, record)
        with self.assertRaisesRegex(ValueError, "chain/identity mismatch"):
            self.migrate()
        self.assertFalse(self.destination.exists())

    def test_reject_checkpoint_prefix_mismatch(self):
        state = copy.deepcopy(self.saved_state)
        state["dataset_state"]["last_record_sha256"] = "0" * 64
        torch.save(state, self.source / "data.pt")
        with self.assertRaisesRegex(ValueError, "Checkpoint journal prefix"):
            self.migrate()
        self.assertFalse(self.destination.exists())

    def test_reject_unbound_evidence_and_unrelated_source_change(self):
        self.write(self.evidence, {"passed": True, "candidate_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "exact new kernel SHA"):
            self.migrate()
        source = self.old_project / "src/cpt_world/trl_environment.py"
        source.write_bytes(source.read_bytes() + b"\n# unrelated change\n")
        with self.assertRaisesRegex(ValueError, "Old source tree"):
            self.migrate()
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
