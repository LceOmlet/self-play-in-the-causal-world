"""Continuous supply and recovery tests; no model generation or optimizer run."""

from __future__ import annotations

import copy
import importlib
import json
import pickle
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import chdir
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from verl.utils.dataset.rl_dataset import RLHFDataset

from cpt_world import verl_streaming_dataset as module
from cpt_world.identification import INTERACTION_SURFACE_VERSION


class FakeStream:
    def __init__(self, start_seed, calls):
        self.calls = calls
        self.state = {
            "version": 1,
            "next_sample_index": start_seed,
            "next_family_offset": 0,
            "best_intervention_balance_slot": 0,
            "accepted_rows": 0,
        }
        self.last_attempts = []

    def __next__(self):
        index = self.state["next_sample_index"]
        self.calls.append(index)
        self.last_attempts = [
            {"sample_index": index, "status": "rejected", "error_type": "FixtureRejection"},
            {"sample_index": index + 1, "status": "accepted"},
        ]
        self.state["next_sample_index"] += 2
        self.state["accepted_rows"] += 1
        self.state["next_family_offset"] = self.state["accepted_rows"] % 5
        return {
            "query_type": "ate",
            "tape_key": f"fixture:{index + 1}",
            "environment_version": INTERACTION_SURFACE_VERSION,
            "prompt": [{"role": "user", "content": str(index + 1)}],
        }

    def state_dict(self):
        return copy.deepcopy(self.state)

    def load_state_dict(self, state):
        self.state = copy.deepcopy(state)


def fake_convert(row, index):
    return {
        "data_source": "cpt_world/ate",
        "prompt": copy.deepcopy(row["prompt"]),
        "extra_info": {"index": index, "tape_key": row["tape_key"]},
    }


class StreamingDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.calls = []
        self.actual_sources = module.current_source_fingerprints()
        self.actual_make_stream = module._make_stream
        self.actual_convert_row = module._convert_row
        self.fingerprints = {"source.py": "a" * 64}
        self.sources = patch.object(
            module, "current_source_fingerprints", return_value=self.fingerprints
        ).start()
        self.addCleanup(patch.stopall)
        patch.object(
            module, "_make_stream", side_effect=lambda start: FakeStream(start, self.calls)
        ).start()
        patch.object(module, "_convert_row", side_effect=fake_convert).start()
        self.descriptor = {
            "version": 1,
            "kind": "cpt_world_continuous",
            "environment_version": INTERACTION_SURFACE_VERSION,
            "start_seed": 1000,
            "stream_start_index": 74,
            "journal_dir": "journal",
            "source_fingerprints": self.fingerprints,
        }
        self.path = self.root / "continuous.json"
        self.path.write_text(json.dumps(self.descriptor), encoding="utf-8")
        self.config = {
            "shuffle": False,
            "dataloader_num_workers": 0,
            "train_batch_size": 1,
            "gen_batch_size": 1,
            "filter_overlong_prompts": False,
        }

    def dataset(self, **kwargs):
        return module.CPTWorldStreamingDataset(
            data_files=str(self.path), tokenizer=None, config=self.config, **kwargs
        )

    def test_descriptor_validation_does_not_create_a_journal(self):
        descriptor = module.validate_stream_descriptor(self.path)
        self.assertEqual(descriptor["journal_dir"], str(self.root / "journal"))
        self.assertFalse((self.root / "journal").exists())
        for field, value in (("version", 2), ("environment_version", "old"), ("start_seed", -1)):
            invalid = dict(self.descriptor, **{field: value})
            self.path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaises(ValueError):
                module.validate_stream_descriptor(self.path)

    def test_sequential_fresh_rows_and_repeated_official_getitem(self):
        dataset = self.dataset()
        self.assertEqual(len(dataset), sys.maxsize)
        self.assertIs(module.CPTWorldStreamingDataset.__getitem__, RLHFDataset.__getitem__)
        first = dataset[74]
        first["raw_prompt"][0]["content"] = "caller mutation"
        self.assertEqual(dataset[74]["raw_prompt"][0]["content"], "1001")
        self.assertEqual(dataset[75]["extra_info"]["tape_key"], "fixture:1003")
        self.assertEqual(self.calls, [1000, 1002])
        record = json.loads((self.root / "journal/rows/00000000000000000075.json").read_bytes())
        self.assertEqual(record["next_generator_state"]["accepted_rows"], 2)
        self.assertEqual([r["status"] for r in record["last_attempts"]], ["rejected", "accepted"])
        self.assertIn("dummy_tensor", dataset[74])

    def test_real_generator_and_converter_round_trip_complete_tool_row(self):
        from cpt_world.registry import TASK_FAMILY_QUERY_TYPES

        started = time.perf_counter()
        rows = []
        with (
            patch.object(module, "_make_stream", self.actual_make_stream),
            patch.object(module, "_convert_row", self.actual_convert_row),
        ):
            dataset = self.dataset()
            for index in range(74, 79):
                row_started = time.perf_counter()
                official_row = dataset[index]
                record_path = self.root / f"journal/rows/{index:020d}.json"
                record = json.loads(record_path.read_bytes())
                tool_row = json.loads(
                    official_row["tools_kwargs"]["act"]["create_kwargs"]["row_json"]
                )
                self.assertEqual(tool_row, record["row"])
                self.assertEqual(official_row["extra_info"]["tape_key"], record["row"]["tape_key"])
                self.assertEqual(record["next_generator_state"]["accepted_rows"], index - 73)
                self.assertEqual(record["last_attempts"][-1]["status"], "accepted")
                rows.append((official_row, time.perf_counter() - row_started))
        self.assertEqual(
            [row["extra_info"]["query_type"] for row, _ in rows], list(TASK_FAMILY_QUERY_TYPES)
        )
        with patch.object(module, "_make_stream", side_effect=AssertionError("must replay")):
            restored = self.dataset()
            for index, (row, _) in enumerate(rows, 74):
                self.assertEqual(restored[index]["tools_kwargs"], row["tools_kwargs"])
                self.assertEqual(restored[index]["raw_prompt"], row["raw_prompt"])
        print(
            "REAL_STREAM_ROUND_TRIP="
            + json.dumps(
                {
                    "start_seed": self.descriptor["start_seed"],
                    "rows": [
                        {"query_type": row["extra_info"]["query_type"], "seconds": seconds}
                        for row, seconds in rows
                    ],
                    "total_seconds": time.perf_counter() - started,
                    "all_committed_rows_replayed_without_generation": True,
                }
            ),
            flush=True,
        )

    def test_out_of_order_or_premigration_indices_do_not_generate(self):
        dataset = self.dataset()
        with self.assertRaises(IndexError):
            dataset[73]
        with self.assertRaises(ValueError):
            dataset[75]
        self.assertEqual(self.calls, [])

    def test_two_instances_concurrently_generate_the_same_index_once(self):
        left, right = self.dataset(), self.dataset()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(dataset.__getitem__, 74) for dataset in (left, right)]
            results = [future.result() for future in futures]
        self.assertEqual(self.calls, [1000])
        self.assertEqual(results[0]["extra_info"], results[1]["extra_info"])
        self.assertEqual(left.state_dict(), right.state_dict())

    def test_checkpoint_replay_preserves_rows_generated_after_checkpoint(self):
        before = self.dataset()
        before[74]
        saved = before.state_dict()
        later = before[75]
        restored = self.dataset()
        restored.load_state_dict(saved)
        self.assertEqual(restored[75]["extra_info"], later["extra_info"])
        self.assertEqual(self.calls, [1000, 1002])
        restored[76]
        self.assertEqual(self.calls, [1000, 1002, 1004])

    def test_pickle_reopens_journal_without_regenerating_a_row(self):
        dataset = self.dataset()
        dataset[74]
        restored_rows = pickle.loads(pickle.dumps(dataset.dataframe))
        self.assertEqual(restored_rows.state_dict(), dataset.state_dict())
        self.assertEqual(restored_rows[74]["extra_info"]["tape_key"], "fixture:1001")
        self.assertEqual(self.calls, [1000])
        restored_dataset = pickle.loads(pickle.dumps(dataset))
        self.assertFalse(hasattr(restored_dataset, "dataframe"))
        restored_dataset.resume_dataset_state()
        self.assertEqual(restored_dataset.state_dict(), dataset.state_dict())
        self.assertEqual(restored_dataset[74]["extra_info"]["tape_key"], "fixture:1001")
        self.assertEqual(self.calls, [1000])
        dataset.serialize_dataset = True
        self.assertEqual(pickle.loads(pickle.dumps(dataset)).state_dict(), dataset.state_dict())

    def test_official_factory_loads_custom_file_and_uses_official_getitem(self):
        from omegaconf import OmegaConf
        from verl.trainer.ppo.utils import create_rl_dataset

        self.path.write_text(
            json.dumps(dict(self.descriptor, source_fingerprints=self.actual_sources)),
            encoding="utf-8",
        )
        config = OmegaConf.create(
            dict(
                self.config,
                custom_cls={"path": str(Path(module.__file__)), "name": "CPTWorldStreamingDataset"},
            )
        )
        dataset = create_rl_dataset([str(self.path)], config, None, None, is_train=True)
        self.assertIs(dataset.__class__.__getitem__, RLHFDataset.__getitem__)
        self.assertNotEqual(dataset.__class__.__module__, module.__name__)
        namespace = dataset.dataframe.__class__.__getitem__.__globals__
        with patch.dict(
            namespace,
            {
                "_make_stream": lambda start: FakeStream(start, self.calls),
                "_convert_row": fake_convert,
            },
        ):
            self.assertEqual(dataset[74]["extra_info"]["tape_key"], "fixture:1001")
            self.assertEqual(self.calls, [1000])

    def test_official_file_factory_real_families_with_external_scripts_and_cwd(self):
        from omegaconf import OmegaConf
        from verl.trainer.ppo.utils import create_rl_dataset

        from cpt_world.registry import TASK_FAMILY_QUERY_TYPES

        self.path.write_text(
            json.dumps(dict(self.descriptor, source_fingerprints=self.actual_sources)),
            encoding="utf-8",
        )
        external_scripts = ModuleType("scripts")
        external_scripts.__path__ = [str(self.root / "external-package/scripts")]
        config = OmegaConf.create(
            dict(
                self.config,
                custom_cls={"path": str(Path(module.__file__)), "name": "CPTWorldStreamingDataset"},
            )
        )
        started = time.perf_counter()
        with patch.dict(sys.modules, {"scripts": external_scripts}), chdir(self.root):
            sys.modules.pop("scripts.prepare_verl_cpt_data", None)
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module("scripts.prepare_verl_cpt_data")
            dataset = create_rl_dataset([str(self.path)], config, None, None, is_train=True)
            self.assertIs(dataset.__class__.__getitem__, RLHFDataset.__getitem__)
            rows = [dataset[index] for index in range(74, 79)]
            self.assertEqual(
                [row["extra_info"]["query_type"] for row in rows], list(TASK_FAMILY_QUERY_TYPES)
            )
            namespace = dataset.dataframe.__class__.__getitem__.__globals__
            converter = namespace["_row_converter"]()
            self.assertEqual(
                Path(converter.__code__.co_filename),
                Path(module.__file__).resolve().parents[2] / "scripts/prepare_verl_cpt_data.py",
            )
            self.assertIs(namespace["_row_converter"](), converter)
            restored = create_rl_dataset([str(self.path)], config, None, None, is_train=True)
            restored.load_state_dict(dataset.state_dict())
            replay_namespace = restored.dataframe.__class__.__getitem__.__globals__
            with patch.dict(
                replay_namespace,
                {"_make_stream": lambda _: self.fail("committed rows must replay")},
            ):
                for index, expected in enumerate(rows, 74):
                    self.assertEqual(restored[index]["tools_kwargs"], expected["tools_kwargs"])
                    self.assertEqual(restored[index]["raw_prompt"], expected["raw_prompt"])
            self.assertIs(sys.modules["scripts"], external_scripts)
            self.assertNotIn("scripts.prepare_verl_cpt_data", sys.modules)
        print(
            "EXTERNAL_SCRIPTS_REAL_STREAM="
            + json.dumps(
                {
                    "query_types": [row["extra_info"]["query_type"] for row in rows],
                    "total_seconds": time.perf_counter() - started,
                    "converter_loaded_from_fingerprinted_absolute_path": True,
                    "all_committed_rows_replayed_without_generation": True,
                }
            ),
            flush=True,
        )

    def test_committed_row_survives_a_crash_before_head_publication(self):
        dataset = self.dataset()
        atomic = module._atomic_json

        def fail_head(path, value):
            if path.name == "head.json":
                raise OSError("fixture interrupted head publication")
            return atomic(path, value)

        with patch.object(module, "_atomic_json", side_effect=fail_head):
            with self.assertRaises(OSError):
                dataset[74]
        restored = self.dataset()
        self.assertEqual(restored[74]["extra_info"]["tape_key"], "fixture:1001")
        self.assertEqual(self.calls, [1000])
        self.assertEqual(restored.state_dict()["next_index"], 75)

    def test_incomplete_temporary_row_is_not_a_committed_task(self):
        dataset = self.dataset()
        incomplete = self.root / "journal/rows/.00000000000000000074.json.crashed.tmp"
        incomplete.write_bytes(b'{"row":')
        self.assertEqual(dataset[74]["extra_info"]["tape_key"], "fixture:1001")
        self.assertEqual(self.calls, [1000])

    def test_missing_or_modified_checkpoint_prefix_is_rejected(self):
        dataset = self.dataset()
        dataset[74]
        saved = dataset.state_dict()
        changed = dict(saved, last_record_sha256="b" * 64)
        with self.assertRaises(ValueError):
            dataset.load_state_dict(changed)
        (self.root / "journal/rows/00000000000000000074.json").unlink()
        with self.assertRaises(ValueError):
            dataset.load_state_dict(saved)

    def test_recreated_journal_cannot_impersonate_a_checkpoint_dependency(self):
        dataset = self.dataset()
        saved = dataset.state_dict()
        (self.root / "journal").rename(self.root / "old-journal")
        recreated = self.dataset()
        with self.assertRaises(ValueError):
            recreated.load_state_dict(saved)

    def test_source_changes_are_rejected_before_new_generation(self):
        dataset = self.dataset()
        self.sources.return_value = {"source.py": "b" * 64}
        with self.assertRaises(ValueError):
            dataset[74]
        with self.assertRaises(ValueError):
            self.dataset()
        self.assertEqual(self.calls, [])

    def test_stream_incompatible_data_options_are_rejected(self):
        for key, value in (
            ("shuffle", True),
            ("dataloader_num_workers", 1),
            ("train_batch_size", 2),
            ("gen_batch_size", 2),
            ("filter_overlong_prompts", True),
        ):
            with self.subTest(key=key), patch.dict(self.config, {key: value}):
                with self.assertRaises(ValueError):
                    self.dataset()
        with self.assertRaises(ValueError):
            self.dataset(max_samples=10)

    def test_validation_files_follow_original_download_read_and_getitem(self):
        path = self.root / "validation.parquet"
        path.write_bytes(b"fixture parent loader")

        def parent_read(dataset):
            dataset.dataframe = [fake_convert({"prompt": [], "tape_key": "validation"}, 0)]

        with (
            patch.object(RLHFDataset, "_download") as download,
            patch.object(RLHFDataset, "_read_files_and_tokenize", parent_read),
        ):
            dataset = module.CPTWorldStreamingDataset(
                data_files=str(path), tokenizer=None, config={"shuffle": True}
            )
            download.assert_called_once_with(use_origin_parquet=False)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["extra_info"]["tape_key"], "validation")
            self.assertIsNone(dataset.state_dict())
            dataset.load_state_dict(None)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
