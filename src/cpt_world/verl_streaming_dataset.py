"""Journal the existing fresh-task generator behind verl's map-style data API.

The index space is not a materialized or repeating task collection. Each new
sequential index asks the existing generator for a fresh row. Completed rows
are atomically journaled before delivery, so repeated access and recovery do
not regenerate an accepted counterfactual. verl still owns row processing,
sampling, rollout, reward, filtering, and all training updates.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
import threading
from contextlib import contextmanager, suppress
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from verl.utils.dataset.rl_dataset import RLHFDataset

from cpt_world.identification import INTERACTION_SURFACE_VERSION

_KIND = "cpt_world_continuous"
_PROJECT = Path(__file__).resolve().parents[2]


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def current_source_fingerprints():
    """Fingerprint the environment/generator and the reused verl row converter."""
    paths = sorted((_PROJECT / "src/cpt_world").rglob("*.py"))
    paths.append(_PROJECT / "scripts/prepare_verl_cpt_data.py")
    return {path.relative_to(_PROJECT).as_posix(): _digest(path.read_bytes()) for path in paths}


def validate_stream_descriptor(path):
    """Validate a local descriptor without creating or modifying its journal."""
    path = Path(path).resolve()
    descriptor = json.loads(path.read_bytes())
    if not isinstance(descriptor, dict) or descriptor.get("kind") != _KIND:
        raise ValueError("Not a CPT-World continuous-task descriptor")
    if type(descriptor.get("version")) is not int or descriptor["version"] != 1:
        raise ValueError("Unsupported continuous-task descriptor version")
    if descriptor.get("environment_version") != INTERACTION_SURFACE_VERSION:
        raise ValueError("Continuous-task descriptor has an obsolete environment contract")
    for key in ("start_seed", "stream_start_index"):
        if type(descriptor.get(key)) is not int or descriptor[key] < 0:
            raise ValueError(f"Descriptor {key} must be a nonnegative integer")
    if descriptor["stream_start_index"] >= sys.maxsize:
        raise ValueError("Continuous-task start index is outside Python's index space")
    journal = descriptor.get("journal_dir")
    if not isinstance(journal, str) or not journal:
        raise ValueError("Descriptor journal_dir must be a nonempty path")
    journal = Path(journal).expanduser()
    if not journal.is_absolute():
        journal = path.parent / journal
    descriptor["journal_dir"] = str(journal.resolve())
    if descriptor.get("source_fingerprints") != current_source_fingerprints():
        raise ValueError("Continuous-task source fingerprints changed")
    return descriptor


def _make_stream(start_seed):
    from cpt_world.trl_environment import BalancedTrainingRowStream

    return BalancedTrainingRowStream(start_seed=start_seed)


@lru_cache(maxsize=1)
def _row_converter():
    # The official TaskRunner runs from verl, where another installed `scripts`
    # package can shadow this project's directory. Load the fingerprinted file
    # itself without changing sys.path or depending on that generic namespace.
    path = _PROJECT / "scripts/prepare_verl_cpt_data.py"
    name = "_cpt_world_row_converter_" + _digest(str(path).encode("utf-8"))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the CPT-World row converter from {path}")
    converter_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter_module)
    return converter_module.convert_row


def _convert_row(row, index):
    return _row_converter()(row, index)


def _atomic_json(path, value):
    """Publish one complete JSON file; incomplete temporary files are never read."""
    raw = _json_bytes(value)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return _digest(raw)


class JournaledTaskRows:
    """A synchronous, recoverable row store with one strictly increasing frontier."""

    def __init__(self, descriptor):
        self.descriptor = copy.deepcopy(descriptor)
        self.descriptor_sha256 = _digest(_json_bytes(descriptor))
        self.directory = Path(descriptor["journal_dir"])
        self.rows_directory = self.directory / "rows"
        self.rows_directory.mkdir(parents=True, exist_ok=True)
        self.start_index = descriptor["stream_start_index"]
        self._thread_lock = threading.RLock()
        with self._locked():
            metadata_path = self.directory / "journal.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_bytes())
                if (
                    metadata.get("version") != 1
                    or metadata.get("descriptor_sha256") != self.descriptor_sha256
                    or metadata.get("descriptor") != self.descriptor
                    or not isinstance(metadata.get("journal_id"), str)
                ):
                    raise ValueError("Continuous-task journal belongs to another descriptor")
            else:
                if (
                    any(self.rows_directory.glob("*.json"))
                    or (self.directory / "head.json").exists()
                ):
                    raise ValueError("Continuous-task journal metadata is missing")
                metadata = {
                    "version": 1,
                    "journal_id": uuid4().hex,
                    "descriptor_sha256": self.descriptor_sha256,
                    "descriptor": self.descriptor,
                }
                _atomic_json(metadata_path, metadata)
            self.journal_id = metadata["journal_id"]
            self._head_locked()

    def __len__(self):
        return sys.maxsize

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_thread_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._thread_lock = threading.RLock()
        # Pickling transports a reference to the committed journal, never its
        # contents or a live generator. Reopening cannot recreate a lost journal.
        with self._locked():
            self._verify_sources()
            metadata = json.loads((self.directory / "journal.json").read_bytes())
            if (
                metadata.get("journal_id") != self.journal_id
                or metadata.get("descriptor_sha256") != self.descriptor_sha256
                or metadata.get("descriptor") != self.descriptor
            ):
                raise ValueError("Serialized dataset requires its original continuous journal")
            self._head_locked()

    @contextmanager
    def _locked(self):
        with self._thread_lock, (self.directory / ".lock").open("a+b") as stream:
            if os.name == "nt":
                import msvcrt

                if stream.tell() == 0:
                    stream.write(b"\0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _verify_sources(self):
        if self.descriptor["source_fingerprints"] != current_source_fingerprints():
            raise ValueError("Continuous-task source fingerprints changed")

    def _record_path(self, index):
        return self.rows_directory / f"{index:020d}.json"

    def _read_record(self, index):
        path = self._record_path(index)
        if not path.exists():
            raise ValueError(f"Continuous-task journal row {index} is missing")
        raw = path.read_bytes()
        record = json.loads(raw)
        if (
            record.get("version") != 1
            or record.get("index") != index
            or record.get("journal_id") != self.journal_id
            or record.get("descriptor_sha256") != self.descriptor_sha256
            or not isinstance(record.get("row"), dict)
            or not isinstance(record.get("converted_row"), dict)
            or record.get("row_sha256") != _digest(_json_bytes(record["row"]))
            or record.get("converted_row_sha256") != _digest(_json_bytes(record["converted_row"]))
            or not isinstance(record.get("next_generator_state"), dict)
            or record.get("next_generator_state", {}).get("accepted_rows")
            != index - self.start_index + 1
        ):
            raise ValueError(f"Continuous-task journal row {index} has invalid identity or state")
        return record, _digest(raw)

    def _head_locked(self):
        path = self.directory / "head.json"
        if path.exists():
            head = json.loads(path.read_bytes())
        else:
            head = {"version": 1, "next_index": self.start_index, "last_record_sha256": None}
        index = head.get("next_index")
        if head.get("version") != 1 or type(index) is not int or index < self.start_index:
            raise ValueError("Invalid continuous-task journal frontier")
        if index == self.start_index:
            if head.get("last_record_sha256") is not None:
                raise ValueError("An empty continuous-task journal has a predecessor")
        else:
            _, previous_hash = self._read_record(index - 1)
            if head.get("last_record_sha256") != previous_hash:
                raise ValueError("Continuous-task journal frontier disagrees with its last row")
        # A crash can publish a complete row before publishing the head. Recover
        # that committed row, including its original CF result, instead of rerunning it.
        advanced = False
        while self._record_path(head["next_index"]).exists():
            record, record_hash = self._read_record(head["next_index"])
            if record.get("previous_record_sha256") != head["last_record_sha256"]:
                raise ValueError("Continuous-task journal chain is inconsistent")
            head = {
                "version": 1,
                "next_index": head["next_index"] + 1,
                "last_record_sha256": record_hash,
            }
            advanced = True
        if advanced or not path.exists():
            _atomic_json(path, head)
        return head

    def __getitem__(self, index):
        if type(index) is not int or not self.start_index <= index < sys.maxsize:
            raise IndexError("Continuous-task index precedes migration or is invalid")
        with self._locked():
            self._verify_sources()
            head = self._head_locked()
            if index > head["next_index"]:
                raise ValueError("Continuous tasks must be requested in sequential index order")
            if index < head["next_index"]:
                record, _ = self._read_record(index)
                return copy.deepcopy(record["converted_row"])
            stream = _make_stream(self.descriptor["start_seed"])
            if index > self.start_index:
                previous, _ = self._read_record(index - 1)
                stream.load_state_dict(previous["next_generator_state"])
            row = next(stream)
            next_state = stream.state_dict()
            if next_state.get("accepted_rows") != index - self.start_index + 1:
                raise ValueError("Continuous generator cursor disagrees with the journal index")
            converted = _convert_row(row, index)
            record = {
                "version": 1,
                "index": index,
                "journal_id": self.journal_id,
                "descriptor_sha256": self.descriptor_sha256,
                "previous_record_sha256": head["last_record_sha256"],
                "row": row,
                "row_sha256": _digest(_json_bytes(row)),
                "converted_row": converted,
                "converted_row_sha256": _digest(_json_bytes(converted)),
                "next_generator_state": next_state,
                "last_attempts": stream.last_attempts,
            }
            self._verify_sources()
            record_hash = _atomic_json(self._record_path(index), record)
            _atomic_json(
                self.directory / "head.json",
                {"version": 1, "next_index": index + 1, "last_record_sha256": record_hash},
            )
            return copy.deepcopy(converted)

    def state_dict(self):
        with self._locked():
            self._verify_sources()
            head = self._head_locked()
            return {
                "version": 1,
                "kind": "cpt_world_continuous_journal",
                "journal_id": self.journal_id,
                "descriptor_sha256": self.descriptor_sha256,
                "next_index": head["next_index"],
                "last_record_sha256": head["last_record_sha256"],
            }

    def load_state_dict(self, state):
        with self._locked():
            self._verify_sources()
            head = self._head_locked()
            if (
                not isinstance(state, dict)
                or state.get("version") != 1
                or state.get("kind") != "cpt_world_continuous_journal"
                or state.get("journal_id") != self.journal_id
                or state.get("descriptor_sha256") != self.descriptor_sha256
                or type(state.get("next_index")) is not int
                or not self.start_index <= state["next_index"] <= head["next_index"]
            ):
                raise ValueError("Checkpoint requires a missing or incompatible continuous journal")
            if state["next_index"] == self.start_index:
                expected_hash = None
            else:
                _, expected_hash = self._read_record(state["next_index"] - 1)
            if state.get("last_record_sha256") != expected_hash:
                raise ValueError("Checkpoint's committed continuous-task journal prefix changed")
            # The StatefulDataLoader sampler restores the consumed index. Rows
            # journaled after that checkpoint remain available for stable replay.


class CPTWorldStreamingDataset(RLHFDataset):
    """Use official row processing for a continuous descriptor or ordinary files."""

    def _download(self, use_origin_parquet=False):
        files = self.original_data_files if use_origin_parquet else self.data_files
        candidates = []
        for filename in files:
            path = Path(filename)
            if path.is_file() and path.suffix == ".json":
                value = json.loads(path.read_bytes())
                if isinstance(value, dict) and value.get("kind") == _KIND:
                    candidates.append(path)
        if candidates:
            if len(files) != 1 or len(candidates) != 1:
                raise ValueError("Use one continuous descriptor without additional data files")
            self._continuous_descriptor = validate_stream_descriptor(candidates[0])
            return
        self._continuous_descriptor = None
        return super()._download(use_origin_parquet=use_origin_parquet)

    def _read_files_and_tokenize(self):
        if self._continuous_descriptor is None:
            return super()._read_files_and_tokenize()
        if (
            self.config.get("shuffle") is not False
            or self.config.get("dataloader_num_workers") != 0
        ):
            raise ValueError("Continuous tasks require shuffle=false and dataloader_num_workers=0")
        if (
            self.config.get("train_batch_size") != 1
            or (self.config.get("gen_batch_size") or self.config.get("train_batch_size")) != 1
            or self.max_samples > 0
            or self.filter_overlong_prompts
        ):
            raise ValueError(
                "Continuous tasks require train/gen batch size 1, no max_samples cap, "
                "and filter_overlong_prompts=false"
            )
        self.dataframe = JournaledTaskRows(self._continuous_descriptor)

    def state_dict(self):
        if self._continuous_descriptor is None:
            return None
        return self.dataframe.state_dict()

    def load_state_dict(self, state):
        if self._continuous_descriptor is None:
            if state is not None:
                raise ValueError("A continuous-task state cannot restore an ordinary file dataset")
            return
        self.dataframe.load_state_dict(state)
