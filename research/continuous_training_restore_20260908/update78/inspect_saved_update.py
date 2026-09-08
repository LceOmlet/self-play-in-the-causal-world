"""Read-only CPU inspection of already saved update 78 and committed task rows."""

import hashlib
import json
import math
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def json_digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    return hashlib.sha256(raw.encode()).hexdigest()


def main():
    torch.set_num_threads(1)
    base = Path("/home/chen/runs/continuous-training-restore-20260908")
    run = base / "resume-stream-02"
    source = base / "hold-01/global_step_77"
    target = run / "checkpoints/global_step_78"
    descriptor = read(run / "stream.json")
    progress = read(target / "dapo_progress.json")
    state = torch.load(target / "data.pt", map_location="cpu", weights_only=False)
    assert progress["completed_updates"] == 78 and progress["generated_batches"] == 87
    assert progress["data_epoch"] == 0
    assert state["_num_yielded"] == state["_sampler_iter_yielded"] == 87
    assert state["_sampler_iter_state"] == {"samples_yielded": 87}
    assert not state["_iterator_finished"]
    journal = run / "task-journal"
    previous_path = journal / "rows/00000000000000000086.json"
    cf_path = journal / "rows/00000000000000000087.json"
    previous, cf, head = read(previous_path), read(cf_path), read(journal / "head.json")
    saved_journal = state["dataset_state"]
    assert saved_journal["next_index"] == 87
    assert saved_journal["last_record_sha256"] == sha(previous_path)
    assert saved_journal["journal_id"] == previous["journal_id"] == cf["journal_id"]
    assert saved_journal["descriptor_sha256"] == json_digest(descriptor)
    assert cf["previous_record_sha256"] == sha(previous_path)
    assert head["next_index"] >= 88
    assert cf["row_sha256"] == json_digest(cf["row"])
    assert cf["converted_row_sha256"] == json_digest(cf["converted_row"])
    forwarded = json.loads(
        cf["converted_row"]["extra_info"]["tools_kwargs"]["act"]["create_kwargs"]["row_json"]
    )
    assert forwarded == cf["row"]
    assert cf["row"]["query_type"] == "individual_counterfactual_probability"
    truth = json.loads(cf["row"]["terminal_truth_json"])
    assert truth["certification"] == "exact" and truth["interval_source"] == "exact_markovian"
    assert 0 <= truth["lower"] <= truth["upper"] <= 1
    assert cf["next_generator_state"]["accepted_rows"] == 2
    models = [path / "actor/model_world_size_1_rank_0.pt" for path in (source, target)]
    before, after = [torch.load(path, map_location="cpu", weights_only=False) for path in models]
    assert before.keys() == after.keys() and len(before) == 496
    groups = {
        part: {"tensors": 0, "changed_tensors": 0, "elements": 0,
               "changed_elements": 0, "max_abs_change": 0.0, "squared_change": 0.0}
        for part in ("lora_A", "lora_B")
    }
    for name in before:
        a, b = [value.to_local() if isinstance(value, DTensor) else value
                for value in (before[name], after[name])]
        assert a.device.type == b.device.type == "cpu" and a.shape == b.shape
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        part = "lora_A" if "lora_A" in name else "lora_B"
        assert part in name
        item = groups[part]
        delta = b.float() - a.float()
        changed = int(torch.count_nonzero(delta))
        item["tensors"] += 1
        item["changed_tensors"] += int(changed > 0)
        item["elements"] += a.numel()
        item["changed_elements"] += changed
        item["max_abs_change"] = max(item["max_abs_change"], float(delta.abs().max()))
        item["squared_change"] += float(delta.double().square().sum())
    assert sum(item["changed_elements"] for item in groups.values()) > 0
    rollouts = [json.loads(line) for line in (run / "rollouts/78.jsonl").read_text().splitlines()]
    assert len(rollouts) == 4 and all(row["step"] == 78 for row in rollouts)
    assert len({row["input"] for row in rollouts}) == 1
    assert not torch.cuda.is_initialized()
    artifacts = {"data.pt": target / "data.pt", "dapo_progress.json": target / "dapo_progress.json",
                 "78.jsonl": run / "rollouts/78.jsonl", cf_path.name: cf_path}
    result = {
        "passed": True,
        "scope": "Saved-state inspection only: CPU LoRA comparison, sampler/journal consistency "
                 "and already logged rollouts. No new task, inference or optimizer step executed; "
                 "this establishes a real parameter update, not improved learned capability.",
        "source_checkpoint": str(source), "target_checkpoint": str(target),
        "model_files_sha256": {str(path): sha(path) for path in models},
        "parameter_differences": groups,
        "parameter_difference_l2": math.sqrt(sum(row["squared_change"] for row in groups.values())),
        "all_496_lora_tensors_finite": True,
        "progress": progress, "native_data_state": state,
        "journal_head_observed": head,
        "journal_consistency": "Saved next index87 follows committed row86; row87 links to "
                               "that same prefix. A journal ahead of the checkpoint is supported.",
        "completed_update_task": {key: previous["row"][key]
                                  for key in ("sample_index", "query_type", "tape_key")},
        "next_cf_task": {"logical_index": 87, "sample_index": cf["row"]["sample_index"],
                         "truth": truth, "truth_present_in_journal_and_forwarded_tool_row": True,
                         "cached_record_sha256": sha(cf_path)},
        "rollouts": {"count": 4, "same_input": True,
                     "scores": [row["score"] for row in rollouts],
                     "completed": [row["completed"] for row in rollouts],
                     "overlong": [row["overlong"] for row in rollouts]},
        "artifact_sha256": {name: sha(path) for name, path in artifacts.items()},
        "inspector_sha256": sha(Path(__file__)), "torch_version": torch.__version__,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
