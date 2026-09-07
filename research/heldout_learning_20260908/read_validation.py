"""Join frozen official validation outputs to public tasks and owner events.

Read-only evidence processing. A partial event snapshot is never reported as a
complete validation pass, and unfinished output errors are not filled with zero.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq

    dataset_sha = digest(args.dataset.read_bytes())
    assert dataset_sha == "4508652e399973610611368545a92b1c0a2be8faa4fe1f3ec547fac8e3183900"
    tasks = pq.read_table(args.dataset).to_pylist()
    assert len(tasks) == 25
    assert len({t["prompt"][-1]["content"] for t in tasks}) == 25
    assert len({t["extra_info"]["tape_key"] for t in tasks}) == 25
    reader = module(
        args.project / "research/training_submission_20260907/capture_progress.py",
        "captured_training_reader",
    )
    owner = module(
        args.project / "research/base_signal_diagnostic_20260907/analyze_events.py",
        "owner_event_reader",
    )
    events, fingerprints = owner.read_events(args.snapshot)
    task_tapes = {t["extra_info"]["tape_key"] for t in tasks}
    traces = defaultdict(list)
    for event in events:
        if event["event"] == "act" and event["tape_key"] in task_tapes:
            traces[event["trajectory_id"]].append(event)
    ended = defaultdict(list)
    for event in events:
        if event["event"] == "raw_score" and event.get("tape_key") in task_tapes:
            ended[event["tape_key"]].append(event)
    # This frozen pass has one rollout per original validation task. The owner
    # score is emitted after generation ends, even without a terminal answer.
    assert all(len(values) == 1 for values in ended.values())
    summaries = {key: owner.summarize_trajectory(trace) for key, trace in traces.items()}
    report = {
        "dataset_sha256": dataset_sha,
        "step": args.step,
        "reader_sha256": digest(Path(__file__).read_bytes()),
        "snapshot": str(args.snapshot),
        "event_prefixes": fingerprints,
        "observed_validation_requests": len(traces),
        "observed_completed_answers": sum(t["completed"] for t in summaries.values()),
        "ended_owner_scores_with_task_identity": len(ended),
        "ended_without_terminal_answer_with_task_identity": sum(
            not values[0]["completed"] for values in ended.values()
        ),
        "ended_score_scope": (
            "Known original validation tape only; no request ID inferred from score. "
            "No-action scores without tape metadata await the complete official dump. "
            "These counts are not a complete-pass failure rate."
        ),
        "complete_official_pass": False,
        "scope": "Fixed 25 tasks, one greedy trajectory per task. A complete official pass "
        "requires 25 saved outputs and finished official validation metrics. "
        "This is not a before/after learning-gain estimate.",
    }
    official = {}
    for name in ["train.log", "task-runner-stdout.log"]:
        path = args.snapshot / name
        if path.exists():
            for row in reader.metrics_from_log(path.read_bytes()):
                if row["step"] == args.step:
                    for key, value in row.items():
                        assert key not in official or official[key] == value
                        official[key] = value
    dump = args.snapshot / "validation" / f"{args.step}.jsonl"
    rows = [json.loads(line) for line in dump.read_bytes().splitlines()] if dump.exists() else []
    report["official_output_rows"] = len(rows)
    report["official_validation_metrics_logged"] = any(k.startswith("val-") for k in official)
    assert len(rows) <= 25
    if len(rows) == 25 and report["official_validation_metrics_logged"]:
        assert all(math.isfinite(value) for value in official.values())
        records = []
        for output in rows:
            assert output["step"] == args.step
            candidates = [t for t in tasks if t["prompt"][-1]["content"] in output["input"]]
            assert len(candidates) == 1, "Public input must identify exactly one original task"
            task = candidates[0]
            extra = task["extra_info"]
            tape = extra["tape_key"]
            commands = reader.commands_from_output(output["output"])
            possible = {key: trace for key, trace in traces.items() if trace[0]["tape_key"] == tape}
            matching = [
                key
                for key, trace in possible.items()
                if commands == [event["command"] for event in trace]
            ]
            raw, penalty = float(output["raw_reward"]), float(output["overlong_reward"])
            assert math.isfinite(raw) and math.isfinite(penalty) and penalty <= 0
            assert output["score"] == output["acc"] == raw
            if commands:
                assert len(matching) == 1, "Full executed command sequence must match uniquely"
                request = matching[0]
                trajectory = summaries[request]
                assert trajectory["completed"] == bool(output["completed"])
                assert raw == (trajectory["quality"] if trajectory["completed"] else 0.0)
            else:
                assert not possible and not output["completed"] and raw == 0
                request, trajectory = None, None
            records.append(
                {
                    "index": extra["index"],
                    "tape_key": tape,
                    "query_type": extra["query_type"],
                    "task_join": "unique_full_public_user_message",
                    "public_user_message_sha256": digest(task["prompt"][-1]["content"].encode()),
                    "input_sha256": digest(output["input"].encode()),
                    "output_sha256": digest(output["output"].encode()),
                    "request_id": request,
                    "event_join": "unique_full_executed_commands"
                    if request
                    else "no_act_no_events",
                    "raw_quality": raw,
                    "official_length_penalty": penalty,
                    "shaped_reward": raw + penalty,
                    "completed": bool(output["completed"]),
                    "trajectory": trajectory,
                }
            )
        assert len({r["tape_key"] for r in records}) == 25
        assert {r["tape_key"] for r in records} == task_tapes
        groups = defaultdict(list)
        for record in records:
            groups[record["query_type"]].append(record)
        families = {}
        strict = {
            "backadj_minimal_sets": ["valid_adjustment_set"],
            "best_intervention": ["optimal_action"],
            "mediator_set": ["mediators_exact_match", "order_exact_match"],
        }
        for family, group in groups.items():
            assert len(group) == 5
            metrics = defaultdict(list)
            for record in group:
                trajectory = record["trajectory"]
                for key, value in ((trajectory or {}).get("task_metrics") or {}).items():
                    metrics[key].append(value)
            summary = {
                "tasks": len(group),
                "completed_answers": sum(r["completed"] for r in group),
                "mean_raw_quality": sum(r["raw_quality"] for r in group) / 5,
                "mean_shaped_reward": sum(r["shaped_reward"] for r in group) / 5,
                "task_metrics": {
                    k: {"observed": len(v), "mean": sum(v) / len(v), "values": v}
                    for k, v in metrics.items()
                },
            }
            if family in strict:
                summary["strict_success_count"] = sum(
                    bool(
                        r["completed"]
                        and all(r["trajectory"]["task_metrics"][k] for k in strict[family])
                    )
                    for r in group
                )
            raw_keys = [k for k in official if k.endswith(f"cpt_world/{family}/acc/mean@1")]
            reward_keys = [k for k in official if k.endswith(f"cpt_world/{family}/reward/mean@1")]
            assert len(raw_keys) == len(reward_keys) == 1
            assert abs(official[raw_keys[0]] - summary["mean_raw_quality"]) < 1e-6
            assert abs(official[reward_keys[0]] - summary["mean_shaped_reward"]) < 1e-6
            families[family] = summary
        report.update(
            complete_official_pass=True,
            outputs_sha256=digest(dump.read_bytes()),
            official_metrics=official,
            records=sorted(records, key=lambda r: r["index"]),
            families=families,
        )
    else:
        report["scope"] += " Current snapshot is partial: no aggregate quality or failure rate."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key
                in {
                    "complete_official_pass",
                    "official_output_rows",
                    "observed_validation_requests",
                    "observed_completed_answers",
                    "families",
                }
            }
        )
    )


if __name__ == "__main__":
    main()
