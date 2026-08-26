#!/usr/bin/env python3
"""Summarize captured GRPO rollout A/B runs and optionally enforce P0 parity."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import statistics
from pathlib import Path

METRIC_DICT = re.compile(r"\{[^{}\r\n]+\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--p0-baseline", type=Path)
    parser.add_argument("--p0-candidate", type=Path)
    parser.add_argument("--bf16-logprob-tolerance", type=float, default=2**-7)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in METRIC_DICT.finditer(text):
        try:
            row = ast.literal_eval(match.group())
        except (SyntaxError, ValueError):
            continue
        if isinstance(row, dict) and "step_time" in row:
            rows.append(row)
    return rows


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize(run: Path) -> dict:
    checkpoint_dir = run / "checkpoints"
    events = read_jsonl(checkpoint_dir / "vllm-lifecycle.jsonl")
    metrics = read_metrics(run / "train.log")
    event_rows = [row for row in events if row["event"] != "request"]
    request_rows = [row for row in events if row["event"] == "request"]
    event_summary = {}
    for event_name in sorted({row["event"] for row in event_rows}):
        durations = [row["seconds"] for row in event_rows if row["event"] == event_name]
        event_summary[event_name] = {
            "count": len(durations),
            "sum_seconds": sum(durations),
            "median_seconds": statistics.median(durations),
        }
    completion_tokens = sum(row["completion_tokens"] for row in request_rows)
    generation_seconds = sum(
        row["seconds"] for row in event_rows if row["event"] == "generate"
    )
    step_times = [float(row["step_time"]) for row in metrics]
    cached_tokens = sum(row.get("cached_tokens") or 0 for row in request_rows)
    prompt_tokens = sum(row["prompt_tokens"] for row in request_rows)
    telemetry = run / "gpu-telemetry.csv"
    temperatures = []
    memory_used_mib = []
    if telemetry.exists():
        for line in telemetry.read_text(encoding="utf-8", errors="replace").splitlines():
            columns = [column.strip() for column in line.split(",")]
            try:
                if len(columns) >= 6:
                    temperatures.append(float(columns[1]))
                    memory_used_mib.append(float(columns[2]))
                elif len(columns) >= 5:
                    memory_used_mib.append(float(columns[1]))
            except ValueError:
                continue
    temperature_log = run / "gpu-temperature.csv"
    if temperature_log.exists():
        for line in temperature_log.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                temperatures.append(float(line.rsplit(",", 1)[1]))
            except (IndexError, ValueError):
                continue
    return {
        "run": str(run),
        "steps": len(step_times),
        "step_time_median_seconds": statistics.median(step_times) if step_times else None,
        "step_time_p90_seconds": quantile(step_times, 0.9),
        "step_time_total_seconds": sum(step_times),
        "completion_tokens": completion_tokens,
        "generation_tokens_per_second": (
            completion_tokens / generation_seconds if generation_seconds else None
        ),
        "effective_generation_tokens_per_second": (
            completion_tokens / sum(step_times) if step_times else None
        ),
        "prefix_cache_hit_fraction": cached_tokens / prompt_tokens if prompt_tokens else None,
        "gpu_peak_memory_mib": max(memory_used_mib) if memory_used_mib else None,
        "gpu_peak_temperature_c": max(temperatures) if temperatures else None,
        "events": event_summary,
    }


def flatten_numbers(value):
    if isinstance(value, list):
        for item in value:
            yield from flatten_numbers(item)
    elif value is not None:
        yield float(value)


def p0_parity(baseline: Path, candidate: Path, tolerance: float) -> dict:
    baseline_rows = read_jsonl(baseline / "checkpoints" / "rollout-artifacts.jsonl")
    candidate_rows = read_jsonl(candidate / "checkpoints" / "rollout-artifacts.jsonl")
    if len(baseline_rows) != len(candidate_rows):
        raise AssertionError(
            f"rollout count differs: baseline={len(baseline_rows)}, candidate={len(candidate_rows)}"
        )
    exact_keys = ("prompt_ids", "completion_ids", "tool_mask", "tool_trace")
    max_logprob_difference = 0.0
    for index, (left, right) in enumerate(zip(baseline_rows, candidate_rows, strict=True)):
        for key in exact_keys:
            if left[key] != right[key]:
                raise AssertionError(f"P0 {key} differs at rollout {index}")
        left_logprobs = list(flatten_numbers(left["sampling_logprobs"]))
        right_logprobs = list(flatten_numbers(right["sampling_logprobs"]))
        if len(left_logprobs) != len(right_logprobs):
            raise AssertionError(f"P0 logprob shape differs at rollout {index}")
        if left_logprobs:
            max_logprob_difference = max(
                max_logprob_difference,
                max(abs(a - b) for a, b in zip(left_logprobs, right_logprobs, strict=True)),
            )
    if max_logprob_difference > tolerance:
        raise AssertionError(
            f"P0 selected-token logprob max difference {max_logprob_difference} exceeds {tolerance}"
        )
    return {
        "rollouts": len(baseline_rows),
        "token_ids_tool_trace_exact": True,
        "max_selected_token_logprob_difference": max_logprob_difference,
        "tolerance": tolerance,
    }


def main() -> None:
    args = parse_args()
    report = {"runs": [summarize(run) for run in args.runs]}
    if args.p0_baseline or args.p0_candidate:
        if not args.p0_baseline or not args.p0_candidate:
            raise ValueError("--p0-baseline and --p0-candidate must be supplied together")
        report["p0_parity"] = p0_parity(
            args.p0_baseline,
            args.p0_candidate,
            args.bf16_logprob_tolerance,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
