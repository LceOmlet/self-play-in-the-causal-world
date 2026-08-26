#!/usr/bin/env python3
"""Run or compare fixed-token greedy vLLM semantics for APC and MTP-1."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--adapter")
    parser.add_argument("--prompt-artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--mtp-tokens", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--compare", nargs="+", type=Path)
    parser.add_argument("--repeat-control", type=Path)
    parser.add_argument("--logprob-tolerance", type=float, default=2**-7)
    args = parser.parse_args()
    if args.compare:
        return args
    missing = [
        name
        for name in ("model", "adapter", "prompt_artifact", "output")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(f"missing run arguments: {', '.join(missing)}")
    return args


def extract_sampled_logprobs(completion) -> list[float]:
    values = []
    for token_id, token_logprobs in zip(
        completion.token_ids,
        completion.logprobs,
        strict=True,
    ):
        values.append(float(token_logprobs[token_id].logprob))
    return values


def run(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    first_rollout = json.loads(
        args.prompt_artifact.read_text(encoding="utf-8").splitlines()[0]
    )
    prompts = [{"prompt_token_ids": ids} for ids in first_rollout["prompt_ids"]]
    speculative_config = (
        {"method": "mtp", "num_speculative_tokens": args.mtp_tokens}
        if args.mtp_tokens
        else None
    )
    llm = LLM(
        model=args.model,
        max_model_len=32768,
        gpu_memory_utilization=0.50,
        max_num_seqs=len(prompts),
        max_num_batched_tokens=4096,
        enable_prefix_caching=args.enable_prefix_caching,
        speculative_config=speculative_config,
        enable_lora=True,
        max_lora_rank=8,
        logprobs_mode="processed_logprobs",
        seed=0,
    )
    outputs = llm.generate(
        prompts,
        sampling_params=SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            logprobs=0,
        ),
        lora_request=LoRARequest("checkpoint-50", 1, args.adapter),
        use_tqdm=False,
    )
    payload = {
        "enable_prefix_caching": args.enable_prefix_caching,
        "mtp_tokens": args.mtp_tokens,
        "outputs": [
            {
                "prompt_ids": output.prompt_token_ids,
                "completion_ids": completion.token_ids,
                "sampled_logprobs": extract_sampled_logprobs(completion),
            }
            for output in outputs
            for completion in output.outputs
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(args.output)


def _differences_against_baseline(
    baseline: list[dict], outputs: list[dict], label: Path
) -> tuple[list[float], bool, list[str]]:
    failures = []
    differences = []
    token_ids_exact = True
    for index, (left, right) in enumerate(zip(baseline, outputs, strict=True)):
        if left["prompt_ids"] != right["prompt_ids"]:
            failures.append(f"prompt IDs differ for {label} output {index}")
        if left["completion_ids"] != right["completion_ids"]:
            token_ids_exact = False
            failures.append(f"greedy token IDs differ for {label} output {index}")
            continue
        left_logprobs = left["sampled_logprobs"]
        right_logprobs = right["sampled_logprobs"]
        if len(left_logprobs) != len(right_logprobs):
            failures.append(f"logprob lengths differ for {label} output {index}")
            continue
        differences.extend(
            abs(a - b)
            for a, b in zip(left_logprobs, right_logprobs, strict=True)
            if math.isfinite(a) and math.isfinite(b)
        )
    differences.sort()
    return differences, token_ids_exact, failures


def _difference_stats(differences: list[float]) -> dict[str, float]:
    return {
        "mean": sum(differences) / len(differences) if differences else 0.0,
        "p99": (
            differences[int(0.99 * (len(differences) - 1))]
            if differences
            else 0.0
        ),
        "max": differences[-1] if differences else 0.0,
    }


def compare(paths: list[Path], tolerance: float, repeat_control: Path | None) -> None:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    baseline = rows[0]["outputs"]
    report = []
    failures = []
    control_stats = None
    if repeat_control is not None:
        control = json.loads(repeat_control.read_text(encoding="utf-8"))["outputs"]
        if len(control) != len(baseline):
            raise AssertionError(f"output count differs for {repeat_control}")
        control_differences, control_tokens_exact, control_failures = (
            _differences_against_baseline(baseline, control, repeat_control)
        )
        if not control_tokens_exact or control_failures:
            raise AssertionError("; ".join(control_failures))
        control_stats = _difference_stats(control_differences)
    for path, row in zip(paths[1:], rows[1:], strict=True):
        outputs = row["outputs"]
        if len(baseline) != len(outputs):
            failures.append(f"output count differs for {path}")
            report.append({"candidate": str(path), "output_count_exact": False})
            continue
        differences, token_ids_exact, output_failures = _differences_against_baseline(
            baseline, outputs, path
        )
        failures.extend(output_failures)
        stats = _difference_stats(differences)
        if control_stats is None and stats["max"] > tolerance:
            failures.append(
                f"{path} max selected-token logprob difference "
                f"{stats['max']} exceeds {tolerance}"
            )
        if control_stats is not None:
            for metric in ("mean", "p99", "max"):
                if stats[metric] > control_stats[metric] + tolerance:
                    failures.append(
                        f"{path} {metric} selected-token logprob difference "
                        f"{stats[metric]} exceeds repeat-control envelope "
                        f"{control_stats[metric]} + {tolerance}"
                    )
        report.append(
            {
                "candidate": str(path),
                "greedy_token_ids_exact": token_ids_exact,
                "selected_token_count": len(differences),
                "mean_selected_token_logprob_difference": stats["mean"],
                "p99_selected_token_logprob_difference": stats["p99"],
                "max_selected_token_logprob_difference": stats["max"],
                "selected_token_logprobs_within_tolerance": sum(
                    difference <= tolerance for difference in differences
                ),
                "repeat_control_envelope": control_stats,
            }
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise AssertionError("; ".join(failures))


def main() -> None:
    args = parse_args()
    if args.compare:
        if len(args.compare) < 2:
            raise ValueError("--compare needs a baseline and at least one candidate")
        compare(args.compare, args.logprob_tolerance, args.repeat_control)
    else:
        run(args)


if __name__ == "__main__":
    main()
