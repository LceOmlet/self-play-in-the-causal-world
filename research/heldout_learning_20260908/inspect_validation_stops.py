"""Inspect complete official outputs against their uniquely joined owner events.

No generation or reward code is replaced. Visible decoded text is not claimed
to preserve the original token IDs or exact response-mask length.
"""

import argparse
import hashlib
import importlib.util
import json
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verl-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["complete_official_pass"] and len(report["records"]) == 25
    from omegaconf import OmegaConf

    config = OmegaConf.load(args.config)
    reward_config = config.reward.reward_kwargs
    buffer = reward_config.overlong_buffer_cfg
    assert buffer.enable and buffer.log and buffer.len == 4096 and buffer.penalty_factor == 1.0
    assert reward_config.max_resp_len == config.actor_rollout_ref.rollout.response_length == 30720
    turns = config.actor_rollout_ref.rollout.multi_turn
    assert turns.max_assistant_turns is None and turns.max_user_turns is None
    assert turns.max_parallel_calls == 1
    reward_source = args.verl_root / "verl/workers/reward_manager/dapo.py"
    loop_source = args.verl_root / "verl/experimental/agent_loop/tool_agent_loop.py"
    adapter_source = args.project / "src/cpt_world/verl_environment.py"
    expected_sources = {
        reward_source: "b01ee41902a7766b877bc3f35ec6d56bc5818960adc105364be0d01db6b4c5ed",
        loop_source: "52c38bef0152c6f219fa69b654e77f96085433be1ceb5091713cb49c49fd3859",
        adapter_source: "9deed2b23c4b07f8327de46f9bff81be2e484cf3d402e8e7ac1130eaa4836fdb",
    }
    assert all(sha(path.read_bytes()) == value for path, value in expected_sources.items())
    dump = args.snapshot / "validation" / f"{report['step']}.jsonl"
    assert sha(dump.read_bytes()) == report["outputs_sha256"]
    outputs = {
        sha(row["input"].encode()): row
        for row in [json.loads(line) for line in dump.read_bytes().splitlines()]
    }
    assert len(outputs) == 25
    owner = module(
        args.project / "research/base_signal_diagnostic_20260907/analyze_events.py", "owner"
    )
    events, fingerprints = owner.read_events(args.snapshot)
    assert fingerprints == report["event_prefixes"]
    traces = defaultdict(list)
    for event in events:
        if event["event"] == "act":
            traces[event["trajectory_id"]].append(event)
    call = re.compile(
        r"<tool_call>\s*<function=act>\s*<parameter=command>\s*"
        r"(.*?)\s*</parameter>\s*</function>\s*</tool_call>",
        re.DOTALL,
    )
    records = []
    for record in report["records"]:
        output = outputs[record["input_sha256"]]["output"]
        assert sha(output.encode()) == record["output_sha256"]
        trace = traces.get(record["request_id"], [])
        calls = list(call.finditer(output))
        assert len(calls) == len(trace)
        matches = []
        for i, (match, event) in enumerate(zip(calls, trace, strict=True)):
            segment = output[match.end() : calls[i + 1].start() if i + 1 < len(calls) else None]
            matches.append(event["feedback"] in segment)
        last = trace[-1] if trace else None
        suffix = output[calls[-1].end() :] if calls else output
        observation = (
            "terminal_answer_executed"
            if record["completed"]
            else "ends_at_executed_nonterminal_call_without_feedback"
            if calls and not suffix.strip() and not matches[-1]
            else "other_unfinished_output"
        )
        # The official negative penalty has an invertible affine expression:
        # p = -(L - (M-B))/B for the verified factor=1. All integers here are
        # below 2**24 and B=4096, so these recorded FP32 values are exact dyadics.
        # With zero penalty only L <= M-B is known; never invent an exact length.
        penalty = Fraction(record["official_length_penalty"])
        length = (
            reward_config.max_resp_len - buffer.len - penalty * buffer.len if penalty < 0 else None
        )
        if length is not None:
            assert length.denominator == 1 and 0 < length <= reward_config.max_resp_len
            length = int(length)
        if record["completed"]:
            stop_path = "terminal_answer"
        elif length == reward_config.max_resp_len:
            stop_path = "generation_response_limit"
        elif observation == "ends_at_executed_nonterminal_call_without_feedback":
            assert last is not None and not last["completed"]
            assert len(last["feedback"]) < turns.max_tool_response_length
            assert all(matches[:-1]) and length is not None
            stop_path = "tool_feedback_response_limit"
        else:
            stop_path = "unresolved"
        remaining_budget = None
        if last and not last["completed"]:
            remaining_budget = owner.feedback_of(last).get("remaining_budget")
        counts = Counter(json.dumps(event["command"], sort_keys=True) for event in trace)
        records.append(
            {
                "index": record["index"],
                "tape_key": record["tape_key"],
                "query_type": record["query_type"],
                "request_id": record["request_id"],
                "completed": record["completed"],
                "observation": observation,
                "stop_path_from_verified_control_flow": stop_path,
                "response_attention_length_from_negative_penalty": length,
                "response_attention_length_upper_bound": reward_config.max_resp_len - buffer.len
                if length is None
                else length,
                "remaining_environment_budget": remaining_budget,
                "output_sha256": record["output_sha256"],
                "output_characters": len(output),
                "executed_calls": len(trace),
                "feedback_exactly_present_after_each_call": matches,
                "observations_used": last["observations_used"] if last else 0,
                "last_command": last["command"] if last else None,
                "last_feedback_characters": len(last["feedback"]) if last else 0,
                "last_feedback_sha256": sha(last["feedback"].encode()) if last else None,
                "most_repeated_command": json.loads(counts.most_common(1)[0][0])
                if counts
                else None,
                "most_repeated_command_count": counts.most_common(1)[0][1] if counts else 0,
                "output_tail": output[-1800:],
            }
        )
    result = {
        "reader_sha256": sha(Path(__file__).read_bytes()),
        "validation_report_sha256": sha(args.report.read_bytes()),
        "configuration_sha256": sha(args.config.read_bytes()),
        "verified_sources": {str(path): value for path, value in expected_sources.items()},
        "observations": dict(Counter(record["observation"] for record in records)),
        "stop_paths": dict(
            Counter(record["stop_path_from_verified_control_flow"] for record in records)
        ),
        "scope": "Observations from full decoded official output and matched owner events. "
        "Exact token IDs were not preserved by the official text dump. Negative official length "
        "penalties recover the exact response attention length via the verified upstream formula; "
        "this includes tool text, not just model action tokens. "
        "Zero penalties give an upper bound. "
        "Stop paths are inferred from these observations and the inspected exact source, not "
        "new runtime instrumentation. No before/after learning effect is established here.",
        "records": records,
    }
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"observations": result["observations"], "stop_paths": result["stop_paths"]}))


if __name__ == "__main__":
    main()
