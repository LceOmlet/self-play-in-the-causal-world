"""Run independent 9B tool conversations against the opaque owner HTTP API.

Only eval100's public task_ids are read. Prompts and tools come from the owner.
The owner URL (including its role token) is supplied at runtime and never logged.
No model request is retried. Owner transport retries reuse the exact request UUID.
Importing this module does not load a model, read tasks, or start evaluation.
Run ``python run_9b.py --self-test`` for synthetic API/tool checks only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable


def post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("HTTP response must be a JSON object")
    return value


def owner_request(url: str, payload: dict, timeout: float, attempts: int = 3) -> dict:
    """Transport recovery only: every attempt sends the identical UUID/payload."""
    request = {**payload, "request_id": payload.get("request_id", str(uuid.uuid4()))}
    for attempt in range(attempts):
        try:
            return post_json(url, request, timeout)
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt + 1 == attempts:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt + 1 == attempts:
                raise
        time.sleep(0.25 * (attempt + 1))
    raise AssertionError("unreachable")


def count_tokens(tokenizer: Any, messages: list, tools: list, *, generation: bool) -> int:
    rendered_messages = copy.deepcopy(messages)
    # HF Qwen templates require mappings; keep OpenAI wire records untouched.
    for message in rendered_messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                except ValueError:
                    parsed = None
                function["arguments"] = parsed if isinstance(parsed, dict) else {"invalid_raw_arguments": arguments}
    tokens = tokenizer.apply_chat_template(
        rendered_messages, tools=tools, tokenize=True, add_generation_prompt=generation,
        enable_thinking=False, return_dict=True,
    )
    ids = tokens["input_ids"] if hasattr(tokens, "keys") else tokens
    if hasattr(ids,"tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0],list):
        if len(ids)!=1:
            raise ValueError("Expected one conversation for token accounting")
        ids=ids[0]
    return len(ids)


def run_task(
    task_id: str, *, tokenizer: Any, profile: dict, model: str, owner_url: str,
    model_api: str, output_dir: Path, timeout: float = 3600,
    owner: Callable[..., dict] = owner_request,
    completion: Callable[..., dict] = post_json,
    resume_first_responses: Path | None = None,
) -> dict:
    """Dependency injection supports synthetic tests with no private data/API."""
    path = output_dir / (hashlib.sha256(task_id.encode()).hexdigest() + ".json")
    if path.exists():
        raise FileExistsError("refusing to overwrite an existing task transcript")
    record: dict = {
        "task_id": task_id, "model": model, "profile": copy.deepcopy(profile),
        "started_unix": time.time(), "messages": [], "tools": [], "events": [],
        "model_requests": 0, "api_requests_this_run": 0, "reused_responses": 0,
        "usage": {}, "termination_reason": None,
    }
    messages = record["messages"]
    ended = False
    finish_attempted = False

    def save() -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def finish(reason: str) -> None:
        nonlocal ended, finish_attempted
        record["termination_reason"] = reason
        if not ended and not finish_attempted:
            finish_attempted = True
            request = {"op": "finish", "task_id": task_id, "reason": reason,
                       "request_id": str(uuid.uuid4())}
            result = owner(owner_url, request, timeout)
            record["events"].append({"kind": "finish", "request": request,
                                     "response": result, "time_unix": time.time()})
            ended = True

    try:
        prompt = owner(owner_url, {"op": "prompt", "task_id": task_id}, timeout)
        if prompt.get("task_id") != task_id:
            raise ValueError("owner returned a different task_id")
        if not isinstance(prompt.get("messages"), list) or not isinstance(prompt.get("tools"), list):
            raise ValueError("owner prompt must contain messages and tools arrays")
        messages.extend(copy.deepcopy(prompt["messages"]))
        tools = record["tools"] = copy.deepcopy(prompt["tools"])
        pending_original = None
        if resume_first_responses is not None:
            source = resume_first_responses/path.name
            source_bytes = source.read_bytes()
            original = json.loads(source_bytes)
            original_turns = [e for e in original["events"] if e["kind"]=="completion"]
            if not (original["task_id"]==task_id and original["model"]==model
                    and original["profile"]==profile and original["model_requests"]==1
                    and original.get("error_type")=="TypeError"
                    and original["termination_reason"]=="runner_or_owner_failure"
                    and len(original_turns)==1 and "response" in original_turns[0]
                    and not any(e["kind"]=="tool" for e in original["events"])
                    and original["messages"][:-1]==messages and original["tools"]==tools):
                raise ValueError("Original first-response recovery preconditions failed")
            pending_original = original_turns[0]
            record["recovery_source"] = {"filename":source.name,
                "sha256":hashlib.sha256(source_bytes).hexdigest(),
                "original_failure":"TypeError before any tool dispatch"}
        save()
        while not ended:
            prompt_tokens = count_tokens(tokenizer, messages, tools, generation=True)
            remaining = profile["max_model_len"] - prompt_tokens
            if remaining <= 0:
                finish("context_exhausted")
                break
            payload = {
                "model": profile["models"][model], "messages": copy.deepcopy(messages),
                "tools": tools, "tool_choice": "auto", "temperature": profile["temperature"],
                "seed": profile["seed"], "stream": False,
                "max_tokens": min(profile["max_tokens_per_turn"], remaining),
                "chat_template_kwargs": {"enable_thinking": False},
            }
            turn = {"kind": "completion", "started_unix": time.time(),
                    "prompt_tokens_tokenizer": prompt_tokens, "remaining_tokens": remaining,
                    "request": payload}
            record["events"].append(turn)
            record["model_requests"] += 1
            save()
            try:
                if pending_original is not None:
                    if pending_original["request"]!=payload:
                        raise ValueError("Recovery payload differs; refusing to resample")
                    response = copy.deepcopy(pending_original["response"])
                    turn["reused_original_completion"] = {
                        "source_sha256":record["recovery_source"]["sha256"],
                        "original_started_unix":pending_original["started_unix"],
                        "original_ended_unix":pending_original["ended_unix"]}
                    record["reused_responses"] += 1
                    pending_original = None
                else:
                    record["api_requests_this_run"] += 1
                    response = completion(model_api, payload, timeout)
            except Exception as error:
                # Exception text may contain URLs; persist only its safe type.
                turn["error_type"] = type(error).__name__
                finish("model_api_failure")
                break
            turn.update(response=response, ended_unix=time.time())
            usage = response.get("usage", {})
            for key, value in usage.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    record["usage"][key] = record["usage"].get(key, 0) + value
            choices = response.get("choices", [])
            if len(choices) != 1 or not isinstance(choices[0].get("message"), dict):
                finish("invalid_model_response")
                break
            assistant = copy.deepcopy(choices[0]["message"])
            if assistant.get("role") != "assistant":
                finish("invalid_model_role")
                break
            messages.append(assistant)
            save()
            if count_tokens(tokenizer, messages, tools, generation=False) > profile["max_model_len"]:
                finish("context_exhausted")
                break
            calls = assistant.get("tool_calls") or []
            if not calls:
                finish("model_length_limit" if choices[0].get("finish_reason") == "length"
                       else "assistant_without_tool_call")
                break
            for call in calls:
                function = call.get("function", {})
                name, raw_arguments = function.get("name"), function.get("arguments")
                if call.get("type") != "function" or not isinstance(call.get("id"), str):
                    finish("invalid_tool_call")
                    break
                try:
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else None
                except (ValueError, TypeError):
                    arguments = None
                if not isinstance(arguments, dict):
                    finish("invalid_tool_arguments")
                    break
                request = {"op": name, "task_id": task_id, "request_id": str(uuid.uuid4())}
                if name == "act" and isinstance(arguments.get("command"), dict):
                    request["command"] = arguments["command"]
                elif name == "numerics" and isinstance(arguments.get("code"), str):
                    request["code"] = arguments["code"]
                else:
                    finish("unsupported_tool_or_arguments")
                    break
                receipt = {"kind": "tool", "tool_call_id": call["id"],
                           "request": request, "started_unix": time.time()}
                record["events"].append(receipt)
                save()
                result = owner(owner_url, request, timeout)
                receipt.update(response=result, ended_unix=time.time())
                if not isinstance(result.get("output"), str) or not isinstance(result.get("ended"), bool):
                    raise ValueError("invalid owner tool receipt")
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": result["output"]})
                ended = result["ended"]
                if ended:
                    record["termination_reason"] = "owner_ended"
                save()
                if ended:
                    break
                if count_tokens(tokenizer, messages, tools, generation=False) > profile["max_model_len"]:
                    finish("context_exhausted")
                    break
    except Exception as error:
        record["error_type"] = type(error).__name__
        if not ended and not finish_attempted:
            try:
                finish("runner_or_owner_failure")
            except Exception as finish_error:
                record["finish_error_type"] = type(finish_error).__name__
    finally:
        record["ended_unix"] = time.time()
        save()
    return {"task_id": task_id, "termination_reason": record["termination_reason"],
            "transcript": path.name, "model_requests": record["model_requests"]}


def main() -> None:
    import sys

    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("base", "checkpoint2350"), default="checkpoint2350")
    parser.add_argument("--owner-url", required=True, help="runtime owner URL including opaque role token")
    parser.add_argument("--index", type=Path, required=True, help="public eval100.json only")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-api", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--resume-first-responses", type=Path,
        help="Recover saved first response of the known pre-tool TypeError without resampling")
    args = parser.parse_args()
    profile = json.loads(Path(__file__).with_name("model_profile.json").read_text())
    if not 1 <= args.concurrency <= 4:
        parser.error("concurrency must be between 1 and 4")
    task_ids = json.loads(args.index.read_text(encoding="utf-8"))["task_ids"]
    if not isinstance(task_ids, list) or len(task_ids) != 100:
        parser.error("public eval100 index must contain 100 unique task IDs")
    if any(not isinstance(task_id, str) for task_id in task_ids):
        parser.error("task IDs must be strings")
    if len(set(task_ids)) != 100:
        parser.error("public eval100 index must contain 100 unique task IDs")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or profile["base_model_path"])
    summaries = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(
            run_task, task_id, tokenizer=tokenizer, profile=profile, model=args.model,
            owner_url=args.owner_url, model_api=args.model_api or profile["model_api"],
            output_dir=args.output_dir, timeout=args.timeout,
            resume_first_responses=args.resume_first_responses,
        ) for task_id in task_ids]
        for future in as_completed(futures):
            summary = future.result()
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    (args.output_dir / "run_summary.json").write_text(
        json.dumps({"model": args.model, "tasks": summaries}, indent=2), encoding="utf-8",
    )


def self_test() -> None:
    """Synthetic fixtures only; no tokenizer weights, benchmark files, or HTTP."""
    import tempfile

    profile = json.loads(Path(__file__).with_name("model_profile.json").read_text())

    class Tokenizer:
        def apply_chat_template(self, messages: list, **kwargs: Any) -> list:
            assert kwargs["enable_thinking"] is False and "tools" in kwargs
            return [0] * (100 + len(messages))

    calls = [
        {"id": "first", "type": "function", "function": {
            "name": "numerics", "arguments": '{ "code" : "print(2)" }'}},
        {"id": "second", "type": "function", "function": {
            "name": "act", "arguments": '{"command":{"answer":"synthetic"}}'}},
        {"id": "third", "type": "function", "function": {
            "name": "numerics", "arguments": '{"code":"never execute after ended"}'}},
    ]
    for scenario in ("ordered_tools", "plain_text", "context_exhausted", "api_failure",
                     "tool_feedback_exhausts_context"):
        operations, model_requests = [], []

        def owner(url: str, payload: dict, timeout: float) -> dict:
            operations.append(payload)
            if payload["op"] == "prompt":
                return {"task_id": "synthetic", "messages": [
                    {"role": "user", "content": "Synthetic only"}], "tools": []}
            return {"output": "synthetic receipt", "ended": payload["op"] != "numerics"}

        def model_api(url: str, payload: dict, timeout: float) -> dict:
            model_requests.append(payload)
            assert payload["max_tokens"] == min(8192, config["max_model_len"] - 101)
            if scenario == "api_failure":
                raise ConnectionError("sensitive transport details must not be persisted")
            message = {"role": "assistant", "content": None, "tool_calls": calls}
            if scenario == "plain_text":
                message = {"role": "assistant", "content": '{"answer":"never parse text"}'}
            return {"choices": [{"message": message, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 101, "completion_tokens": 7}}

        config = dict(profile)
        if scenario == "context_exhausted":
            config["max_model_len"] = 100
        elif scenario == "tool_feedback_exhausts_context":
            config["max_model_len"] = 102
        with tempfile.TemporaryDirectory() as temporary:
            result = run_task(
                "synthetic", tokenizer=Tokenizer(), profile=config, model="checkpoint2350",
                owner_url="http://opaque-test-secret", model_api="http://synthetic",
                output_dir=Path(temporary), owner=owner, completion=model_api,
            )
            transcript = json.loads(next(Path(temporary).glob("*.json")).read_text())
            assert "opaque-test-secret" not in json.dumps(transcript)
            if scenario == "ordered_tools":
                assert [op["op"] for op in operations] == ["prompt", "numerics", "act"]
                assert result["termination_reason"] == "owner_ended"
                assert transcript["messages"][1]["tool_calls"] == calls
                assert [message["tool_call_id"] for message in transcript["messages"]
                        if message["role"] == "tool"] == ["first", "second"]
            elif scenario == "tool_feedback_exhausts_context":
                assert [op["op"] for op in operations] == ["prompt", "numerics", "finish"]
                assert operations[-1]["reason"] == "context_exhausted"
                assert model_requests[0]["max_tokens"] == 1
            else:
                expected = {"plain_text": "assistant_without_tool_call",
                            "context_exhausted": "context_exhausted",
                            "api_failure": "model_api_failure"}[scenario]
                assert operations[-1]["reason"] == expected
                assert len(model_requests) == (0 if scenario == "context_exhausted" else 1)
            print(f"{scenario}: PASS")


if __name__ == "__main__":
    main()
