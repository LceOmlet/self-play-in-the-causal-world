"""On-demand CPU evaluator using the running official rollout server.

No training hooks, weight changes, additional GPU model, or periodic scheduler.
The real ToolAgentLoop and CPTWorldTool own tokenization, tools and termination.
An evaluation turn yields its own request when training fills the server or its
generation window ends. A checkpoint change censors the whole diagnostic.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path
from uuid import uuid4


def server_snapshot(server, run_path):
    # Synchronous, read-only Ray RPC; native generate remains an async RPC.
    import json
    from pathlib import Path

    run = Path(run_path)
    tracker = run / "checkpoints/latest_checkpointed_iteration.txt"
    step = (
        int(tracker.read_text())
        if tracker.exists()
        else json.loads((run / "prepared.json").read_text())["step"]
    )
    other = sum(
        not key.startswith("cpt-diagnostic-")
        for key in server.engine.output_processor.request_states
    )
    return {
        "step": step,
        "other_requests": other,
        "capacity": server.config.max_num_seqs,
        "paused": server._submission_paused,
    }


async def borrowed_turn(
    server, *, run_path, request_id, expected_step, cancel_request=None, **kwargs
):
    import contextlib

    if cancel_request is None:
        import ray

        cancel_request = ray.cancel

    async def snapshot():
        return await server.__ray_call__.remote(server_snapshot, run_path)

    def available(state):
        return not state["paused"] and 0 < state["other_requests"] < state["capacity"]

    state = await snapshot()
    step = state["step"]
    if expected_step is not None and step != expected_step:
        return {"status": "version_changed", "step": step}
    if not available(state):
        return {"status": "yielded", "step": step}
    admission_count = state["other_requests"]
    started = time.monotonic()
    reference = server.generate.remote(request_id=request_id, **kwargs)
    task = asyncio.ensure_future(reference)
    try:
        while not task.done():
            await asyncio.wait([task], timeout=0.2)
            state = await snapshot()
            if state["step"] != step or not available(state):
                cancel_request(reference)
                # Ray async-actor cancellation reaches native AsyncLLM.generate,
                # which aborts only this request and releases its KV state.
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
                return {"status": "yielded", "step": step, "seconds": time.monotonic() - started}
        output = await task
        state = await snapshot()
        if state["step"] != step:
            return {"status": "version_changed", "step": state["step"]}
        if output.stop_reason in ("aborted", "abort"):
            return {"status": "yielded", "step": step}
        return {
            "status": "complete",
            "step": step,
            "output": output,
            "seconds": time.monotonic() - started,
            "training_requests_at_admission": admission_count,
        }
    finally:
        if not task.done():
            cancel_request(reference)
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


class DiagnosticInterrupted(RuntimeError):
    pass


class BorrowedClient:
    def __init__(self, server, run, directory, deadline):
        self.server, self.run, self.directory = server, run, directory
        self.deadline = deadline
        self.step = None
        self.turns, self.yields, self.seconds = 0, 0, 0.0

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        del request_id
        # Leave vLLM's existing FCFS/priority configuration alone. Admission and
        # cancellation below give training precedence without changing it.
        kwargs.pop("priority", None)
        while time.time() < self.deadline:
            result = await borrowed_turn(
                self.server,
                run_path=str(self.run),
                request_id="cpt-diagnostic-" + uuid4().hex,
                expected_step=self.step,
                prompt_ids=prompt_ids,
                sampling_params=dict(sampling_params),
                **kwargs,
            )
            self.seconds += result.get("seconds", 0)
            if result["status"] == "version_changed":
                raise DiagnosticInterrupted(f"Policy changed: {self.step} -> {result['step']}")
            if result["status"] == "complete":
                self.step = result["step"]
                self.turns += 1
                output = result.pop("output")
                output.extra_fields.update(
                    diagnostic_checkpoint_step=self.step,
                    min_global_steps=self.step,
                    max_global_steps=self.step,
                )
                with (self.directory / "turns.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {**result, "turn": self.turns, "token_count": len(output.token_ids)}
                        )
                        + "\n"
                    )
                return output
            self.yields += 1
            await asyncio.sleep(1)
        raise DiagnosticInterrupted("Diagnostic time allowance elapsed; not a task failure")


async def evaluate(args, server, config):
    import pyarrow.parquet as pq
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopWorker,
        DictConfigWrap,
        ToolListWrap,
    )
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

    # Initializing this official CPU worker loads only tokenizer/config/tools.
    client = BorrowedClient(server, args.run, args.output, time.time() + args.seconds)
    worker = AgentLoopWorker(config=config, llm_client=client)
    rows = pq.read_table(args.dataset).to_pylist()
    row = rows[args.index]
    loop = ToolAgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=client,
        tokenizer=worker.tokenizer,
        processor=worker.processor,
        hf_model_type=worker.hf_model_type,
        dataset_cls=worker.dataset_cls,
        data_config=DictConfigWrap(config.data),
        tools=ToolListWrap(worker.tools),
    )
    result = {
        "index": args.index,
        "task": row["extra_info"],
        "started": time.time(),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "scope": "On-demand shared-server diagnostic; not a complete fixed validation pass. "
        "Interruption is censoring, not zero reward. Completion can favor shorter tasks.",
    }
    try:
        output = await loop.run(
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "logprobs": config.actor_rollout_ref.rollout.calculate_log_probs,
            },
            raw_prompt=row["prompt"],
            tools_kwargs=row["extra_info"]["tools_kwargs"],
            extra_info=row["extra_info"],
        )
        result.update(
            status="complete",
            owner=output.extra_fields.get("cpt_world"),
            response_tokens=len(output.response_ids),
            num_turns=output.num_turns,
        )
        (args.output / "output.json").write_text(output.model_dump_json(indent=2))
        (args.output / "response.txt").write_text(worker.tokenizer.decode(output.response_ids))
    except DiagnosticInterrupted as error:
        result.update(status="censored", reason=str(error))
    except Exception as error:
        result.update(status="error", reason=repr(error))
        raise
    finally:
        result.update(
            ended=time.time(),
            checkpoint_step=client.step,
            completed_generation_turns=client.turns,
            yielded_requests=client.yields,
            request_seconds=client.seconds,
        )
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=600)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        CPT_WORLD_VERL_AUDIT_DIR=str(args.output / "environment"),
    )
    import psutil
    import ray
    from omegaconf import OmegaConf

    identity = json.loads((args.run / "process.json").read_text())
    process = psutil.Process(identity["pid"])
    assert abs(process.create_time() - identity["create_time"]) < 0.01
    gcs = [p for p in process.children(recursive=True) if p.name() == "gcs_server"]
    assert len(gcs) == 1
    port = next(s.split("=", 1)[1] for s in gcs[0].cmdline() if s.startswith("--gcs_server_port="))
    if port == "0":
        logdir = next(s.split("=", 1)[1] for s in gcs[0].cmdline() if s.startswith("--log_dir="))
        port = re.search(
            r"GcsServer server started, listening on port (\d+)",
            (Path(logdir) / "gcs_server.out").read_text(),
        ).group(1)
    ray.init(address="127.0.0.1:" + port, log_to_driver=False)
    named = [
        s for s in ray.util.list_named_actors(all_namespaces=True) if s["name"] == "vllm_server_0_0"
    ]
    assert len(named) == 1, named
    server = ray.get_actor(named[0]["name"], namespace=named[0]["namespace"])
    config = OmegaConf.load(args.run / "resolved.yaml")
    assert config.trainer.test_freq == -1 and not config.trainer.val_before_train
    assert config.actor_rollout_ref.rollout.n == 4
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "run": str(args.run),
                "trainer": identity,
                "server": named[0],
                "resolved_sha256": hashlib.sha256(
                    (args.run / "resolved.yaml").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    try:
        asyncio.run(evaluate(args, server, config))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
