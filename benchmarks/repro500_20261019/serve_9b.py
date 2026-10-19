"""Start an isolated inference server; never train or edit checkpoint weights.

Run with the existing venv's bin directory on PATH (FlashInfer JIT needs ninja).
The process-local platform fallback uses vLLM's existing NonNvmlCudaPlatform.
This module is re-executed in spawned engine processes, so keep the patch outside
the __main__ guard. No installed package or system driver is modified.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def configure_platform() -> None:
    import vllm.platforms as platforms

    platforms.builtin_platform_plugins["cuda"] = (
        lambda: "vllm.platforms.cuda.NonNvmlCudaPlatform"
    )
    platforms._current_platform = None


configure_platform()


def main() -> None:
    import argparse
    import json
    import runpy
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("base", "checkpoint2350"), default="checkpoint2350")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18821)
    args = parser.parse_args()
    profile = json.loads(Path(__file__).with_name("model_profile.json").read_text())
    command = [
        "vllm.entrypoints.openai.api_server",
        "--model", profile["base_model_path"],
        "--served-model-name", "base",
        "--host", args.host, "--port", str(args.port),
        "--max-model-len", str(profile["max_model_len"]),
        "--gpu-memory-utilization", str(profile["gpu_memory_utilization"]),
        "--max-num-seqs", str(profile["concurrency"]),
        "--max-num-batched-tokens", str(profile["max_num_batched_tokens"]),
        "--enable-auto-tool-choice", "--tool-call-parser", profile["tool_call_parser"],
        "--reasoning-parser", profile["reasoning_parser"],
        "--seed", str(profile["seed"]), "--enforce-eager",
        "--limit-mm-per-prompt", '{"image":0,"video":0}',
    ]
    if args.model == "checkpoint2350":
        command += [
            "--enable-lora", "--max-lora-rank", str(profile["max_lora_rank"]),
            "--lora-modules", f"checkpoint2350={profile['adapter_path']}",
        ]
    print(json.dumps({"event": "server_start", "pid": os.getpid(), "argv": command}), flush=True)
    sys.argv = command
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
