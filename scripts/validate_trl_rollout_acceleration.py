#!/usr/bin/env python3
"""CPU-only contract checks for the pinned TRL rollout-acceleration patch."""

from __future__ import annotations

from contextlib import nullcontext
from types import MethodType, SimpleNamespace

from torch import nn
from trl.generation.vllm_generation import VLLMGeneration


class FakeLLM:
    def __init__(self, *, fail_generate: bool = False) -> None:
        self.events: list[tuple[str, object]] = []
        self.fail_generate = fail_generate

    def wake_up(self, *, tags: list[str]) -> None:
        self.events.append(("wake", tuple(tags)))

    def sleep(self, *, level: int) -> None:
        self.events.append(("sleep", level))

    def generate(self, prompts, *, sampling_params, use_tqdm):
        del sampling_params, use_tqdm
        self.events.append(("generate", len(prompts)))
        if self.fail_generate:
            raise RuntimeError("injected generation failure")
        return [
            SimpleNamespace(
                prompt_token_ids=prompt["prompt_token_ids"],
                outputs=[
                    SimpleNamespace(
                        token_ids=[7],
                        logprobs=[{7: SimpleNamespace(rank=1, logprob=-0.25)}],
                    )
                ],
            )
            for prompt in prompts
        ]


def build_backend(*, residency: bool, sleep_level: int = 2, fail_generate: bool = False):
    backend = object.__new__(VLLMGeneration)
    backend.mode = "colocate"
    backend.enable_sleep_mode = True
    backend.rollout_residency = residency
    backend.sleep_level = sleep_level
    backend._rollout_session_depth = 0
    backend._llm_weights_sleeping = True
    backend._llm_kv_cache_sleeping = True
    backend.llm = FakeLLM(fail_generate=fail_generate)
    backend.accelerator = SimpleNamespace(local_process_index=0)
    backend.temperature = 1.0
    backend.top_p = 1.0
    backend.top_k = 0
    backend.min_p = None
    backend.repetition_penalty = 1.0
    backend.max_completion_length = 8
    backend.logprobs = 0
    backend.generation_kwargs = {}
    backend.structured_outputs_regex = None
    backend.tensor_parallel_size = 1
    backend.model = nn.Module()

    def sync_weights(self) -> None:
        self.llm.events.append(("sync", None))
        self._llm_weights_sleeping = False

    backend.sync_weights = MethodType(sync_weights, backend)
    return backend


def generate_once(backend: VLLMGeneration):
    return backend.generate(
        prompts=[[1, 2, 3]],
        images=None,
        num_generations=1,
        profiler=nullcontext(),
    )


def main() -> None:
    baseline = build_backend(residency=False)
    generate_once(baseline)
    generate_once(baseline)
    assert baseline.llm.events == [
        ("sync", None),
        ("wake", ("kv_cache",)),
        ("generate", 1),
        ("sleep", 2),
        ("sync", None),
        ("wake", ("kv_cache",)),
        ("generate", 1),
        ("sleep", 2),
    ]

    resident = build_backend(residency=True)
    with resident.rollout_session():
        generate_once(resident)
        with resident.rollout_session():
            generate_once(resident)
        assert resident._rollout_session_depth == 1
    assert resident.llm.events == [
        ("sync", None),
        ("wake", ("kv_cache",)),
        ("generate", 1),
        ("generate", 1),
        ("sleep", 2),
    ]

    mtp = build_backend(residency=True, sleep_level=1)
    with mtp.rollout_session():
        generate_once(mtp)
    assert mtp.llm.events[-1] == ("sleep", 1)

    failing = build_backend(residency=True, fail_generate=True)
    try:
        with failing.rollout_session():
            generate_once(failing)
    except RuntimeError as error:
        assert str(error) == "injected generation failure"
    else:
        raise AssertionError("injected generation failure did not propagate")
    assert failing.llm.events[-1] == ("sleep", 2)
    assert failing._rollout_session_depth == 0

    print("TRL rollout acceleration lifecycle contracts: PASS")


if __name__ == "__main__":
    main()
