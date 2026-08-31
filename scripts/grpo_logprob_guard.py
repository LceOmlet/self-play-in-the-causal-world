"""Validate the sampling log-prob boundary returned by TRL/vLLM."""

from __future__ import annotations

import math
from collections.abc import Sequence


def require_finite_sampling_logprobs(
    sampling_logprobs: Sequence[Sequence[float | None]] | None,
    *,
    global_step: int,
) -> None:
    """Reject missing or non-finite sampling log-probs before GRPO tensorization."""

    if sampling_logprobs is None:
        raise RuntimeError(
            "TRL/vLLM returned no sampling log-probs while importance-sampling "
            f"correction is enabled at global_step={global_step}"
        )
    for sequence_index, sequence in enumerate(sampling_logprobs):
        if sequence is None:
            raise RuntimeError(
                "TRL/vLLM returned a missing sampling log-prob sequence while "
                "importance-sampling correction is enabled at "
                f"global_step={global_step}, sequence_index={sequence_index}"
            )
        for token_index, value in enumerate(sequence):
            if value is None or not math.isfinite(float(value)):
                raise RuntimeError(
                    "TRL/vLLM returned a non-finite sampling log-prob while "
                    "importance-sampling correction is enabled at "
                    f"global_step={global_step}, sequence_index={sequence_index}, "
                    f"token_index={token_index}, value={value!r}. Do not impute this "
                    "value; disable speculative decoding and regenerate the rollout."
                )
