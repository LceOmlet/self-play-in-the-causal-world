#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${1:-/home/chen/projects/self-play-in-the-causal-world}
MODEL_DIR=${2:-/home/chen/models/Qwen/Qwen3.5-9B}
ENV_DIR=${3:-/home/chen/.venvs/dolens-rl}
RUN_DIR=${4:-/home/chen/runs/cpt-world-grpo-etv2-10k}
MAX_STEPS=${MAX_STEPS:-10000}
MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-32768}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-31497}
VLLM_MEMORY_UTILIZATION=${VLLM_MEMORY_UTILIZATION:-0.50}
VLLM_ROLLOUT_RESIDENCY=${VLLM_ROLLOUT_RESIDENCY:-1}
VLLM_ENABLE_PREFIX_CACHING=${VLLM_ENABLE_PREFIX_CACHING:-1}
VLLM_MTP_SPECULATIVE_TOKENS=${VLLM_MTP_SPECULATIVE_TOKENS:-0}
VLLM_SLEEP_LEVEL=${VLLM_SLEEP_LEVEL:-1}
CAPTURE_ROLLOUTS=${CAPTURE_ROLLOUTS:-0}
SAVE_STEPS=${SAVE_STEPS:-50}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-200}
FLA_KERNEL_DIR=${FLA_KERNEL_DIR:-/home/chen/kernels/fla-v1-398dfa8c}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}

PROJECT_DIR=$(realpath "$PROJECT_DIR")
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export CPT_WORLD_EXPECTED_SOURCE="$PROJECT_DIR/src/cpt_world"

"$ENV_DIR/bin/python" - <<'PY'
import json
import os
from fractions import Fraction
from pathlib import Path

import cpt_world
from cpt_world import (
    TASK_FAMILY_QUERY_TYPES,
    TERMINAL_QUALITY_REWARD_VERSION,
    task_advantage_utility,
    terminal_quality_reward,
)

expected_source = Path(os.environ["CPT_WORLD_EXPECTED_SOURCE"]).resolve()
loaded_source = Path(cpt_world.__file__).resolve()
if expected_source not in loaded_source.parents:
    raise RuntimeError(
        f"cpt_world import escaped the requested project: {loaded_source} not under "
        f"{expected_source}"
    )
if TERMINAL_QUALITY_REWARD_VERSION != "terminal-quality-v9":
    raise RuntimeError(
        "training requires terminal-quality-v9, got "
        f"{TERMINAL_QUALITY_REWARD_VERSION}"
    )
for query_type in TASK_FAMILY_QUERY_TYPES:
    for quality in (0.0, 0.25, 0.75, 1.0):
        if task_advantage_utility(quality, query_type) != quality:
            raise RuntimeError(
                f"training utility altered {query_type} terminal quality {quality}"
            )
backdoor_contract = {
    "exact": terminal_quality_reward(
        {"kind": "backadj", "edit_distance": 0}
    ),
    "one_edit": terminal_quality_reward(
        {"kind": "backadj", "edit_distance": 1}
    ),
    "two_edits": terminal_quality_reward(
        {"kind": "backadj", "edit_distance": 2}
    ),
}
if backdoor_contract != {
    "exact": Fraction(1),
    "one_edit": Fraction(1, 2),
    "two_edits": Fraction(1, 3),
}:
    raise RuntimeError(f"unexpected backdoor reward contract: {backdoor_contract}")
print(
    "CPT_WORLD_PREFLIGHT="
    + json.dumps(
        {
            "source": str(loaded_source),
            "reward_version": TERMINAL_QUALITY_REWARD_VERSION,
            "task_families": list(TASK_FAMILY_QUERY_TYPES),
            "utility": "identity",
            "backdoor_reward": {
                key: str(value) for key, value in backdoor_contract.items()
            },
        },
        separators=(",", ":"),
    ),
    flush=True,
)
PY

mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Keep Liger's chunked loss active while allowing PyTorch to fall back to
# eager execution for an individual dynamic shape if Dynamo itself fails.
export TORCHDYNAMO_SUPPRESS_ERRORS=1

nvidia-smi \
  --query-gpu=timestamp,temperature.gpu,memory.used,memory.free,utilization.gpu,power.draw \
  --format=csv,noheader,nounits \
  --loop=1 >"$RUN_DIR/gpu-telemetry.csv" &
MONITOR_PID=$!

cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

resume_args=()
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  resume_args+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

rollout_acceleration_args=(--vllm-sleep-level "$VLLM_SLEEP_LEVEL")
if [[ "$VLLM_ROLLOUT_RESIDENCY" == "1" ]]; then
  rollout_acceleration_args+=(--vllm-rollout-residency)
fi
if [[ "$VLLM_ENABLE_PREFIX_CACHING" == "1" ]]; then
  rollout_acceleration_args+=(--vllm-enable-prefix-caching)
elif [[ "$VLLM_ENABLE_PREFIX_CACHING" == "0" ]]; then
  rollout_acceleration_args+=(--no-vllm-enable-prefix-caching)
elif [[ "$VLLM_ENABLE_PREFIX_CACHING" != "default" ]]; then
  echo "VLLM_ENABLE_PREFIX_CACHING must be default, 0, or 1" >&2
  exit 2
fi
if [[ "$VLLM_MTP_SPECULATIVE_TOKENS" != "0" ]]; then
  rollout_acceleration_args+=(--vllm-mtp-speculative-tokens "$VLLM_MTP_SPECULATIVE_TOKENS")
fi
if [[ "$CAPTURE_ROLLOUTS" == "1" ]]; then
  rollout_acceleration_args+=(--capture-rollouts)
fi

cd "$PROJECT_DIR"
"$ENV_DIR/bin/python" scripts/train_grpo_resource_smoke.py \
  --model "$MODEL_DIR" \
  --output-dir "$RUN_DIR/checkpoints" \
  --max-steps "$MAX_STEPS" \
  --max-completion-length "$MAX_COMPLETION_LENGTH" \
  --max-model-length "$MAX_MODEL_LENGTH" \
  --vllm-memory-utilization "$VLLM_MEMORY_UTILIZATION" \
  "${rollout_acceleration_args[@]}" \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit "$SAVE_TOTAL_LIMIT" \
  --fla-kernel-dir "$FLA_KERNEL_DIR" \
  --use-liger-kernel \
  "${resume_args[@]}" \
  2>&1 | tee "$RUN_DIR/train.log"
