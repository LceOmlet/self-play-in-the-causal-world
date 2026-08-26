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
SAVE_STEPS=${SAVE_STEPS:-50}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}
REWARD_UTILITY_EPSILON=${REWARD_UTILITY_EPSILON:-0.02}
FLA_KERNEL_DIR=${FLA_KERNEL_DIR:-/home/chen/kernels/fla-v1-398dfa8c}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}

mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Keep Liger's chunked loss active while allowing PyTorch to fall back to
# eager execution for an individual dynamic shape if Dynamo itself fails.
export TORCHDYNAMO_SUPPRESS_ERRORS=1

nvidia-smi \
  --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,power.draw \
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

cd "$PROJECT_DIR"
"$ENV_DIR/bin/python" scripts/train_grpo_resource_smoke.py \
  --model "$MODEL_DIR" \
  --output-dir "$RUN_DIR/checkpoints" \
  --max-steps "$MAX_STEPS" \
  --max-completion-length "$MAX_COMPLETION_LENGTH" \
  --max-model-length "$MAX_MODEL_LENGTH" \
  --vllm-memory-utilization "$VLLM_MEMORY_UTILIZATION" \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit "$SAVE_TOTAL_LIMIT" \
  --reward-utility-epsilon "$REWARD_UTILITY_EPSILON" \
  --fla-kernel-dir "$FLA_KERNEL_DIR" \
  --use-liger-kernel \
  "${resume_args[@]}" \
  2>&1 | tee "$RUN_DIR/train.log"
