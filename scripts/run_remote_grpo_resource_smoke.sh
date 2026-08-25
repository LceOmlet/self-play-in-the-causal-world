#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${1:-/home/chen/projects/self-play-in-the-causal-world}
MODEL_DIR=${2:-/home/chen/models/Qwen/Qwen3.5-9B}
ENV_DIR=${3:-/home/chen/.venvs/dolens-rl}
RUN_DIR=${4:-/home/chen/runs/cpt-world-grpo-resource-smoke}
MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-9216}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-7168}
VLLM_MEMORY_UTILIZATION=${VLLM_MEMORY_UTILIZATION:-0.50}
FLA_KERNEL_DIR=${FLA_KERNEL_DIR:-/home/chen/kernels/fla-v1-398dfa8c}

mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

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

cd "$PROJECT_DIR"
"$ENV_DIR/bin/python" scripts/train_grpo_resource_smoke.py \
  --model "$MODEL_DIR" \
  --output-dir "$RUN_DIR/checkpoint" \
  --max-steps 1 \
  --max-completion-length "$MAX_COMPLETION_LENGTH" \
  --max-model-length "$MAX_MODEL_LENGTH" \
  --vllm-memory-utilization "$VLLM_MEMORY_UTILIZATION" \
  --fla-kernel-dir "$FLA_KERNEL_DIR" \
  --use-liger-kernel \
  2>&1 | tee "$RUN_DIR/train.log"
