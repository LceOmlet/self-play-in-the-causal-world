#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${1:?project directory is required}
MODEL_DIR=${2:?model directory is required}
ENV_DIR=${3:?virtual environment directory is required}
RUN_DIR=${4:?run directory is required}
MODE=${5:?mode must be kv or train}
TOTAL_LENGTH=${6:?total sequence length is required}
FLA_KERNEL_DIR=${FLA_KERNEL_DIR:-/home/chen/kernels/fla-v1-398dfa8c}
VLLM_MEMORY_UTILIZATION=${VLLM_MEMORY_UTILIZATION:-0.50}

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
"$ENV_DIR/bin/python" scripts/stress_grpo_context.py "$MODE" \
  --model "$MODEL_DIR" \
  --output-dir "$RUN_DIR/output" \
  --total-length "$TOTAL_LENGTH" \
  --fla-kernel-dir "$FLA_KERNEL_DIR" \
  --vllm-memory-utilization "$VLLM_MEMORY_UTILIZATION" \
  2>&1 | tee "$RUN_DIR/stress.log"
