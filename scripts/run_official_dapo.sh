#!/usr/bin/env bash
set -euo pipefail

export CPT_WORLD_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export VERL_ROOT="${VERL_ROOT:-/home/chen/vendor/dapo-official-20260906/verl-tool-termination-v1}"
export VERL_RECIPE_ROOT="${VERL_RECIPE_ROOT:-/home/chen/vendor/dapo-official-20260906/verl-recipe-mask-v1}"
DAPO_PYTHON="${DAPO_PYTHON:-/home/chen/.venvs/dolens-dapo-official/bin/python}"
export CPT_WORLD_MODEL="${CPT_WORLD_MODEL:-/home/chen/models/Qwen/Qwen3.5-9B}"
if [[ -n "${CPT_WORLD_INITIAL_ADAPTER:-}" || -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  echo "Historical RL checkpoints are disqualified. Unset checkpoint initialization/resume variables and start from the original base model." >&2
  exit 2
fi
: "${CPT_WORLD_TRAIN_DATA:?Set the continuous training stream descriptor JSON path}"
: "${CPT_WORLD_VAL_DATA:?Set the prepared validation parquet path}"
: "${CPT_WORLD_RUN_DIR:?Set an isolated output directory}"
export CPT_WORLD_TRAIN_DATA CPT_WORLD_VAL_DATA CPT_WORLD_RUN_DIR
export CPT_WORLD_VERL_AUDIT_DIR="${CPT_WORLD_RUN_DIR}/environment"
export CPT_WORLD_EXPECTED_SOURCE="${CPT_WORLD_PROJECT}/src/cpt_world"
export PYTHONPATH="${CPT_WORLD_PROJECT}/src:${VERL_RECIPE_ROOT}:${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
mkdir -p -- "${CPT_WORLD_RUN_DIR}"
"${DAPO_PYTHON}" "${CPT_WORLD_PROJECT}/scripts/verify_official_dapo.py" \
  --output "${CPT_WORLD_RUN_DIR}/official-source-verification.json"

# The official module owns the entire training loop. This launcher only passes
# the project configuration and the caller's explicit Hydra overrides.
cd -- "${VERL_ROOT}"
exec "${DAPO_PYTHON}" -m dapo.main_dapo \
  "hydra.searchpath=[file://${VERL_ROOT}/verl/trainer/config,file://${CPT_WORLD_PROJECT}/configs/verl]" \
  +profiles@_global_=cpt_world_dapo "$@"
