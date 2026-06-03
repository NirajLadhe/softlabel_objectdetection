#!/usr/bin/env bash
set -euo pipefail

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/haaroon/softlabel_object_detection"
VENV="${PROJECT_ROOT}/.venv"

HARD_CONFIG="${PROJECT_ROOT}/configs/vindrcxr_hardlabel_config_albers.yaml"
SOFT_CONFIG="${PROJECT_ROOT}/configs/vindrcxr_1d_softlabel_config_albers.yaml"

LOG_DIR="${PROJECT_ROOT}/outputs/training_logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
HARD_LOG="${LOG_DIR}/hardlabel_${TIMESTAMP}.log"
SOFT_LOG="${LOG_DIR}/softlabel_1d_${TIMESTAMP}.log"

# ── environment ────────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"
source "${VENV}/bin/activate"

# Optional: restrict GPUs (uncomment if needed)
# export CUDA_VISIBLE_DEVICES=0,1

echo "============================================================"
echo "Sequential training started at $(date)"
echo "Hard config: ${HARD_CONFIG}"
echo "Soft config: ${SOFT_CONFIG}"
echo "============================================================"

# ── 1) Hard-label training ─────────────────────────────────────────────────
echo ""
echo "[1/2] Starting HARD-LABEL training..."
echo "Log: ${HARD_LOG}"

softlabel-train-lightning --config "${HARD_CONFIG}" 2>&1 | tee "${HARD_LOG}"
HARD_EXIT="${PIPESTATUS[0]}"

if [[ "${HARD_EXIT}" -ne 0 ]]; then
  echo "ERROR: Hard-label training failed (exit code ${HARD_EXIT}). Aborting."
  exit "${HARD_EXIT}"
fi

echo ""
echo "[1/2] Hard-label training finished successfully at $(date)"
echo "  Models:  ${PROJECT_ROOT}/outputs/saved_models/vindrcxr/hardlabels/"
echo "  Metrics: ${PROJECT_ROOT}/outputs/vindrcxr/training_metric_logs/"

# ── 2) Soft-label (1D) training ────────────────────────────────────────────
echo ""
echo "[2/2] Starting SOFT-LABEL (1D) training..."
echo "Log: ${SOFT_LOG}"

softlabel-train-lightning --config "${SOFT_CONFIG}" 2>&1 | tee "${SOFT_LOG}"
SOFT_EXIT="${PIPESTATUS[0]}"

if [[ "${SOFT_EXIT}" -ne 0 ]]; then
  echo "ERROR: Soft-label training failed (exit code ${SOFT_EXIT})."
  exit "${SOFT_EXIT}"
fi

echo ""
echo "[2/2] Soft-label training finished successfully at $(date)"
echo "  Models:  ${PROJECT_ROOT}/outputs/saved_models/vindrcxr/1d_gaussian_softlabels/"
echo "  Metrics: ${PROJECT_ROOT}/outputs/training_metric_logs/"

echo ""
echo "============================================================"
echo "All training runs completed at $(date)"
echo "============================================================"