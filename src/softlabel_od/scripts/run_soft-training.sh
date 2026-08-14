#!/usr/bin/env bash
set -euo pipefail

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/haaroon/softlabel_object_detection"
VENV="${PROJECT_ROOT}/.venv"

CONFIGS=(
  "${PROJECT_ROOT}/configs/vindrcxr_s-r50-iomin-cg.yaml"
  "${PROJECT_ROOT}/configs/vindrcxr_s-r50-iomin.yaml"
  "${PROJECT_ROOT}/configs/vindrcxr_s-r50-iou-.yaml"
  "${PROJECT_ROOT}/configs/vindrcxr_s-r101-iou.yaml"
)

RUN_NAMES=(
  "vindrcxr_s-r50-iomin-cg"
  "vindrcxr_s-r50-iomin"
  "vindrcxr_s-r50-iou"
  "vindrcxr_s-r101-iou"
)

LOG_DIR="${PROJECT_ROOT}/outputs/training_logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ── environment ────────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"
source "${VENV}/bin/activate"

# Optional: restrict GPUs (uncomment if needed)
# export CUDA_VISIBLE_DEVICES=0,1

TOTAL="${#CONFIGS[@]}"

echo "============================================================"
echo "Sequential soft-label training started at $(date)"
echo "Total runs: ${TOTAL}"
for i in "${!CONFIGS[@]}"; do
  echo "  [$((i + 1))/${TOTAL}] ${CONFIGS[$i]}"
done
echo "============================================================"

# ── soft-label training runs ───────────────────────────────────────────────
for i in "${!CONFIGS[@]}"; do
  RUN_NUM="$((i + 1))"
  CONFIG="${CONFIGS[$i]}"
  RUN_NAME="${RUN_NAMES[$i]}"
  LOG_FILE="${LOG_DIR}/softlabel_${RUN_NAME}_${TIMESTAMP}.log"

  echo ""
  echo "[${RUN_NUM}/${TOTAL}] Starting SOFT-LABEL training: ${RUN_NAME}"
  echo "Config: ${CONFIG}"
  echo "Log:    ${LOG_FILE}"

  softlabel-train-lightning --config "${CONFIG}" 2>&1 | tee "${LOG_FILE}"
  EXIT_CODE="${PIPESTATUS[0]}"

  if [[ "${EXIT_CODE}" -ne 0 ]]; then
    echo "ERROR: Soft-label training failed for ${RUN_NAME} (exit code ${EXIT_CODE}). Aborting."
    exit "${EXIT_CODE}"
  fi

  echo ""
  echo "[${RUN_NUM}/${TOTAL}] Finished successfully at $(date): ${RUN_NAME}"
done

echo ""
echo "============================================================"
echo "All ${TOTAL} soft-label training runs completed at $(date)"
echo "  Models:  ${PROJECT_ROOT}/outputs/saved_models/vindrcxr/1d_gaussian_softlabels/"
echo "  Metrics: ${PROJECT_ROOT}/outputs/training_metric_logs/"
echo "============================================================"