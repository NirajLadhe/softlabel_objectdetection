#!/usr/bin/env bash
set -euo pipefail

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/haaroon/softlabel_object_detection"
VENV="${PROJECT_ROOT}/.venv"

WEIGHTS_DIR="${PROJECT_ROOT}/outputs/saved_models/vindrcxr/hardlabels"
LABELS_DIR="${PROJECT_ROOT}/outputs/vindrcxr/hard_labels"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/vindrcxr/evaluation_results"

# Shared hard-label training config (no per-run h-* yaml configs in repo)
HARD_CONFIG="${PROJECT_ROOT}/configs/vindrcxr_hardlabel_config_albers.yaml"

LOG_DIR="${PROJECT_ROOT}/outputs/evaluation_logs"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

METRIC_TYPE=("default" "tide" "lrp")

# ── run definitions (parallel arrays) ──────────────────────────────────────
EVAL_TAGS=(
  "h-r50-iomin-cg-30Epoch"
  "h-r50-iomin-30Epoch"
  "h-r50-iou-30Epoch"
  "h-r101-iou-30Epoch"
)

WEIGHTS=(
  "${WEIGHTS_DIR}/vindrcxr_h-r50-iomin-cg_30Epoch.pth"
  "${WEIGHTS_DIR}/vindrcxr_h-r50-iomin_30Epoch.pth"
  "${WEIGHTS_DIR}/vindrcxr_h-r50-iou_30Epoch.pth"
  "${WEIGHTS_DIR}/vindrcxr_h-r101-iou_30Epoch.pth"
)

LABEL_FILES=(
  "${LABELS_DIR}/HardLabels_MajorityVote_iomin_center_gate.json"
  "${LABELS_DIR}/HardLabels_MajorityVote_iomin.json"
  "${LABELS_DIR}/HardLabels_MajorityVote_iou0_5.json"
  "${LABELS_DIR}/HardLabels_MajorityVote_iou0_5.json"
)

BACKBONES=(
  "resnet50"
  "resnet50"
  "resnet50"
  "resnet101"
)

# ── environment ────────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"
source "${VENV}/bin/activate"

# Optional: restrict GPUs (uncomment if needed)
# export CUDA_VISIBLE_DEVICES=0

TOTAL="${#EVAL_TAGS[@]}"

echo "============================================================"
echo "Sequential HARD-LABEL evaluation started at $(date)"
echo "Metric type: ${METRIC_TYPE[@]}"
echo "Config:      ${HARD_CONFIG}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Total runs:  ${TOTAL}"
for i in "${!EVAL_TAGS[@]}"; do
  echo "  [$((i + 1))/${TOTAL}] ${EVAL_TAGS[$i]}"
done
echo "============================================================"

# ── evaluation runs ────────────────────────────────────────────────────────
for i in "${!EVAL_TAGS[@]}"; do
  RUN_NUM="$((i + 1))"
  EVAL_TAG="${EVAL_TAGS[$i]}"
  WEIGHT="${WEIGHTS[$i]}"
  LABEL_FILE="${LABEL_FILES[$i]}"
  BACKBONE="${BACKBONES[$i]}"
  LOG_FILE="${LOG_DIR}/hardlabel_eval_${EVAL_TAG}_${TIMESTAMP}.log"

  if [[ ! -f "${WEIGHT}" ]]; then
    echo "ERROR: Weights not found: ${WEIGHT}"
    exit 1
  fi
  if [[ ! -f "${LABEL_FILE}" ]]; then
    echo "ERROR: Label file not found: ${LABEL_FILE}"
    exit 1
  fi

  echo ""
  echo "[${RUN_NUM}/${TOTAL}] Evaluating: ${EVAL_TAG}"
  echo "Config:     ${HARD_CONFIG}"
  echo "Weights:    ${WEIGHT}"
  echo "Label file: ${LABEL_FILE}"
  echo "Backbone:   ${BACKBONE}"
  echo "Log:        ${LOG_FILE}"

  softlabel-evaluate \
    --metric-type "${METRIC_TYPE[@]}" \
    --eval-tag "${EVAL_TAG}" \
    --config "${HARD_CONFIG}" \
    --weights "${WEIGHT}" \
    --label-file "${LABEL_FILE}" \
    --backbone "${BACKBONE}" \
    --gmm-type HARD_LABEL \
    2>&1 | tee "${LOG_FILE}"

  EXIT_CODE="${PIPESTATUS[0]}"
  if [[ "${EXIT_CODE}" -ne 0 ]]; then
    echo "ERROR: Evaluation failed for ${EVAL_TAG} (exit code ${EXIT_CODE}). Aborting."
    exit "${EXIT_CODE}"
  fi

  echo "[${RUN_NUM}/${TOTAL}] Finished successfully at $(date): ${EVAL_TAG}"
  echo "  Results: ${OUTPUT_DIR}/eval_results_${EVAL_TAG}.json"
done

echo ""
echo "============================================================"
echo "All ${TOTAL} hard-label evaluations completed at $(date)"
echo "Results: ${OUTPUT_DIR}/"
echo "============================================================"