#!/usr/bin/env bash
set -euo pipefail

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/haaroon/softlabel_object_detection"
VENV="${PROJECT_ROOT}/.venv"

WEIGHTS_DIR="${PROJECT_ROOT}/outputs/saved_models/vindrcxr/1d_gaussian_softlabels"
LABELS_DIR="${PROJECT_ROOT}/outputs/vindrcxr/1d_gaussian_softlabels"
PRIORS_DIR="${LABELS_DIR}/priors"
OUTPUT_DIR="${LABELS_DIR}/evaluation_results"

LOG_DIR="${PROJECT_ROOT}/outputs/evaluation_logs"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# evaluate.py accepts one metric or "all" (default + tide + lrp + pdq in one JSON)
METRIC_TYPE=("default" "tide" "lrp")

# ── run definitions (parallel arrays) ──────────────────────────────────────
EVAL_TAGS=(
  "s-r50-iomin-cg-30Epoch"
  "s-r50-iomin-30Epoch"
  "s-r50-iou-30Epoch"
  "s-r101-iou-30Epoch"
)

CONFIGS=(
  "${PROJECT_ROOT}/configs/vindrcxr_s-r50-iomin-cg.yaml"
  "${PROJECT_ROOT}/configs/vindrcxr_s-r50-iomin.yaml"
  "${PROJECT_ROOT}/configs/vindrcxr_s-r50-iou.yaml"
  "${PROJECT_ROOT}/configs/vindrcxr_s-r101-iou.yaml"
)

WEIGHTS=(
  "${WEIGHTS_DIR}/vindrcxr_s-r50-iomin-cg_30Epoch.pth"
  "${WEIGHTS_DIR}/vindrcxr_s-r50-iomin_30Epoch.pth"
  "${WEIGHTS_DIR}/vindrcxr_s-r50-iou_30Epoch.pth"
  "${WEIGHTS_DIR}/vindrcxr_s-r101-iou_30Epoch.pth"
)

LABEL_FILES=(
  "${LABELS_DIR}/1DGMM_softLabels_iomin_cg0_5_0_5_with_priors.json"
  "${LABELS_DIR}/1DGMM_softLabels_iomin0_5_with_priors.json"
  "${LABELS_DIR}/1DGMM_softLabels_iou0_5_with_priors.json"
  "${LABELS_DIR}/1DGMM_softLabels_iou0_5_with_priors.json"
)

PRIORS_FILES=(
  "${PRIORS_DIR}/class_priors_1d_iomin_cg0_5_0_5.json"
  "${PRIORS_DIR}/class_priors_1d_iomin0_5.json"
  "${PRIORS_DIR}/class_priors_1d_iou0_5.json"
  "${PRIORS_DIR}/class_priors_1d_iou0_5.json"
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
echo "Sequential SOFT-LABEL evaluation started at $(date)"
echo "Metric type: ${METRIC_TYPE[@]}"
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
  CONFIG="${CONFIGS[$i]}"
  WEIGHT="${WEIGHTS[$i]}"
  LABEL_FILE="${LABEL_FILES[$i]}"
  PRIORS_FILE="${PRIORS_FILES[$i]}"
  BACKBONE="${BACKBONES[$i]}"
  LOG_FILE="${LOG_DIR}/softlabel_eval_${EVAL_TAG}_${TIMESTAMP}.log"

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
  echo "Config:      ${CONFIG}"
  echo "Weights:     ${WEIGHT}"
  echo "Label file:  ${LABEL_FILE}"
  echo "Priors file: ${PRIORS_FILE}"
  echo "Backbone:    ${BACKBONE}"
  echo "Log:         ${LOG_FILE}"

  softlabel-evaluate \
    --metric-type "${METRIC_TYPE[@]}" \
    --eval-tag "${EVAL_TAG}" \
    --config "${CONFIG}" \
    --weights "${WEIGHT}" \
    --label-file "${LABEL_FILE}" \
    --priors-file "${PRIORS_FILE}" \
    --backbone "${BACKBONE}" \
    --gmm-type 1D \
    --no-finding-eval-policy drop \
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
echo "All ${TOTAL} soft-label evaluations completed at $(date)"
echo "Results: ${OUTPUT_DIR}/"
echo "============================================================"