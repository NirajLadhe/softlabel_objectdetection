#!/usr/bin/env bash
set -euo pipefail

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/haaroon/softlabel_object_detection"
VENV="${PROJECT_ROOT}/.venv"

TEST_DIR="/media/klee_ro/public_datasets/vindr_cxr/test"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/vindrcxr/Kaggle_Submissions"
LOG_DIR="${PROJECT_ROOT}/outputs/kaggle_inference_logs"

SOFT_WEIGHTS_DIR="${PROJECT_ROOT}/outputs/saved_models/vindrcxr/1d_gaussian_softlabels"
HARD_WEIGHTS_DIR="${PROJECT_ROOT}/outputs/saved_models/vindrcxr/hardlabels"

# Optional: enforce Kaggle row order / catch missing IDs
# SAMPLE_SUBMISSION="/path/to/sample_submission.csv"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ── run definitions (parallel arrays) ──────────────────────────────────────
RUN_TAGS=(
  "s-r50-iomin-cg"
  "s-r50-iomin"
  "s-r50-iou"
  "s-r101-iou"
  "h-r50-iomin-cg"
  "h-r50-iomin"
  "h-r50-iou"
  "h-r101-iou"
)

CONFIGS=(
  "${PROJECT_ROOT}/configs/vindrcxr_s-r50-iomin-cg.yaml"
  "${PROJECT_ROOT}/configs/vindrcxr_s-r50-iomin.yaml"
  "${PROJECT_ROOT}/configs/vindrcxr_s-r50-iou.yaml"
  "${PROJECT_ROOT}/configs/vindrcxr_s-r101-iou.yaml"
  "${PROJECT_ROOT}/configs/vindrcxr_h-r50-iomin-cg.yaml"
  "${PROJECT_ROOT}/configs/vindrcxr_h-r50-iomin.yaml"
  "${PROJECT_ROOT}/configs/vindrcxr_h-r50-iou.yaml"
  "${PROJECT_ROOT}/configs/vindrcxr_h-r101-iou.yaml"
)

WEIGHTS=(
  "${SOFT_WEIGHTS_DIR}/vindrcxr_s-r50-iomin-cg_30Epoch.pth"
  "${SOFT_WEIGHTS_DIR}/vindrcxr_s-r50-iomin_30Epoch.pth"
  "${SOFT_WEIGHTS_DIR}/vindrcxr_s-r50-iou_30Epoch.pth"
  "${SOFT_WEIGHTS_DIR}/vindrcxr_s-r101-iou_30Epoch.pth"
  "${HARD_WEIGHTS_DIR}/vindrcxr_h-r50-iomin-cg_30Epoch.pth"
  "${HARD_WEIGHTS_DIR}/vindrcxr_h-r50-iomin_30Epoch.pth"
  "${HARD_WEIGHTS_DIR}/vindrcxr_h-r50-iou_30Epoch.pth"
  "${HARD_WEIGHTS_DIR}/vindrcxr_h-r101-iou_30Epoch.pth"
)

OUTPUTS=(
  "${OUTPUT_DIR}/submission_s-r50-iomin-cg.csv"
  "${OUTPUT_DIR}/submission_s-r50-iomin.csv"
  "${OUTPUT_DIR}/submission_s-r50-iou.csv"
  "${OUTPUT_DIR}/submission_s-r101-iou.csv"
  "${OUTPUT_DIR}/submission_h-r50-iomin-cg.csv"
  "${OUTPUT_DIR}/submission_h-r50-iomin.csv"
  "${OUTPUT_DIR}/submission_h-r50-iou.csv"
  "${OUTPUT_DIR}/submission_h-r101-iou.csv"
)

# ── environment ────────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"
source "${VENV}/bin/activate"

# Optional: restrict GPUs
# export CUDA_VISIBLE_DEVICES=0

TOTAL="${#RUN_TAGS[@]}"

echo "============================================================"
echo "Kaggle inference started at $(date)"
echo "Test dir:    ${TEST_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Total runs:  ${TOTAL}"
for i in "${!RUN_TAGS[@]}"; do
  echo "  [$((i + 1))/${TOTAL}] ${RUN_TAGS[$i]}"
done
echo "============================================================"

# ── inference runs ─────────────────────────────────────────────────────────
for i in "${!RUN_TAGS[@]}"; do
  RUN_NUM="$((i + 1))"
  RUN_TAG="${RUN_TAGS[$i]}"
  CONFIG="${CONFIGS[$i]}"
  WEIGHT="${WEIGHTS[$i]}"
  OUTPUT="${OUTPUTS[$i]}"
  LOG_FILE="${LOG_DIR}/kaggle_infer_${RUN_TAG}_${TIMESTAMP}.log"

  if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config not found: ${CONFIG}"
    exit 1
  fi
  if [[ ! -d "${TEST_DIR}" ]]; then
    echo "ERROR: Test directory not found: ${TEST_DIR}"
    exit 1
  fi
  if [[ ! -f "${WEIGHT}" ]]; then
    echo "ERROR: Weights not found: ${WEIGHT}"
    exit 1
  fi

  echo ""
  echo "[${RUN_NUM}/${TOTAL}] Running Kaggle inference: ${RUN_TAG}"
  echo "Config:  ${CONFIG}"
  echo "Weights: ${WEIGHT}"
  echo "Output:  ${OUTPUT}"
  echo "Log:     ${LOG_FILE}"

  CMD=(
    python "${PROJECT_ROOT}/src/softlabel_od/scripts/infer_kaggle.py"
    --config "${CONFIG}"
    --weights "${WEIGHT}"
    --test-dir "${TEST_DIR}"
    --output "${OUTPUT}"
    --batch-size 4
    --num-workers 4
  )

  # Uncomment if you have Kaggle sample_submission.csv
  # CMD+=(--sample-submission "${SAMPLE_SUBMISSION}")

  "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"

  EXIT_CODE="${PIPESTATUS[0]}"
  if [[ "${EXIT_CODE}" -ne 0 ]]; then
    echo "ERROR: Inference failed for ${RUN_TAG} (exit code ${EXIT_CODE}). Aborting."
    exit "${EXIT_CODE}"
  fi

  echo "[${RUN_NUM}/${TOTAL}] Finished successfully at $(date): ${RUN_TAG}"
done

echo ""
echo "============================================================"
echo "All ${TOTAL} Kaggle inference runs completed at $(date)"
echo "Submissions: ${OUTPUT_DIR}/"
echo "Logs:        ${LOG_DIR}/"
echo "============================================================"