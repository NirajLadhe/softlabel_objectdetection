#!/usr/bin/env bash
# =============================================================================
# run_eval_hardlabel.sh
#   Waits for the hard-label training to finish, then evaluates the best
#   checkpoint with all 4 metric groups (default, tide, lrp, pdq).
#
#   Each metric runs as a SEPARATE invocation so:
#     - one failure (e.g. PDQ on the non-probabilistic hard model) does not
#       abort the others, and
#     - output files match the v5/v6/v7 naming the visualization tooling reads:
#         eval_results_hardlabel_v1_best_{default,tide,lrp,pdq}.json
#
#   Usage:
#     bash run_eval_hardlabel.sh [GPU_INDEX]      # default GPU 1
#   Run detached so it survives terminal close:
#     nohup bash run_eval_hardlabel.sh 1 > outputs/eval_hardlabel_v1.log 2>&1 &
# =============================================================================
set -u

PROJ=/home/niraj/softlabel_objectdetection
cd "$PROJ" || exit 1

GPU=${1:-1}                                   # GPU index (first arg, default 1)
RUN_NAME=lvis_hardlabel_v1_resnet50
TAG=hardlabel_v1_best

WEIGHTS=$PROJ/outputs/saved_models/lvis/1d_gaussian_softlabels/${RUN_NAME}_best_model.pth
LABEL_FILE=$PROJ/outputs/lvis/hard_labels/LVIS_HardLabels_val.json
PRIORS=$PROJ/outputs/lvis/1d_gaussian_softlabels/priors/class_priors_1d_iou0_5.json
OUT=$PROJ/outputs/lvis/1d_gaussian_softlabels/evaluation_results
# evaluate.py does not pass image_subdir, and the adapter joins root + "" + file;
# so DATA_ROOT must include the val2017 subdir (training passes it via OPTIONS instead).
DATA_ROOT=/media/klee_ro/public_datasets/coco_lvis/val2017

TRAIN_MATCH="softlabel-train-lightning --config configs/config_lvis.yaml"

# ── 1) wait for training to finish ──────────────────────────────────────────
echo "[watch] $(date) waiting for training to finish ..."
while pgrep -f "$TRAIN_MATCH" >/dev/null; do
    sleep 60
done
echo "[watch] $(date) training process is gone."

# ── 2) verify the best checkpoint exists ────────────────────────────────────
if [[ ! -f "$WEIGHTS" ]]; then
    echo "[error] best model not found: $WEIGHTS" >&2
    exit 1
fi
echo "[ok] evaluating: $WEIGHTS"

# ── 3) run each metric separately (failures are non-fatal) ──────────────────
for METRIC in default tide lrp pdq; do
    echo "============================================================"
    echo "[eval] $(date)  metric=$METRIC"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=$GPU softlabel-evaluate \
        --config       configs/config_lvis.yaml \
        --eval-tag     ${TAG}_${METRIC} \
        --metric-type  $METRIC \
        --weights      "$WEIGHTS" \
        --gmm-type     HARD_LABEL \
        --dataset      lvis \
        --label-file   "$LABEL_FILE" \
        --priors-file  "$PRIORS" \
        --data-root    "$DATA_ROOT" \
        --output       "$OUT" \
        --num-classes  825 \
        --backbone     resnet50 \
        --test-split-ratio 0.20 \
        || echo "[warn] metric=$METRIC FAILED (continuing with the rest)"
done

echo "[done] $(date) eval files in: $OUT/eval_results_${TAG}_*.json"
