#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python src/softlabel_od/scripts/evaluate.py \
    --config configs/nucls_hardlabel_config.yaml \
    --weights outputs/saved_models/nucls_transform_0.001/hard_labels/nucls_hardlabels_iou0_25_100epochs_v1.pth \
    --gmm-type HARD_LABEL \
    --dataset nucls \
    --label-file outputs/nucls_transform_0.001/hard_labels/HardLabels_MajorityVote_iou.json \
    --data-root /media/klee_ro/public_datasets/nucls \
    --output outputs/nucls_transform_0.001/evaluation_results/hard_transform_2 \
    --num-classes 14 \
    --backbone resnet50 \
    --test-split-ratio 0.10 \
    --val-split-ratio 0.10 \
    --eval-tag nucls_hard_v1_lrp \
    --metric-type lrp
    2>&1 | tee outputs/nucls_transform_0.001/evaluation_results/hard_transform_2/hard_transform_2_lrp_eval_log.log