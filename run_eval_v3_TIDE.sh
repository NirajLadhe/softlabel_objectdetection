#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python src/softlabel_od/scripts/evaluate.py \
    --config configs/nucls_1d_softlabel_config.yaml \
    --weights outputs/saved_models/nucls_transform_0.001/1d_gaussian_softlabels/nucls_softlabels_transform_0001_200epochs_soft_transform_13_best_model.pth \
    --gmm-type 1D \
    --dataset nucls \
    --label-file outputs/nucls_transform_0.001/1d_gaussian_softlabels/1DGMM_softLabels_iou0_25_with_priors.json \
    --priors-file outputs/nucls_transform_0.001/1d_gaussian_softlabels/priors/class_priors_1d_iou0_25.json \
    --data-root /media/klee_ro/public_datasets/nucls \
    --output outputs/nucls_transform_0.001/evaluation_results/soft_transform_14 \
    --num-classes 14 \
    --test-split-ratio 0.10\
    --val-split-ratio 0.10 \
    --eval-tag nucls_v13_TIDE \
    --metric-type tide \
    2>&1 | tee outputs/nucls_transform_0.001/evaluation_results/soft_transform_14/soft_transform_14_TIDE_eval_log.log
