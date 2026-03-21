"""
Training loop — single authoritative entry point per training mode.

train_one_epoch                — generic per-epoch loop (vanilla PyTorch)
train_and_evaluate_model_1d    — 1D GMM probabilistic training + validation
train_and_evaluate_model_2d    — 2D GMM vertex training + validation
train_and_evaluate_model_hard  — hard label training + validation
"""

from softlabel_od.object_detection.core.training.train_epoch import train_one_epoch
from softlabel_od.object_detection.core.training.train_1d import train_and_evaluate_model_1d
from softlabel_od.object_detection.core.training.train_2d import train_and_evaluate_model_2d
from softlabel_od.object_detection.core.training.train_hard import train_and_evaluate_model_hard
