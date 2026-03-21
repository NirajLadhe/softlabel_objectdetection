"""
softlabel_generation
====================
Pre-training pipeline: annotation clustering, inter-annotator reliability,
class-specific priors, and GMM-based soft label generation.

Sub-packages
------------
core        — dataset-agnostic algorithms (reliability, label assembly, preprocessing)
ports       — abstract interfaces (CorrespondenceSolver)
adapters    — dataset-specific extensions (VinDr-CXR, add new datasets here)
utils       — shared helpers (AHC clustering variants)
"""
