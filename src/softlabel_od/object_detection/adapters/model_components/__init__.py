"""
Model component adapters: custom heads, RoI heads, and backbone wrappers.

faster_rcnn_1d  — Probabilistic Faster R-CNN with 1D GMM head (primary model).
faster_rcnn_2d  — Faster R-CNN with 2D vertex-based GMM head.

Adding a new architecture
--------------------------
Implement a get_model_<name>() factory function and register it in
object_detection/core/model/factory.py.
"""
