"""
Evaluator adapters.

evaluate_model  — standalone evaluation script for trained models.

Adding a new evaluator
-----------------------
Implement a function that accepts (model, data_loader, device) and returns
a metrics dict. Register it here.
"""

from softlabel_od.object_detection.adapters.evaluators.evaluate_model import evaluate, load_model
