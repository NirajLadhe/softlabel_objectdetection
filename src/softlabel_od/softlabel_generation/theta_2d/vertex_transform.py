"""
vertex_transform.py

Shared, single-source-of-truth implementation of the theta -> vertex
covariance transform proved and empirically verified earlier:

    v_k = A_k . theta                  (k in {top_left, top_right, bottom_right, bottom_left})
    Sigma_{v_k} = A_k . Sigma_theta . A_k^T

Every script that needs to go from a 4x4 Sigma_theta to a vertex's 2x2
covariance (or the reverse: build Sigma_theta from raw annotator boxes)
should import from here, rather than redefining the A matrices locally.
This avoids the two independent implementations drifting apart, which is
exactly the kind of bug the earlier Stage A / Stage B verification exists
to catch.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

VERTEX_NAMES = ["top_left", "top_right", "bottom_right", "bottom_left"]
THETA_NAMES = ["x", "y", "w", "h"]

# Fixed 2x4 linear maps: corner_xy = A_k . [x, y, w, h]
A_MATRICES: Dict[str, np.ndarray] = {
    "top_left":     np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0]], dtype=float),
    "top_right":    np.array([[1, 0, 1, 0],
                               [0, 1, 0, 0]], dtype=float),
    "bottom_right": np.array([[1, 0, 1, 0],
                               [0, 1, 0, 1]], dtype=float),
    "bottom_left":  np.array([[1, 0, 0, 0],
                               [0, 1, 0, 1]], dtype=float),
}


def weighted_mean_cov_theta(
    boxes: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the weighted mean and 4x4 covariance of theta = (x, y, w, h)
    directly from raw annotator boxes. This is THE canonical instance-level
    estimate -- verified (Stage A) to transform exactly into every vertex
    covariance via sigma_v_from_theta() below, to floating-point precision.

    Args:
        boxes:   [N, 4] array of raw annotator boxes, columns = x, y, w, h
        weights: [N]    array of per-annotator reliability weights (already
                        filtered to > 1e-6 by the caller)

    Returns:
        (mean_theta [4], sigma_theta [4,4])
    """
    w = weights / weights.sum()
    mean_theta = np.average(boxes, axis=0, weights=w)
    diff = boxes - mean_theta
    sigma_theta = np.einsum("n,ni,nj->ij", w, diff, diff)
    return mean_theta, sigma_theta


def sigma_v_from_theta(sigma_theta: np.ndarray, vertex_name: str) -> np.ndarray:
    """Sigma_vertex = A_k . Sigma_theta . A_k^T -- the proved, verified transform."""
    A = A_MATRICES[vertex_name]
    return A @ sigma_theta @ A.T


def mean_v_from_theta(mean_theta: np.ndarray, vertex_name: str) -> np.ndarray:
    """corner_mean = A_k . mean_theta"""
    A = A_MATRICES[vertex_name]
    return A @ mean_theta


def all_vertex_covariances(sigma_theta: np.ndarray) -> Dict[str, np.ndarray]:
    """Convenience: project one Sigma_theta into all four vertex covariances at once."""
    return {v: sigma_v_from_theta(sigma_theta, v) for v in VERTEX_NAMES}


def all_vertex_means(mean_theta: np.ndarray) -> Dict[str, np.ndarray]:
    return {v: mean_v_from_theta(mean_theta, v) for v in VERTEX_NAMES}


def sigma_theta_to_named_dict(sigma_theta: np.ndarray) -> Dict[str, float]:
    """
    Unpack the 10 unique entries of a 4x4 Sigma_theta into a named dict,
    matching the specification document's naming exactly. Useful for
    logging, JSON serialisation, and documentation cross-checks.
    """
    return {
        "Var(x)": float(sigma_theta[0, 0]),
        "Var(y)": float(sigma_theta[1, 1]),
        "Var(w)": float(sigma_theta[2, 2]),
        "Var(h)": float(sigma_theta[3, 3]),
        "Cov(x,y)": float(sigma_theta[0, 1]),
        "Cov(x,w)": float(sigma_theta[0, 2]),
        "Cov(x,h)": float(sigma_theta[0, 3]),
        "Cov(y,w)": float(sigma_theta[1, 2]),
        "Cov(y,h)": float(sigma_theta[1, 3]),
        "Cov(w,h)": float(sigma_theta[2, 3]),
    }


def named_dict_to_sigma_theta(d: Dict[str, float]) -> np.ndarray:
    """Inverse of sigma_theta_to_named_dict -- rebuild the full 4x4 matrix."""
    a, b, c, dd = d["Var(x)"], d["Var(y)"], d["Var(w)"], d["Var(h)"]
    p, q, r = d["Cov(x,y)"], d["Cov(x,w)"], d["Cov(x,h)"]
    s, t, u = d["Cov(y,w)"], d["Cov(y,h)"], d["Cov(w,h)"]
    return np.array([
        [a, p, q, r],
        [p, b, s, t],
        [q, s, c, u],
        [r, t, u, dd],
    ], dtype=float)
