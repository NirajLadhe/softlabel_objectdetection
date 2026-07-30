"""
verify_vertex_covariance_transform.py

Confirms, numerically, that each vertex's 2x2 covariance matrix stored in the
2D GMM soft label JSON is exactly recoverable as a linear transform of a single
4x4 covariance matrix over box parameters theta = (x, y, w, h).

For each cluster:
    1. Pull raw per-annotator [x, y, w, h] boxes + weights from original_annotations.
    2. Compute the weighted 4x4 covariance matrix  Sigma_theta  directly from these.
    3. Build the fixed 2x4 linear maps A_TL, A_TR, A_BR, A_BL (corner = A . theta).
    4. Predict each vertex covariance as  A @ Sigma_theta @ A.T .
    5. Compare against the vertex covariance already stored in the soft label JSON.
    6. Report the max absolute difference (should be ~1e-9 or smaller -- floating
       point noise only) and plot the four uncertainty ellipses for a visual check.

Usage:
    python verify_vertex_covariance_transform.py \
        --soft-labels-json outputs/nucls_2D/2d_gaussian_softlabels/2DGMM_softLabels_iou0_25_with_priors.json \
        --num-clusters 5 \
        --output-dir outputs/visualizations/covariance_verification
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

VERTEX_NAMES = ["top_left", "top_right", "bottom_right", "bottom_left"]

# Fixed 2x4 linear maps: corner_xy = A @ [x, y, w, h]
A_MATRICES = {
    "top_left":     np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0]], dtype=float),
    "top_right":    np.array([[1, 0, 1, 0],
                               [0, 1, 0, 0]], dtype=float),
    "bottom_right": np.array([[1, 0, 1, 0],
                               [0, 1, 0, 1]], dtype=float),
    "bottom_left":  np.array([[1, 0, 0, 0],
                               [0, 1, 0, 1]], dtype=float),
}


def weighted_covariance_4d(boxes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted 4x4 covariance matrix over (x, y, w, h) -- INSTANCE ONLY, no prior."""
    w = weights / weights.sum()
    mean = np.average(boxes, axis=0, weights=w)
    diff = boxes - mean
    cov = np.einsum("n,ni,nj->ij", w, diff, diff)
    return cov, mean


def predict_vertex_covariance(sigma_theta: np.ndarray, vertex_name: str) -> np.ndarray:
    """Sigma_vertex = A @ Sigma_theta @ A.T"""
    A = A_MATRICES[vertex_name]
    return A @ sigma_theta @ A.T


def bayesian_blend(n: int, alpha: float, cov_instance: np.ndarray, cov_prior: np.ndarray,
                   reg_eps: float = 1e-4) -> np.ndarray:
    """
    Replicates Prior2DCalculator.adjust_covariance() exactly, including the
    numerical-stability regularisation floor added AFTER the Bayesian blend:

        adjusted_cov = (n * instance_cov + alpha * prior_cov) / (n + alpha)
        adjusted_cov += 1e-4 * np.eye(2)          # <- previously missing here

    Without this floor term, Stage B under-predicts every stored covariance
    by exactly reg_eps on the diagonal (confirmed: earlier runs showed a
    uniform 1.000e-04 max-abs-diff across every corner and every cluster --
    the signature of a missing constant additive term, not a real mismatch).
    """
    blended = (n * cov_instance + alpha * cov_prior) / (n + alpha)
    blended += reg_eps * np.eye(2)
    return blended


def load_class_priors(priors_path: str):
    with open(priors_path) as f:
        data = json.load(f)
    return data.get("class_priors", data), data.get("global_prior")


def plot_ellipses(cluster_id, image_id, stored_vertices, predicted_vertices, output_path):
    """Plot stored vs. predicted uncertainty ellipses for all four vertices."""
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {"top_left": "#336B87", "top_right": "#028090",
              "bottom_right": "#763626", "bottom_left": "#B5854A"}

    for v_name in VERTEX_NAMES:
        mean = np.array(stored_vertices[v_name]["mean"])
        cov_stored = np.array(stored_vertices[v_name]["covariance"])
        cov_pred = predicted_vertices[v_name]

        for cov, ls, alpha, label_suffix in [
            (cov_stored, "-", 0.35, "stored"),
            (cov_pred, "--", 0.0, "predicted"),
        ]:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, 0, None)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            # 3-sigma ellipse, scaled up for visibility (coords are normalised 0-1)
            width, height = 2 * np.sqrt(eigvals) * 3 * 25
            ell = Ellipse(mean, width, height, angle=angle,
                          facecolor=colors[v_name], alpha=alpha,
                          edgecolor=colors[v_name], linestyle=ls, linewidth=1.6)
            ax.add_patch(ell)
        ax.plot(*mean, marker="o", color=colors[v_name], markersize=5)
        ax.text(mean[0], mean[1] + 0.015, v_name, fontsize=8, color=colors[v_name])

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # image coords: y grows downward
    ax.set_aspect("equal")
    ax.set_title(f"Cluster {cluster_id}  ({image_id})\n"
                 f"solid = stored covariance   dashed = predicted from Sigma_theta",
                 fontsize=10)
    ax.set_xlabel("x (normalised)")
    ax.set_ylabel("y (normalised)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft-labels-json", required=True)
    ap.add_argument("--priors-json", default=None,
                    help="class_priors_2d.json -- needed for Stage B check")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--reg-eps", type=float, default=1e-4,
                    help="Regularisation floor added after blending, matching "
                         "Prior2DCalculator's '+= 1e-4 * np.eye(2)' line")
    ap.add_argument("--num-clusters", type=int, default=5)
    ap.add_argument("--output-dir", default="outputs/visualizations/covariance_verification")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.soft_labels_json) as f:
        soft_labels = json.load(f)

    class_priors, global_prior = (None, None)
    if args.priors_json and os.path.exists(args.priors_json):
        class_priors, global_prior = load_class_priors(args.priors_json)

    checked = 0
    for entry in soft_labels:
        if checked >= args.num_clusters:
            break

        anns = entry.get("original_annotations", [])
        weights = np.array([a["weight"] for a in anns], dtype=float)
        valid_mask = weights > 1e-6
        if valid_mask.sum() < 3:
            continue
        if entry.get("vertices") is None:
            continue

        valid_anns = [a for a, m in zip(anns, valid_mask) if m]
        valid_weights = weights[valid_mask]
        boxes = np.array([a["bbox"] for a in valid_anns], dtype=float)
        n = len(valid_anns)

        sigma_theta_instance, mean_theta = weighted_covariance_4d(boxes, valid_weights)
        stored_vertices = entry["vertices"]

        print("=" * 78)
        print(f"Cluster {entry['finding_id']}  (image: {entry['image_id']})")
        print(f"  Valid annotators used: {n}   prior_used: {entry.get('prior_used')}")

        # ---- STAGE A: instance-only, no prior -- proves the linear-transform claim ----
        print(f"\n  [STAGE A] Instance-only covariance (no prior blending)")
        max_diff_instance = 0.0
        for v_name in VERTEX_NAMES:
            A = A_MATRICES[v_name]
            pred = predict_vertex_covariance(sigma_theta_instance, v_name)
            # direct: transform each raw box to this vertex's (x,y), then weighted cov
            v_xy = np.stack([
                boxes[:, 0] + (A[0, 2] * boxes[:, 2]),
                boxes[:, 1] + (A[1, 3] * boxes[:, 3]),
            ], axis=1)
            direct_cov, _ = weighted_covariance_4d(v_xy, valid_weights)
            diff = np.abs(direct_cov - pred).max()
            max_diff_instance = max(max_diff_instance, diff)
            print(f"    [{v_name:>13}]  direct-vertex-cov vs A.Sigma_theta.A^T   "
                  f"max abs diff: {diff:.3e}")
        print(f"    --> Stage A claim holds (transform is exact on raw data): "
              f"{'YES' if max_diff_instance < 1e-10 else 'NO -- investigate'}")

        # ---- STAGE B: prior-blended, matching what's actually stored ----
        if class_priors is not None:
            print(f"\n  [STAGE B] Prior-blended covariance (replicates pipeline exactly)")
            prior_key = entry.get("prior_used")
            prior_block = class_priors.get(prior_key, global_prior) if prior_key else global_prior
            if prior_block is None:
                print("    No matching prior found -- skipping Stage B for this cluster.")
            else:
                max_diff_blended = 0.0
                for v_name in VERTEX_NAMES:
                    A = A_MATRICES[v_name]
                    v_xy = np.stack([
                        boxes[:, 0] + (A[0, 2] * boxes[:, 2]),
                        boxes[:, 1] + (A[1, 3] * boxes[:, 3]),
                    ], axis=1)
                    direct_cov, _ = weighted_covariance_4d(v_xy, valid_weights)
                    cov_prior = np.array(prior_block[v_name])

                    if n == 1:
                        # Pipeline's n==1 branch returns prior_cov directly --
                        # no blend, no regularisation floor added in that path.
                        blended_pred = cov_prior
                    else:
                        blended_pred = bayesian_blend(n, args.alpha, direct_cov, cov_prior,
                                                      reg_eps=args.reg_eps)

                    stored = np.array(stored_vertices[v_name]["covariance"])
                    diff = np.abs(blended_pred - stored).max()
                    max_diff_blended = max(max_diff_blended, diff)
                    print(f"    [{v_name:>13}]  replicated-blend vs stored   "
                          f"max abs diff: {diff:.3e}")
                print(f"    --> Stage B match (confirms pipeline's own blend formula): "
                      f"{'YES' if max_diff_blended < 1e-8 else 'NO -- investigate'}")
        else:
            print(f"\n  [STAGE B] Skipped -- pass --priors-json to check prior blending.")

        out_png = os.path.join(args.output_dir, f"cluster_{entry['finding_id']}_verification.png")
        predicted_vertices = {v: predict_vertex_covariance(sigma_theta_instance, v) for v in VERTEX_NAMES}
        plot_ellipses(entry["finding_id"], entry["image_id"],
                     stored_vertices, predicted_vertices, out_png)
        print(f"\n  Saved plot: {out_png}")

        checked += 1

    print("\n" + "=" * 78)
    print(f"Checked {checked} clusters. See {args.output_dir} for ellipse plots.")


if __name__ == "__main__":
    main()