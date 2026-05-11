# pdq.py

import numpy as np
from tqdm import tqdm

try:
    from scipy.optimize import linear_sum_assignment
    from scipy.special import ndtr  # Normal CDF
except ImportError as e:
    raise ImportError(
        "PDQEvaluator requires scipy. Install it with: pip install scipy"
    ) from e


class PDQEvaluator:
    """
    Minimal dataset-agnostic PDQ evaluator.

    Expected prediction format per image:
    {
        "boxes": np.ndarray[N, 4],        # x1, y1, x2, y2
        "scores": np.ndarray[N],          # optional, not directly used here
        "labels": np.ndarray[N],          # optional, not directly used here
        "bbox_stds": np.ndarray[N, 4],    # dx, dy, dw, dh stds
        "class_probs": np.ndarray[N, C],  # class probabilities
        "image_size": optional
    }

    Expected ground truth format per image:
    {
        "boxes": np.ndarray[M, 4],
        "labels": np.ndarray[M],
        "image_size": (H, W)
    }
    """

    def __init__(self, eps=1e-6, min_std=1.0):
        self.eps = eps
        self.min_std = min_std

    def _convert_stds_to_pixel_space(self, boxes, bbox_stds):
        """
        Convert bbox stds from box-regression delta space to pixel space.

        bbox_stds are assumed to be:
        [std_dx, std_dy, std_dw, std_dh]

        Output:
        [std_x1, std_y1, std_x2, std_y2]
        """

        boxes = np.asarray(boxes, dtype=np.float32)
        bbox_stds = np.asarray(bbox_stds, dtype=np.float32)

        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        pred_w = np.maximum(boxes[:, 2] - boxes[:, 0], 1.0)
        pred_h = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)

        std_dx = bbox_stds[:, 0]
        std_dy = bbox_stds[:, 1]
        std_dw = bbox_stds[:, 2]
        std_dh = bbox_stds[:, 3]

        sigma_cx = std_dx * pred_w
        sigma_cy = std_dy * pred_h
        sigma_w = std_dw * pred_w
        sigma_h = std_dh * pred_h

        std_x1 = np.sqrt(sigma_cx ** 2 + 0.25 * sigma_w ** 2)
        std_y1 = np.sqrt(sigma_cy ** 2 + 0.25 * sigma_h ** 2)
        std_x2 = std_x1.copy()
        std_y2 = std_y1.copy()

        pixel_stds = np.stack([std_x1, std_y1, std_x2, std_y2], axis=1)

        pixel_stds = np.maximum(pixel_stds, self.min_std)

        return pixel_stds.astype(np.float32)

    def _pixel_probability_map(self, box, box_std, image_size):
        """
        Builds a spatial probability map for one predicted box.

        Each pixel receives the probability that it lies inside the uncertain box.

        box:
            [x1, y1, x2, y2]

        box_std:
            [std_x1, std_y1, std_x2, std_y2]

        image_size:
            (H, W)
        """

        if image_size is None:
            raise ValueError(
                "image_size is missing. Use gt.get('image_size') in runner/evaluation."
            )

        H, W = image_size
        H = int(H)
        W = int(W)

        x1, y1, x2, y2 = box.astype(np.float32)
        std_x1, std_y1, std_x2, std_y2 = box_std.astype(np.float32)

        std_x1 = max(float(std_x1), self.min_std)
        std_y1 = max(float(std_y1), self.min_std)
        std_x2 = max(float(std_x2), self.min_std)
        std_y2 = max(float(std_y2), self.min_std)

        xs = np.arange(W, dtype=np.float32) + 0.5
        ys = np.arange(H, dtype=np.float32) + 0.5

        prob_left = ndtr((xs - x1) / std_x1)
        prob_right = ndtr((x2 - xs) / std_x2)

        prob_top = ndtr((ys - y1) / std_y1)
        prob_bottom = ndtr((y2 - ys) / std_y2)

        prob_x = prob_left * prob_right
        prob_y = prob_top * prob_bottom

        prob_map = prob_y[:, None] * prob_x[None, :]

        prob_map = np.clip(prob_map, self.eps, 1.0 - self.eps)

        return prob_map.astype(np.float32)

    def _gt_box_to_mask(self, gt_box, image_size):
        """
        Converts one GT bbox into a binary mask.
        """

        H, W = image_size
        H = int(H)
        W = int(W)

        x1, y1, x2, y2 = gt_box.astype(np.float32)

        xs = np.arange(W, dtype=np.float32) + 0.5
        ys = np.arange(H, dtype=np.float32) + 0.5

        x_mask = (xs >= x1) & (xs < x2)
        y_mask = (ys >= y1) & (ys < y2)

        mask = y_mask[:, None] & x_mask[None, :]

        return mask

    def _compute_pairwise_pdq(
        self,
        pred_box,
        pred_box_std,
        pred_class_probs,
        gt_box,
        gt_label,
        image_size,
    ):
        """
        Computes PDQ score between one prediction and one ground-truth object.

        PDQ = sqrt(spatial_quality * label_quality)
        """

        prob_map = self._pixel_probability_map(
            pred_box,
            pred_box_std,
            image_size,
        )

        gt_mask = self._gt_box_to_mask(gt_box, image_size)

        gt_area = int(gt_mask.sum())

        if gt_area == 0:
            return 0.0, 0.0, 0.0

        prob_map = np.clip(prob_map, self.eps, 1.0 - self.eps)

        fg_probs = np.clip(
            prob_map[gt_mask].astype(np.float64),
            self.eps,
            1.0 - self.eps,
        )

        bg_probs = np.clip(
            prob_map[~gt_mask].astype(np.float64),
            self.eps,
            1.0 - self.eps,
        )

        fg_loss = -np.sum(np.log(fg_probs)) / gt_area
        bg_loss = -np.sum(np.log1p(-bg_probs)) / gt_area

        spatial_quality = float(np.exp(-(fg_loss + bg_loss)))
        spatial_quality = float(np.clip(spatial_quality, 0.0, 1.0))

        gt_label = int(gt_label)

        if gt_label < 0 or gt_label >= len(pred_class_probs):
            label_quality = 0.0
        else:
            label_quality = float(pred_class_probs[gt_label])

        label_quality = float(np.clip(label_quality, 0.0, 1.0))

        pdq_score = float(np.sqrt(spatial_quality * label_quality))

        return pdq_score, spatial_quality, label_quality

    def _evaluate_single_image(self, pred, gt):
        """
        Evaluates PDQ for one image.
        """

        pred_boxes = np.asarray(pred.get("boxes", []), dtype=np.float32)
        pred_stds = np.asarray(pred.get("bbox_stds", []), dtype=np.float32)
        class_probs = np.asarray(pred.get("class_probs", []), dtype=np.float32)

        gt_boxes = np.asarray(gt.get("boxes", []), dtype=np.float32)
        gt_labels = np.asarray(gt.get("labels", []), dtype=np.int64)

        image_size = gt.get("image_size")

        if image_size is None:
            image_size = pred.get("image_size")

        if image_size is None:
            raise ValueError(
                "Missing image_size. Expected gt['image_size'] = (H, W). "
                "Do not hard-code image size."
            )

        num_pred = len(pred_boxes)
        num_gt = len(gt_boxes)

        result = {
            "pdq_sum": 0.0,
            "spatial_sum": 0.0,
            "label_sum": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }

        if num_gt == 0 and num_pred == 0:
            return result

        if num_gt == 0:
            result["fp"] = num_pred
            return result

        if num_pred == 0:
            result["fn"] = num_gt
            return result

        if pred_stds.shape[0] != num_pred:
            raise ValueError(
                f"bbox_stds count mismatch: got {pred_stds.shape[0]}, "
                f"expected {num_pred}"
            )

        if class_probs.shape[0] != num_pred:
            raise ValueError(
                f"class_probs count mismatch: got {class_probs.shape[0]}, "
                f"expected {num_pred}"
            )

        pixel_stds = self._convert_stds_to_pixel_space(pred_boxes, pred_stds)

        pdq_matrix = np.zeros((num_gt, num_pred), dtype=np.float32)
        spatial_matrix = np.zeros((num_gt, num_pred), dtype=np.float32)
        label_matrix = np.zeros((num_gt, num_pred), dtype=np.float32)

        for gt_idx in range(num_gt):
            for pred_idx in range(num_pred):
                pdq_score, spatial_quality, label_quality = self._compute_pairwise_pdq(
                    pred_box=pred_boxes[pred_idx],
                    pred_box_std=pixel_stds[pred_idx],
                    pred_class_probs=class_probs[pred_idx],
                    gt_box=gt_boxes[gt_idx],
                    gt_label=gt_labels[gt_idx],
                    image_size=image_size,
                )

                pdq_matrix[gt_idx, pred_idx] = pdq_score
                spatial_matrix[gt_idx, pred_idx] = spatial_quality
                label_matrix[gt_idx, pred_idx] = label_quality

        cost_matrix = -pdq_matrix

        matched_gt_indices, matched_pred_indices = linear_sum_assignment(cost_matrix)

        matched_gt = set()
        matched_pred = set()

        for gt_idx, pred_idx in zip(matched_gt_indices, matched_pred_indices):
            score = float(pdq_matrix[gt_idx, pred_idx])

            if score <= 0.0:
                continue

            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)

            result["pdq_sum"] += score
            result["spatial_sum"] += float(spatial_matrix[gt_idx, pred_idx])
            result["label_sum"] += float(label_matrix[gt_idx, pred_idx])
            result["tp"] += 1

        result["fp"] += num_pred - len(matched_pred)
        result["fn"] += num_gt - len(matched_gt)

        return result

    def evaluate(self, predictions, ground_truths):
        """
        Evaluates all images.

        Returns:
        {
            "pdq": float,
            "avg_pPDQ": float,
            "avg_spatial_quality": float,
            "avg_label_quality": float,
            "tp": int,
            "fp": int,
            "fn": int
        }
        """

        if len(predictions) != len(ground_truths):
            raise ValueError(
                f"Number of predictions and ground truths must match. "
                f"Got {len(predictions)} predictions and {len(ground_truths)} GT entries."
            )

        total_pdq_sum = 0.0
        total_spatial_sum = 0.0
        total_label_sum = 0.0

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred, gt in tqdm(
            zip(predictions, ground_truths),
            total=len(predictions),
            desc="Computing PDQ",
        ):
            image_result = self._evaluate_single_image(pred, gt)

            total_pdq_sum += image_result["pdq_sum"]
            total_spatial_sum += image_result["spatial_sum"]
            total_label_sum += image_result["label_sum"]

            total_tp += image_result["tp"]
            total_fp += image_result["fp"]
            total_fn += image_result["fn"]

        denominator = total_tp + total_fp + total_fn

        if denominator == 0:
            pdq = 0.0
        else:
            pdq = total_pdq_sum / denominator

        if total_tp == 0:
            avg_pPDQ = 0.0
            avg_spatial_quality = 0.0
            avg_label_quality = 0.0
        else:
            avg_pPDQ = total_pdq_sum / total_tp
            avg_spatial_quality = total_spatial_sum / total_tp
            avg_label_quality = total_label_sum / total_tp

        return {
            "pdq": float(pdq),
            "avg_pPDQ": float(avg_pPDQ),
            "avg_spatial_quality": float(avg_spatial_quality),
            "avg_label_quality": float(avg_label_quality),
            "tp": int(total_tp),
            "fp": int(total_fp),
            "fn": int(total_fn),
        }