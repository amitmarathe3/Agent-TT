"""Bounding box IoU computation and greedy score-ordered matching.

This module provides model-agnostic matching of predicted to ground-truth
bounding boxes at a given IoU threshold, in both class-aware and
class-agnostic modes.

Prediction format (per image): list of [x, y, w, h, score, class_id]
Ground-truth format (per image): list of [x, y, w, h, class_id]
All coordinates are in absolute pixels (xywh format).
"""

from __future__ import annotations

from typing import NamedTuple
import numpy as np


# ---------------------------------------------------------------------------
# Core IoU
# ---------------------------------------------------------------------------

def compute_iou_xywh(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes in [x, y, w, h] format.

    Args:
        box_a: Array [x, y, w, h].
        box_b: Array [x, y, w, h].

    Returns:
        Scalar IoU in [0, 1].
    """
    ax1, ay1 = float(box_a[0]), float(box_a[1])
    ax2, ay2 = ax1 + float(box_a[2]), ay1 + float(box_a[3])
    bx1, by1 = float(box_b[0]), float(box_b[1])
    bx2, by2 = bx1 + float(box_b[2]), by1 + float(box_b[3])

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = float(box_a[2]) * float(box_a[3])
    area_b = float(box_b[2]) * float(box_b[3])
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def compute_iou_matrix(preds_xywh: np.ndarray, gts_xywh: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix between predictions and ground-truth boxes.

    Args:
        preds_xywh: Array of shape [P, 4] with columns [x, y, w, h].
        gts_xywh: Array of shape [G, 4] with columns [x, y, w, h].

    Returns:
        IoU matrix of shape [P, G].
    """
    if len(preds_xywh) == 0 or len(gts_xywh) == 0:
        return np.zeros((len(preds_xywh), len(gts_xywh)), dtype=np.float32)

    # Convert to x1y1x2y2
    p = np.array(preds_xywh, dtype=np.float32)
    g = np.array(gts_xywh, dtype=np.float32)

    px1 = p[:, 0]
    py1 = p[:, 1]
    px2 = p[:, 0] + p[:, 2]
    py2 = p[:, 1] + p[:, 3]
    area_p = p[:, 2] * p[:, 3]

    gx1 = g[:, 0]
    gy1 = g[:, 1]
    gx2 = g[:, 0] + g[:, 2]
    gy2 = g[:, 1] + g[:, 3]
    area_g = g[:, 2] * g[:, 3]

    # Broadcasting: [P, G]
    ix1 = np.maximum(px1[:, None], gx1[None, :])
    iy1 = np.maximum(py1[:, None], gy1[None, :])
    ix2 = np.minimum(px2[:, None], gx2[None, :])
    iy2 = np.minimum(py2[:, None], gy2[None, :])

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    union = area_p[:, None] + area_g[None, :] - inter
    union = np.maximum(union, 1e-9)
    return (inter / union).astype(np.float32)


# ---------------------------------------------------------------------------
# Match result types
# ---------------------------------------------------------------------------

class MatchResult(NamedTuple):
    """Result of matching predictions to ground truths for one image.

    Attributes:
        tp_pred_indices: Indices into preds that are TPs.
        tp_gt_indices:   Corresponding GT indices for each TP.
        tp_ious:         IoU value for each matched pair.
        fp_pred_indices: Indices into preds that are FPs.
        fn_gt_indices:   Indices into GTs that are unmatched (FNs).
    """
    tp_pred_indices: list[int]
    tp_gt_indices:   list[int]
    tp_ious:         list[float]
    fp_pred_indices: list[int]
    fn_gt_indices:   list[int]


# ---------------------------------------------------------------------------
# Greedy score-ordered matching
# ---------------------------------------------------------------------------

def match_predictions(
    preds: list[list[float]],
    gts: list[list[float]],
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> MatchResult:
    """Greedy, score-ordered (descending) matching of predictions to ground truths.

    Each GT box may only be matched once. Each prediction may only be matched once.
    Predictions are sorted by score (descending) before matching.

    Args:
        preds: List of [x, y, w, h, score, class_id] per predicted box.
        gts:   List of [x, y, w, h, class_id] per ground-truth box.
        iou_threshold: Minimum IoU for a positive match (default 0.5).
        class_agnostic: If True, class labels are ignored when matching.

    Returns:
        MatchResult with TP/FP/FN index lists and matched IoUs.
    """
    if len(preds) == 0 and len(gts) == 0:
        return MatchResult([], [], [], [], [])
    if len(preds) == 0:
        return MatchResult([], [], [], [], list(range(len(gts))))
    if len(gts) == 0:
        return MatchResult([], [], [], list(range(len(preds))), [])

    preds_arr = np.array(preds, dtype=np.float64)  # [P, 6]
    gts_arr = np.array(gts, dtype=np.float64)       # [G, 5]

    # Sort predictions by score descending
    order = np.argsort(-preds_arr[:, 4])
    preds_sorted = preds_arr[order]
    original_pred_idx = order.tolist()

    iou_mat = compute_iou_matrix(preds_sorted[:, :4], gts_arr[:, :4])

    matched_gt = set()
    tp_pred_indices: list[int] = []
    tp_gt_indices:   list[int] = []
    tp_ious:         list[float] = []
    fp_pred_indices: list[int] = []

    for i in range(len(preds_sorted)):
        pred_cls = int(preds_sorted[i, 5])
        best_iou = -1.0
        best_gt = -1

        for j in range(len(gts_arr)):
            if j in matched_gt:
                continue
            if not class_agnostic and int(gts_arr[j, 4]) != pred_cls:
                continue
            if iou_mat[i, j] >= iou_threshold and iou_mat[i, j] > best_iou:
                best_iou = iou_mat[i, j]
                best_gt = j

        orig_i = original_pred_idx[i]
        if best_gt >= 0:
            matched_gt.add(best_gt)
            tp_pred_indices.append(orig_i)
            tp_gt_indices.append(best_gt)
            tp_ious.append(float(best_iou))
        else:
            fp_pred_indices.append(orig_i)

    fn_gt_indices = [j for j in range(len(gts_arr)) if j not in matched_gt]

    return MatchResult(
        tp_pred_indices=tp_pred_indices,
        tp_gt_indices=tp_gt_indices,
        tp_ious=tp_ious,
        fp_pred_indices=fp_pred_indices,
        fn_gt_indices=fn_gt_indices,
    )


def match_dataset(
    predictions: dict[str, list[list[float]]],
    ground_truths: dict[str, list[list[float]]],
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> dict[str, MatchResult]:
    """Apply per-image greedy matching across the full dataset.

    Args:
        predictions: Dict mapping image_id -> list of [x,y,w,h,score,class_id].
        ground_truths: Dict mapping image_id -> list of [x,y,w,h,class_id].
        iou_threshold: IoU threshold for positive matching.
        class_agnostic: If True, class labels are ignored when matching.

    Returns:
        Dict mapping image_id -> MatchResult.
    """
    all_ids = set(predictions.keys()) | set(ground_truths.keys())
    results: dict[str, MatchResult] = {}

    for img_id in all_ids:
        preds = predictions.get(img_id, [])
        gts = ground_truths.get(img_id, [])
        results[img_id] = match_predictions(
            preds, gts,
            iou_threshold=iou_threshold,
            class_agnostic=class_agnostic,
        )

    return results
