"""Model-agnostic object detection metrics for CADICA coronary angiography.

All metric functions accept predictions and ground truths as plain Python
lists in the format:
    predictions: dict[image_id -> list of [x, y, w, h, score, class_id]]
    ground_truths: dict[image_id -> list of [x, y, w, h, class_id]]
Coordinates are in absolute pixels (xywh). class_id is an integer.

Key outputs:
- Per-class and aggregate TP/FP/FN, Precision, Recall, F1
- AP per class (COCO 101-point interpolation), mAP@0.5, mAP@0.5:0.95
- Confusion matrix (classes + background)
- Frame-level classification metrics with ROC/PR curves
- Patient-level aggregation
- Bootstrap 95% confidence intervals by patient
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.matching import compute_iou_matrix, match_predictions, match_dataset

# NumPy 2.0 renamed trapz -> trapezoid; support both.
_trapezoid = getattr(np, "trapezoid", np.trapz)


# ---------------------------------------------------------------------------
# COCO-style 101-point AP interpolation
# ---------------------------------------------------------------------------

def _interpolate_ap_101(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using 101-point COCO-style interpolation.

    Args:
        recalls: Array of recall values in ascending order.
        precisions: Array of precision values corresponding to recalls.

    Returns:
        Average Precision (AP) as a float in [0, 1].
    """
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    ap = 0.0
    for thr in recall_thresholds:
        prec_at_thr = precisions[recalls >= thr]
        ap += float(np.max(prec_at_thr)) if len(prec_at_thr) > 0 else 0.0
    return ap / 101.0


# ---------------------------------------------------------------------------
# Precision-Recall curve builder
# ---------------------------------------------------------------------------

def build_pr_curve(
    all_scores: np.ndarray,
    all_tp_flags: np.ndarray,
    n_gt: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build precision-recall curve by sweeping score threshold.

    Args:
        all_scores: Confidence scores for each prediction [P].
        all_tp_flags: Boolean TP flag for each prediction [P].
        n_gt: Total number of ground-truth boxes for this class.

    Returns:
        (recalls, precisions) arrays sorted by ascending recall.
    """
    if len(all_scores) == 0 or n_gt == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0])

    order = np.argsort(-all_scores)
    tp_cumsum = np.cumsum(all_tp_flags[order].astype(float))
    fp_cumsum = np.cumsum((~all_tp_flags[order]).astype(float))

    recalls = tp_cumsum / max(n_gt, 1)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)

    # Prepend (0, 1) and append (1, 0) for completeness
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])
    return recalls, precisions


# ---------------------------------------------------------------------------
# Per-class AP computation
# ---------------------------------------------------------------------------

def compute_class_ap(
    predictions: dict[str, list[list[float]]],
    ground_truths: dict[str, list[list[float]]],
    class_id: int,
    iou_threshold: float = 0.5,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute Average Precision for a single class at a given IoU threshold.

    Args:
        predictions: Dict image_id -> [[x,y,w,h,score,class_id], ...].
        ground_truths: Dict image_id -> [[x,y,w,h,class_id], ...].
        class_id: Integer class ID to evaluate.
        iou_threshold: IoU threshold for positive matching.

    Returns:
        (ap, recalls, precisions) tuple.
    """
    # Collect all predictions for this class across images
    all_scores: list[float] = []
    all_tp_flags: list[bool] = []
    n_gt_total = 0

    all_img_ids = set(predictions.keys()) | set(ground_truths.keys())

    for img_id in all_img_ids:
        preds = predictions.get(img_id, [])
        gts = ground_truths.get(img_id, [])

        # Filter to this class
        cls_preds = [p for p in preds if int(p[5]) == class_id]
        cls_gts = [g for g in gts if int(g[4]) == class_id]
        n_gt_total += len(cls_gts)

        if len(cls_preds) == 0:
            continue

        if len(cls_gts) == 0:
            # All predictions are FP
            for p in cls_preds:
                all_scores.append(float(p[4]))
                all_tp_flags.append(False)
            continue

        # Score-ordered matching for this image (class-agnostic since already filtered)
        result = match_predictions(cls_preds, cls_gts, iou_threshold=iou_threshold, class_agnostic=True)

        tp_set = set(result.tp_pred_indices)
        for i, p in enumerate(cls_preds):
            all_scores.append(float(p[4]))
            all_tp_flags.append(i in tp_set)

    if len(all_scores) == 0 or n_gt_total == 0:
        return 0.0, np.array([0.0, 1.0]), np.array([1.0, 0.0])

    recalls, precisions = build_pr_curve(
        np.array(all_scores, dtype=np.float64),
        np.array(all_tp_flags, dtype=bool),
        n_gt_total,
    )
    ap = _interpolate_ap_101(recalls, precisions)
    return ap, recalls, precisions


# ---------------------------------------------------------------------------
# mAP @ 0.5 and mAP @ 0.5:0.95
# ---------------------------------------------------------------------------

def compute_map(
    predictions: dict[str, list[list[float]]],
    ground_truths: dict[str, list[list[float]]],
    class_ids: list[int],
    iou_threshold: float = 0.5,
) -> dict[int, float]:
    """Compute per-class AP at a single IoU threshold.

    Args:
        predictions: Dict image_id -> [[x,y,w,h,score,class_id], ...].
        ground_truths: Dict image_id -> [[x,y,w,h,class_id], ...].
        class_ids: List of integer class IDs.
        iou_threshold: IoU threshold.

    Returns:
        Dict class_id -> AP.
    """
    return {
        cls: compute_class_ap(predictions, ground_truths, cls, iou_threshold)[0]
        for cls in class_ids
    }


def compute_map_50_95(
    predictions: dict[str, list[list[float]]],
    ground_truths: dict[str, list[list[float]]],
    class_ids: list[int],
) -> dict[str, Any]:
    """Compute mAP@0.5 and mAP@0.5:0.95 (COCO-style).

    Args:
        predictions: Dict image_id -> [[x,y,w,h,score,class_id], ...].
        ground_truths: Dict image_id -> [[x,y,w,h,class_id], ...].
        class_ids: List of class IDs.

    Returns:
        Dict with keys 'mAP_50', 'mAP_50_95', 'per_class_AP_50',
        'per_class_AP_50_95', and per-class PR curves.
    """
    thresholds = np.arange(0.5, 1.0, 0.05)  # [0.50, 0.55, ..., 0.95] -> 10 values

    class_aps_per_thr: dict[float, dict[int, float]] = {}
    pr_curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # Collect AP@0.5 with PR curves
    ap50_per_class: dict[int, float] = {}
    for cls in class_ids:
        ap, recalls, precisions = compute_class_ap(predictions, ground_truths, cls, 0.5)
        ap50_per_class[cls] = ap
        pr_curves[cls] = (recalls, precisions)
    class_aps_per_thr[0.5] = ap50_per_class

    # Collect APs for remaining thresholds
    for thr in thresholds[1:]:
        thr = round(float(thr), 2)
        class_aps_per_thr[thr] = compute_map(predictions, ground_truths, class_ids, thr)

    # mAP@0.5
    mAP_50 = float(np.mean(list(ap50_per_class.values()))) if class_ids else 0.0

    # mAP@0.5:0.95 — mean over all thresholds and all classes
    all_aps = []
    per_class_ap_50_95: dict[int, float] = {}
    for cls in class_ids:
        cls_aps = [class_aps_per_thr[round(float(t), 2)][cls] for t in thresholds]
        mean_cls_ap = float(np.mean(cls_aps))
        per_class_ap_50_95[cls] = mean_cls_ap
        all_aps.extend(cls_aps)
    mAP_50_95 = float(np.mean(all_aps)) if all_aps else 0.0

    return {
        "mAP_50": mAP_50,
        "mAP_50_95": mAP_50_95,
        "per_class_AP_50": ap50_per_class,
        "per_class_AP_50_95": per_class_ap_50_95,
        "pr_curves": pr_curves,
    }


# ---------------------------------------------------------------------------
# Object-level metrics at fixed confidence threshold
# ---------------------------------------------------------------------------

def compute_detection_metrics(
    predictions: dict[str, list[list[float]]],
    ground_truths: dict[str, list[list[float]]],
    class_ids: list[int],
    class_names: dict[int, str],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute TP/FP/FN, Precision, Recall, F1 per class and aggregated.

    Also returns mean IoU of matched boxes.

    Args:
        predictions: Dict image_id -> [[x,y,w,h,score,class_id], ...].
        ground_truths: Dict image_id -> [[x,y,w,h,class_id], ...].
        class_ids: List of integer class IDs.
        class_names: Dict class_id -> human-readable name.
        iou_threshold: IoU threshold for positive matching.
        conf_threshold: Minimum confidence score to consider a prediction.

    Returns:
        Dict with per-class and aggregated metrics.
    """
    # Apply confidence threshold filter
    filtered_preds = {
        img_id: [p for p in plist if float(p[4]) >= conf_threshold]
        for img_id, plist in predictions.items()
    }

    per_class_tp: dict[int, int] = {c: 0 for c in class_ids}
    per_class_fp: dict[int, int] = {c: 0 for c in class_ids}
    per_class_fn: dict[int, int] = {c: 0 for c in class_ids}
    per_class_iou_sum: dict[int, float] = {c: 0.0 for c in class_ids}

    all_ids = set(filtered_preds.keys()) | set(ground_truths.keys())

    for img_id in all_ids:
        preds = filtered_preds.get(img_id, [])
        gts = ground_truths.get(img_id, [])

        result = match_predictions(preds, gts, iou_threshold=iou_threshold, class_agnostic=False)

        # TPs: count per predicted class
        for idx, gt_idx, iou in zip(result.tp_pred_indices, result.tp_gt_indices, result.tp_ious):
            cls = int(preds[idx][5])
            if cls in per_class_tp:
                per_class_tp[cls] += 1
                per_class_iou_sum[cls] += iou

        # FPs: count per predicted class
        for idx in result.fp_pred_indices:
            cls = int(preds[idx][5])
            if cls in per_class_fp:
                per_class_fp[cls] += 1

        # FNs: count per GT class
        for gt_idx in result.fn_gt_indices:
            cls = int(gts[gt_idx][4])
            if cls in per_class_fn:
                per_class_fn[cls] += 1

    # Compute per-class derived metrics
    per_class: dict[int, dict[str, float]] = {}
    all_tp = all_fp = all_fn = 0
    total_iou_sum = 0.0

    for cls in class_ids:
        tp = per_class_tp[cls]
        fp = per_class_fp[cls]
        fn = per_class_fn[cls]

        precision = tp / (tp + fp + 1e-9) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn + 1e-9) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-9) if (precision + recall) > 0 else 0.0
        mean_iou = per_class_iou_sum[cls] / tp if tp > 0 else 0.0

        per_class[cls] = {
            "name": class_names.get(cls, str(cls)),
            "TP": tp, "FP": fp, "FN": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "mean_iou": round(mean_iou, 6),
        }
        all_tp += tp
        all_fp += fp
        all_fn += fn
        total_iou_sum += per_class_iou_sum[cls]

    # Macro averages (mean over classes with GT support)
    supported_cls = [c for c in class_ids if (per_class_tp[c] + per_class_fn[c]) > 0]
    macro_prec = float(np.mean([per_class[c]["precision"] for c in supported_cls])) if supported_cls else 0.0
    macro_rec = float(np.mean([per_class[c]["recall"] for c in supported_cls])) if supported_cls else 0.0
    macro_f1 = float(np.mean([per_class[c]["f1"] for c in supported_cls])) if supported_cls else 0.0

    # Weighted averages (weighted by GT count)
    total_gt = sum(per_class_tp[c] + per_class_fn[c] for c in class_ids)
    w_prec = w_rec = w_f1 = 0.0
    if total_gt > 0:
        for c in class_ids:
            weight = (per_class_tp[c] + per_class_fn[c]) / total_gt
            w_prec += weight * per_class[c]["precision"]
            w_rec += weight * per_class[c]["recall"]
            w_f1 += weight * per_class[c]["f1"]

    # Overall (micro) metrics
    micro_prec = all_tp / (all_tp + all_fp + 1e-9) if (all_tp + all_fp) > 0 else 0.0
    micro_rec = all_tp / (all_tp + all_fn + 1e-9) if (all_tp + all_fn) > 0 else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec + 1e-9) if (micro_prec + micro_rec) > 0 else 0.0
    mean_iou_all = total_iou_sum / all_tp if all_tp > 0 else 0.0

    return {
        "per_class": per_class,
        "micro": {
            "TP": all_tp, "FP": all_fp, "FN": all_fn,
            "precision": round(micro_prec, 6),
            "recall": round(micro_rec, 6),
            "f1": round(micro_f1, 6),
            "mean_iou": round(mean_iou_all, 6),
        },
        "macro": {
            "precision": round(macro_prec, 6),
            "recall": round(macro_rec, 6),
            "f1": round(macro_f1, 6),
        },
        "weighted": {
            "precision": round(w_prec, 6),
            "recall": round(w_rec, 6),
            "f1": round(w_f1, 6),
        },
        "iou_threshold": iou_threshold,
        "conf_threshold": conf_threshold,
    }


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    predictions: dict[str, list[list[float]]],
    ground_truths: dict[str, list[list[float]]],
    class_ids: list[int],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.5,
) -> np.ndarray:
    """Compute confusion matrix with background row/column.

    Rows = ground-truth class (background = last row for FNs).
    Columns = predicted class (background = last col for FPs).
    Diagonal = TPs. Background-background cell = unmatched negatives (not tracked here).

    Args:
        predictions: Dict image_id -> [[x,y,w,h,score,class_id], ...].
        ground_truths: Dict image_id -> [[x,y,w,h,class_id], ...].
        class_ids: List of integer class IDs (ordered).
        iou_threshold: IoU threshold.
        conf_threshold: Confidence threshold.

    Returns:
        Square confusion matrix of shape [n_classes+1, n_classes+1].
        Last index is background.
    """
    n = len(class_ids) + 1  # +1 for background
    bg_idx = len(class_ids)
    cls_to_idx = {c: i for i, c in enumerate(class_ids)}

    cm = np.zeros((n, n), dtype=np.int64)

    filtered_preds = {
        img_id: [p for p in plist if float(p[4]) >= conf_threshold]
        for img_id, plist in predictions.items()
    }

    all_ids = set(filtered_preds.keys()) | set(ground_truths.keys())

    for img_id in all_ids:
        preds = filtered_preds.get(img_id, [])
        gts = ground_truths.get(img_id, [])

        result = match_predictions(preds, gts, iou_threshold=iou_threshold, class_agnostic=False)

        # TPs: GT class -> predicted class (should be same, but record both)
        for pred_idx, gt_idx, _ in zip(result.tp_pred_indices, result.tp_gt_indices, result.tp_ious):
            gt_cls = int(gts[gt_idx][4])
            pred_cls = int(preds[pred_idx][5])
            r = cls_to_idx.get(gt_cls, bg_idx)
            c = cls_to_idx.get(pred_cls, bg_idx)
            cm[r, c] += 1

        # FPs: predicted class -> background GT
        for pred_idx in result.fp_pred_indices:
            pred_cls = int(preds[pred_idx][5])
            c = cls_to_idx.get(pred_cls, bg_idx)
            cm[bg_idx, c] += 1

        # FNs: GT class -> background prediction
        for gt_idx in result.fn_gt_indices:
            gt_cls = int(gts[gt_idx][4])
            r = cls_to_idx.get(gt_cls, bg_idx)
            cm[r, bg_idx] += 1

    return cm


# ---------------------------------------------------------------------------
# Frame-level classification metrics
# ---------------------------------------------------------------------------

def compute_frame_metrics(
    predictions: dict[str, list[list[float]]],
    ground_truths: dict[str, list[list[float]]],
    conf_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute frame-level binary lesion classification metrics.

    A frame is classified as 'has lesion' if max prediction score >= conf_threshold.
    Ground truth: frame has lesion if it has >=1 GT box.

    Returns ROC-AUC, PR-AUC, Accuracy, Precision, Recall, Specificity, F1,
    plus raw (scores, labels) for plotting.

    Args:
        predictions: Dict image_id -> [[x,y,w,h,score,class_id], ...].
        ground_truths: Dict image_id -> [[x,y,w,h,class_id], ...].
        conf_threshold: Confidence threshold for binary frame classification.

    Returns:
        Dict with frame-level metrics and raw arrays.
    """
    all_ids = set(predictions.keys()) | set(ground_truths.keys())

    y_scores: list[float] = []
    y_true: list[int] = []

    for img_id in sorted(all_ids):
        preds = predictions.get(img_id, [])
        gts = ground_truths.get(img_id, [])
        max_score = max((float(p[4]) for p in preds), default=0.0)
        gt_pos = 1 if len(gts) > 0 else 0
        y_scores.append(max_score)
        y_true.append(gt_pos)

    y_scores_arr = np.array(y_scores, dtype=np.float64)
    y_true_arr = np.array(y_true, dtype=np.int32)
    y_pred_arr = (y_scores_arr >= conf_threshold).astype(np.int32)

    # Confusion at threshold
    tp = int(np.sum((y_pred_arr == 1) & (y_true_arr == 1)))
    fp = int(np.sum((y_pred_arr == 1) & (y_true_arr == 0)))
    tn = int(np.sum((y_pred_arr == 0) & (y_true_arr == 0)))
    fn = int(np.sum((y_pred_arr == 0) & (y_true_arr == 1)))

    precision = tp / (tp + fp + 1e-9) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn + 1e-9) if (tp + fn) > 0 else 0.0  # sensitivity
    specificity = tn / (tn + fp + 1e-9) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-9) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-9)

    # ROC curve
    roc_auc, roc_fpr, roc_tpr = _compute_roc(y_scores_arr, y_true_arr)

    # PR curve
    pr_auc, pr_rec, pr_prec = _compute_pr(y_scores_arr, y_true_arr)

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "sensitivity": round(float(recall), 6),
        "specificity": round(float(specificity), 6),
        "f1": round(float(f1), 6),
        "roc_auc": round(float(roc_auc), 6),
        "pr_auc": round(float(pr_auc), 6),
        "conf_threshold": conf_threshold,
        # Raw arrays for plotting (stored as lists for JSON serialization)
        "_roc_fpr": roc_fpr.tolist(),
        "_roc_tpr": roc_tpr.tolist(),
        "_pr_recall": pr_rec.tolist(),
        "_pr_precision": pr_prec.tolist(),
        "_y_scores": y_scores_arr.tolist(),
        "_y_true": y_true_arr.tolist(),
    }


def _compute_roc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute ROC curve and AUC from scratch."""
    thresholds = np.sort(np.unique(scores))[::-1]
    fprs: list[float] = []
    tprs: list[float] = []

    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5, np.array([0.0, 1.0]), np.array([0.0, 1.0])

    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fprs.append(float(fp / n_neg))
        tprs.append(float(tp / n_pos))

    # Add boundary points
    fpr = np.array([0.0] + fprs + [1.0])
    tpr = np.array([0.0] + tprs + [1.0])

    # Sort by fpr ascending for AUC calculation
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]

    auc = float(_trapezoid(tpr, fpr))
    return auc, fpr, tpr


def _compute_pr(scores: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute Precision-Recall curve and AUC from scratch."""
    thresholds = np.sort(np.unique(scores))[::-1]
    precs: list[float] = []
    recs: list[float] = []

    n_pos = np.sum(labels == 1)
    if n_pos == 0:
        return 0.0, np.array([0.0, 1.0]), np.array([1.0, 0.0])

    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))
        precs.append(float(tp / (tp + fp + 1e-9)))
        recs.append(float(tp / (tp + fn + 1e-9)))

    rec = np.array([1.0] + recs + [0.0])
    prec = np.array([n_pos / (n_pos + np.sum(labels == 0) + 1e-9)] + precs + [1.0])

    auc = float(_trapezoid(prec[::-1], rec[::-1]))
    return auc, rec, prec


# ---------------------------------------------------------------------------
# Best confidence threshold by max F1 on validation set
# ---------------------------------------------------------------------------

def find_best_conf_threshold(
    predictions: dict[str, list[list[float]]],
    ground_truths: dict[str, list[list[float]]],
    class_ids: list[int],
    iou_threshold: float = 0.5,
    n_thresholds: int = 50,
) -> dict[str, float]:
    """Find the confidence threshold that maximises macro F1 on the given split.

    Args:
        predictions: Val set predictions.
        ground_truths: Val set ground truths.
        class_ids: Class IDs to include.
        iou_threshold: IoU threshold for matching.
        n_thresholds: Number of confidence thresholds to sweep.

    Returns:
        Dict with 'best_conf_threshold', 'best_f1', and 'sweep' list.
    """
    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    sweep = []
    best_f1 = -1.0
    best_thr = 0.5
    class_names = {c: str(c) for c in class_ids}

    for thr in thresholds:
        metrics = compute_detection_metrics(
            predictions, ground_truths, class_ids, class_names,
            iou_threshold=iou_threshold, conf_threshold=float(thr),
        )
        f1 = metrics["macro"]["f1"]
        sweep.append({"conf_threshold": round(float(thr), 4), "macro_f1": round(f1, 6)})
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return {
        "best_conf_threshold": round(best_thr, 4),
        "best_macro_f1": round(best_f1, 6),
        "sweep": sweep,
    }


# ---------------------------------------------------------------------------
# F1-vs-confidence curve per class
# ---------------------------------------------------------------------------

def compute_f1_conf_curve(
    predictions: dict[str, list[list[float]]],
    ground_truths: dict[str, list[list[float]]],
    class_ids: list[int],
    iou_threshold: float = 0.5,
    n_thresholds: int = 50,
) -> dict[int, dict[str, list]]:
    """Compute F1-vs-confidence sweep per class.

    Returns:
        Dict class_id -> {"conf_thresholds": [...], "f1_values": [...]}.
    """
    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    result: dict[int, dict[str, list]] = {c: {"conf_thresholds": [], "f1_values": []} for c in class_ids}

    for thr in thresholds:
        for cls in class_ids:
            filtered_preds = {
                img_id: [p for p in plist if float(p[4]) >= thr and int(p[5]) == cls]
                for img_id, plist in predictions.items()
            }
            cls_gts = {
                img_id: [g for g in glist if int(g[4]) == cls]
                for img_id, glist in ground_truths.items()
            }
            tp = fp = fn = 0
            for img_id in set(filtered_preds) | set(cls_gts):
                r = match_predictions(
                    filtered_preds.get(img_id, []),
                    cls_gts.get(img_id, []),
                    iou_threshold=iou_threshold,
                    class_agnostic=True,
                )
                tp += len(r.tp_pred_indices)
                fp += len(r.fp_pred_indices)
                fn += len(r.fn_gt_indices)

            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)

            result[cls]["conf_thresholds"].append(round(float(thr), 4))
            result[cls]["f1_values"].append(round(float(f1), 6))

    return result


# ---------------------------------------------------------------------------
# Patient-level aggregation
# ---------------------------------------------------------------------------

def compute_patient_metrics(
    predictions: dict[str, list[list[float]]],
    ground_truths: dict[str, list[list[float]]],
    frame_to_patient: dict[str, str],
    conf_threshold: float = 0.5,
    significant_class_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Aggregate frame-level predictions to patient-level lesion detection.

    A patient is positive ('has lesion') if any frame in their study has >=1 GT box.
    A patient is predicted positive if max-score across all frames >= conf_threshold.

    Also computes 'significant stenosis' metric (>=70% stenosis, class IDs provided).

    Args:
        predictions: Dict frame_id -> [[x,y,w,h,score,class_id], ...].
        ground_truths: Dict frame_id -> [[x,y,w,h,class_id], ...].
        frame_to_patient: Dict frame_id -> patient_id.
        conf_threshold: Min score for frame-level positive prediction.
        significant_class_ids: Class IDs representing >=70% stenosis (e.g. p70_90..p100).

    Returns:
        Dict with patient-level TP/FP/FN/TN and derived metrics, plus
        'significant_stenosis' sub-dict if significant_class_ids provided.
    """
    all_ids = set(predictions.keys()) | set(ground_truths.keys())

    patient_gt_scores: dict[str, dict[str, float]] = {}  # patient -> {max_score, has_lesion, has_sig}

    for img_id in all_ids:
        patient = frame_to_patient.get(img_id, img_id)
        preds = predictions.get(img_id, [])
        gts = ground_truths.get(img_id, [])

        max_score = max((float(p[4]) for p in preds), default=0.0)
        has_lesion = int(len(gts) > 0)

        sig = 0
        if significant_class_ids:
            sig = int(any(int(g[4]) in significant_class_ids for g in gts))

        if patient not in patient_gt_scores:
            patient_gt_scores[patient] = {
                "max_score": max_score,
                "has_lesion": has_lesion,
                "has_sig": sig,
            }
        else:
            patient_gt_scores[patient]["max_score"] = max(
                patient_gt_scores[patient]["max_score"], max_score
            )
            patient_gt_scores[patient]["has_lesion"] = max(
                patient_gt_scores[patient]["has_lesion"], has_lesion
            )
            patient_gt_scores[patient]["has_sig"] = max(
                patient_gt_scores[patient]["has_sig"], sig
            )

    # Binary metrics
    def _binary_metrics(scores_labels: list[tuple[float, int]]) -> dict[str, Any]:
        y_scores = np.array([s for s, _ in scores_labels])
        y_true = np.array([l for _, l in scores_labels])
        y_pred = (y_scores >= conf_threshold).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        precision = tp / (tp + fp + 1e-9) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn + 1e-9) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-9) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-9)
        roc_auc, _, _ = _compute_roc(y_scores, y_true)
        return {
            "n_patients": len(scores_labels),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "accuracy": round(float(accuracy), 6),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "roc_auc": round(float(roc_auc), 6),
        }

    lesion_pairs = [
        (patient_gt_scores[p]["max_score"], patient_gt_scores[p]["has_lesion"])
        for p in patient_gt_scores
    ]
    result = {"lesion_detection": _binary_metrics(lesion_pairs)}

    if significant_class_ids:
        sig_pairs = [
            (patient_gt_scores[p]["max_score"], patient_gt_scores[p]["has_sig"])
            for p in patient_gt_scores
        ]
        result["significant_stenosis"] = _binary_metrics(sig_pairs)

    return result


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (patient-level resampling)
# ---------------------------------------------------------------------------

def bootstrap_confidence_intervals(
    predictions: dict[str, list[list[float]]],
    ground_truths: dict[str, list[list[float]]],
    class_ids: list[int],
    frame_to_patient: dict[str, str],
    class_names: dict[int, str],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.5,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap 95% CIs by patient resampling for F1, mAP@0.5, sensitivity.

    Each bootstrap sample draws patients with replacement. Frames belonging to
    sampled patients form the bootstrap evaluation set.

    Args:
        predictions: Dict frame_id -> preds.
        ground_truths: Dict frame_id -> gts.
        class_ids: Class IDs.
        frame_to_patient: Dict frame_id -> patient_id.
        class_names: Dict class_id -> name.
        iou_threshold: IoU threshold.
        conf_threshold: Confidence threshold.
        n_bootstrap: Number of bootstrap iterations.
        ci_level: Confidence interval level (default 0.95).
        seed: Random seed.

    Returns:
        Dict with point estimates and CI bounds for macro_f1, mAP_50, sensitivity.
    """
    rng = np.random.RandomState(seed)

    # Group frames by patient
    patient_to_frames: dict[str, list[str]] = {}
    for frame_id, patient in frame_to_patient.items():
        patient_to_frames.setdefault(patient, []).append(frame_id)

    all_patients = sorted(patient_to_frames.keys())
    n_patients = len(all_patients)

    if n_patients < 2:
        return {"error": "Not enough patients for bootstrap"}

    boot_macro_f1: list[float] = []
    boot_map50: list[float] = []
    boot_sensitivity: list[float] = []

    for _ in range(n_bootstrap):
        sampled_patients = rng.choice(all_patients, size=n_patients, replace=True)
        boot_preds: dict[str, list[list[float]]] = {}
        boot_gts: dict[str, list[list[float]]] = {}

        for pat in sampled_patients:
            for frame_id in patient_to_frames.get(pat, []):
                # Suffix to create unique keys for duplicate patients
                key = f"{frame_id}__{pat}"
                boot_preds[key] = predictions.get(frame_id, [])
                boot_gts[key] = ground_truths.get(frame_id, [])

        # Detection metrics
        det = compute_detection_metrics(
            boot_preds, boot_gts, class_ids, class_names,
            iou_threshold=iou_threshold, conf_threshold=conf_threshold,
        )
        boot_macro_f1.append(det["macro"]["f1"])
        boot_sensitivity.append(det["micro"]["recall"])

        # mAP@0.5
        ap_per_cls = compute_map(boot_preds, boot_gts, class_ids, iou_threshold=0.5)
        map50 = float(np.mean(list(ap_per_cls.values()))) if class_ids else 0.0
        boot_map50.append(map50)

    alpha = (1.0 - ci_level) / 2.0

    def _ci(values: list[float]) -> dict[str, float]:
        arr = np.array(values)
        return {
            "mean": round(float(np.mean(arr)), 6),
            "lower": round(float(np.percentile(arr, alpha * 100)), 6),
            "upper": round(float(np.percentile(arr, (1 - alpha) * 100)), 6),
        }

    return {
        "n_bootstrap": n_bootstrap,
        "ci_level": ci_level,
        "macro_f1": _ci(boot_macro_f1),
        "mAP_50": _ci(boot_map50),
        "sensitivity": _ci(boot_sensitivity),
    }
