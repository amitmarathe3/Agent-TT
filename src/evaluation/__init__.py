"""Evaluation package for CADICA coronary angiography object detection.

Model-agnostic evaluation framework accepting predictions and ground truths
as per-image lists of [x, y, w, h, score, class_id] / [x, y, w, h, class_id].
"""

from src.evaluation.matching import (
    compute_iou_xywh,
    compute_iou_matrix,
    match_predictions,
    match_dataset,
    MatchResult,
)
from src.evaluation.metrics import (
    compute_detection_metrics,
    compute_map,
    compute_map_50_95,
    compute_confusion_matrix,
    compute_frame_metrics,
    compute_patient_metrics,
    bootstrap_confidence_intervals,
    find_best_conf_threshold,
    compute_f1_conf_curve,
)
from src.evaluation.report import (
    generate_report,
    generate_markdown_report,
    save_json_metrics,
    plot_confusion_matrix,
    plot_pr_curves,
    plot_roc_curve,
)

__all__ = [
    # matching
    "compute_iou_xywh",
    "compute_iou_matrix",
    "match_predictions",
    "match_dataset",
    "MatchResult",
    # metrics
    "compute_detection_metrics",
    "compute_map",
    "compute_map_50_95",
    "compute_confusion_matrix",
    "compute_frame_metrics",
    "compute_patient_metrics",
    "bootstrap_confidence_intervals",
    "find_best_conf_threshold",
    "compute_f1_conf_curve",
    # report
    "generate_report",
    "generate_markdown_report",
    "save_json_metrics",
    "plot_confusion_matrix",
    "plot_pr_curves",
    "plot_roc_curve",
]
