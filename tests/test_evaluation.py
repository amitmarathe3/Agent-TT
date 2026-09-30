"""Unit and integration tests for the CADICA evaluation framework.

Tests hand-computed toy cases:
- Perfect predictions yield metrics = 1.0
- No predictions yield precision/AP = 0
- Known IoU values match expectations
- PR/ROC curves have correct boundary behavior
- Confusion matrix correctly tracks TP/FP/FN/background
- Frame-level and patient-level metrics
- Bootstrap CIs are within [0, 1] and CI width is positive
"""

from __future__ import annotations

import pytest
import numpy as np

from src.evaluation.matching import (
    compute_iou_xywh,
    compute_iou_matrix,
    match_predictions,
    match_dataset,
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
    build_pr_curve,
    _interpolate_ap_101,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_case():
    """Perfect detection: predictions exactly match ground truth."""
    gts = {
        "img1": [[10.0, 10.0, 50.0, 50.0, 0]],
        "img2": [[100.0, 100.0, 30.0, 30.0, 1], [20.0, 20.0, 15.0, 15.0, 2]],
        "img3": [],
    }
    # Same boxes, high confidence
    preds = {
        "img1": [[10.0, 10.0, 50.0, 50.0, 0.99, 0]],
        "img2": [[100.0, 100.0, 30.0, 30.0, 0.95, 1], [20.0, 20.0, 15.0, 15.0, 0.90, 2]],
        "img3": [],
    }
    return preds, gts


@pytest.fixture
def no_pred_case():
    """No predictions at all."""
    gts = {
        "img1": [[10.0, 10.0, 50.0, 50.0, 0]],
        "img2": [[100.0, 100.0, 30.0, 30.0, 1]],
    }
    preds: dict[str, list] = {"img1": [], "img2": []}
    return preds, gts


@pytest.fixture
def class_ids_3():
    return [0, 1, 2]


@pytest.fixture
def class_names_3():
    return {0: "mild", 1: "moderate", 2: "severe"}


# ---------------------------------------------------------------------------
# IoU tests
# ---------------------------------------------------------------------------

class TestIoU:
    def test_perfect_overlap(self):
        box = [0.0, 0.0, 100.0, 100.0]
        assert compute_iou_xywh(box, box) == pytest.approx(1.0, abs=1e-6)

    def test_no_overlap(self):
        a = [0.0, 0.0, 10.0, 10.0]
        b = [100.0, 100.0, 10.0, 10.0]
        assert compute_iou_xywh(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_partial_overlap(self):
        # Two 10x10 boxes overlapping by 5x10 = 50 px
        a = [0.0, 0.0, 10.0, 10.0]
        b = [5.0, 0.0, 10.0, 10.0]
        # union = 200 - 50 = 150; iou = 50/150 = 1/3
        assert compute_iou_xywh(a, b) == pytest.approx(1 / 3, rel=1e-4)

    def test_zero_area_box(self):
        a = [0.0, 0.0, 0.0, 0.0]
        b = [0.0, 0.0, 10.0, 10.0]
        assert compute_iou_xywh(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_iou_matrix_shape(self):
        preds = np.array([[0, 0, 10, 10], [20, 20, 5, 5]], dtype=np.float32)
        gts = np.array([[0, 0, 10, 10], [50, 50, 10, 10], [0, 0, 5, 5]], dtype=np.float32)
        mat = compute_iou_matrix(preds, gts)
        assert mat.shape == (2, 3)
        assert mat[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert mat[0, 1] == pytest.approx(0.0, abs=1e-5)

    def test_iou_matrix_empty_preds(self):
        mat = compute_iou_matrix(np.zeros((0, 4)), np.ones((3, 4)))
        assert mat.shape == (0, 3)

    def test_iou_matrix_empty_gts(self):
        mat = compute_iou_matrix(np.ones((2, 4)), np.zeros((0, 4)))
        assert mat.shape == (2, 0)


# ---------------------------------------------------------------------------
# Matching tests
# ---------------------------------------------------------------------------

class TestMatching:
    def test_perfect_match_single_box(self):
        preds = [[10.0, 10.0, 50.0, 50.0, 0.9, 0]]
        gts = [[10.0, 10.0, 50.0, 50.0, 0]]
        result = match_predictions(preds, gts, iou_threshold=0.5)
        assert len(result.tp_pred_indices) == 1
        assert len(result.fp_pred_indices) == 0
        assert len(result.fn_gt_indices) == 0
        assert result.tp_ious[0] == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap_gives_fp_fn(self):
        preds = [[0.0, 0.0, 10.0, 10.0, 0.8, 0]]
        gts = [[100.0, 100.0, 10.0, 10.0, 0]]
        result = match_predictions(preds, gts, iou_threshold=0.5)
        assert len(result.tp_pred_indices) == 0
        assert len(result.fp_pred_indices) == 1
        assert len(result.fn_gt_indices) == 1

    def test_class_mismatch_causes_fp_fn(self):
        preds = [[10.0, 10.0, 50.0, 50.0, 0.9, 1]]  # class 1
        gts = [[10.0, 10.0, 50.0, 50.0, 0]]           # class 0
        result = match_predictions(preds, gts, iou_threshold=0.5, class_agnostic=False)
        # Wrong class -> FP prediction, FN gt
        assert len(result.tp_pred_indices) == 0
        assert len(result.fp_pred_indices) == 1
        assert len(result.fn_gt_indices) == 1

    def test_class_agnostic_ignores_class(self):
        preds = [[10.0, 10.0, 50.0, 50.0, 0.9, 1]]  # class 1
        gts = [[10.0, 10.0, 50.0, 50.0, 0]]           # class 0
        result = match_predictions(preds, gts, iou_threshold=0.5, class_agnostic=True)
        # Agnostic: should match by IoU
        assert len(result.tp_pred_indices) == 1

    def test_each_gt_matched_at_most_once(self):
        # Two predictions pointing at the same GT
        preds = [
            [10.0, 10.0, 50.0, 50.0, 0.9, 0],
            [11.0, 11.0, 48.0, 48.0, 0.8, 0],
        ]
        gts = [[10.0, 10.0, 50.0, 50.0, 0]]
        result = match_predictions(preds, gts, iou_threshold=0.5)
        assert len(result.tp_pred_indices) == 1
        assert len(result.fp_pred_indices) == 1

    def test_empty_preds(self):
        gts = [[0.0, 0.0, 10.0, 10.0, 0]]
        result = match_predictions([], gts, iou_threshold=0.5)
        assert len(result.fn_gt_indices) == 1
        assert len(result.tp_pred_indices) == 0

    def test_empty_gts(self):
        preds = [[0.0, 0.0, 10.0, 10.0, 0.9, 0]]
        result = match_predictions(preds, [], iou_threshold=0.5)
        assert len(result.fp_pred_indices) == 1
        assert len(result.tp_pred_indices) == 0

    def test_score_ordering_matters(self):
        """Higher score prediction should get the match."""
        preds = [
            [10.0, 10.0, 50.0, 50.0, 0.6, 0],  # lower score
            [10.0, 10.0, 48.0, 48.0, 0.9, 0],  # higher score -> should match
        ]
        gts = [[10.0, 10.0, 50.0, 50.0, 0]]
        result = match_predictions(preds, gts, iou_threshold=0.5)
        # The higher-score pred (index 1) should be TP
        assert 1 in result.tp_pred_indices


# ---------------------------------------------------------------------------
# Detection metrics: perfect case
# ---------------------------------------------------------------------------

class TestDetectionMetricsPerfect:
    def test_perfect_precision_recall_f1(self, perfect_case, class_ids_3, class_names_3):
        preds, gts = perfect_case
        metrics = compute_detection_metrics(
            preds, gts, class_ids_3, class_names_3, conf_threshold=0.5
        )
        assert metrics["micro"]["precision"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["micro"]["recall"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["micro"]["f1"] == pytest.approx(1.0, abs=1e-4)

    def test_perfect_per_class_f1(self, perfect_case, class_ids_3, class_names_3):
        preds, gts = perfect_case
        metrics = compute_detection_metrics(
            preds, gts, class_ids_3, class_names_3, conf_threshold=0.5
        )
        for cls in class_ids_3:
            assert metrics["per_class"][cls]["TP"] >= 0
        # Class 0 has 1 GT and 1 pred -> TP=1
        assert metrics["per_class"][0]["TP"] == 1
        assert metrics["per_class"][0]["FP"] == 0
        assert metrics["per_class"][0]["FN"] == 0

    def test_perfect_mean_iou_is_one(self, perfect_case, class_ids_3, class_names_3):
        preds, gts = perfect_case
        metrics = compute_detection_metrics(
            preds, gts, class_ids_3, class_names_3, conf_threshold=0.5
        )
        assert metrics["micro"]["mean_iou"] == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Detection metrics: no predictions
# ---------------------------------------------------------------------------

class TestDetectionMetricsNoPreds:
    def test_no_preds_zero_precision(self, no_pred_case, class_ids_3, class_names_3):
        preds, gts = no_pred_case
        metrics = compute_detection_metrics(
            preds, gts, [0, 1], class_names_3, conf_threshold=0.5
        )
        assert metrics["micro"]["TP"] == 0
        assert metrics["micro"]["precision"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["micro"]["recall"] == pytest.approx(0.0, abs=1e-5)
        assert metrics["micro"]["f1"] == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# AP and mAP tests
# ---------------------------------------------------------------------------

class TestAP:
    def test_perfect_ap(self):
        """Perfect classifier should have AP = 1.0."""
        preds = {
            "img1": [[0, 0, 10, 10, 0.99, 0]],
            "img2": [[20, 20, 10, 10, 0.95, 0]],
        }
        gts = {
            "img1": [[0, 0, 10, 10, 0]],
            "img2": [[20, 20, 10, 10, 0]],
        }
        ap_map = compute_map(preds, gts, [0], iou_threshold=0.5)
        assert ap_map[0] == pytest.approx(1.0, abs=1e-3)

    def test_no_preds_ap_zero(self):
        """No predictions -> AP = 0."""
        preds = {"img1": []}
        gts = {"img1": [[0, 0, 10, 10, 0]]}
        ap_map = compute_map(preds, gts, [0], iou_threshold=0.5)
        assert ap_map[0] == pytest.approx(0.0, abs=1e-6)

    def test_map_50_95_leq_map_50(self):
        """mAP@0.5:0.95 should be <= mAP@0.5 (stricter thresholds)."""
        preds = {
            "img1": [[0, 0, 10, 10, 0.99, 0]],
            "img2": [[0, 0, 10, 10, 0.80, 1]],
        }
        gts = {
            "img1": [[0, 0, 10, 10, 0]],
            "img2": [[0, 0, 10, 10, 1]],
        }
        result = compute_map_50_95(preds, gts, [0, 1])
        assert result["mAP_50_95"] <= result["mAP_50"] + 1e-6

    def test_pr_curve_boundary_points(self):
        """PR curve must start at recall=0 and reach achieved max recall."""
        scores = np.array([0.9, 0.7, 0.5])
        tp_flags = np.array([True, False, True])
        recs, precs = build_pr_curve(scores, tp_flags, n_gt=3)
        assert recs[0] == pytest.approx(0.0, abs=1e-6)
        assert recs[-1] == pytest.approx(2 / 3, abs=1e-6)

    def test_interpolated_ap_perfect_recall(self):
        """Curve with perfect recall at all thresholds = AP ~1."""
        recs = np.linspace(0, 1, 101)
        precs = np.ones(101)
        ap = _interpolate_ap_101(recs, precs)
        assert ap == pytest.approx(1.0, abs=1e-4)

    def test_interpolated_ap_random_recall(self):
        """Curve with zero precision at all thresholds = AP ~0."""
        recs = np.linspace(0, 1, 101)
        precs = np.zeros(101)
        ap = _interpolate_ap_101(recs, precs)
        assert ap == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

class TestConfusionMatrix:
    def test_perfect_predictions_diagonal(self, perfect_case, class_ids_3):
        preds, gts = perfect_case
        cm = compute_confusion_matrix(preds, gts, class_ids_3, conf_threshold=0.5)
        n = len(class_ids_3) + 1
        assert cm.shape == (n, n)
        # Diagonal TPs: class 0 has 1 TP, class 1 has 1 TP, class 2 has 1 TP
        assert cm[0, 0] == 1
        assert cm[1, 1] == 1
        assert cm[2, 2] == 1

    def test_no_preds_fn_in_background_col(self, no_pred_case, class_ids_3):
        preds, gts = no_pred_case
        cm = compute_confusion_matrix(preds, gts, [0, 1], conf_threshold=0.5)
        bg_idx = 2  # len([0,1]) = background
        # All GTs are unmatched -> FN column (last column)
        assert cm[0, bg_idx] == 1  # class 0 FN
        assert cm[1, bg_idx] == 1  # class 1 FN


# ---------------------------------------------------------------------------
# Frame-level metrics
# ---------------------------------------------------------------------------

class TestFrameMetrics:
    def test_perfect_frame_classifier(self):
        """All positive frames have high score, negatives have 0."""
        preds = {
            "f1": [[0, 0, 10, 10, 0.9, 0]],
            "f2": [[0, 0, 10, 10, 0.8, 0]],
            "f3": [],  # Negative frame
        }
        gts = {
            "f1": [[0, 0, 10, 10, 0]],
            "f2": [[0, 0, 10, 10, 0]],
            "f3": [],
        }
        metrics = compute_frame_metrics(preds, gts, conf_threshold=0.5)
        assert metrics["TP"] == 2
        assert metrics["TN"] == 1
        assert metrics["FP"] == 0
        assert metrics["FN"] == 0
        assert metrics["accuracy"] == pytest.approx(1.0, abs=1e-5)
        assert metrics["f1"] == pytest.approx(1.0, abs=1e-4)

    def test_roc_auc_between_0_and_1(self):
        preds = {"f1": [[0, 0, 10, 10, 0.7, 0]], "f2": [], "f3": [[0, 0, 5, 5, 0.3, 0]]}
        gts = {"f1": [[0, 0, 10, 10, 0]], "f2": [[0, 0, 10, 10, 0]], "f3": []}
        metrics = compute_frame_metrics(preds, gts, conf_threshold=0.5)
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["pr_auc"] <= 1.0

    def test_all_false_negatives(self):
        """No predictions -> all GT frames are FN, precision=0."""
        preds = {"f1": [], "f2": []}
        gts = {"f1": [[0, 0, 10, 10, 0]], "f2": [[0, 0, 10, 10, 0]]}
        metrics = compute_frame_metrics(preds, gts, conf_threshold=0.5)
        assert metrics["FN"] == 2
        assert metrics["FP"] == 0
        assert metrics["f1"] == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Patient-level metrics
# ---------------------------------------------------------------------------

class TestPatientMetrics:
    def test_patient_aggregation(self):
        frame_to_patient = {"f1": "p1", "f2": "p1", "f3": "p2", "f4": "p2"}
        preds = {
            "f1": [[0, 0, 10, 10, 0.9, 0]],
            "f2": [],
            "f3": [],
            "f4": [],
        }
        gts = {
            "f1": [[0, 0, 10, 10, 0]],
            "f2": [],
            "f3": [],
            "f4": [],
        }
        metrics = compute_patient_metrics(preds, gts, frame_to_patient, conf_threshold=0.5)
        ld = metrics["lesion_detection"]
        assert ld["n_patients"] == 2
        assert ld["TP"] == 1  # p1 has lesion and is predicted positive
        assert ld["TN"] == 1  # p2 has no lesion and is predicted negative


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

class TestBootstrap:
    def test_ci_bounds_are_valid(self, perfect_case, class_ids_3, class_names_3):
        preds, gts = perfect_case
        frame_to_patient = {"img1": "p1", "img2": "p2", "img3": "p3"}
        boot = bootstrap_confidence_intervals(
            preds, gts, class_ids_3, frame_to_patient, class_names_3,
            n_bootstrap=50, seed=42,
        )
        for key in ["macro_f1", "mAP_50", "sensitivity"]:
            ci = boot[key]
            assert 0.0 <= ci["lower"] <= ci["mean"] + 1e-6
            assert ci["lower"] <= ci["upper"] + 1e-6
            assert 0.0 <= ci["upper"] <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Known IoU value: verified by hand
# ---------------------------------------------------------------------------

class TestKnownIoU:
    def test_known_iou_50_percent_overlap(self):
        """Box A=[0,0,10,10], Box B=[5,0,10,10] -> overlap=5x10=50, union=150, IoU=1/3."""
        iou = compute_iou_xywh([0, 0, 10, 10], [5, 0, 10, 10])
        assert iou == pytest.approx(1 / 3, rel=1e-4)

    def test_known_iou_contained_box(self):
        """Small box inside big box -> IoU = small_area / big_area."""
        big = [0, 0, 20, 20]   # 400 px
        small = [5, 5, 10, 10]  # 100 px (fully inside big)
        iou = compute_iou_xywh(big, small)
        # union = 400, inter = 100 -> IoU = 100/400 = 0.25
        assert iou == pytest.approx(0.25, rel=1e-4)


# ---------------------------------------------------------------------------
# Sanity check baseline predictors (Phase 2.4)
# ---------------------------------------------------------------------------

class TestSanityCheckBaselines:
    def test_oracle_builder_gives_perfect_scores(self, perfect_case, class_ids_3, class_names_3):
        from scripts.sanity_check import build_oracle_predictions
        _, gts = perfect_case
        oracle_preds = build_oracle_predictions(gts)
        det = compute_detection_metrics(
            oracle_preds, gts, class_ids_3, class_names_3,
            iou_threshold=0.5, conf_threshold=0.5,
        )
        assert det["micro"]["f1"] == 1.0
        assert det["macro"]["f1"] == 1.0
        map_res = compute_map_50_95(oracle_preds, gts, class_ids_3)
        assert map_res["mAP_50"] == pytest.approx(1.0, abs=1e-4)
        assert map_res["mAP_50_95"] == pytest.approx(1.0, abs=1e-4)

    def test_random_builder_generates_valid_format(self, class_ids_3):
        from scripts.sanity_check import build_random_predictions
        gts = {"img1": [[10, 10, 20, 20, 0]], "img2": []}
        random_preds = build_random_predictions(gts, class_ids_3, seed=123)
        assert set(random_preds.keys()) == set(gts.keys())
        for fid, boxes in random_preds.items():
            for b in boxes:
                assert len(b) == 6
                x, y, w, h, score, cls_id = b
                assert 0.0 <= x <= 512.0
                assert 0.0 <= y <= 512.0
                assert 0.0 <= score <= 1.0
                assert cls_id in class_ids_3

    def test_zero_tp_gives_ap_exactly_zero(self, class_ids_3):
        """Zero TP must yield AP exactly 0.0 (fixing the 0.0099 envelope bug)."""
        # Ground truths present, but predictions are completely disjoint
        gts = {
            "img1": [[10.0, 10.0, 50.0, 50.0, 0]],
            "img2": [[100.0, 100.0, 30.0, 30.0, 1]],
        }
        preds = {
            "img1": [[400.0, 400.0, 50.0, 50.0, 0.99, 0]],  # disjoint
            "img2": [[400.0, 400.0, 30.0, 30.0, 0.95, 1]],  # disjoint
        }
        res_50 = compute_map(preds, gts, [0, 1], iou_threshold=0.5)
        assert res_50[0] == 0.0
        assert res_50[1] == 0.0
        res_50_95 = compute_map_50_95(preds, gts, [0, 1])
        assert res_50_95["mAP_50"] == 0.0
        assert res_50_95["mAP_50_95"] == 0.0

    def test_jittered_oracle_behavior(self, perfect_case, class_ids_3):
        """Jittered oracle (IoU ~ 0.64) yields AP@0.5 == 1.0 and AP@0.5:0.95 < AP@0.5."""
        from scripts.sanity_check import build_jittered_oracle_predictions
        _, gts = perfect_case
        jittered = build_jittered_oracle_predictions(gts, shift_factor=0.22)
        res_50_95 = compute_map_50_95(jittered, gts, class_ids_3)
        assert res_50_95["mAP_50"] == pytest.approx(1.0, abs=1e-5)
        assert res_50_95["mAP_50_95"] < res_50_95["mAP_50"]
        # With shift=0.22, IoU is ~0.64, so thresholds >= 0.65 fail (matches at 0.50, 0.55, 0.60 -> 3/10 = 0.3)
        assert res_50_95["mAP_50_95"] == pytest.approx(0.30, abs=0.05)

    def test_oracle_frame_pr_auc_is_one(self, perfect_case):
        """Oracle frame-level PR-AUC must be exactly 1.0 (fixing the 0.6004 bug)."""
        from scripts.sanity_check import build_oracle_predictions
        _, gts = perfect_case
        oracle = build_oracle_predictions(gts)
        fmetrics = compute_frame_metrics(oracle, gts, conf_threshold=0.5)
        assert fmetrics["pr_auc"] == pytest.approx(1.0, abs=1e-5)
        assert fmetrics["roc_auc"] == pytest.approx(1.0, abs=1e-5)
        assert fmetrics["accuracy"] == pytest.approx(1.0, abs=1e-5)


class TestCanonicalEvalSets:
    def test_canonical_frame_counts(self):
        """Canonical eval sets must match exactly the YOLO dataset directory frames."""
        import os
        from src.data.eval_sets import get_canonical_eval_frame_ids
        if os.path.exists("outputs/yolo_dataset/images/val"):
            val_ids = get_canonical_eval_frame_ids("val")
            assert len(val_ids) == 722
        if os.path.exists("outputs/yolo_dataset/images/test"):
            test_ids = get_canonical_eval_frame_ids("test")
            assert len(test_ids) == 852

    def test_full_frames_warning(self):
        """load_eval_ground_truth with full_frames=True must warn."""
        import warnings
        from src.data.eval_sets import load_eval_ground_truth
        with pytest.warns(UserWarning, match=r"--full-frames"):
            load_eval_ground_truth("val", scheme="7class", full_frames=True)

