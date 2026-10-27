"""Detector Control Experiment (inference only, run1 weights, 544_padded).

Evaluates detector frame score (max box score) for:
1. Annotated keyframes vs matched nonlesion keyframes (239 val / 275 test)
2. Annotated keyframes vs distant non-keyframes (125 val / 163 test)

Threshold chosen on VAL; TEST evaluated once; 500-iteration patient cluster bootstrap 95% CIs.
Saves metrics to outputs/eval/detector_control/metrics.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.frame_classifier import patient_bootstrap_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("detector_control")


def max_box_score(boxes: list[list[float]]) -> float:
    if not boxes:
        return 0.0
    # box format: [x, y, w, h, score, class_id]
    scores = [b[4] for b in boxes if len(b) >= 5]
    return float(max(scores)) if scores else 0.0


def main() -> None:
    t0 = time.time()
    device = "0" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # 1. Load data index and splits
    fi = pd.read_csv(ROOT / "data_index" / "frames_index.csv")
    sp = pd.read_csv(ROOT / "data_index" / "splits" / "split_v1.csv")
    p2s = dict(zip(sp["patient_id"], sp["split"]))
    fi["split"] = fi["patient_id"].map(p2s)
    fi = fi.drop_duplicates(subset=["frame_id"]).set_index("frame_id")

    # 2. Load existing 544_padded predictions for canonical sets
    pv = json.load(open(ROOT / "outputs" / "preds_val_544.json"))
    pt = json.load(open(ROOT / "outputs" / "preds_test_544.json"))

    det_scores: dict[str, float] = {}
    for fid, boxes in pv.items():
        det_scores[fid] = max_box_score(boxes)
    for fid, boxes in pt.items():
        det_scores[fid] = max_box_score(boxes)

    # 3. Check for matched nonlesion keyframes missing in preds
    missing_items: list[tuple[str, str, Path]] = []
    for split in ["val", "test"]:
        nkf = fi[(fi["split"] == split) & (fi["is_selected_keyframe"] == True) & (fi["has_lesion_video"] == False)]
        for fid in nkf.index:
            if fid not in det_scores:
                img_path = ROOT / "outputs" / "extracted_keyframes" / split / f"{fid}.png"
                if not img_path.exists():
                    img_path = ROOT / "outputs" / "yolo_dataset" / "images" / split / f"{fid}.png"
                if img_path.exists():
                    missing_items.append((split, fid, img_path))
                else:
                    logger.warning("Image missing on disk: %s", fid)

    logger.info("Found %d nonlesion keyframe images to predict with 544_padded", len(missing_items))

    # 4. Predict missing images with YOLO run1 weights
    matched_preds_cache = ROOT / "outputs" / "runs" / "exp_7class_yolo11s_oversample_run1" / "preds_matched_negatives_544.json"
    if matched_preds_cache.exists():
        logger.info("Loading cached matched predictions from %s", matched_preds_cache)
        cached = json.load(open(matched_preds_cache))
        for fid, boxes in cached.items():
            det_scores[fid] = max_box_score(boxes)
    elif missing_items:
        weights_path = ROOT / "outputs" / "runs" / "exp_7class_yolo11s_oversample_run1" / "weights" / "best.pt"
        logger.info("Loading YOLO model from %s...", weights_path)
        model = YOLO(str(weights_path))
        lb = LetterBox(new_shape=(544, 544), auto=False, scaleup=False, center=True)

        batch_size = 16
        new_preds: dict[str, list[list[float]]] = {}
        for i in range(0, len(missing_items), batch_size):
            chunk = missing_items[i : i + batch_size]
            chunk_imgs = [lb(image=cv2.imread(str(item[2]))) for item in chunk]
            results = model.predict(
                source=chunk_imgs,
                conf=0.001,
                iou=0.6,
                max_det=50,
                imgsz=544,
                device=device,
                verbose=False,
            )
            for r, item in zip(results, chunk):
                fid = item[1]
                boxes_list: list[list[float]] = []
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    scores = r.boxes.conf.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy()
                    for j in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[j]
                        x1_512 = float(np.clip(x1 - 16.0, 0.0, 512.0))
                        y1_512 = float(np.clip(y1 - 16.0, 0.0, 512.0))
                        x2_512 = float(np.clip(x2 - 16.0, 0.0, 512.0))
                        y2_512 = float(np.clip(y2 - 16.0, 0.0, 512.0))
                        boxes_list.append([
                            round(x1_512, 2),
                            round(y1_512, 2),
                            round(max(0.0, x2_512 - x1_512), 2),
                            round(max(0.0, y2_512 - y1_512), 2),
                            round(float(scores[j]), 6),
                            int(classes[j]),
                        ])
                new_preds[fid] = boxes_list
                det_scores[fid] = max_box_score(boxes_list)

        matched_preds_cache.write_text(json.dumps(new_preds, indent=2), encoding="utf-8")
        logger.info("Saved %d predictions to %s", len(new_preds), matched_preds_cache)

    # 5. Build evaluation subsets
    # Setup 1: Annotated keyframes vs Matched nonlesion keyframes
    # Setup 2: Annotated keyframes vs Distant non-keyframes (unannotated, is_selected_keyframe == False)
    eval_results = {}

    for setup_name, neg_filter in [
        ("matched_keyframes", lambda row: row["is_selected_keyframe"] and (not row["has_lesion_video"])),
        ("distant_non_keyframes", lambda row: (not row["has_groundtruth"]) and (not row["is_selected_keyframe"])),
    ]:
        logger.info("\n=== Evaluating Setup: %s ===", setup_name)
        dfs = {}
        for split in ["val", "test"]:
            sub_fi = fi[fi["split"] == split]
            pos_fids = sub_fi[sub_fi["has_groundtruth"] & sub_fi["is_selected_keyframe"]].index
            neg_fids = [fid for fid in sub_fi.index if neg_filter(sub_fi.loc[fid]) and fid in det_scores]

            rows = []
            for fid in pos_fids:
                if fid in det_scores:
                    row_info = sub_fi.loc[fid]
                    rows.append({
                        "frame_id": fid,
                        "patient_id": str(row_info["patient_id"]),
                        "video_id": str(row_info["video_id"]),
                        "target": 1,
                        "score": det_scores[fid],
                    })
            for fid in neg_fids:
                row_info = sub_fi.loc[fid]
                rows.append({
                    "frame_id": fid,
                    "patient_id": str(row_info["patient_id"]),
                    "video_id": str(row_info["video_id"]),
                    "target": 0,
                    "score": det_scores[fid],
                })
            dfs[split] = pd.DataFrame(rows)
            logger.info("%s %s: total=%d (pos=%d, neg=%d)", setup_name, split, len(dfs[split]), len(pos_fids), len(neg_fids))

        df_val = dfs["val"]
        df_test = dfs["test"]

        # Val ROC-AUC
        val_auc = float(roc_auc_score(df_val["target"], df_val["score"]))

        # Find optimal threshold on VAL maximizing balanced accuracy
        best_thresh = 0.05
        best_val_bal_acc = -1.0
        # Sweep unique scores or fine grid
        thresholds = np.unique(np.clip(df_val["score"].values, 0.001, 0.999))
        if len(thresholds) > 100:
            thresholds = np.quantile(thresholds, np.linspace(0.01, 0.99, 100))
        for th in thresholds:
            pred_binary = (df_val["score"] >= th).astype(int)
            ba = balanced_accuracy_score(df_val["target"], pred_binary)
            if ba > best_val_bal_acc:
                best_val_bal_acc = ba
                best_thresh = float(th)

        val_preds_binary = (df_val["score"] >= best_thresh).astype(int)
        val_metrics = {
            "n_frames": int(len(df_val)),
            "n_pos": int(df_val["target"].sum()),
            "n_neg": int((df_val["target"] == 0).sum()),
            "roc_auc": round(val_auc, 4),
            "balanced_accuracy": round(best_val_bal_acc, 4),
            "threshold": round(best_thresh, 4),
            "sensitivity": round(float(recall_score(df_val["target"], val_preds_binary)), 4),
            "specificity": round(float((val_preds_binary[df_val["target"] == 0] == 0).mean()), 4),
        }

        # Evaluate TEST ONCE using best_thresh
        test_auc = float(roc_auc_score(df_test["target"], df_test["score"]))
        test_preds_binary = (df_test["score"] >= best_thresh).astype(int)
        test_bal_acc = float(balanced_accuracy_score(df_test["target"], test_preds_binary))
        test_sens = float(recall_score(df_test["target"], test_preds_binary))
        test_spec = float((test_preds_binary[df_test["target"] == 0] == 0).mean())

        # 500-iteration patient cluster bootstrap on test
        ci_test = patient_bootstrap_ci(
            df=df_test,
            patient_col="patient_id",
            score_col="score",
            label_col="target",
            threshold=best_thresh,
            n_bootstraps=500,
            seed=42,
        )

        test_metrics = {
            "n_frames": int(len(df_test)),
            "n_pos": int(df_test["target"].sum()),
            "n_neg": int((df_test["target"] == 0).sum()),
            "roc_auc": round(test_auc, 4),
            "balanced_accuracy": round(test_bal_acc, 4),
            "threshold": round(best_thresh, 4),
            "sensitivity": round(test_sens, 4),
            "specificity": round(test_spec, 4),
            "ci_95": ci_test,
        }

        eval_results[setup_name] = {
            "val": val_metrics,
            "test": test_metrics,
        }

    out_dir = ROOT / "outputs" / "eval" / "detector_control"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "metrics.json"
    out_file.write_text(json.dumps(eval_results, indent=2), encoding="utf-8")
    logger.info("Saved detector control metrics to %s", out_file)
    print("\nDETECTOR CONTROL METRICS (RAW OUTPUT):")
    print(json.dumps(eval_results, indent=2))


if __name__ == "__main__":
    main()
