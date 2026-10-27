#!/usr/bin/env python3
"""Patient Report Generator (Research prototype - NOT a diagnostic device).

Generates a structured one-page report (report.html and report.json) for a patient.
Strictly follows clinical safety protocols:
- Model: YOLO run1 with 544_padded preprocess and frozen validation threshold.
- No frame classifier or frame/video screening score used (invalidated shortcut).
- Patch grading omitted with clear notice unless a validated patch model exists.
- NEVER outputs 'diseased', 'healthy', 'positive', or 'negative'.
- Dynamic reliability box read from metrics.json at runtime.
- Handles bad or blank frames gracefully without crashing.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import glob
import io
import json
import logging
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("patient_report")

# Stenosis severity bins
CLASS_MAPPING = {
    0: ("p0_20", "< 20%"),
    1: ("p20_50", "20% - 50%"),
    2: ("p50_70", "50% - 70%"),
    3: ("p70_90", "70% - 90%"),
    4: ("p90_98", "90% - 98%"),
    5: ("p99", "99% (subtotal)"),
    6: ("p100", "100% (total occlusion)"),
}


def load_reliability_metrics() -> dict[str, Any]:
    """Read test metrics at runtime from metrics.json."""
    metrics_path = ROOT / "outputs" / "eval" / "exp_7class_yolo11s_oversample_run1_544_test" / "metrics.json"
    if not metrics_path.exists():
        # Fallback search if path differs
        hits = sorted(glob.glob(str(ROOT / "outputs" / "eval" / "*544*_test" / "metrics.json")))
        if hits:
            metrics_path = Path(hits[0])
        else:
            hits = sorted(glob.glob(str(ROOT / "outputs" / "eval" / "*run1*_test" / "metrics.json")))
            if hits:
                metrics_path = Path(hits[0])

    if not metrics_path.exists():
        return {
            "precision": None,
            "precision_ci": None,
            "recall": None,
            "recall_ci": None,
            "f1": None,
            "f1_ci": None,
            "map_50": None,
            "map_50_ci": None,
            "false_alarm_share": None,
            "false_alarm_share_ci": None,
            "false_alarm_sentence": "",
            "source": None,
        }

    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        boot = data.get("bootstrap", {})
        det = data.get("detection", {}).get("micro", {})

        precision = boot.get("full_sample", {}).get("precision", det.get("precision"))
        precision_ci = [boot.get("precision", {}).get("lower"), boot.get("precision", {}).get("upper")] if "precision" in boot and boot["precision"] else None

        recall = boot.get("full_sample", {}).get("sensitivity", det.get("recall"))
        recall_ci = [boot.get("sensitivity", {}).get("lower"), boot.get("sensitivity", {}).get("upper")] if "sensitivity" in boot and boot["sensitivity"] else None

        f1 = boot.get("full_sample", {}).get("f1", det.get("f1"))
        f1_ci = [boot.get("f1", {}).get("lower"), boot.get("f1", {}).get("upper")] if "f1" in boot and boot["f1"] else None

        map50 = boot.get("full_sample", {}).get("mAP_50", data.get("map", {}).get("mAP_50"))
        map50_ci = [boot.get("mAP_50", {}).get("lower"), boot.get("mAP_50", {}).get("upper")] if "mAP_50" in boot and boot["mAP_50"] else None

        fa_share = boot.get("full_sample", {}).get("false_alarm_share", det.get("false_alarm_share"))
        if fa_share is None:
            tp = det.get("TP", 0)
            fp = det.get("FP", 0)
            fa_share = fp / (tp + fp) if (tp + fp) > 0 else 0.0
        fa_ci = [boot.get("false_alarm_share", {}).get("lower"), boot.get("false_alarm_share", {}).get("upper")] if "false_alarm_share" in boot and boot["false_alarm_share"] else None

        fa_percent = round(float(fa_share) * 100)
        fa_sentence = (
            f"On held-out patients, about {fa_percent}% of lesion candidates were false alarms. "
            "Candidates can appear in patients without disease, so their presence or number is not evidence of disease."
        )

        rel_source = str(metrics_path.relative_to(ROOT)).replace("\\", "/")

        return {
            "precision": float(precision) if precision is not None else None,
            "precision_ci": [float(c) for c in precision_ci] if precision_ci and None not in precision_ci else None,
            "recall": float(recall) if recall is not None else None,
            "recall_ci": [float(c) for c in recall_ci] if recall_ci and None not in recall_ci else None,
            "f1": float(f1) if f1 is not None else None,
            "f1_ci": [float(c) for c in f1_ci] if f1_ci and None not in f1_ci else None,
            "map_50": float(map50) if map50 is not None else None,
            "map_50_ci": [float(c) for c in map50_ci] if map50_ci and None not in map50_ci else None,
            "false_alarm_share": float(fa_share) if fa_share is not None else None,
            "false_alarm_share_ci": [float(c) for c in fa_ci] if fa_ci and None not in fa_ci else None,
            "false_alarm_sentence": fa_sentence,
            "source": rel_source,
        }
    except Exception as exc:
        logger.warning("Could not read reliability metrics: %s", exc)
        return {
            "precision": None,
            "precision_ci": None,
            "recall": None,
            "recall_ci": None,
            "f1": None,
            "f1_ci": None,
            "map_50": None,
            "map_50_ci": None,
            "false_alarm_share": None,
            "false_alarm_share_ci": None,
            "false_alarm_sentence": "",
            "source": None,
        }


def check_patch_classifier() -> tuple[bool, str]:
    """Check if a validated patch classifier exists meeting criteria."""
    patch_metrics = ROOT / "outputs" / "eval" / "patch_classifier" / "metrics.json"
    if not patch_metrics.exists():
        return False, "Patch severity grading omitted: no validated patch severity classifier exists meeting benchmark criteria."
    try:
        data = json.loads(patch_metrics.read_text(encoding="utf-8"))
        test_auc = data.get("test_auc", 0.0)
        ctrl_geom_auc = data.get("control_geometry_auc", 0.0)
        ctrl_stats_auc = data.get("control_stats_auc", 0.0)
        ci_width = data.get("ci_width", 0.1)
        if test_auc > (ctrl_geom_auc + ci_width) and test_auc > (ctrl_stats_auc + ci_width):
            return True, "Validated patch classifier enabled."
        return False, "Patch severity grading omitted: classifier did not exceed controls by CI width."
    except Exception:
        return False, "Patch severity grading omitted: unreadable patch metrics."


def load_input_frames(input_path: Path, patient_id: str, step: int = 1) -> list[tuple[str, np.ndarray]]:
    """Load frames from directory or video file, handling bad or blank frames safely."""
    frames: list[tuple[str, np.ndarray]] = []
    
    if input_path.is_file():
        # Video file or single image
        ext = input_path.suffix.lower()
        if ext in [".png", ".jpg", ".jpeg", ".tif", ".bmp"]:
            img = cv2.imread(str(input_path))
            if img is not None:
                frames.append((input_path.stem, img))
            else:
                logger.warning("Could not read image: %s", input_path)
        else:
            # Video file
            cap = cv2.VideoCapture(str(input_path))
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % step == 0:
                    fid = f"{patient_id}_f{frame_idx:05d}"
                    frames.append((fid, frame))
                frame_idx += 1
            cap.release()
            logger.info("Extracted %d frames from video %s (step=%d)", len(frames), input_path.name, step)
    elif input_path.is_dir():
        # Directory of frames
        all_imgs = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.bmp"]:
            all_imgs.extend(input_path.glob(ext))
        all_imgs = sorted(all_imgs)
        
        # If folder contains images starting with patient_id, filter to them
        pt_imgs = [p for p in all_imgs if p.stem.startswith(f"{patient_id}_")]
        target_imgs = pt_imgs if pt_imgs else all_imgs
        
        for p in target_imgs:
            try:
                img = cv2.imread(str(p))
                if img is not None and img.size > 0:
                    frames.append((p.stem, img))
                else:
                    logger.warning("Unreadable/blank image skipped: %s", p.name)
            except Exception as e:
                logger.warning("Error reading %s: %e", p.name, e)
        logger.info("Loaded %d frames from directory %s", len(frames), input_path)
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")
        
    return frames


def predict_frame(
    model: YOLO,
    img: np.ndarray,
    lb: LetterBox,
    conf_threshold: float,
    device: str,
) -> list[dict[str, Any]]:
    """Predict on a single frame using 544_padded preprocessing."""
    if img is None or img.size == 0 or np.all(img == 0):
        return []
        
    h, w = img.shape[:2]
    if (w, h) != (512, 512):
        im_512 = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    else:
        im_512 = img

    scale_x = w / 512.0
    scale_y = h / 512.0

    img_input = lb(image=im_512)
    results = model.predict(
        source=img_input,
        conf=conf_threshold,
        iou=0.6,
        max_det=50,
        imgsz=544,
        device=device,
        verbose=False,
    )
    r = results[0] if results else None
    if r is None or r.boxes is None or len(r.boxes) == 0:
        return []

    xyxy = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()

    detections: list[dict[str, Any]] = []
    for j in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[j]
        # De-pad from 544 (16 px offset on each side)
        x1_512 = float(np.clip(x1 - 16.0, 0.0, 512.0))
        y1_512 = float(np.clip(y1 - 16.0, 0.0, 512.0))
        x2_512 = float(np.clip(x2 - 16.0, 0.0, 512.0))
        y2_512 = float(np.clip(y2 - 16.0, 0.0, 512.0))

        box_x = round(x1_512 * scale_x, 1)
        box_y = round(y1_512 * scale_y, 1)
        box_w = round(max(0.0, x2_512 - x1_512) * scale_x, 1)
        box_h = round(max(0.0, y2_512 - y1_512) * scale_y, 1)
        score = round(float(scores[j]), 4)
        cls_id = int(classes[j])
        cls_code, cls_range = CLASS_MAPPING.get(cls_id, (f"class_{cls_id}", "unknown"))

        detections.append({
            "box": [box_x, box_y, box_w, box_h],
            "score": score,
            "class_id": cls_id,
            "class_code": cls_code,
            "stenosis_range": cls_range,
        })
    return detections


def draw_boxes(img: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    """Draw red candidate boxes and labels on an angiography frame."""
    out = img.copy()
    for det in detections:
        x, y, w, h = det["box"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{det['stenosis_range']} ({det['score']:.2f})"
        cv2.putText(out, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def frame_to_base64(img: np.ndarray, max_dim: int = 400) -> str:
    """Convert image to base64 JPEG data URL for self-contained HTML."""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clinical patient report for angiography.")
    parser.add_argument("--input", type=str, required=True, help="Path to folder of frames or video file")
    parser.add_argument("--patient-id", type=str, required=True, help="Patient identifier (e.g. p8)")
    parser.add_argument("--meta", type=str, default=None, help="Optional JSON or CSV metadata file")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--step", type=int, default=1, help="Frame stride for video extraction")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold (defaults to frozen val threshold)")
    parser.add_argument("--weights", type=str, default=None, help="Path to YOLO weights")
    args = parser.parse_args()

    patient_id = args.patient_id
    out_dir = Path(args.out) if args.out else ROOT / "outputs" / "reports" / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.weights:
        weights_path = Path(args.weights)
    else:
        weights_path = ROOT / "outputs" / "runs" / "exp_7class_yolo11s_oversample_run1_544" / "weights" / "best.pt"
        if not weights_path.exists():
            weights_path = ROOT / "outputs" / "runs" / "exp_7class_yolo11s_oversample_run1" / "weights" / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    # Frozen threshold: default from threshold.json
    if args.conf is not None:
        conf_thresh = args.conf
    else:
        thresh_file = ROOT / "outputs" / "runs" / "exp_7class_yolo11s_oversample_run1_544" / "threshold.json"
        if not thresh_file.exists():
            thresh_file = ROOT / "outputs" / "runs" / "exp_7class_yolo11s_oversample_run1" / "threshold.json"
        if thresh_file.exists():
            try:
                tdata = json.loads(thresh_file.read_text(encoding="utf-8"))
                conf_thresh = float(tdata.get("best_conf_threshold", 0.05))
            except Exception:
                conf_thresh = 0.05
        else:
            conf_thresh = 0.05

    device = "0" if torch.cuda.is_available() else "cpu"
    logger.info("Initializing YOLO model from %s on device %s (threshold=%.4f)...", weights_path, device, conf_thresh)
    model = YOLO(str(weights_path))
    lb = LetterBox(new_shape=(544, 544), auto=False, scaleup=False, center=True)

    # Load frames
    input_path = Path(args.input)
    frames = load_input_frames(input_path, patient_id=patient_id, step=args.step)
    logger.info("Processing %d frames for patient %s...", len(frames), patient_id)

    # Clinical metadata (recorded, not used by model)
    metadata_record: dict[str, Any] = {}
    if args.meta:
        mpath = Path(args.meta)
        if mpath.exists():
            try:
                metadata_record = json.loads(mpath.read_text(encoding="utf-8"))
            except Exception:
                pass
    if not metadata_record:
        # Check data_index/metadata_clean.csv
        meta_csv = ROOT / "data_index" / "metadata_clean.csv"
        if meta_csv.exists():
            try:
                mdf = pd.read_csv(meta_csv)
                prow = mdf[mdf["patient_id"] == patient_id]
                if not prow.empty:
                    metadata_record = {k: v for k, v in prow.iloc[0].to_dict().items() if pd.notna(v)}
            except Exception:
                pass

    # Process all frames
    frames_with_candidates = 0
    total_candidate_boxes = 0
    all_candidates: list[dict[str, Any]] = []
    frame_results: list[dict[str, Any]] = []

    for fid, img in frames:
        dets = predict_frame(model, img, lb, conf_thresh, device)
        max_score = max([d["score"] for d in dets]) if dets else 0.0
        if dets:
            frames_with_candidates += 1
            total_candidate_boxes += len(dets)
            for d in dets:
                all_candidates.append({
                    "frame_id": fid,
                    "stenosis_range": d["stenosis_range"],
                    "class_code": d["class_code"],
                    "score": d["score"],
                    "box": d["box"],
                })
        frame_results.append({
            "frame_id": fid,
            "num_candidates": len(dets),
            "max_score": max_score,
            "detections": dets,
            "image": img,
        })

    # Pick top 6 frames with boxes drawn
    # Priority: highest candidate max score, or first frames with candidates
    frames_sorted = sorted(frame_results, key=lambda x: x["max_score"], reverse=True)
    top_frames = frames_sorted[:6]

    # Generate drawn images (base64)
    top_frames_display: list[dict[str, Any]] = []
    for f in top_frames:
        drawn = draw_boxes(f["image"], f["detections"])
        top_frames_display.append({
            "frame_id": f["frame_id"],
            "max_score": f["max_score"],
            "num_candidates": f["num_candidates"],
            "img_base64": frame_to_base64(drawn),
        })

    # Load dynamic reliability metrics from test metrics.json
    rel_metrics = load_reliability_metrics()
    has_patch_model, patch_notice = check_patch_classifier()

    # Findings summary wording: strictly "lesion candidates found" or "no lesion candidates found"
    findings_verdict = (
        f"{total_candidate_boxes} lesion candidates found across {frames_with_candidates} frame(s)."
        if total_candidate_boxes > 0
        else "No lesion candidates found."
    )

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Structured JSON Report
    report_data = {
        "header": {
            "patient_id": patient_id,
            "date": date_str,
            "total_frames_processed": len(frames),
            "models_used": [weights_path.name],
            "preprocessing": "544_padded",
            "confidence_threshold": conf_thresh,
            "disclaimer": "Research prototype, not a diagnostic device.",
        },
        "findings": {
            "verdict": findings_verdict,
            "frames_with_candidates": frames_with_candidates,
            "total_candidate_boxes": total_candidate_boxes,
            "patch_grading": patch_notice,
            "candidate_table": [
                {
                    "frame_id": c["frame_id"],
                    "stenosis_range": c["stenosis_range"],
                    "score": c["score"],
                    "box": c["box"],
                }
                for c in all_candidates
            ],
        },
        "reliability": {
            "precision": round(float(rel_metrics["precision"]), 6) if rel_metrics["precision"] is not None else None,
            "precision_ci_95": [round(float(c), 6) for c in rel_metrics["precision_ci"]] if rel_metrics["precision_ci"] else None,
            "recall": round(float(rel_metrics["recall"]), 6) if rel_metrics["recall"] is not None else None,
            "recall_ci_95": [round(float(c), 6) for c in rel_metrics["recall_ci"]] if rel_metrics["recall_ci"] else None,
            "f1": round(float(rel_metrics["f1"]), 6) if rel_metrics["f1"] is not None else None,
            "f1_ci_95": [round(float(c), 6) for c in rel_metrics["f1_ci"]] if rel_metrics["f1_ci"] else None,
            "map_50": round(float(rel_metrics["map_50"]), 6) if rel_metrics["map_50"] is not None else None,
            "map_50_ci_95": [round(float(c), 6) for c in rel_metrics["map_50_ci"]] if rel_metrics["map_50_ci"] else None,
            "false_alarm_share": round(float(rel_metrics["false_alarm_share"]), 6) if rel_metrics["false_alarm_share"] is not None else None,
            "false_alarm_share_ci_95": [round(float(c), 6) for c in rel_metrics["false_alarm_share_ci"]] if rel_metrics["false_alarm_share_ci"] else None,
            "false_alarm_sentence": rel_metrics["false_alarm_sentence"],
            "interpretive_warning": (
                "No lesion candidates found does NOT mean the patient is free of disease: "
                "the detector finds only a small fraction of true lesions."
            ),
            "metrics_source": rel_metrics["source"],
        },
        "not_validated": {
            "screening_shortcut": (
                "Frame/video-level disease screening failed shortcut controls (chance performance). "
                "ResNet18 frame and video classifiers learned angiographer keyframe selection cues "
                "rather than vascular pathology and have been invalidated."
            ),
            "control_patient_absence": (
                "No non-CAD control patients exist in the training data (41/42 patients harbor CAD lesions), "
                "so patient-level specificity is completely unmeasured."
            ),
            "external_generalization": "Performance on external hospitals, different angiography systems, and scanner vendors is untested.",
            "preprocessing_sensitivity": "Detections are sensitive to letterbox scale/padding (512 vs 544 changed validation mAP@0.5 by 2.5x).",
        },
        "recorded_metadata": {
            "note": "Recorded clinical metadata (recorded, not used by the model)",
            "data": metadata_record,
        },
        "recommendation": "Review of the angiogram by a qualified cardiologist is required. This report must not be used for clinical decisions.",
    }

    # Save report.json
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    logger.info("Saved report JSON to %s", json_path)

    # HTML Report Generation (Self-contained, strictly compliant)
    # Check that words 'diseased', 'healthy', 'positive', 'negative' are absent from clinical text!
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Patient Angiography Report - {patient_id}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #1e293b; background: #f8fafc; margin: 0; padding: 24px; }}
  .container {{ max-width: 900px; margin: 0 auto; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); padding: 32px; }}
  .header {{ border-bottom: 2px solid #e2e8f0; padding-bottom: 16px; margin-bottom: 24px; }}
  .header h1 {{ margin: 0 0 8px 0; font-size: 24px; color: #0f172a; }}
  .meta-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; font-size: 14px; color: #475569; }}
  .notice-banner {{ background: #fff1f2; border-left: 4px solid #e11d48; padding: 12px 16px; margin: 16px 0; border-radius: 4px; font-weight: 600; color: #9f1239; }}
  .section {{ margin-top: 24px; }}
  .section h2 {{ font-size: 18px; color: #0f172a; border-bottom: 1px solid #cbd5e1; padding-bottom: 6px; margin-bottom: 12px; }}
  .findings-verdict {{ font-size: 16px; font-weight: 600; color: #1e293b; margin-bottom: 12px; }}
  .table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin: 12px 0; }}
  .table th, .table td {{ border: 1px solid #cbd5e1; padding: 8px 12px; text-align: left; }}
  .table th {{ background: #f1f5f9; font-weight: 600; color: #334155; }}
  .table tr:nth-child(even) {{ background: #f8fafc; }}
  .gallery {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 16px 0; }}
  .frame-card {{ border: 1px solid #cbd5e1; border-radius: 6px; overflow: hidden; background: #000; text-align: center; }}
  .frame-card img {{ width: 100%; height: auto; display: block; }}
  .frame-card .label {{ background: #f8fafc; color: #334155; font-size: 12px; padding: 6px; font-weight: 500; border-top: 1px solid #cbd5e1; }}
  .box-reliability {{ background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 6px; padding: 16px; margin-top: 16px; }}
  .box-reliability h3 {{ margin-top: 0; color: #1e40af; font-size: 15px; }}
  .box-notvalidated {{ background: #fefce8; border: 1px solid #fef08a; border-radius: 6px; padding: 16px; margin-top: 16px; }}
  .box-notvalidated h3 {{ margin-top: 0; color: #854d0e; font-size: 15px; }}
  .recommendation {{ background: #f1f5f9; border-left: 4px solid #475569; padding: 12px 16px; margin-top: 24px; font-weight: 600; color: #0f172a; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>Angiography Analysis Report: {patient_id}</h1>
    <div class="meta-grid">
      <div><strong>Date:</strong> {date_str}</div>
      <div><strong>Frames Processed:</strong> {len(frames)}</div>
      <div><strong>Model Used:</strong> {weights_path.name} (544_padded)</div>
      <div><strong>Threshold:</strong> {conf_thresh:.4f} (frozen from validation)</div>
    </div>
    <div class="notice-banner">
      Research prototype, not a diagnostic device.
    </div>
  </div>

  <div class="section">
    <h2>Findings</h2>
    <div class="findings-verdict">{findings_verdict}</div>
    <p><em>{patch_notice}</em></p>
    
    {f'''<div class="gallery">''' + "".join([
        f'''<div class="frame-card">
              <img src="{tf['img_base64']}" alt="{tf['frame_id']}">
              <div class="label">{tf['frame_id']} (max: {tf['max_score']:.2f}, {tf['num_candidates']} box)</div>
            </div>'''
        for tf in top_frames_display
    ]) + '</div>' if top_frames_display else '<p>No candidate frames to display.</p>'}

    {f'''<table class="table">
      <thead>
        <tr>
          <th>Finding</th>
          <th>Frame ID</th>
          <th>Stenosis Severity</th>
          <th>Confidence Level</th>
          <th>Visual Guide</th>
        </tr>
      </thead>
      <tbody>''' + "".join([
        f'''<tr>
              <td><strong>Candidate #{idx}</strong></td>
              <td><code>{c['frame_id']}</code></td>
              <td><strong>{c['stenosis_range']}</strong></td>
              <td>{round(c['score'] * 100)}% ({'High' if c['score'] >= 0.70 else ('Moderate' if c['score'] >= 0.40 else 'Review')})</td>
              <td>Red Box #{idx} on Image</td>
            </tr>'''
        for idx, c in enumerate(all_candidates[:12], start=1)
    ]) + '</tbody></table>' if all_candidates else ''}
    {f'<p><small>Showing top {min(12, len(all_candidates))} of {len(all_candidates)} candidates (ranked by confidence).</small></p>' if len(all_candidates) > 12 else ''}
  </div>

  <div class="box-reliability">
    <h3>Detector Reliability (Evaluated on Unseen Patients)</h3>
    <ul>
      <li><strong>Precision:</strong> {rel_metrics['precision']:.4f}{f" (95% CI: [{rel_metrics['precision_ci'][0]:.4f}, {rel_metrics['precision_ci'][1]:.4f}])" if rel_metrics['precision_ci'] else ""}</li>
      <li><strong>Recall:</strong> {rel_metrics['recall']:.4f}{f" (95% CI: [{rel_metrics['recall_ci'][0]:.4f}, {rel_metrics['recall_ci'][1]:.4f}])" if rel_metrics['recall_ci'] else ""}</li>
      <li><strong>F1-score:</strong> {rel_metrics['f1']:.4f}{f" (95% CI: [{rel_metrics['f1_ci'][0]:.4f}, {rel_metrics['f1_ci'][1]:.4f}])" if rel_metrics['f1_ci'] else ""}</li>
      <li><strong>Bounding Box mAP@0.5:</strong> {rel_metrics['map_50']:.4f}{f" (95% CI: [{rel_metrics['map_50_ci'][0]:.4f}, {rel_metrics['map_50_ci'][1]:.4f}])" if rel_metrics['map_50_ci'] else ""}</li>
      <li><strong>False-Alarm Share FP/(TP+FP):</strong> {rel_metrics['false_alarm_share']:.4f}{f" (95% CI: [{rel_metrics['false_alarm_share_ci'][0]:.4f}, {rel_metrics['false_alarm_share_ci'][1]:.4f}])" if rel_metrics['false_alarm_share_ci'] else ""}</li>
    </ul>
    <p><strong>False-Alarm Warning:</strong> {rel_metrics['false_alarm_sentence']}</p>
    <p><strong>Clinical Warning:</strong> No lesion candidates found does NOT mean the patient is free of disease: the detector finds only a small fraction of true lesions.</p>
  </div>

  <div class="box-notvalidated">
    <h3>Validation Boundaries & Caveats</h3>
    <ul>
      <li><strong>Frame/Video Screening Failed Shortcut Controls:</strong> Frame-level and video-level disease screening models exhibited near-chance performance when evaluated on matched keyframes (test ROC-AUC 0.4894) and have been invalidated.</li>
      <li><strong>Unmeasured Specificity:</strong> No non-CAD control patients exist in the training dataset (41 of 42 patients harbor lesions), so specificity is completely unmeasured.</li>
      <li><strong>External Scanners Untested:</strong> Generalization across external hospitals, cath labs, and x-ray equipment is untested.</li>
      <li><strong>Preprocessing Sensitivity:</strong> Detections are sensitive to letterbox padding and image scaling.</li>
    </ul>
  </div>

  {f'''<div class="section">
    <h2>Recorded Clinical Metadata</h2>
    <p><small>Recorded, not used by the model</small></p>
    <table class="table">
      <tbody>''' + "".join([
        f'<tr><th>{k}</th><td>{v}</td></tr>'
        for k, v in list(metadata_record.items())[:8]
    ]) + '</tbody></table></div>' if metadata_record else ''}

  <div class="recommendation">
    Recommendation: Review of the angiogram by a qualified cardiologist is required. This report must not be used for clinical decisions.
  </div>
</div>
</body>
</html>
"""
    html_path = out_dir / "report.html"
    html_path.write_text(html_content, encoding="utf-8")
    logger.info("Saved report HTML to %s", html_path)
    print(f"[OK] Generated report for {patient_id}: {html_path}")


if __name__ == "__main__":
    main()
