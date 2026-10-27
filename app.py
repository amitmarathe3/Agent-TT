#!/usr/bin/env python3
"""Research prototype: YOLO stenosis detection demo (NOT a diagnostic device).

Loads the single trained run's weights, runs inference at imgsz=512 / iou=0.7 /
max_det=100, and shows annotated output plus a detection table.

Numbers in the banner are read at runtime from
outputs/eval/*run1*_val/metrics.json -- never typed by hand.

Run: scripts/run_app.bat
"""

from __future__ import annotations

import glob
import json
import random
import threading
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- config
_cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
_WEIGHTS_DEFAULT = "outputs/runs/exp_7class_yolo11s_oversample_run1/weights/best.pt"
WEIGHTS = ROOT / _cfg.get("yolo_weights", _WEIGHTS_DEFAULT)
_SCHEME: dict[str, int] = dict(_cfg["class_schemes"]["schemes"]["7class"])
ID2NAME = {v: k for k, v in _SCHEME.items()}

# Readable stenosis ranges for clinical presentation.
RANGE = {
    "p0_20": "Minimal Narrowing (< 20%)",
    "p20_50": "Mild Stenosis (20% - 50%)",
    "p50_70": "Moderate Stenosis (50% - 70%)",
    "p70_90": "Severe Stenosis (70% - 90%)",
    "p90_98": "Critical Stenosis (90% - 98%)",
    "p99": "Subtotal Occlusion (99%)",
    "p100": "Total Occlusion (100%)",
}

SHORT_RANGE = {
    "p0_20": "< 20%",
    "p20_50": "20-50%",
    "p50_70": "50-70%",
    "p70_90": "70-90%",
    "p90_98": "90-98%",
    "p99": "99%",
    "p100": "100%",
}

RUN_GLOB = "*run1*"
YOLO_DIR = ROOT / "outputs" / "yolo_dataset"
TEST_IMAGES = (
    (YOLO_DIR / "test" / "images")
    if (YOLO_DIR / "test" / "images").exists()
    else (YOLO_DIR / "images" / "test")
)
TEST_LABELS = (
    (YOLO_DIR / "test" / "labels")
    if (YOLO_DIR / "test" / "labels").exists()
    else (YOLO_DIR / "labels" / "test")
)
THRESH_FILE = ROOT / "outputs" / "runs" / (
    _cfg.get("run_name", "exp_7class_yolo11s_oversample_run1") + "/threshold.json"
)

IMGSZ, NMS_IOU, MAX_DET = 512, 0.7, 100


# ------------------------------------------------- runtime-read banner
def _val_metrics_path() -> Path | None:
    hits = sorted(glob.glob(str(ROOT / "outputs" / "eval" / f"{RUN_GLOB}_val" / "metrics.json")))
    return Path(hits[0]) if hits else None


def banner_text() -> str:
    """Banner with numbers read at runtime from the val metrics.json."""
    p = _val_metrics_path()
    if p is None or not p.exists():
        return (
            "Research prototype, not a diagnostic device. Validation metrics "
            "unavailable. Frame-level screening is unvalidated. "
            "Results depend on preprocessing (512 vs 544 changed val mAP50 about 2.5x). "
            "Detections are NOT reliable."
        )
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
        map50 = float(m["map"]["mAP_50"])
        return (
            f"**Research prototype, not a diagnostic device.** "
            f"Val mAP@0.5 **{map50:.4f}**. Frame-level screening is unvalidated. "
            f"Results depend on preprocessing (512 vs 544 changed val mAP50 about 2.5x). "
            f"**Detections are NOT reliable.**"
        )
    except Exception:  # pragma: no cover - banner must never crash
        return (
            "Research prototype, not a diagnostic device. Validation metrics "
            "could not be read. Frame-level screening is unvalidated. "
            "Results depend on preprocessing (512 vs 544 changed val mAP50 about 2.5x). "
            "Detections are NOT reliable."
        )


def frozen_threshold() -> float:
    """Frozen threshold from the run's threshold.json (0.05 if unreadable)."""
    try:
        return float(json.loads(THRESH_FILE.read_text(encoding="utf-8"))["best_conf_threshold"])
    except Exception:
        return 0.05


# ------------------------------------------------------------- model
_device: str | None = None
_model = None
_lock = threading.Lock()


def _pick_device() -> str:
    """GPU if this 3.14 Python reports cuda True, else CPU."""
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_model():
    """Load (and cache) the YOLO model. Falls back CPU->GPU OOM safely."""
    global _model, _device
    with _lock:
        if _model is not None:
            return _model
        from ultralytics import YOLO

        if not WEIGHTS.exists():
            raise FileNotFoundError(f"weights not found: {WEIGHTS}")
        _device = _pick_device()
        model = YOLO(str(WEIGHTS))
        try:
            model.to(_device)
        except Exception:
            _device = "cpu"
        _model = model
        return _model


def active_device() -> str:
    get_model()
    return str(_device)


# ------------------------------------------------- prediction core
def _read_gt(fid: str) -> list[list[float]]:
    """Ground-truth boxes for a test frame -> [[x,y,w,h,cls], ...] absolute px."""
    lbl = TEST_LABELS / f"{fid}.txt"
    if not lbl.exists():
        return []
    try:
        with Image.open(TEST_IMAGES / f"{fid}.png") as im:
            W, H = im.size
    except Exception:
        return []
    out: list[list[float]] = []
    for line in lbl.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            c = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        out.append([(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H, c])
    return out


def _box_iou(box1: list[float], box2: list[float]) -> float:
    """Compute IoU between two [x, y, w, h] boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def deduplicate_detections(detections: list[dict[str, Any]], iou_thresh: float = 0.35) -> list[dict[str, Any]]:
    """Suppress redundant overlapping boxes around the same lesion, keeping the highest confidence."""
    sorted_dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    keep: list[dict[str, Any]] = []
    for d in sorted_dets:
        box_d = [d["x"], d["y"], d["w"], d["h"]]
        if not any(_box_iou(box_d, [k["x"], k["y"], k["w"], k["h"]]) > iou_thresh for k in keep):
            keep.append(d)
    return keep


def run_prediction(
    image: Any, threshold: float | None = None, preprocess: str = "544_padded"
) -> dict[str, Any]:
    """Run inference on one image. NEVER raises -- returns a result dict.

    Returns keys: ok, error, n_detections, detections, frame_score, threshold,
    verdict, image_id.
    """
    thr = frozen_threshold() if threshold is None else float(threshold)
    result: dict[str, Any] = {
        "ok": True,
        "error": None,
        "image_id": None,
        "n_detections": 0,
        "detections": [],
        "frame_score": None,
        "threshold": thr,
        "verdict": "NOT FLAGGED",
    }
    try:
        if image is None:
            result["error"] = "no image provided"
            return result

        if isinstance(image, (str, Path)):
            result["image_id"] = Path(image).stem
            image = Image.open(image)

        # Accept PIL / numpy / raw bytes-ish input defensively.
        if isinstance(image, np.ndarray):
            if image.size == 0:
                result["error"] = "blank image"
                return result
            if image.dtype != np.uint8:
                image = np.clip(image.astype("float32") * (255.0 if image.max() <= 1.0 else 1.0), 0, 255)
                image = image.astype(np.uint8)
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            result["error"] = f"unsupported input type: {type(image).__name__}"
            return result

        image = image.convert("RGB")
        if image.size[0] == 0 or image.size[1] == 0:
            result["error"] = "blank image"
            return result

        arr = np.asarray(image)
        if arr.size == 0 or not np.isfinite(arr).all():
            result["error"] = "blank image"
            return result

        from src.models.yolo_infer import predict_single_image

        model = get_model()
        raw_dets = predict_single_image(
            model=model,
            image=arr,
            conf=0.001,
            iou=NMS_IOU,
            max_det=MAX_DET,
            device=_device,
            preprocess=preprocess,
        )

        raw_dets.sort(key=lambda d: d[4], reverse=True)
        all_dets: list[dict[str, Any]] = []
        for x, y, w, h, sc, cls_id in raw_dets:
            cname = ID2NAME.get(cls_id, "")
            all_dets.append(
                {
                    "class": cname,
                    "stenosis_range": RANGE.get(cname, "Unknown"),
                    "short_range": SHORT_RANGE.get(cname, "Lesion"),
                    "score": round(float(sc), 4),
                    "x": round(float(x), 1),
                    "y": round(float(y), 1),
                    "w": round(float(w), 1),
                    "h": round(float(h), 1),
                }
            )

        frame_score = max((d["score"] for d in all_dets), default=0.0)
        result["frame_score"] = round(float(frame_score), 4) if all_dets else None

        # Filter detections at chosen threshold and deduplicate overlapping candidate boxes
        filtered = [d for d in all_dets if d["score"] >= thr]
        distinct_dets = deduplicate_detections(filtered, iou_thresh=0.35)
        for idx, d in enumerate(distinct_dets, start=1):
            d["index"] = idx

        result["n_detections"] = len(distinct_dets)
        result["detections"] = distinct_dets
        result["verdict"] = "CANDIDATE DETECTED" if (frame_score is not None and frame_score >= thr) else "NO CANDIDATES"
        return result
    except Exception as exc:  # noqa: BLE001 - prototype must never crash
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["detections"] = []
        result["n_detections"] = 0
        result["frame_score"] = None
        result["verdict"] = "ERROR"
        return result


def annotate(image: Any, result: dict[str, Any], draw_gt: list[list[float]] | None = None) -> Any:
    """Draw detections (red) and optional ground truth (green)."""
    try:
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        img = image.convert("RGB").copy()
        d = ImageDraw.Draw(img)

        for gt in (draw_gt or []):
            x, y, w, h, _c = gt
            d.rectangle([x, y, x + w, y + h], outline=(0, 220, 0), width=3)

        for det in result.get("detections", []):
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            idx = det.get("index", 1)
            sc = det["score"]
            short_sev = det.get("short_range", det.get("stenosis_range", "").split("(")[0].strip())
            d.rectangle([x, y, x + w, y + h], outline=(255, 40, 40), width=3)
            # Clean label on the box: e.g. "#1: 70-90% (83%)"
            badge_text = f"#{idx}: {short_sev} ({round(sc * 100)}%)"
            d.rectangle([x, max(0, y - 16), x + len(badge_text) * 7 + 6, max(0, y)], fill=(220, 20, 60))
            d.text((x + 3, max(0, y - 14)), badge_text, fill=(255, 255, 255))
        return img
    except Exception:
        return image


def random_test_sample() -> tuple[Any, str, list[list[float]]]:
    """Pick a random test-split image (returns image, id, ground-truth boxes)."""
    if TEST_IMAGES.exists():
        files = sorted(p for p in TEST_IMAGES.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    else:
        files = []
    if not files:
        return None, "no test images found", []
    p = random.choice(files)
    try:
        img = Image.open(p).convert("RGB")
    except Exception:
        return None, f"could not read {p.name}", []
    return img, p.stem, _read_gt(p.stem)


# ------------------------------------------------------------- UI
_TABLE_HEADERS = ["Finding", "Narrowing Severity", "Confidence Level", "Visual Marker"]


def _to_table(res: dict[str, Any]):
    rows = []
    for idx, d in enumerate(res.get("detections", []), start=1):
        sc = d["score"]
        conf_pct = f"{round(sc * 100)}%"
        conf_desc = "High" if sc >= 0.70 else ("Moderate" if sc >= 0.35 else "Low")
        conf_str = f"{conf_pct} ({conf_desc})"
        rows.append([
            f"Candidate #{idx}",
            d["stenosis_range"],
            conf_str,
            f"Box #{idx} (Red)",
        ])
    if not rows:
        rows = [["(No lesions detected)", "No arterial narrowing identified above threshold", "—", "—"]]
    return rows


def build_app():
    import gradio as gr

    with gr.Blocks(title="Coronary Angiogram Lesion Visualizer") as demo:
        gr.Markdown("# 🫀 Coronary Angiogram Lesion Visualizer")
        gr.Markdown(
            "**Visual Assistant for Coronary Artery Stenosis Detection**  \n"
            "*Upload an angiography frame or load a sample test image to visualize suspected coronary stenosis locations and severity.*"
        )
        with gr.Row():
            with gr.Column():
                in_img = gr.Image(type="pil", label="Angiogram Input (Upload or Select Sample)")
                with gr.Row():
                    rnd_btn = gr.Button("🎲 Load Sample Test Image", variant="secondary")
                    img_id = gr.Textbox(label="Sample ID", value="", interactive=False)
                thr = gr.Slider(
                    minimum=0.01, maximum=0.90, value=frozen_threshold(), step=0.01,
                    label="Detection Sensitivity Threshold (Frozen validation default: 0.05)",
                )
                run_btn = gr.Button("Analyze Frame", variant="primary")
            with gr.Column():
                out_img = gr.Image(type="pil", label="Analyzed Angiogram (Red: AI Detection, Green: Verified Reference)")
                verdict = gr.Markdown("### Ready for analysis\nUpload an angiogram or click **Load Sample Test Image**.")
                status = gr.Markdown("")
        tbl = gr.Dataframe(headers=_TABLE_HEADERS, label="Detected Lesion Candidates (Ranked by Severity & Confidence)", interactive=False)
        gr.Markdown(
            "<small>*Clinical Note: Automated detections are intended for visual review and educational assistance. "
            "Candidates can appear in non-diseased segments, and negative scans do not rule out pathology. Specialist review is required.*</small>"
        )

        state = gr.State(None)

        def _on_random():
            img, fid, gt = random_test_sample()
            return img, fid, gt

        def _run(img, t, gt):
            if img is None:
                return None, _to_table({"detections": []}), "### ⚠️ No Image Provided\nPlease upload an angiogram image or click 'Load Sample Test Image'.", ""
            res = run_prediction(img, t)
            if not res.get("ok"):
                return None, _to_table(res), f"### ⚠️ Error\n{res.get('error') or ''}", ""
            annotated = annotate(img, res, gt or [])
            if res["n_detections"] == 0:
                v = (
                    "### No Significant Lesion Candidates Detected\n"
                    f"No arterial narrowing detected above sensitivity threshold ({res['threshold']:.2f})."
                )
                st = "Finding: **Unremarkable above threshold**"
            else:
                top_d = res["detections"][0]
                n_det = res["n_detections"]
                v = (
                    f"### ⚠️ Suspected Lesion Identified ({n_det} distinct site{'s' if n_det > 1 else ''})\n"
                    f"**Primary finding:** {top_d['stenosis_range']} with **{round(top_d['score'] * 100)}% confidence**."
                )
                st = f"Identified **{n_det}** candidate lesion site(s) on this angiogram frame."
            return annotated, _to_table(res), v, st

        rnd_btn.click(_on_random, outputs=[in_img, img_id, state])
        run_btn.click(_run, inputs=[in_img, thr, state],
                      outputs=[out_img, tbl, verdict, status])
        thr.change(_run, inputs=[in_img, thr, state],
                   outputs=[out_img, tbl, verdict, status])
        in_img.change(_run, inputs=[in_img, thr, state],
                      outputs=[out_img, tbl, verdict, status])
    return demo


if __name__ == "__main__":
    print(banner_text().replace("**", ""))
    print(f"weights : {WEIGHTS}")
    print(f"device  : {_pick_device()} (cuda reported: {_pick_device() != 'cpu'})")
    print(f"frozen threshold: {frozen_threshold()}")
    build_app().launch(server_name="127.0.0.1", server_port=7860, show_error=True)
