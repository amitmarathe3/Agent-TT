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

# Readable stenosis ranges for the 7class scheme.
RANGE = {
    "p0_20": "0-20%",
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
            all_dets.append(
                {
                    "class": ID2NAME.get(cls_id, str(cls_id)),
                    "stenosis_range": RANGE.get(ID2NAME.get(cls_id, ""), "?"),
                    "score": round(float(sc), 4),
                    "x": round(float(x), 1),
                    "y": round(float(y), 1),
                    "w": round(float(w), 1),
                    "h": round(float(h), 1),
                }
            )

        frame_score = max((d["score"] for d in all_dets), default=0.0)
        result["frame_score"] = round(float(frame_score), 4) if all_dets else None

        # Filter detections at the chosen threshold
        dets = [d for d in all_dets if d["score"] >= thr]
        result["n_detections"] = len(dets)
        result["detections"] = dets
        result["verdict"] = "FLAGGED" if (frame_score is not None and frame_score >= thr) else "NOT FLAGGED"
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
            d.rectangle([x, y, x + w, y + h], outline=(255, 40, 40), width=3)
            d.text((x + 2, max(0, y - 12)), f"{det['stenosis_range']} {det['score']:.2f}",
                   fill=(255, 220, 0))
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
def _to_table(res: dict[str, Any]):
    rows = [
        [d["class"], d["stenosis_range"], d["score"], d["x"], d["y"], d["w"], d["h"]]
        for d in res.get("detections", [])
    ]
    if not rows:
        rows = [["(no detections)", "", "", "", "", "", ""]]
    return rows


_TABLE_HEADERS = ["Class", "Stenosis range", "Score", "x", "y", "w", "h"]


def build_app():
    import gradio as gr

    with gr.Blocks(title="CADICA research prototype") as demo:
        gr.Markdown(banner_text())
        gr.Markdown(
            f"weights `{WEIGHTS.name}` · imgsz {IMGSZ} · iou {NMS_IOU} · "
            f"max_det {MAX_DET} · frozen threshold `{frozen_threshold():.4f}`"
        )
        with gr.Row():
            with gr.Column():
                in_img = gr.Image(type="pil", label="Upload image")
                with gr.Row():
                    rnd_btn = gr.Button("🎲 Random test sample", variant="secondary")
                    img_id = gr.Textbox(label="Image id", value="", interactive=False)
                thr = gr.Slider(
                    minimum=0.001, maximum=0.95, value=frozen_threshold(), step=0.001,
                    label="Confidence threshold (frozen default from threshold.json)",
                )
                run_btn = gr.Button("Run detection", variant="primary")
            with gr.Column():
                out_img = gr.Image(type="pil", label="Annotated (red = detection, green = ground truth)")
                frame_score = gr.Number(label="Frame score (max box score)")
                verdict = gr.Markdown("### waiting for input")
                status = gr.Markdown("")
        tbl = gr.Dataframe(headers=_TABLE_HEADERS, label="Detections", interactive=False)

        state = gr.State(None)

        def _on_random():
            img, fid, gt = random_test_sample()
            return img, fid, gt

        def _run(img, t, gt):
            if img is None:
                return None, _to_table({"detections": []}), None, "### ⚠️ no image", "no image provided"
            res = run_prediction(img, t)
            if not res.get("ok"):
                return None, _to_table(res), None, "### ⚠️ error", res.get("error") or ""
            annotated = annotate(img, res, gt or [])
            if res["n_detections"] == 0:
                v = f"### ✅ NOT FLAGGED — **no detections**\nthreshold {res['threshold']:.3f}"
            elif res["verdict"] == "FLAGGED":
                v = f"### 🚩 FLAGGED\nthreshold {res['threshold']:.3f}"
            else:
                v = f"### ✅ NOT FLAGGED\nthreshold {res['threshold']:.3f}"
            st = (
                f"detections: **{res['n_detections']}** · frame score (max box): "
                f"**{res['frame_score']}**"
            )
            return annotated, _to_table(res), res["frame_score"], v, st

        rnd_btn.click(_on_random, outputs=[in_img, img_id, state])
        run_btn.click(_run, inputs=[in_img, thr, state],
                      outputs=[out_img, tbl, frame_score, verdict, status])
        thr.change(_run, inputs=[in_img, thr, state],
                   outputs=[out_img, tbl, frame_score, verdict, status])
        in_img.change(_run, inputs=[in_img, thr, state],
                      outputs=[out_img, tbl, frame_score, verdict, status])
    return demo


if __name__ == "__main__":
    print(banner_text().replace("**", ""))
    print(f"weights : {WEIGHTS}")
    print(f"device  : {_pick_device()} (cuda reported: {_pick_device() != 'cpu'})")
    print(f"frozen threshold: {frozen_threshold()}")
    build_app().launch(server_name="127.0.0.1", server_port=7860, show_error=True)
