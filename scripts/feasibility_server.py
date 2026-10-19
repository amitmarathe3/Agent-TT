"""
Feasibility Check API Server
Run: python scripts/feasibility_server.py
Then open: http://localhost:5050
"""
import os, sys, time, json, tempfile, traceback, base64, io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify, send_from_directory
import torch

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB max upload

DASHBOARD_DIR = Path(__file__).parent.parent / "outputs" / "feasibility_ui"
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

GPU_BUDGET_MB  = 4295   # GTX 1650 total VRAM
LATENCY_BUDGET = 100.0  # ms
N_WARMUP       = 5
N_RUNS         = 20
IMGSZ          = 512


def run_feasibility(pt_path: str) -> dict:
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    total_vram = (
        torch.cuda.get_device_properties(0).total_memory / 1e6
        if device == "cuda" else 0
    )

    pt_size_mb = os.path.getsize(pt_path) / 1e6

    model = YOLO(pt_path)
    model.to(device)

    n_params = sum(p.numel() for p in model.model.parameters())

    dummy = torch.zeros(1, 3, IMGSZ, IMGSZ).to(device)
    for _ in range(N_WARMUP):
        model.model(dummy)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    latencies = []
    for _ in range(N_RUNS):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_ms  = sum(latencies) / len(latencies)
    min_ms  = min(latencies)
    max_ms  = max(latencies)
    fps     = 1000.0 / avg_ms

    gpu_alloc   = torch.cuda.memory_allocated()  / 1e6 if device == "cuda" else 0
    gpu_reserved= torch.cuda.memory_reserved()   / 1e6 if device == "cuda" else 0
    gpu_peak    = torch.cuda.max_memory_allocated()/ 1e6 if device == "cuda" else 0

    feasible = gpu_peak < GPU_BUDGET_MB and avg_ms < LATENCY_BUDGET

    return {
        "model_path":       str(pt_path),
        "device":           device,
        "gpu_name":         gpu_name,
        "total_vram_mb":    round(total_vram, 0),
        "imgsz":            IMGSZ,
        "weights_size_mb":  round(pt_size_mb, 2),
        "n_parameters":     n_params,
        "latency_avg_ms":   round(avg_ms, 2),
        "latency_min_ms":   round(min_ms, 2),
        "latency_max_ms":   round(max_ms, 2),
        "throughput_fps":   round(fps, 1),
        "gpu_allocated_mb": round(gpu_alloc, 2),
        "gpu_reserved_mb":  round(gpu_reserved, 2),
        "gpu_peak_mb":      round(gpu_peak, 2),
        "feasible":         feasible,
        "latency_budget_ms":LATENCY_BUDGET,
        "gpu_budget_mb":    GPU_BUDGET_MB,
    }


@app.route("/")
def index():
    return send_from_directory(str(DASHBOARD_DIR), "index.html")


@app.route("/api/check", methods=["POST"])
def api_check():
    if "model" not in request.files:
        return jsonify({"error": "No model file uploaded"}), 400

    f = request.files["model"]
    if not f.filename.endswith(".pt"):
        return jsonify({"error": "Only .pt files are accepted"}), 400

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
        f.save(tmp_path)

    try:
        results = run_feasibility(tmp_path)
        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.route("/api/check-path", methods=["POST"])
def api_check_path():
    """Check a model by server-side path (for pre-loaded best.pt)."""
    data = request.get_json(force=True)
    pt_path = data.get("path", "")
    if not pt_path or not Path(pt_path).exists():
        return jsonify({"error": f"File not found: {pt_path}"}), 404
    try:
        results = run_feasibility(str(pt_path))
        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/infer", methods=["POST"])
def api_infer():
    """Run YOLO inference on a retina image and return annotated result as base64."""
    from ultralytics import YOLO
    from PIL import Image
    import numpy as np

    is_tmp = False
    img_path = None

    if "image" in request.files and request.files["image"].filename:
        img_file = request.files["image"]
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:
            img_path = tmp_img.name
            img_file.save(img_path)
            is_tmp = True
        model_path = request.form.get(
            "model_path",
            str(Path(__file__).parent.parent /
                "outputs/runs/exp_7class_yolo11s_oversample_run1/weights/best.pt")
        )
        conf_thresh = float(request.form.get("conf", 0.10))
    else:
        # Check json or form for image_path
        data = request.get_json(silent=True) or request.form or {}
        img_path = data.get("image_path")
        if not img_path:
            return jsonify({"error": "No image file uploaded or image_path specified"}), 400
        model_path = data.get(
            "model_path",
            str(Path(__file__).parent.parent /
                "outputs/runs/exp_7class_yolo11s_oversample_run1/weights/best.pt")
        )
        conf_thresh = float(data.get("conf", 0.10))

    if not Path(img_path).exists():
        return jsonify({"error": f"Image file not found: {img_path}"}), 404

    if not Path(model_path).exists():
        return jsonify({"error": f"Model not found: {model_path}"}), 404

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(model_path)
        results = model.predict(img_path, conf=conf_thresh, device=device, verbose=False)
        result = results[0]

        # Render annotated image
        annotated = result.plot()  # numpy BGR array
        annotated_rgb = annotated[:, :, ::-1]  # BGR -> RGB
        pil_img = Image.fromarray(annotated_rgb)

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        boxes = result.boxes
        detections = []
        if boxes is not None and len(boxes) > 0:
            names = result.names
            for box in boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                xyxy   = box.xyxy[0].tolist()
                detections.append({
                    "class_id":   cls_id,
                    "class_name": names.get(cls_id, str(cls_id)),
                    "confidence": round(conf, 4),
                    "bbox":       [round(x, 1) for x in xyxy]
                })

        return jsonify({
            "image_b64":  b64,
            "detections": detections,
            "n_det":      len(detections),
            "conf_thresh": conf_thresh,
            "model_path": str(model_path),
            "image_name": Path(img_path).name
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if is_tmp and img_path:
            try:
                os.unlink(img_path)
            except Exception:
                pass


if __name__ == "__main__":
    print("=" * 60)
    print("  CADICA Feasibility Server")
    print("  Open: http://localhost:5050")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5050, debug=False)
