import sys
from pathlib import Path

# Make src importable from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch

from src.models.yolo_infer import predict_canonical_split
from src.data.eval_sets import load_eval_ground_truth
from src.evaluation.metrics import compute_map_50_95

weights_path = Path("outputs/runs/exp_7class_yolo11s_oversample_run1/weights/best.pt")
yolo_dir = Path("outputs/yolo_dataset")
device = "0" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")
print(f"Weights: {weights_path}")

# --- VALIDATION EVALUATION ---
gts_val, _ = load_eval_ground_truth(split="val", scheme="7class", yolo_dir=yolo_dir, index_dir="data_index", config={})
print(f"\n[VAL] Loaded {len(gts_val)} GT frames")

print("\n--- [VAL] Running 512 Preprocess ---")
preds_val_512 = predict_canonical_split(
    weights_path=weights_path,
    split="val",
    output_json_path="outputs/preds_val_512.json",
    yolo_dir=yolo_dir,
    conf=0.001,
    iou=0.6,
    max_det=50,
    imgsz=512,
    device=device,
    preprocess="512",
    batch_size=16,
)
map50_val_512 = float(compute_map_50_95(preds_val_512, gts_val, list(range(7)))["mAP_50"])

print("\n--- [VAL] Running 544_padded Preprocess ---")
preds_val_544 = predict_canonical_split(
    weights_path=weights_path,
    split="val",
    output_json_path="outputs/preds_val_544.json",
    yolo_dir=yolo_dir,
    conf=0.001,
    iou=0.6,
    max_det=50,
    imgsz=544,
    device=device,
    preprocess="544_padded",
    batch_size=16,
)
map50_val_544 = float(compute_map_50_95(preds_val_544, gts_val, list(range(7)))["mAP_50"])

# Samples for val
frames_with_gt = [fid for fid, boxes in gts_val.items() if len(boxes) > 0]
sample_fids = frames_with_gt[:3]

samples_output = []
for fid in sample_fids:
    gt_boxes = gts_val.get(fid, [])
    p512_top = sorted(preds_val_512.get(fid, []), key=lambda x: x[4], reverse=True)[:3]
    p544_top = sorted(preds_val_544.get(fid, []), key=lambda x: x[4], reverse=True)[:3]
    samples_output.append({
        "frame_id": fid,
        "gt": gt_boxes,
        "pred_512_top3": p512_top,
        "pred_544_top3": p544_top,
    })

# --- TEST EVALUATION ---
gts_test, _ = load_eval_ground_truth(split="test", scheme="7class", yolo_dir=yolo_dir, index_dir="data_index", config={})
print(f"\n[TEST] Loaded {len(gts_test)} GT frames")

print("\n--- [TEST] Running 512 Preprocess ---")
preds_test_512 = predict_canonical_split(
    weights_path=weights_path,
    split="test",
    output_json_path="outputs/preds_test_512.json",
    yolo_dir=yolo_dir,
    conf=0.001,
    iou=0.6,
    max_det=50,
    imgsz=512,
    device=device,
    preprocess="512",
    batch_size=16,
)
map50_test_512 = float(compute_map_50_95(preds_test_512, gts_test, list(range(7)))["mAP_50"])

print("\n--- [TEST] Running 544_padded Preprocess ---")
preds_test_544 = predict_canonical_split(
    weights_path=weights_path,
    split="test",
    output_json_path="outputs/preds_test_544.json",
    yolo_dir=yolo_dir,
    conf=0.001,
    iou=0.6,
    max_det=50,
    imgsz=544,
    device=device,
    preprocess="544_padded",
    batch_size=16,
)
map50_test_544 = float(compute_map_50_95(preds_test_544, gts_test, list(range(7)))["mAP_50"])

results = {
    "device": device,
    "val_mAP50_512": map50_val_512,
    "val_mAP50_544_padded": map50_val_544,
    "test_mAP50_512": map50_test_512,
    "test_mAP50_544_padded": map50_test_544,
    "samples_val": samples_output,
}

out_file = Path("outputs/decisive_test_results.json")
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("DECISIVE TEST RESULTS:")
print(f"  val  512 mAP@0.5:        {map50_val_512:.6f}")
print(f"  val  544_padded mAP@0.5: {map50_val_544:.6f}")
print(f"  test 512 mAP@0.5:        {map50_test_512:.6f}")
print(f"  test 544_padded mAP@0.5: {map50_test_544:.6f}")
print("=" * 60)

for s in samples_output:
    print(f"\nFrame: {s['frame_id']}")
    print(f"  GT:           {s['gt']}")
    print(f"  512 top3:     {s['pred_512_top3']}")
    print(f"  544_pad top3: {s['pred_544_top3']}")
