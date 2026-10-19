import json
import sys
import time
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import run_prediction, frozen_threshold

test_img_dir = ROOT / "outputs" / "yolo_dataset" / "images" / "test"
if not test_img_dir.exists():
    test_img_dir = ROOT / "outputs" / "yolo_dataset" / "test" / "images"

img_paths = sorted(test_img_dir.glob("*.png"))
if len(img_paths) < 2:
    raise RuntimeError(f"Found less than 2 test images in {test_img_dir}")

img1_path = img_paths[0]
img2_path = img_paths[1]
blank_img = np.zeros((512, 512, 3), dtype=np.uint8)

thr = frozen_threshold()

res1 = run_prediction(img1_path, threshold=thr)
res2 = run_prediction(img2_path, threshold=thr)
res3 = run_prediction(blank_img, threshold=thr)

results = {
    "url": "http://127.0.0.1:7860",
    "test_image_1": {
        "path": str(img1_path.name),
        "result": res1,
    },
    "test_image_2": {
        "path": str(img2_path.name),
        "result": res2,
    },
    "blank_image": {
        "path": "solid_black_512x512",
        "result": res3,
    },
}

out_path = ROOT / "outputs" / "smoke_test_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("APP URL: http://127.0.0.1:7860")
print("=" * 60)
print("\n--- SAMPLE 1 RESULT (JSON) ---")
print(json.dumps(res1, indent=2))
print("\n--- SAMPLE 2 RESULT (JSON) ---")
print(json.dumps(res2, indent=2))
print("\n--- BLANK IMAGE RESULT (JSON) ---")
print(json.dumps(res3, indent=2))
