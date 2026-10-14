"""Visual check: draw YOLO boxes on random images and save grids."""
import random, cv2
from pathlib import Path

root = Path("outputs/yolo_dataset")

for split in ["train", "val", "test"]:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    imgs = sorted(img_dir.glob("*.png"))
    lbls = {p.stem: p for p in lbl_dir.glob("*.txt")}
    # Only images that have at least one box
    pos = [p for p in imgs if p.stem in lbls and lbls[p.stem].read_text().strip()]
    random.seed(0)
    sample = random.sample(pos, min(8, len(pos)))
    tiles = []
    for p in sample:
        im = cv2.imread(str(p))
        if im is None:
            print(f"WARNING: could not read {p}")
            continue
        h, w = im.shape[:2]
        for line in lbls[p.stem].read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            c = int(parts[0])
            xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im, str(c), (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        tiles.append(cv2.resize(im, (384, 384)))
    if len(tiles) >= 8:
        grid = cv2.vconcat([cv2.hconcat(tiles[:4]), cv2.hconcat(tiles[4:])])
    elif tiles:
        # pad to 8
        while len(tiles) < 8:
            tiles.append(tiles[-1].copy())
        grid = cv2.vconcat([cv2.hconcat(tiles[:4]), cv2.hconcat(tiles[4:])])
    else:
        print(f"No annotated images for {split}")
        continue
    out = f"check_{split}.png"
    cv2.imwrite(out, grid)
    import os
    sz = os.path.getsize(out)
    print(f"Saved {out} ({sz} bytes)")

print("done")
