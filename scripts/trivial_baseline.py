"""Trivial-feature baseline for coronary lesion frame classification.

Extracts simple global/low-level image features:
1. Mean intensity
2. Standard deviation of intensity
3. Percentiles of intensity (5, 10, 25, 50, 75, 90, 95)
4. Fraction of dark vessel pixels after CLAHE (thresholds: < 50, < 75)
5. Edge density (Canny edge density, Sobel edge density)

Trains a Logistic Regression on the exact same train frames as the classifier,
and evaluates ROC-AUC on the canonical validation and test sets.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_frame_classifier import load_split_items


def extract_trivial_features(image_path: str) -> np.ndarray:
    """Extract trivial image statistics and edge/vessel cues."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # 1. Basic intensity statistics
    mean_val = float(np.mean(img))
    std_val = float(np.std(img))
    p5, p10, p25, p50, p75, p90, p95 = np.percentile(img, [5, 10, 25, 50, 75, 90, 95])

    # 2. CLAHE dark vessel pixels
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img)
    frac_dark_50 = float(np.mean(clahe_img < 50))
    frac_dark_75 = float(np.mean(clahe_img < 75))

    # 3. Edge density
    edges_canny = cv2.Canny(img, 50, 150)
    edge_density_canny = float(np.mean(edges_canny > 0))

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    edge_density_sobel = float(np.mean(grad_mag > 30))

    return np.array([
        mean_val,
        std_val,
        p5, p10, p25, p50, p75, p90, p95,
        frac_dark_50,
        frac_dark_75,
        edge_density_canny,
        edge_density_sobel,
    ], dtype=np.float32)


def main() -> None:
    print("=== EXTRACTING TRIVIAL IMAGE FEATURES ===")
    split_items = load_split_items()

    X_train, y_train = [], []
    for it in split_items["train"]:
        feats = extract_trivial_features(it["image_path"])
        X_train.append(feats)
        y_train.append(it["label"])
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_val, y_val = [], []
    for it in split_items["val"]:
        feats = extract_trivial_features(it["image_path"])
        X_val.append(feats)
        y_val.append(it["label"])
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    X_test, y_test = [], []
    for it in split_items["test"]:
        feats = extract_trivial_features(it["image_path"])
        X_test.append(feats)
        y_test.append(it["label"])
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(f"Train feature shape: {X_train.shape}, pos={int(y_train.sum())}, neg={len(y_train)-int(y_train.sum())}")
    print(f"Val feature shape:   {X_val.shape}, pos={int(y_val.sum())}, neg={len(y_val)-int(y_val.sum())}")
    print(f"Test feature shape:  {X_test.shape}, pos={int(y_test.sum())}, neg={len(y_test)-int(y_test.sum())}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    clf.fit(X_train_scaled, y_train)

    val_probs = clf.predict_proba(X_val_scaled)[:, 1]
    test_probs = clf.predict_proba(X_test_scaled)[:, 1]

    val_auc = float(roc_auc_score(y_val, val_probs))
    test_auc = float(roc_auc_score(y_test, test_probs))

    feature_names = [
        "mean", "std", "p5", "p10", "p25", "p50", "p75", "p90", "p95",
        "dark_clahe_50", "dark_clahe_75", "edge_canny", "edge_sobel"
    ]
    coefs = dict(zip(feature_names, clf.coef_[0].round(4).tolist()))

    print("\n=== TRIVIAL BASELINE RESULTS ===")
    print(f"Validation ROC-AUC: {val_auc:.4f}")
    print(f"Test ROC-AUC:       {test_auc:.4f}")
    print("Logistic Regression Coefficients:")
    for k, v in coefs.items():
        print(f"  {k:16s}: {v:+.4f}")

    out_dir = ROOT / "outputs" / "eval" / "trivial_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {
        "val_roc_auc": round(val_auc, 4),
        "test_roc_auc": round(test_auc, 4),
        "coefficients": coefs,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved metrics to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
