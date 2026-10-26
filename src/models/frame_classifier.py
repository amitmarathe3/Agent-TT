"""Frame-level lesion classifier and video-level screening pipeline for CADICA.

Implements:
1. ResNet18 binary classifier for frame-level lesion screening.
2. Training loop with Albumentations augmentation (rotation, scale, CLAHE, brightness/contrast; no flips, no color jitter).
3. Class-weighted BCE loss, AdamW, early stopping on val ROC-AUC.
4. Multi-seed training and ensembling.
5. Scoring variants:
   a) Classifier alone
   b) Detector frame score (max box confidence)
   c) Weighted ensemble (weight tuned on val)
   d) Temporal smoothing over neighbouring frames in the same video
6. Video-level aggregation (mean of top-k frame scores, k tuned on val).
7. Patient bootstrap confidence intervals (500 iterations).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc as calc_auc,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

logger = logging.getLogger("frame_classifier")


def get_transforms(img_size: int = 384) -> tuple[A.Compose, A.Compose]:
    """Get train and eval transforms matching user specification.
    
    Augmentations: small rotation/scale, brightness/contrast, CLAHE;
    no flips, no color jitter.
    """
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Affine(scale=(0.95, 1.05), rotate=(-10, 10), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    eval_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return train_transform, eval_transform


class CADICAFrameDataset(Dataset):
    """Dataset loading CADICA frames from disk with labels."""

    def __init__(
        self,
        frame_items: list[dict[str, Any]],
        transform: A.Compose | None = None,
    ):
        """Args:
            frame_items: List of dicts with 'frame_id', 'image_path', 'label' (0 or 1), 'patient_id', 'video_id'.
            transform: Albumentations compose transform.
        """
        self.frame_items = frame_items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame_items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.frame_items[idx]
        img_path = str(item["image_path"])

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            # Fallback to PIL
            pil_img = Image.open(img_path).convert("RGB")
            img = np.array(pil_img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=img)
            img_tensor = augmented["image"]
        else:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        label = float(item["label"])

        return {
            "image": img_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "frame_id": item["frame_id"],
            "patient_id": item["patient_id"],
            "video_id": item["video_id"],
        }


def build_resnet18(pretrained: bool = True) -> nn.Module:
    """Build ResNet18 model with single output logit."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


def evaluate_model_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Get model probability predictions, true labels, and frame IDs."""
    model.eval()
    all_preds = []
    all_labels = []
    all_fids = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()
            fids = batch["frame_id"]

            logits = model(images).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.extend(probs)
            all_labels.extend(labels)
            all_fids.extend(fids)

    return np.array(all_preds), np.array(all_labels), all_fids


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute standard binary classification metrics."""
    y_pred = (y_score >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)

    # ROC AUC
    try:
        roc_auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        roc_auc = 0.5

    # PR AUC
    try:
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_score)
        pr_auc = float(calc_auc(rec_curve, prec_curve))
    except Exception:
        pr_auc = float(np.mean(y_true))

    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    sens = float(recall_score(y_true, y_pred, zero_division=0))
    
    # Specificity = TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))

    pos_prevalence = float(np.mean(y_true))
    maj_baseline = float(max(pos_prevalence, 1.0 - pos_prevalence))

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "accuracy": round(acc, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "threshold": round(threshold, 4),
        "confusion_matrix": cm.tolist(),
        "positive_prevalence": round(pos_prevalence, 4),
        "majority_baseline": round(maj_baseline, 4),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "balanced_accuracy",
) -> float:
    """Find threshold maximizing chosen metric on validation set."""
    best_th = 0.5
    best_val = -1.0
    thresholds = np.linspace(0.01, 0.99, 99)

    for th in thresholds:
        m = compute_binary_metrics(y_true, y_score, threshold=th)
        val = m.get(metric, 0.0)
        if val > best_val:
            best_val = val
            best_th = th

    return float(best_th)


def patient_bootstrap_ci(
    df: pd.DataFrame,
    patient_col: str,
    score_col: str,
    label_col: str,
    threshold: float,
    n_bootstraps: int = 500,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Compute 95% cluster bootstrap confidence intervals by resampling patients."""
    unique_patients = df[patient_col].unique()
    rng = np.random.RandomState(seed)

    boot_metrics: dict[str, list[float]] = {
        "roc_auc": [],
        "pr_auc": [],
        "balanced_accuracy": [],
        "sensitivity": [],
        "specificity": [],
        "f1": [],
    }

    patient_groups = {p: df[df[patient_col] == p] for p in unique_patients}

    for _ in range(n_bootstraps):
        sampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        sample_dfs = [patient_groups[p] for p in sampled_patients]
        boot_df = pd.concat(sample_dfs, ignore_index=True)

        y_t = boot_df[label_col].values
        y_s = boot_df[score_col].values

        # If resample contains only one class, skip or assign default
        if len(np.unique(y_t)) < 2:
            continue

        m = compute_binary_metrics(y_t, y_s, threshold=threshold)
        for k in boot_metrics:
            boot_metrics[k].append(m[k])

    ci_dict = {}
    for k, vals in boot_metrics.items():
        if len(vals) > 0:
            low = float(np.percentile(vals, 2.5))
            high = float(np.percentile(vals, 97.5))
            ci_dict[k] = (round(low, 4), round(high, 4))
        else:
            ci_dict[k] = (float("nan"), float("nan"))

    return ci_dict


def temporal_smooth_scores(
    df: pd.DataFrame,
    score_col: str,
    window_size: int = 3,
) -> pd.Series:
    """Smooth scores within each video using frame order."""
    # Extract frame number integer from frame_id (e.g. p1_v1_00010 -> 10)
    df = df.copy()
    
    def extract_frame_num(fid: str) -> int:
        parts = str(fid).split("_")
        try:
            return int(parts[-1])
        except Exception:
            return 0

    df["frame_num"] = df["frame_id"].apply(extract_frame_num)
    df = df.sort_values(by=["patient_id", "video_id", "frame_num"])

    smoothed = df.groupby(["patient_id", "video_id"])[score_col].transform(
        lambda s: s.rolling(window=window_size, min_periods=1, center=True).mean()
    )
    return smoothed.loc[df.index]
