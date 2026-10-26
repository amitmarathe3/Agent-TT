"""Unit tests for ResNet18 frame classifier and screening pipeline."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.frame_classifier import (
    build_resnet18,
    compute_binary_metrics,
    find_optimal_threshold,
    get_transforms,
    patient_bootstrap_ci,
    temporal_smooth_scores,
)


def test_synthetic_images_through_classifier():
    """Test 3 synthetic images through the ResNet18 classifier.
    
    Verifies that the model accepts input shape (3, 3, 384, 384) and outputs
    valid single-logit outputs that map to probabilities in [0, 1].
    """
    model = build_resnet18(pretrained=False)
    model.eval()

    # Create 3 synthetic images: batch of 3, 3 channels, 384x384
    synthetic_batch = torch.randn(3, 3, 384, 384)

    with torch.no_grad():
        logits = model(synthetic_batch).squeeze(-1)
        probs = torch.sigmoid(logits)

    assert logits.shape == (3,), f"Expected shape (3,), got {logits.shape}"
    assert probs.shape == (3,), f"Expected shape (3,), got {probs.shape}"
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0), "Probabilities must be in [0, 1]"


def test_no_patient_overlap_between_splits():
    """Verify that patient-wise splits (split_v1.csv) have zero patient leakage."""
    split_path = "data_index/splits/split_v1.csv"
    df = pd.read_csv(split_path)

    train_pts = set(df[df["split"] == "train"]["patient_id"].astype(str))
    val_pts = set(df[df["split"] == "val"]["patient_id"].astype(str))
    test_pts = set(df[df["split"] == "test"]["patient_id"].astype(str))

    # All splits must be non-empty
    assert len(train_pts) > 0, "Train split must have patients"
    assert len(val_pts) > 0, "Val split must have patients"
    assert len(test_pts) > 0, "Test split must have patients"

    # Pairwise disjointness
    assert not (train_pts & val_pts), f"Leakage train & val: {train_pts & val_pts}"
    assert not (train_pts & test_pts), f"Leakage train & test: {train_pts & test_pts}"
    assert not (val_pts & test_pts), f"Leakage val & test: {val_pts & test_pts}"


def test_temporal_smoothing():
    """Verify temporal smoothing preserves order and smooths values within videos."""
    df = pd.DataFrame({
        "patient_id": ["p1", "p1", "p1", "p2", "p2"],
        "video_id": ["v1", "v1", "v1", "v1", "v1"],
        "frame_id": ["p1_v1_00001", "p1_v1_00002", "p1_v1_00003", "p2_v1_00001", "p2_v1_00002"],
        "score": [0.0, 1.0, 0.0, 0.5, 0.5],
    })
    smoothed = temporal_smooth_scores(df, score_col="score", window_size=3)
    assert len(smoothed) == 5
    # Center frame of p1_v1 should be smoothed to (0+1+0)/3 ≈ 0.3333
    assert abs(smoothed.iloc[1] - (1.0 / 3.0)) < 1e-4
