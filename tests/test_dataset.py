"""Unit and integration tests for CADICADataset PyTorch Dataset."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import CADICADataset


@pytest.fixture
def sample_dataset() -> CADICADataset:
    """Fixture providing a CADICADataset initialized with real CADICA.zip index data."""
    frames_df = pd.read_csv("data_index/frames_index.csv")
    ann_df = pd.read_csv("data_index/annotations.csv")

    # Select a small sample of positive and negative frames
    pos_sample = frames_df[frames_df["has_groundtruth"]].head(5)
    neg_sample = frames_df[~frames_df["has_groundtruth"]].head(3)
    sample_frames = pd.concat([pos_sample, neg_sample], ignore_index=True)

    return CADICADataset(
        frames_df=sample_frames,
        annotations_df=ann_df,
        dataset_root="CADICA.zip",
        class_scheme="7class",
        normalize=True,
    )


class TestCADICADataset:
    """Tests asserting correct tensor shapes, types, and functionality of CADICADataset."""

    def test_dataset_length(self, sample_dataset: CADICADataset) -> None:
        """Verify length matches number of frames."""
        assert len(sample_dataset) == 8

    def test_getitem_shapes_and_types(self, sample_dataset: CADICADataset) -> None:
        """Verify __getitem__ returns valid tensors with correct dimensions and ranges."""
        item = sample_dataset[0]

        # Check image tensor
        img = item["image"]
        assert isinstance(img, torch.Tensor)
        assert img.dtype == torch.float32
        assert img.shape == (3, 512, 512)
        assert 0.0 <= img.min() <= img.max() <= 1.0

        # Check boxes
        boxes = item["boxes"]
        assert isinstance(boxes, torch.Tensor)
        assert boxes.dtype == torch.float32
        assert boxes.ndim == 2
        assert boxes.shape[1] == 4

        # Check labels
        labels = item["labels"]
        assert isinstance(labels, torch.Tensor)
        assert labels.dtype == torch.int64
        assert len(labels) == len(boxes)

        # Check identifiers
        assert isinstance(item["patient_id"], str)
        assert isinstance(item["frame_id"], str)

    def test_negative_frame_empty_boxes(self, sample_dataset: CADICADataset) -> None:
        """Verify negative frame returns zero boxes and zero labels."""
        # Index 5 is a negative frame
        item = sample_dataset[5]
        assert item["boxes"].shape == (0, 4)
        assert item["labels"].shape == (0,)

    def test_collate_fn_batching(self, sample_dataset: CADICADataset) -> None:
        """Verify collate_fn stacks images and lists variable-length boxes."""
        batch_items = [sample_dataset[0], sample_dataset[1], sample_dataset[5]]
        batch = CADICADataset.collate_fn(batch_items)

        assert batch["images"].shape == (3, 3, 512, 512)
        assert len(batch["boxes"]) == 3
        assert len(batch["labels"]) == 3
        assert len(batch["patient_ids"]) == 3
        assert len(batch["frame_ids"]) == 3

    def test_class_schemes_mappings(self) -> None:
        """Test dataset category mapping under 7class, binary, and severity3 schemes."""
        frames_df = pd.read_csv("data_index/frames_index.csv").head(2)
        ann_df = pd.read_csv("data_index/annotations.csv")

        # 7class
        ds_7 = CADICADataset(frames_df, ann_df, dataset_root="CADICA.zip", class_scheme="7class")
        assert ds_7.class_mapping["p0_20"] == 0
        assert ds_7.class_mapping["p100"] == 6

        # binary
        ds_bin = CADICADataset(frames_df, ann_df, dataset_root="CADICA.zip", class_scheme="binary")
        assert ds_bin.class_mapping["nonlesion"] == 0
        assert ds_bin.class_mapping["p0_20"] == 1
        assert ds_bin.class_mapping["p100"] == 1

        # severity3
        ds_s3 = CADICADataset(frames_df, ann_df, dataset_root="CADICA.zip", class_scheme="severity3")
        assert ds_s3.class_mapping["p0_20"] == 0
        assert ds_s3.class_mapping["p50_70"] == 1
        assert ds_s3.class_mapping["p100"] == 2
