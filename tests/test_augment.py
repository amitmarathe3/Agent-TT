"""Unit tests for angiography data augmentation pipelines."""

import albumentations as A
import numpy as np
import pytest

from src.data.augment import clip_yolo_boxes, get_train_transforms, get_val_transforms


class TestAugmentationPipelines:
    """Tests verifying behavior, constraints, and box safety of angiography transforms."""

    def test_train_transforms_bbox_safety(self) -> None:
        """Verify bboxes stay within valid [0, 1] range after training augmentation."""
        pipeline = get_train_transforms(image_size=512, bbox_format="yolo")
        image = np.full((512, 512, 3), 128, dtype=np.uint8)
        # Bbox in normalized YOLO format [xc, yc, w, h]
        boxes = [[0.5, 0.5, 0.2, 0.2], [0.3, 0.4, 0.15, 0.1]]
        labels = [0, 2]

        for _ in range(10):
            res = pipeline(image=image, bboxes=boxes, labels=labels)
            out_boxes = res["bboxes"]
            out_labels = res["labels"]

            assert len(out_boxes) == len(out_labels)
            for b in out_boxes:
                xc, yc, w, h = b
                assert 0.0 <= xc <= 1.0, f"xc {xc} out of bounds"
                assert 0.0 <= yc <= 1.0, f"yc {yc} out of bounds"
                assert 0.0 < w <= 1.0, f"w {w} out of bounds"
                assert 0.0 < h <= 1.0, f"h {h} out of bounds"

    def test_val_transforms_deterministic(self) -> None:
        """Verify validation pipeline does not alter bbox coordinates or image content."""
        pipeline = get_val_transforms(image_size=512, bbox_format="yolo")
        image = np.full((512, 512, 3), 100, dtype=np.uint8)
        boxes = [[0.4, 0.6, 0.1, 0.2]]
        labels = [1]

        res = pipeline(image=image, bboxes=boxes, labels=labels)
        np.testing.assert_allclose(res["bboxes"][0], boxes[0], atol=1e-5)
        assert res["labels"] == labels

    def test_no_vertical_flip(self) -> None:
        """Ensure VerticalFlip is strictly absent to preserve catheter top-down orientation."""
        pipeline = get_train_transforms()
        for t in pipeline.transforms:
            assert not isinstance(t, A.VerticalFlip), "VerticalFlip must NOT be present in angiography transforms"

    def test_no_color_or_hue_jitter(self) -> None:
        """Ensure HueSaturationValue or ColorJitter is strictly absent for monochromatic images."""
        pipeline = get_train_transforms()
        for t in pipeline.transforms:
            assert not isinstance(t, A.HueSaturationValue), "Hue/saturation transforms must NOT be applied"
            assert not isinstance(t, A.ColorJitter), "ColorJitter must NOT be applied"

    def test_clip_yolo_boxes(self) -> None:
        """Test clipping edge cases for YOLO bounding boxes."""
        # Normal box
        boxes = [[0.5, 0.5, 0.2, 0.2]]
        clipped = clip_yolo_boxes(boxes)
        assert len(clipped) == 1
        assert np.isclose(clipped[0][0], 0.5)

        # Box overflowing boundary
        overflow_boxes = [[0.95, 0.95, 0.2, 0.2]]
        clipped_overflow = clip_yolo_boxes(overflow_boxes)
        assert len(clipped_overflow) == 1
        xc, yc, w, h = clipped_overflow[0]
        assert xc + w / 2.0 <= 1.0 + 1e-5
        assert yc + h / 2.0 <= 1.0 + 1e-5
