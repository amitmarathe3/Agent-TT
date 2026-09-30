"""Albumentations data augmentation pipelines tailored for angiography medical imaging.

Applies bounding-box safe transformations preserving vessel topology, lesion features,
and anatomical orientation (no vertical flips, no color jitter, no aggressive cropping).
"""

from typing import Any
import albumentations as A
import numpy as np


def get_train_transforms(
    image_size: int = 512,
    bbox_format: str = "yolo",
    horizontal_flip: bool = True,
    min_visibility: float = 0.3,
) -> A.Compose:
    """Build training data augmentation pipeline for coronary angiography frames.

    Transforms applied:
    - Contrast-Limited Adaptive Histogram Equalization (CLAHE) for vessel contrast.
    - Subtle brightness and contrast variations.
    - Mild gamma adjustment.
    - Gaussian noise and subtle Gaussian blur.
    - Small spatial affine transformations (rotation +/-10 deg, scale 0.9-1.1, translation +/-5%).
    - Optional horizontal flip (NO vertical flip to preserve catheter orientation).
    - Strictly NO hue or color jitter (angiograms are monochromatic).
    - Strictly NO aggressive crops that cut off lesions.

    Args:
        image_size: Target image square dimension in pixels (default: 512).
        bbox_format: Bounding box format ('yolo', 'pascal_voc', 'coco', 'albumentations').
        horizontal_flip: Whether to include horizontal flipping (default: True).
        min_visibility: Minimum fraction of bbox that must remain visible after transforms.

    Returns:
        Albumentations Compose pipeline.
    """
    transforms_list: list[Any] = []

    # 1. Pixel-level contrast & illumination enhancement
    transforms_list.extend([
        A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    ])

    # 2. Sensor noise and blur simulation
    # Using GaussNoise with std_range or var_limit compatibility
    try:
        noise_transform = A.GaussNoise(std_range=(0.02, 0.08), p=0.5)
    except Exception:
        noise_transform = A.GaussNoise(var_limit=(10.0, 50.0), p=0.5)

    transforms_list.append(
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            noise_transform,
        ], p=0.3)
    )

    # 3. Geometric / Spatial transformations (preserving lesion integrity)
    transforms_list.append(
        A.Affine(
            scale=(0.9, 1.1),
            rotate=(-10, 10),
            translate_percent=(-0.05, 0.05),
            p=0.5,
        )
    )

    if horizontal_flip:
        transforms_list.append(A.HorizontalFlip(p=0.5))

    # Resize if not already matching target size
    transforms_list.append(A.Resize(image_size, image_size))

    bbox_params = A.BboxParams(
        format=bbox_format,
        label_fields=["labels"],
        min_visibility=min_visibility,
        check_each_transform=True,
    )

    return A.Compose(transforms_list, bbox_params=bbox_params)


def get_val_transforms(
    image_size: int = 512,
    bbox_format: str = "yolo",
    min_visibility: float = 0.1,
) -> A.Compose:
    """Build minimal deterministic evaluation / validation pipeline for angiography frames.

    Contains no stochastic perturbations, only standard resizing and bbox registration.

    Args:
        image_size: Target image square dimension in pixels (default: 512).
        bbox_format: Bounding box format ('yolo', 'pascal_voc', 'coco', 'albumentations').
        min_visibility: Minimum fraction of bbox that must remain visible.

    Returns:
        Albumentations Compose pipeline.
    """
    transforms_list = [
        A.Resize(image_size, image_size),
    ]

    bbox_params = A.BboxParams(
        format=bbox_format,
        label_fields=["labels"],
        min_visibility=min_visibility,
        check_each_transform=True,
    )

    return A.Compose(transforms_list, bbox_params=bbox_params)


def clip_yolo_boxes(boxes: list[list[float]] | np.ndarray) -> list[list[float]]:
    """Ensure YOLO normalized coordinates [x_center, y_center, w, h] stay strictly in [0, 1].

    Args:
        boxes: List or array of YOLO normalized bounding boxes.

    Returns:
        List of clipped bounding boxes.
    """
    valid_boxes = []
    for b in boxes:
        xc, yc, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        # Clip dimensions
        w = max(0.001, min(1.0, w))
        h = max(0.001, min(1.0, h))
        # Ensure box boundaries are within [0, 1]
        x1 = max(0.0, xc - w / 2.0)
        y1 = max(0.0, yc - h / 2.0)
        x2 = min(1.0, xc + w / 2.0)
        y2 = min(1.0, yc + h / 2.0)
        # Recalculate
        new_w = x2 - x1
        new_h = y2 - y1
        if new_w > 0.002 and new_h > 0.002:
            new_xc = x1 + new_w / 2.0
            new_yc = y1 + new_h / 2.0
            valid_boxes.append([new_xc, new_yc, new_w, new_h])
    return valid_boxes
