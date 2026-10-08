"""Visualization tools for qualitative error analysis and model inspection.

Generates 3x4 grids for:
1. Standard validation predictions (Ground Truth in green, predictions in red).
2. The 12 worst False Negatives (missed ground truth lesions).
3. The 12 worst False Positives (high-confidence false alarms).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.matching import match_predictions
from src.utils.logger import get_logger

logger = get_logger("visualization")


def _draw_box_on_axis(
    ax: plt.Axes,
    box: list[float],
    color: str,
    label: str,
    linestyle: str = "-",
    linewidth: float = 2.0,
) -> None:
    """Draw a bounding box with label tag on matplotlib axis.

    Args:
        ax: Matplotlib axes object.
        box: [x, y, w, h] in absolute pixels.
        color: Edge color name or hex.
        label: Text label to draw above the box.
        linestyle: Line style ('-', '--', etc.).
        linewidth: Line width.
    """
    x, y, w, h = box[:4]
    rect = patches.Rectangle(
        (x, y),
        w,
        h,
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
    )
    ax.add_patch(rect)

    # Label box
    ax.text(
        x,
        max(0.0, y - 4.0),
        label,
        fontsize=7,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="square,pad=0.15", facecolor=color, alpha=0.85, edgecolor="none"),
    )


def render_prediction_grid(
    selected_frames: list[tuple[str, list[list[float]], list[list[float]], str]],
    class_names: dict[int, str],
    image_dir: Path | str,
    output_path: Path | str,
    title: str = "CADICA Angiography Detection Inspection",
) -> None:
    """Render a 3x4 grid of images with annotated ground-truth and predicted boxes.

    Args:
        selected_frames: List of up to 12 tuples (frame_id, gts, preds, subtitle).
        class_names: Dict class_id -> name.
        image_dir: Path to directory containing frame images.
        output_path: Destination PNG filepath.
        title: Overall super-title.
    """
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = min(12, len(selected_frames))
    if n_frames == 0:
        logger.warning("No frames provided for grid: %s", output_path)
        return

    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 14))
    axes_flat = axes.flatten()

    for idx in range(n_rows * n_cols):
        ax = axes_flat[idx]
        if idx >= n_frames:
            ax.axis("off")
            continue

        frame_id, gts, preds, subtitle = selected_frames[idx]

        # Load image
        img_p = image_dir / f"{frame_id}.png"
        if not img_p.exists():
            img_p = image_dir / f"{frame_id}.jpg"

        if img_p.exists():
            img_bgr = cv2.imread(str(img_p))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            # Fallback blank canvas
            img_rgb = np.zeros((512, 512, 3), dtype=np.uint8)

        ax.imshow(img_rgb)
        ax.set_title(f"{frame_id}\n{subtitle}", fontsize=8, fontweight="bold")
        ax.axis("off")

        # 1. Draw Ground Truths in GREEN
        for gt in gts:
            cls_name = class_names.get(int(gt[4]), str(int(gt[4])))
            _draw_box_on_axis(
                ax=ax,
                box=gt[:4],
                color="limegreen",
                label=f"GT: {cls_name}",
                linestyle="-",
                linewidth=2.0,
            )

        # 2. Draw Predictions in RED
        for p in preds:
            score = float(p[4])
            cls_name = class_names.get(int(p[5]), str(int(p[5])))
            _draw_box_on_axis(
                ax=ax,
                box=p[:4],
                color="red",
                label=f"{cls_name} ({score:.2f})",
                linestyle="--",
                linewidth=2.0,
            )

    plt.suptitle(
        f"{title}\n(Green: Ground Truth  |  Red: Model Predictions)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved prediction visualization grid to %s", output_path)


def generate_error_grids(
    predictions: dict[str, list[list[float]]],
    ground_truths: dict[str, list[list[float]]],
    class_names: dict[int, str],
    image_dir: Path | str,
    output_dir: Path | str,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> dict[str, Path]:
    """Identify sample validation frames, 12 worst FNs, and 12 worst FPs and generate grids.

    Args:
        predictions: Dict frame_id -> [[x,y,w,h,score,class], ...].
        ground_truths: Dict frame_id -> [[x,y,w,h,class], ...].
        class_names: Dict class_id -> name.
        image_dir: Image directory.
        output_dir: Directory where grids will be saved.
        conf_threshold: Confidence threshold to filter predictions.
        iou_threshold: IoU threshold for matching.

    Returns:
        Dict mapping grid type to saved image path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter predictions at confidence threshold
    filtered_preds: dict[str, list[list[float]]] = {}
    for fid, plist in predictions.items():
        filtered_preds[fid] = [p for p in plist if float(p[4]) >= conf_threshold]

    # Match each frame
    all_fids = sorted(set(predictions.keys()) | set(ground_truths.keys()))

    false_negatives_list: list[dict[str, Any]] = []
    false_positives_list: list[dict[str, Any]] = []
    sample_val_frames: list[tuple[str, list[list[float]], list[list[float]], str]] = []

    for fid in all_fids:
        gts = ground_truths.get(fid, [])
        fpreds = filtered_preds.get(fid, [])

        res = match_predictions(
            fpreds,
            gts,
            iou_threshold=iou_threshold,
            class_agnostic=False,
        )

        # Collect False Negatives
        for fn_idx in res.fn_gt_indices:
            gt_box = gts[fn_idx]
            cls_id = int(gt_box[4])
            area = float(gt_box[2]) * float(gt_box[3])
            false_negatives_list.append({
                "frame_id": fid,
                "gt_box": gt_box,
                "class_id": cls_id,
                "area": area,
                "all_gts": gts,
                "all_fpreds": fpreds,
            })

        # Collect False Positives
        for fp_idx in res.fp_pred_indices:
            p_box = fpreds[fp_idx]
            score = float(p_box[4])
            cls_id = int(p_box[5])
            false_positives_list.append({
                "frame_id": fid,
                "pred_box": p_box,
                "score": score,
                "class_id": cls_id,
                "all_gts": gts,
                "all_fpreds": fpreds,
            })

        # Add to sample candidates if frame has annotations or predictions
        if len(sample_val_frames) < 12 and (len(gts) > 0 or len(fpreds) > 0):
            sub = f"GT: {len(gts)} | Pred: {len(fpreds)}"
            sample_val_frames.append((fid, gts, fpreds, sub))

    # 1. Standard Validation Predictions Grid
    # If not enough annotated frames, pick evenly spaced frames
    if len(sample_val_frames) < 12 and all_fids:
        step = max(1, len(all_fids) // 12)
        sample_val_frames = [
            (fid, ground_truths.get(fid, []), filtered_preds.get(fid, []), f"GT: {len(ground_truths.get(fid, []))} | Pred: {len(filtered_preds.get(fid, []))}")
            for fid in all_fids[::step][:12]
        ]

    val_grid_path = output_dir / "val_predictions_grid.png"
    render_prediction_grid(
        selected_frames=sample_val_frames[:12],
        class_names=class_names,
        image_dir=image_dir,
        output_path=val_grid_path,
        title="Validation Sample Predictions Grid",
    )

    # 2. Worst False Negatives Grid (prioritize high severity and large box area)
    # Sort key: higher class_id (e.g. p100=6, p99=5) first, then larger box area
    false_negatives_list.sort(key=lambda item: (item["class_id"], item["area"]), reverse=True)
    fn_frames: list[tuple[str, list[list[float]], list[list[float]], str]] = []
    seen_fn_fids = set()
    for item in false_negatives_list:
        fid = item["frame_id"]
        if fid not in seen_fn_fids:
            seen_fn_fids.add(fid)
            cls_name = class_names.get(item["class_id"], str(item["class_id"]))
            sub = f"Missed GT: {cls_name} (area={int(item['area'])}px)"
            fn_frames.append((fid, item["all_gts"], item["all_fpreds"], sub))
            if len(fn_frames) == 12:
                break

    fn_grid_path = output_dir / "worst_false_negatives_grid.png"
    render_prediction_grid(
        selected_frames=fn_frames,
        class_names=class_names,
        image_dir=image_dir,
        output_path=fn_grid_path,
        title="12 Worst False Negatives (Missed Lesions)",
    )

    # 3. Worst False Positives Grid (prioritize highest confidence score)
    false_positives_list.sort(key=lambda item: item["score"], reverse=True)
    fp_frames: list[tuple[str, list[list[float]], list[list[float]], str]] = []
    seen_fp_fids = set()
    for item in false_positives_list:
        fid = item["frame_id"]
        if fid not in seen_fp_fids:
            seen_fp_fids.add(fid)
            cls_name = class_names.get(item["class_id"], str(item["class_id"]))
            sub = f"False Alarm: {cls_name} (conf={item['score']:.3f})"
            fp_frames.append((fid, item["all_gts"], item["all_fpreds"], sub))
            if len(fp_frames) == 12:
                break

    fp_grid_path = output_dir / "worst_false_positives_grid.png"
    render_prediction_grid(
        selected_frames=fp_frames,
        class_names=class_names,
        image_dir=image_dir,
        output_path=fp_grid_path,
        title="12 Worst False Positives (High-Confidence False Alarms)",
    )

    return {
        "val_grid": val_grid_path,
        "false_negatives_grid": fn_grid_path,
        "false_positives_grid": fp_grid_path,
    }
