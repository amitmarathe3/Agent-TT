"""Evaluation report generator: figures, Markdown tables, and JSON summaries.

Generates all outputs in outputs/eval/<run_name>/:
- metrics.json
- metrics.md
- Confusion matrix heatmap (raw + normalized)
- Per-class PR curves
- Per-class F1-vs-confidence curves
- Frame-level ROC curve
- Frame-level PR curve
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.utils.logger import get_logger


logger = get_logger("EvalReport")


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# Confusion matrix heatmap
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save a confusion matrix heatmap.

    Args:
        cm: Confusion matrix [n_classes+1, n_classes+1] (last=background).
        class_names: Ordered list of class names (without background).
        output_path: File path for the saved PNG.
        normalize: If True, normalize rows to sum to 1.
        title: Figure title.
    """
    labels = list(class_names) + ["background"]
    n = len(labels)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums > 0, cm.astype(float) / (row_sums + 1e-9), 0.0)
        fmt = ".2f"
        vmax = 1.0
    else:
        cm_plot = cm.astype(float)
        fmt = "d"
        vmax = None

    fig, ax = plt.subplots(figsize=(max(8, n * 1.0), max(6, n * 0.9)))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=vmax)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    thresh = cm_plot.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm_plot[i, j]
            text = f"{val:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=8)

    ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix: %s", output_path)


# ---------------------------------------------------------------------------
# PR curves per class
# ---------------------------------------------------------------------------

def plot_pr_curves(
    pr_curves: dict[int, tuple[np.ndarray, np.ndarray]],
    ap_per_class: dict[int, float],
    class_names: dict[int, str],
    output_path: Path,
) -> None:
    """Plot per-class precision-recall curves on a single figure.

    Args:
        pr_curves: Dict class_id -> (recalls, precisions).
        ap_per_class: Dict class_id -> AP value.
        class_names: Dict class_id -> name.
        output_path: File path for saved PNG.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(pr_curves), 1)))

    for (cls_id, (recalls, precisions)), color in zip(sorted(pr_curves.items()), colors):
        ap = ap_per_class.get(cls_id, 0.0)
        label = f"{class_names.get(cls_id, str(cls_id))} (AP={ap:.3f})"
        ax.plot(recalls, precisions, lw=1.5, color=color, label=label)

    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision–Recall Curves (per class, IoU@0.5)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved PR curves: %s", output_path)


# ---------------------------------------------------------------------------
# F1-vs-confidence curves per class
# ---------------------------------------------------------------------------

def plot_f1_conf_curves(
    f1_curves: dict[int, dict[str, list]],
    class_names: dict[int, str],
    output_path: Path,
) -> None:
    """Plot F1-vs-confidence curves for all classes.

    Args:
        f1_curves: Dict class_id -> {"conf_thresholds": [...], "f1_values": [...]}.
        class_names: Dict class_id -> name.
        output_path: Output file path.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(f1_curves), 1)))

    for (cls_id, curve_data), color in zip(sorted(f1_curves.items()), colors):
        name = class_names.get(cls_id, str(cls_id))
        ax.plot(
            curve_data["conf_thresholds"],
            curve_data["f1_values"],
            lw=1.5, color=color, label=name,
        )

    ax.set_xlabel("Confidence Threshold", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("F1 Score vs Confidence Threshold (per class)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved F1-vs-conf curves: %s", output_path)


# ---------------------------------------------------------------------------
# ROC curve (frame-level)
# ---------------------------------------------------------------------------

def plot_roc_curve(
    fpr: list[float],
    tpr: list[float],
    auc: float,
    output_path: Path,
) -> None:
    """Plot and save a frame-level ROC curve.

    Args:
        fpr: False positive rate values.
        tpr: True positive rate values.
        auc: AUC value.
        output_path: Output file path.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color="#2563eb", label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], lw=1.5, linestyle="--", color="gray", label="Random")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("Frame-Level ROC Curve (Lesion vs Non-Lesion)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curve: %s", output_path)


# ---------------------------------------------------------------------------
# Frame-level PR curve
# ---------------------------------------------------------------------------

def plot_frame_pr_curve(
    recalls: list[float],
    precisions: list[float],
    auc: float,
    output_path: Path,
) -> None:
    """Plot and save a frame-level precision-recall curve.

    Args:
        recalls: Recall values.
        precisions: Precision values.
        auc: PR-AUC value.
        output_path: Output file path.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recalls, precisions, lw=2, color="#16a34a", label=f"PR (AUC = {auc:.4f})")

    ax.set_xlim([-0.01, 1.02])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Frame-Level Precision–Recall Curve", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved frame PR curve: %s", output_path)


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def _fmt(v: Any, decimals: int = 4) -> str:
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def generate_markdown_report(
    all_metrics: dict[str, Any],
    class_names: dict[int, str],
    run_name: str,
    output_path: Path,
) -> None:
    """Write a clean, human-readable Markdown metrics report.

    Args:
        all_metrics: Full metrics dict from evaluate() or collected results.
        class_names: Dict class_id -> name.
        run_name: Name of this evaluation run.
        output_path: File path to write the Markdown report.
    """
    lines: list[str] = []
    lines.append(f"# Evaluation Report: `{run_name}`\n")

    # Detection summary
    if "detection" in all_metrics:
        det = all_metrics["detection"]
        micro = det.get("micro", {})
        macro = det.get("macro", {})
        wtd = det.get("weighted", {})
        iou_thr = det.get("iou_threshold", 0.5)
        conf_thr = det.get("conf_threshold", 0.5)

        lines.append(f"## Object Detection Metrics (IoU@{iou_thr}, Conf≥{conf_thr})\n")
        lines.append("### Overall (Micro)\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for k in ["TP", "FP", "FN", "precision", "recall", "f1", "mean_iou"]:
            lines.append(f"| {k} | {_fmt(micro.get(k, '-'))} |")
        lines.append("")

        lines.append("### Averaged Metrics\n")
        lines.append("| Average | Precision | Recall | F1 |")
        lines.append("|---|---|---|---|")
        lines.append(f"| Macro | {_fmt(macro.get('precision', 0))} | {_fmt(macro.get('recall', 0))} | {_fmt(macro.get('f1', 0))} |")
        lines.append(f"| Weighted | {_fmt(wtd.get('precision', 0))} | {_fmt(wtd.get('recall', 0))} | {_fmt(wtd.get('f1', 0))} |")
        lines.append("")

        lines.append("### Per-Class Detection Metrics\n")
        lines.append("| Class | TP | FP | FN | Precision | Recall | F1 | Mean IoU |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for cls_id, m in sorted(det.get("per_class", {}).items()):
            lines.append(
                f"| {m.get('name', cls_id)} | {m['TP']} | {m['FP']} | {m['FN']} "
                f"| {_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} "
                f"| {_fmt(m['mean_iou'])} |"
            )
        lines.append("")

    # mAP
    if "map" in all_metrics:
        mp = all_metrics["map"]
        lines.append("## Average Precision\n")
        lines.append(f"| **mAP@0.5** | **{_fmt(mp.get('mAP_50', 0))}** |")
        lines.append(f"| **mAP@0.5:0.95** | **{_fmt(mp.get('mAP_50_95', 0))}** |")
        lines.append("")

        lines.append("### Per-Class AP@0.5\n")
        lines.append("| Class | AP@0.5 | AP@0.5:0.95 |")
        lines.append("|---|---|---|")
        for cls_id, ap in sorted(mp.get("per_class_AP_50", {}).items()):
            name = class_names.get(int(cls_id), str(cls_id))
            ap95 = mp.get("per_class_AP_50_95", {}).get(cls_id, 0.0)
            lines.append(f"| {name} | {_fmt(ap)} | {_fmt(ap95)} |")
        lines.append("")

    # Frame-level
    if "frame" in all_metrics:
        fm = all_metrics["frame"]
        lines.append("## Frame-Level Lesion Classification\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for k in ["accuracy", "precision", "recall", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc"]:
            lines.append(f"| {k} | {_fmt(fm.get(k, '-'))} |")
        lines.append(f"| Conf Threshold | {fm.get('conf_threshold', '-')} |")
        lines.append("")

    # Patient-level
    if "patient" in all_metrics:
        pat = all_metrics["patient"]
        lines.append("## Patient-Level Metrics\n")

        for key, label in [("lesion_detection", "Lesion Detection"), ("significant_stenosis", "Significant Stenosis (≥70%)")]:
            if key in pat:
                m = pat[key]
                lines.append(f"### {label}\n")
                lines.append("| Metric | Value |")
                lines.append("|---|---|")
                for mk in ["n_patients", "TP", "FP", "TN", "FN", "accuracy", "precision", "recall", "f1", "roc_auc"]:
                    lines.append(f"| {mk} | {_fmt(m.get(mk, '-'))} |")
                lines.append("")

    # Bootstrap CIs
    if "bootstrap" in all_metrics:
        boot = all_metrics["bootstrap"]
        lines.append(f"## Bootstrap {int(boot.get('ci_level', 0.95) * 100)}% Confidence Intervals (n={boot.get('n_bootstrap', '?')})\n")
        uni = boot.get("universe") or {}
        if uni:
            lines.append(
                f"- **Universe**: split=`{uni.get('split', '-')}` "
                f"{uni.get('n_frames', '?')} frames / {uni.get('n_patients', '?')} patients "
                f"(canonical eval set, NOT frames_index.csv)\n"
            )
        full = boot.get("full_sample") or {}
        lines.append("| Metric | Point estimate | Mean | Lower | Upper |")
        lines.append("|---|---|---|---|---|")
        for k in ["macro_f1", "mAP_50", "sensitivity", "frame_auc"]:
            ci = boot.get(k) or {}
            pt = full.get(k, '-')
            lines.append(
                f"| {k} | {_fmt(pt)} | {_fmt(ci.get('mean', '-'))} | "
                f"{_fmt(ci.get('lower', '-'))} | {_fmt(ci.get('upper', '-'))} |"
            )
        lines.append("")

    # Best threshold
    if "best_threshold" in all_metrics:
        bt = all_metrics["best_threshold"]
        lines.append("## Best Confidence Threshold (by Macro F1)\n")
        lines.append(f"- **Best Threshold**: `{bt.get('best_conf_threshold', '-')}`")
        lines.append(f"- **Best Macro F1**: `{_fmt(bt.get('best_macro_f1', 0))}`\n")

    # Figures
    lines.append("## Figures\n")
    for fname, title in [
        ("confusion_matrix.png", "Confusion Matrix (Raw)"),
        ("confusion_matrix_norm.png", "Confusion Matrix (Normalized)"),
        ("pr_curves.png", "Per-Class PR Curves"),
        ("f1_conf_curves.png", "F1 vs Confidence Curves"),
        ("roc_curve.png", "Frame-Level ROC Curve"),
        ("frame_pr_curve.png", "Frame-Level PR Curve"),
    ]:
        lines.append(f"### {title}\n")
        lines.append(f"![{title}]({fname})\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved Markdown report: %s", output_path)


# ---------------------------------------------------------------------------
def _make_serializable(obj: Any) -> Any:
    """Convert numpy types to native Python, strip plotting arrays, and make JSON-safe."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # Skip private plotting arrays
            if str(k).startswith("_"):
                continue
            # Skip heavy curve arrays in metrics.json
            if str(k) in ("pr_curves",):
                continue
            out[str(k)] = _make_serializable(v)
        return out
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if np.isnan(val) or np.isinf(val) else val
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, float):
        return None if np.isnan(obj) or np.isinf(obj) else obj
    return obj


def save_json_metrics(all_metrics: dict[str, Any], output_path: Path) -> None:
    """Serialize all metrics to a JSON file, stripping raw plotting arrays."""
    clean = _make_serializable(all_metrics)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)
    logger.info("Saved JSON metrics: %s", output_path)


# ---------------------------------------------------------------------------
# Full report generation orchestrator
# ---------------------------------------------------------------------------

def generate_report(
    all_metrics: dict[str, Any],
    class_ids: list[int],
    class_names: dict[int, str],
    run_name: str,
    output_dir: Path,
) -> None:
    """Generate all evaluation outputs for a given run.

    Produces: metrics.json, metrics.md, confusion_matrix.png,
    confusion_matrix_norm.png, pr_curves.png, f1_conf_curves.png,
    roc_curve.png, frame_pr_curve.png.

    Args:
        all_metrics: Full aggregated metrics dict.
        class_ids: Ordered list of class IDs.
        class_names: Dict class_id -> name.
        run_name: Run identifier string.
        output_dir: Directory to write outputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    save_json_metrics(all_metrics, output_dir / "metrics.json")

    # Markdown
    generate_markdown_report(all_metrics, class_names, run_name, output_dir / "metrics.md")

    # Confusion matrix
    if "confusion_matrix" in all_metrics:
        cm = np.array(all_metrics["confusion_matrix"])
        ordered_names = [class_names.get(c, str(c)) for c in class_ids]
        plot_confusion_matrix(cm, ordered_names, output_dir / "confusion_matrix.png",
                              normalize=False, title="Confusion Matrix (Raw)")
        plot_confusion_matrix(cm, ordered_names, output_dir / "confusion_matrix_norm.png",
                              normalize=True, title="Confusion Matrix (Normalized by GT)")

    # PR curves
    if "map" in all_metrics and "pr_curves" in all_metrics.get("map", {}):
        pr_curves = {}
        for cls_id, (recs, precs) in all_metrics["map"]["pr_curves"].items():
            pr_curves[int(cls_id)] = (np.array(recs), np.array(precs))
        ap_per = {int(k): v for k, v in all_metrics["map"]["per_class_AP_50"].items()}
        plot_pr_curves(pr_curves, ap_per, class_names, output_dir / "pr_curves.png")

    # F1-vs-conf curves
    if "f1_conf_curves" in all_metrics:
        plot_f1_conf_curves(all_metrics["f1_conf_curves"], class_names, output_dir / "f1_conf_curves.png")

    # ROC curve
    if "frame" in all_metrics:
        fm = all_metrics["frame"]
        plot_roc_curve(fm["_roc_fpr"], fm["_roc_tpr"], fm["roc_auc"], output_dir / "roc_curve.png")
        plot_frame_pr_curve(fm["_pr_recall"], fm["_pr_precision"], fm["pr_auc"], output_dir / "frame_pr_curve.png")

    logger.info("Evaluation report complete: %s", output_dir)
