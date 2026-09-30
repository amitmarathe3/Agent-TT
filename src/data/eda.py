"""Exploratory Data Analysis (EDA) module for the CADICA coronary angiography dataset.

Reads the structured CSV index tables from ``data_index/`` and generates
publication-ready figures (PNG) and a Markdown summary report covering:

1. Per-patient frame / video / box counts  (bar charts)
2. Lesion category class distribution      (bar chart + table)
3. Bounding-box geometry statistics        (histograms + heatmap)
4. Image resolution distribution           (bar chart)
5. Lesion vs non-lesion & selected vs nonselected splits (pie / bar)
6. Boxes-per-frame histogram
7. Sample annotation grid                  (12 random frames with GT boxes)
8. Metadata–lesion correlation matrix      (if metadata_clean.csv available)
"""

from __future__ import annotations

import io
import json
import logging
import random
import textwrap
import zipfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless rendering

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger

# ────────────────────────────── colour palette ──────────────────────────────

# Curated palette for stenosis categories (ordered by severity)
_CATEGORY_COLOURS: dict[str, str] = {
    "p0_20":  "#2ecc71",   # green – minimal
    "p20_50": "#f1c40f",   # yellow
    "p50_70": "#e67e22",   # orange
    "p70_90": "#e74c3c",   # red
    "p90_98": "#8e44ad",   # purple
    "p99":    "#c0392b",   # dark red
    "p100":   "#1a1a2e",   # near-black – total occlusion
}

_SEVERITY_ORDER = ["p0_20", "p20_50", "p50_70", "p70_90", "p90_98", "p99", "p100"]

# ────────────────────────────── style setup ──────────────────────────────

def _apply_style() -> None:
    """Apply a clean, publication-ready matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "#fafafa",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#cccccc",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


def _natural_patient_sort_key(pid: str) -> int:
    """Extract integer from patient ID for natural sorting."""
    try:
        return int(pid.replace("p", ""))
    except ValueError:
        return 999999


# ═══════════════════════════════════════════════════════════════════════════
#  Main EDA class
# ═══════════════════════════════════════════════════════════════════════════

class DatasetEDA:
    """Generates exploratory data analysis figures and a summary report.

    Reads pre-built CSV index tables from ``data_index/`` and optional
    clinical metadata, then produces PNG figures in ``reports/figures/``
    and an integrated Markdown report at ``reports/eda_summary.md``.
    """

    # Class-imbalance threshold: warn if max_count / min_count > this
    IMBALANCE_THRESHOLD: float = 5.0
    # Tiny box threshold (area in px²)
    TINY_BOX_AREA: int = 400

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        """Initialise the EDA engine from central project configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)

        log_cfg = (
            self.config.get("logging", {})
            if isinstance(self.config.get("logging"), dict)
            else {}
        )
        self.logger = get_logger(
            name="DatasetEDA",
            log_file=log_cfg.get("log_file", "outputs/logs/run.log"),
            level=log_cfg.get("level", "INFO"),
        )

        self.index_dir = Path(self.config.get("index_dir", "data_index"))
        self.reports_dir = Path(self.config.get("reports_dir", "reports"))
        self.figures_dir = self.reports_dir / "figures"
        self.dataset_root = Path(self.config.get("dataset_root", "CADICA.zip"))
        self.random_seed = int(self.config.get("random_seed", 42))

        # DataFrames loaded lazily
        self._frames: pd.DataFrame | None = None
        self._annotations: pd.DataFrame | None = None
        self._videos: pd.DataFrame | None = None
        self._patients: pd.DataFrame | None = None
        self._metadata: pd.DataFrame | None = None

    # ── data loading ─────────────────────────────────────────────────────

    def _load_csv(self, name: str) -> pd.DataFrame:
        """Load a CSV from data_index/."""
        path = self.index_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Index file not found: {path}")
        df = pd.read_csv(path)
        self.logger.info("Loaded %s (%d rows)", name, len(df))
        return df

    @property
    def frames(self) -> pd.DataFrame:
        """Frames index DataFrame."""
        if self._frames is None:
            self._frames = self._load_csv("frames_index.csv")
        return self._frames

    @property
    def annotations(self) -> pd.DataFrame:
        """Annotations DataFrame."""
        if self._annotations is None:
            self._annotations = self._load_csv("annotations.csv")
        return self._annotations

    @property
    def videos(self) -> pd.DataFrame:
        """Videos index DataFrame."""
        if self._videos is None:
            self._videos = self._load_csv("videos_index.csv")
        return self._videos

    @property
    def patients(self) -> pd.DataFrame:
        """Patients index DataFrame."""
        if self._patients is None:
            self._patients = self._load_csv("patients_index.csv")
        return self._patients

    @property
    def metadata(self) -> pd.DataFrame | None:
        """Cleaned clinical metadata DataFrame, or None if unavailable."""
        if self._metadata is None:
            meta_path = self.index_dir / "metadata_clean.csv"
            if meta_path.is_file():
                self._metadata = pd.read_csv(meta_path)
                self.logger.info(
                    "Loaded metadata_clean.csv (%d rows)", len(self._metadata)
                )
            else:
                self.logger.warning(
                    "metadata_clean.csv not found in %s – skipping correlations",
                    self.index_dir,
                )
        return self._metadata

    # ── figure generators ────────────────────────────────────────────────

    def plot_per_patient_stats(self) -> Path:
        """Bar charts: frames, videos, and boxes per patient.

        Returns:
            Path to saved PNG figure.
        """
        df = self.patients.copy()
        df["_sort"] = df["patient_id"].apply(_natural_patient_sort_key)
        df = df.sort_values("_sort")

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle("Per-Patient Dataset Statistics", fontsize=15, fontweight="bold")

        # Frames
        axes[0].bar(df["patient_id"], df["num_frames"], color="#3498db", edgecolor="white", linewidth=0.5)
        axes[0].set_ylabel("Frames")
        axes[0].set_title("Number of Frames per Patient")

        # Videos
        axes[1].bar(df["patient_id"], df["num_videos"], color="#2ecc71", edgecolor="white", linewidth=0.5)
        axes[1].set_ylabel("Videos")
        axes[1].set_title("Number of Videos per Patient")

        # Boxes
        axes[2].bar(df["patient_id"], df["num_boxes"], color="#e74c3c", edgecolor="white", linewidth=0.5)
        axes[2].set_ylabel("Bounding Boxes")
        axes[2].set_title("Number of Bounding Boxes per Patient")
        axes[2].set_xlabel("Patient ID")

        for ax in axes:
            ax.tick_params(axis="x", rotation=90)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = self.figures_dir / "per_patient_stats.png"
        fig.savefig(out)
        plt.close(fig)
        self.logger.info("Saved %s", out)
        return out

    def plot_class_distribution(self) -> tuple[Path, pd.DataFrame, list[str]]:
        """Bar chart and statistics table for lesion category distribution.

        Returns:
            Tuple of (figure_path, counts_dataframe, list_of_warnings).
        """
        ann = self.annotations
        counts = ann["category"].value_counts()

        # Reorder by severity
        ordered_cats = [c for c in _SEVERITY_ORDER if c in counts.index]
        extra_cats = [c for c in counts.index if c not in ordered_cats]
        ordered_cats += extra_cats
        counts = counts.reindex(ordered_cats)

        total = counts.sum()
        pct = (counts / total * 100).round(2)

        # Build summary table
        summary_df = pd.DataFrame({
            "category": counts.index,
            "count": counts.values,
            "percentage": pct.values,
        })

        # Warnings
        warnings: list[str] = []
        if counts.min() > 0:
            ratio = counts.max() / counts.min()
            if ratio > self.IMBALANCE_THRESHOLD:
                warnings.append(
                    f"**Class imbalance detected**: largest / smallest class ratio = "
                    f"{ratio:.1f}x (threshold {self.IMBALANCE_THRESHOLD}x). "
                    f"Largest: `{counts.idxmax()}` ({counts.max():,}), "
                    f"smallest: `{counts.idxmin()}` ({counts.min():,})."
                )

        # Plot
        colours = [_CATEGORY_COLOURS.get(c, "#95a5a6") for c in ordered_cats]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(ordered_cats, counts.values, color=colours, edgecolor="white", linewidth=0.8)
        ax.set_title("Lesion Category Distribution (Stenosis Severity)", fontweight="bold")
        ax.set_xlabel("Category")
        ax.set_ylabel("Number of Bounding Boxes")

        # Annotate counts on bars
        for bar, cnt, p in zip(bars, counts.values, pct.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.005,
                f"{cnt:,}\n({p:.1f}%)",
                ha="center", va="bottom", fontsize=8,
            )

        fig.tight_layout()
        out = self.figures_dir / "class_distribution.png"
        fig.savefig(out)
        plt.close(fig)
        self.logger.info("Saved %s", out)
        return out, summary_df, warnings

    def plot_bbox_statistics(self) -> Path:
        """Histograms of bbox width, height, area, aspect ratio, and centre heatmap.

        Returns:
            Path to saved PNG figure.
        """
        ann = self.annotations.copy()
        ann["area"] = ann["w"] * ann["h"]
        ann["aspect_ratio"] = ann["w"] / ann["h"].replace(0, 1)

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("Bounding Box Geometry Statistics", fontsize=15, fontweight="bold")

        # Width histogram
        axes[0, 0].hist(ann["w"], bins=40, color="#3498db", edgecolor="white", alpha=0.85)
        axes[0, 0].set_title("Box Width (px)")
        axes[0, 0].axvline(ann["w"].median(), color="#e74c3c", linestyle="--", label=f"median={ann['w'].median():.0f}")
        axes[0, 0].legend(fontsize=8)

        # Height histogram
        axes[0, 1].hist(ann["h"], bins=40, color="#2ecc71", edgecolor="white", alpha=0.85)
        axes[0, 1].set_title("Box Height (px)")
        axes[0, 1].axvline(ann["h"].median(), color="#e74c3c", linestyle="--", label=f"median={ann['h'].median():.0f}")
        axes[0, 1].legend(fontsize=8)

        # Area histogram
        axes[0, 2].hist(ann["area"], bins=40, color="#e67e22", edgecolor="white", alpha=0.85)
        axes[0, 2].set_title("Box Area (px²)")
        axes[0, 2].axvline(ann["area"].median(), color="#e74c3c", linestyle="--", label=f"median={ann['area'].median():.0f}")
        axes[0, 2].legend(fontsize=8)

        # Aspect ratio histogram
        axes[1, 0].hist(ann["aspect_ratio"], bins=40, color="#9b59b6", edgecolor="white", alpha=0.85)
        axes[1, 0].set_title("Aspect Ratio (w/h)")
        axes[1, 0].axvline(1.0, color="#e74c3c", linestyle="--", label="1:1")
        axes[1, 0].legend(fontsize=8)

        # Centre location heatmap
        heatmap, xedges, yedges = np.histogram2d(
            ann["x_center"], ann["y_center"], bins=50, range=[[0, 1], [0, 1]]
        )
        im = axes[1, 1].imshow(
            heatmap.T, origin="lower", aspect="auto",
            extent=[0, 1, 0, 1], cmap="hot", interpolation="bilinear",
        )
        axes[1, 1].set_title("Box Centre Heatmap (normalised)")
        axes[1, 1].set_xlabel("x_center")
        axes[1, 1].set_ylabel("y_center")
        fig.colorbar(im, ax=axes[1, 1], shrink=0.8)

        # Area by category box plot
        cat_order = [c for c in _SEVERITY_ORDER if c in ann["category"].unique()]
        ann_ordered = ann[ann["category"].isin(cat_order)]
        cat_groups = [ann_ordered[ann_ordered["category"] == c]["area"].values for c in cat_order]
        bp = axes[1, 2].boxplot(
            cat_groups, tick_labels=cat_order, patch_artist=True, showfliers=False,
        )
        for patch, cat in zip(bp["boxes"], cat_order):
            patch.set_facecolor(_CATEGORY_COLOURS.get(cat, "#95a5a6"))
            patch.set_alpha(0.7)
        axes[1, 2].set_title("Box Area by Category")
        axes[1, 2].set_ylabel("Area (px²)")
        axes[1, 2].tick_params(axis="x", rotation=45)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = self.figures_dir / "bbox_statistics.png"
        fig.savefig(out)
        plt.close(fig)
        self.logger.info("Saved %s", out)
        return out

    def plot_resolution_distribution(self) -> Path:
        """Bar chart of image resolution distribution.

        Returns:
            Path to saved PNG figure.
        """
        frames = self.frames
        res_col = frames["image_width"].astype(str) + "×" + frames["image_height"].astype(str)
        res_counts = res_col.value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(res_counts.index, res_counts.values, color="#1abc9c", edgecolor="white")
        ax.set_title("Image Resolution Distribution", fontweight="bold")
        ax.set_xlabel("Resolution (W×H)")
        ax.set_ylabel("Number of Frames")

        for i, (res, cnt) in enumerate(res_counts.items()):
            ax.text(i, cnt + cnt * 0.01, f"{cnt:,}", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        out = self.figures_dir / "resolution_distribution.png"
        fig.savefig(out)
        plt.close(fig)
        self.logger.info("Saved %s", out)
        return out

    def plot_lesion_selected_splits(self) -> Path:
        """Pie / bar charts for lesion vs non-lesion and selected vs nonselected.

        Returns:
            Path to saved PNG figure.
        """
        videos = self.videos
        frames = self.frames

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle("Dataset Split Analysis", fontsize=15, fontweight="bold")

        # 1. Lesion vs non-lesion videos
        lesion_vids = videos["is_lesion_video"].sum()
        non_lesion_vids = len(videos) - lesion_vids
        axes[0, 0].pie(
            [lesion_vids, non_lesion_vids],
            labels=[f"Lesion ({lesion_vids})", f"Non-lesion ({non_lesion_vids})"],
            colors=["#e74c3c", "#3498db"], autopct="%1.1f%%",
            startangle=90, explode=(0.03, 0),
            textprops={"fontsize": 10},
        )
        axes[0, 0].set_title("Videos: Lesion vs Non-Lesion")

        # 2. Lesion vs non-lesion frames (has_lesion_video)
        lesion_frames = frames["has_lesion_video"].sum()
        non_lesion_frames = len(frames) - lesion_frames
        axes[0, 1].pie(
            [lesion_frames, non_lesion_frames],
            labels=[f"Lesion-video ({lesion_frames:,})", f"Non-lesion ({non_lesion_frames:,})"],
            colors=["#e74c3c", "#3498db"], autopct="%1.1f%%",
            startangle=90, explode=(0.03, 0),
            textprops={"fontsize": 10},
        )
        axes[0, 1].set_title("Frames: From Lesion vs Non-Lesion Videos")

        # 3. Selected vs nonselected frames
        sel = (frames["split_type"] == "selected").sum()
        nonsel = len(frames) - sel
        axes[1, 0].bar(
            ["Selected", "Nonselected"], [sel, nonsel],
            color=["#2ecc71", "#95a5a6"], edgecolor="white",
        )
        for i, v in enumerate([sel, nonsel]):
            axes[1, 0].text(i, v + v * 0.01, f"{v:,}", ha="center", va="bottom", fontsize=10)
        axes[1, 0].set_title("Frames: Selected vs Nonselected")
        axes[1, 0].set_ylabel("Frame Count")

        # 4. Annotated vs unannotated frames
        annotated = frames["has_groundtruth"].sum()
        unannotated = len(frames) - annotated
        axes[1, 1].bar(
            ["Annotated", "Unannotated"], [annotated, unannotated],
            color=["#e67e22", "#bdc3c7"], edgecolor="white",
        )
        for i, v in enumerate([annotated, unannotated]):
            axes[1, 1].text(i, v + v * 0.01, f"{v:,}", ha="center", va="bottom", fontsize=10)
        axes[1, 1].set_title("Frames: With vs Without Ground Truth")
        axes[1, 1].set_ylabel("Frame Count")

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = self.figures_dir / "lesion_selected_splits.png"
        fig.savefig(out)
        plt.close(fig)
        self.logger.info("Saved %s", out)
        return out

    def plot_boxes_per_frame(self) -> Path:
        """Histogram of number of bounding boxes per annotated frame.

        Returns:
            Path to saved PNG figure.
        """
        ann = self.annotations
        bpf = ann.groupby("frame_id").size().reset_index(name="num_boxes")

        fig, ax = plt.subplots(figsize=(8, 5))
        max_boxes = int(bpf["num_boxes"].max())
        bins = np.arange(0.5, max_boxes + 1.5, 1)
        ax.hist(bpf["num_boxes"], bins=bins, color="#8e44ad", edgecolor="white", alpha=0.85, rwidth=0.85)
        ax.set_title("Boxes per Annotated Frame", fontweight="bold")
        ax.set_xlabel("Number of Boxes")
        ax.set_ylabel("Number of Frames")
        ax.set_xticks(range(1, max_boxes + 1))

        # Annotate counts
        counts = bpf["num_boxes"].value_counts().sort_index()
        for n, c in counts.items():
            ax.text(n, c + c * 0.02, str(c), ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        out = self.figures_dir / "boxes_per_frame.png"
        fig.savefig(out)
        plt.close(fig)
        self.logger.info("Saved %s", out)
        return out

    def plot_sample_annotations(self, n_samples: int = 12) -> Path:
        """Grid of random annotated frames with ground-truth boxes drawn.

        Reads frames directly from the dataset zip archive. Falls back
        to directory structure if zip is not available.

        Args:
            n_samples: Number of sample frames to display (default 12).

        Returns:
            Path to saved PNG figure.
        """
        ann = self.annotations
        rng = random.Random(self.random_seed)

        # Select n_samples unique annotated frames
        unique_frames = ann["frame_id"].unique().tolist()
        n_samples = min(n_samples, len(unique_frames))
        sampled_ids = rng.sample(unique_frames, n_samples)

        # Determine grid layout
        ncols = 4
        nrows = (n_samples + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
        fig.suptitle(
            f"Sample Annotated Frames ({n_samples} random)",
            fontsize=15, fontweight="bold",
        )
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        # Load images
        is_zip = self.dataset_root.is_file() and self.dataset_root.suffix.lower() == ".zip"

        if is_zip:
            zf = zipfile.ZipFile(self.dataset_root, "r")
        else:
            zf = None

        try:
            for idx, fid in enumerate(sampled_ids):
                ax = axes_flat[idx]
                frame_rows = ann[ann["frame_id"] == fid]
                frame_path = frame_rows.iloc[0]["frame_path"]

                # Load the image
                img = self._load_frame_image(frame_path, zf)

                if img is not None:
                    ax.imshow(img)
                else:
                    # Placeholder grey image
                    ax.imshow(np.ones((512, 512, 3), dtype=np.uint8) * 200)
                    ax.text(256, 256, "Image\nNot Found", ha="center", va="center", fontsize=12)

                # Draw bounding boxes
                for _, row in frame_rows.iterrows():
                    colour = _CATEGORY_COLOURS.get(row["category"], "#ffffff")
                    rect = patches.Rectangle(
                        (row["x"], row["y"]), row["w"], row["h"],
                        linewidth=2, edgecolor=colour, facecolor="none",
                    )
                    ax.add_patch(rect)
                    ax.text(
                        row["x"], row["y"] - 3,
                        row["category"],
                        fontsize=7, color=colour, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6),
                    )

                ax.set_title(fid, fontsize=8)
                ax.axis("off")

            # Hide unused axes
            for idx in range(n_samples, len(axes_flat)):
                axes_flat[idx].axis("off")

        finally:
            if zf is not None:
                zf.close()

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = self.figures_dir / "sample_annotations.png"
        fig.savefig(out)
        plt.close(fig)
        self.logger.info("Saved %s", out)
        return out

    def _load_frame_image(
        self, frame_path: str, zf: zipfile.ZipFile | None
    ) -> np.ndarray | None:
        """Load a frame image as an RGB numpy array.

        Args:
            frame_path: Relative path within dataset (zip or directory).
            zf: Open ZipFile object, or None for directory access.

        Returns:
            HxWx3 uint8 numpy array, or None if image cannot be loaded.
        """
        try:
            from PIL import Image

            if zf is not None:
                raw = zf.read(frame_path)
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            else:
                # Try dataset_root / frame_path or just frame_path
                fp = self.dataset_root / frame_path
                if not fp.exists():
                    fp = Path(frame_path)
                if not fp.exists():
                    return None
                img = Image.open(fp).convert("RGB")

            return np.array(img)
        except Exception as e:
            self.logger.warning("Could not load frame '%s': %s", frame_path, e)
            return None

    def plot_metadata_correlations(self) -> Path | None:
        """Correlation heatmap between numeric metadata features and lesion counts.

        Returns:
            Path to saved PNG figure, or None if metadata is unavailable.
        """
        meta = self.metadata
        if meta is None:
            return None

        patients = self.patients.copy()

        # Merge
        merged = patients.merge(meta, on="patient_id", how="inner")
        if merged.empty:
            self.logger.warning("No matching patients between metadata and index.")
            return None

        # Select numeric columns only
        numeric_cols = merged.select_dtypes(include=["number"]).columns.tolist()

        # Remove patient-index stats that are obviously redundant
        feature_cols = [
            c for c in numeric_cols
            if c not in ("patient_id",)
        ]

        if len(feature_cols) < 3:
            self.logger.warning("Too few numeric features for correlation matrix.")
            return None

        corr = merged[feature_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

        ax.set_xticks(range(len(feature_cols)))
        ax.set_yticks(range(len(feature_cols)))
        ax.set_xticklabels(feature_cols, rotation=90, fontsize=7)
        ax.set_yticklabels(feature_cols, fontsize=7)

        # Annotate correlation values
        for i in range(len(feature_cols)):
            for j in range(len(feature_cols)):
                val = corr.values[i, j]
                text_color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5.5, color=text_color)

        fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson Correlation")
        ax.set_title(
            "Correlation: Clinical Metadata & Lesion Statistics",
            fontweight="bold", fontsize=13,
        )
        fig.tight_layout()
        out = self.figures_dir / "metadata_correlations.png"
        fig.savefig(out)
        plt.close(fig)
        self.logger.info("Saved %s", out)
        return out

    # ── summary report generator ─────────────────────────────────────────

    def generate_report(
        self,
        figure_paths: dict[str, Path | None],
        class_summary: pd.DataFrame,
        class_warnings: list[str],
        findings: list[str],
    ) -> str:
        """Generate an integrated Markdown summary report.

        Args:
            figure_paths: Mapping of section name to PNG figure path.
            class_summary: Class distribution DataFrame.
            class_warnings: Imbalance warning messages.
            findings: Key findings and potential problems.

        Returns:
            Markdown report content string.
        """
        lines: list[str] = []

        lines.append("# CADICA Dataset — Exploratory Data Analysis Report")
        lines.append("")
        lines.append(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Dataset**: {self.dataset_root}")
        lines.append(f"**Total Patients**: {len(self.patients)}")
        lines.append(f"**Total Videos**: {len(self.videos)}")
        lines.append(f"**Total Frames**: {len(self.frames):,}")
        lines.append(f"**Total Bounding Boxes**: {len(self.annotations):,}")
        lines.append("")

        # Key findings
        lines.append("---")
        lines.append("")
        lines.append("## Key Findings & Potential Issues")
        lines.append("")
        if findings:
            for f in findings:
                lines.append(f"- {f}")
        else:
            lines.append("- No significant issues detected.")
        lines.append("")

        # Section 1: Per-patient stats
        lines.append("---")
        lines.append("")
        lines.append("## 1. Per-Patient Statistics")
        lines.append("")
        if figure_paths.get("per_patient"):
            lines.append(f"![Per-Patient Statistics]({figure_paths['per_patient'].as_posix()})")
        lines.append("")

        # Patient summary table
        df_p = self.patients.copy()
        df_p["_sort"] = df_p["patient_id"].apply(_natural_patient_sort_key)
        df_p = df_p.sort_values("_sort").drop(columns=["_sort"])
        lines.append("| Patient | Videos | Frames | Lesion Videos | Boxes |")
        lines.append("|---------|--------|--------|---------------|-------|")
        for _, row in df_p.iterrows():
            lines.append(
                f"| {row['patient_id']} | {row['num_videos']} | "
                f"{row['num_frames']} | {row['num_lesion_videos']} | "
                f"{row['num_boxes']} |"
            )
        lines.append("")

        # Section 2: Class distribution
        lines.append("---")
        lines.append("")
        lines.append("## 2. Lesion Category Distribution")
        lines.append("")
        if figure_paths.get("class_dist"):
            lines.append(f"![Class Distribution]({figure_paths['class_dist'].as_posix()})")
        lines.append("")

        lines.append("| Category | Count | Percentage |")
        lines.append("|----------|-------|------------|")
        for _, row in class_summary.iterrows():
            lines.append(f"| `{row['category']}` | {int(row['count']):,} | {row['percentage']:.2f}% |")
        lines.append("")

        if class_warnings:
            lines.append("> [!WARNING]")
            for w in class_warnings:
                lines.append(f"> {w}")
            lines.append("")

        # Section 3: Bbox statistics
        lines.append("---")
        lines.append("")
        lines.append("## 3. Bounding Box Geometry Statistics")
        lines.append("")
        if figure_paths.get("bbox_stats"):
            lines.append(f"![Bbox Statistics]({figure_paths['bbox_stats'].as_posix()})")
        lines.append("")

        ann = self.annotations
        ann_area = ann["w"] * ann["h"]
        lines.append("| Metric | Min | Median | Mean | Max | Std |")
        lines.append("|--------|-----|--------|------|-----|-----|")
        for label, series in [("Width (px)", ann["w"]), ("Height (px)", ann["h"]),
                               ("Area (px²)", ann_area),
                               ("Aspect Ratio", ann["w"] / ann["h"].replace(0, 1))]:
            lines.append(
                f"| {label} | {series.min():.1f} | {series.median():.1f} | "
                f"{series.mean():.1f} | {series.max():.1f} | {series.std():.1f} |"
            )
        lines.append("")

        # Section 4: Resolution
        lines.append("---")
        lines.append("")
        lines.append("## 4. Image Resolution Distribution")
        lines.append("")
        if figure_paths.get("resolution"):
            lines.append(f"![Resolution Distribution]({figure_paths['resolution'].as_posix()})")
        lines.append("")

        res_col = self.frames["image_width"].astype(str) + "×" + self.frames["image_height"].astype(str)
        res_counts = res_col.value_counts()
        lines.append("| Resolution | Count | Percentage |")
        lines.append("|------------|-------|------------|")
        for res, cnt in res_counts.items():
            lines.append(f"| {res} | {cnt:,} | {cnt / len(self.frames) * 100:.1f}% |")
        lines.append("")

        # Section 5: Splits
        lines.append("---")
        lines.append("")
        lines.append("## 5. Lesion vs Non-Lesion & Selected vs Nonselected Splits")
        lines.append("")
        if figure_paths.get("splits"):
            lines.append(f"![Split Analysis]({figure_paths['splits'].as_posix()})")
        lines.append("")

        # Section 6: Boxes per frame
        lines.append("---")
        lines.append("")
        lines.append("## 6. Boxes per Frame Histogram")
        lines.append("")
        if figure_paths.get("boxes_per_frame"):
            lines.append(f"![Boxes per Frame]({figure_paths['boxes_per_frame'].as_posix()})")
        lines.append("")

        bpf = ann.groupby("frame_id").size()
        lines.append(f"- **Annotated frames**: {len(bpf):,}")
        lines.append(f"- **Boxes per frame range**: {bpf.min()} – {bpf.max()}")
        lines.append(f"- **Mean boxes per frame**: {bpf.mean():.2f}")
        lines.append(f"- **Frames with multiple boxes**: {(bpf > 1).sum():,} ({(bpf > 1).sum() / len(bpf) * 100:.1f}%)")
        lines.append("")

        # Section 7: Sample annotations
        lines.append("---")
        lines.append("")
        lines.append("## 7. Sample Annotated Frames")
        lines.append("")
        if figure_paths.get("samples"):
            lines.append(f"![Sample Annotations]({figure_paths['samples'].as_posix()})")
        lines.append("")

        # Section 8: Metadata correlations
        lines.append("---")
        lines.append("")
        lines.append("## 8. Metadata–Lesion Correlations")
        lines.append("")
        if figure_paths.get("correlations"):
            lines.append(f"![Metadata Correlations]({figure_paths['correlations'].as_posix()})")
            lines.append("")
            lines.append(
                "The heatmap above shows Pearson correlations between numeric clinical metadata "
                "features (age, BMI, comorbidities, etc.) and per-patient lesion statistics "
                "(number of boxes, lesion videos, etc.)."
            )
        else:
            lines.append("*Metadata correlations unavailable (no `metadata_clean.csv` found).*")
        lines.append("")

        return "\n".join(lines)

    # ── orchestrator ─────────────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        """Execute the full EDA pipeline.

        Generates all figures, compiles key findings, and writes the
        Markdown summary report to ``reports/eda_summary.md``.

        Returns:
            Dict containing figure_paths, class_summary, warnings, and
            the report text.
        """
        _apply_style()
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        figure_paths: dict[str, Path | None] = {}
        findings: list[str] = []

        # 1. Per-patient stats
        self.logger.info("Generating per-patient statistics...")
        figure_paths["per_patient"] = self.plot_per_patient_stats()

        # 2. Class distribution
        self.logger.info("Generating class distribution...")
        class_path, class_summary, class_warnings = self.plot_class_distribution()
        figure_paths["class_dist"] = class_path
        findings.extend(class_warnings)

        # 3. Bbox statistics
        self.logger.info("Generating bounding box statistics...")
        figure_paths["bbox_stats"] = self.plot_bbox_statistics()

        # Check for tiny boxes
        ann = self.annotations
        ann_area = ann["w"] * ann["h"]
        tiny_count = int((ann_area < self.TINY_BOX_AREA).sum())
        if tiny_count > 0:
            pct = tiny_count / len(ann) * 100
            findings.append(
                f"**Tiny boxes**: {tiny_count:,} boxes ({pct:.1f}%) have area < "
                f"{self.TINY_BOX_AREA} px² — may be hard for models to detect."
            )

        # 4. Resolution distribution
        self.logger.info("Generating resolution distribution...")
        figure_paths["resolution"] = self.plot_resolution_distribution()

        # 5. Lesion / selected splits
        self.logger.info("Generating split analysis...")
        figure_paths["splits"] = self.plot_lesion_selected_splits()

        # 6. Boxes per frame
        self.logger.info("Generating boxes-per-frame histogram...")
        figure_paths["boxes_per_frame"] = self.plot_boxes_per_frame()

        # 7. Sample annotations
        self.logger.info("Generating sample annotation grid...")
        try:
            figure_paths["samples"] = self.plot_sample_annotations(n_samples=12)
        except Exception as e:
            self.logger.warning("Could not generate sample annotations: %s", e)
            figure_paths["samples"] = None

        # 8. Metadata correlations
        self.logger.info("Generating metadata correlations...")
        figure_paths["correlations"] = self.plot_metadata_correlations()

        # Check for patients with zero lesions
        no_lesion_patients = self.patients[self.patients["num_boxes"] == 0]
        if len(no_lesion_patients) > 0:
            pid_list = ", ".join(
                sorted(no_lesion_patients["patient_id"].tolist(), key=_natural_patient_sort_key)
            )
            findings.append(
                f"**Patients with zero bounding boxes**: {len(no_lesion_patients)} patient(s) "
                f"have no annotated lesions: {pid_list}."
            )

        # Check imbalance between frames with / without annotations
        annotated_frames = int(self.frames["has_groundtruth"].sum())
        total_frames = len(self.frames)
        ann_pct = annotated_frames / total_frames * 100
        findings.append(
            f"**Annotation sparsity**: only {annotated_frames:,} / {total_frames:,} frames "
            f"({ann_pct:.1f}%) have ground-truth annotations."
        )

        # Generate report
        self.logger.info("Compiling EDA summary report...")
        report_text = self.generate_report(
            figure_paths=figure_paths,
            class_summary=class_summary,
            class_warnings=class_warnings,
            findings=findings,
        )

        report_path = self.reports_dir / "eda_summary.md"
        report_path.write_text(report_text, encoding="utf-8")
        self.logger.info("Saved EDA summary report → %s", report_path)

        return {
            "figure_paths": figure_paths,
            "class_summary": class_summary,
            "class_warnings": class_warnings,
            "findings": findings,
            "report_text": report_text,
            "report_path": report_path,
        }
