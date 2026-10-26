#!/usr/bin/env python3
"""Training and evaluation runner for CADICA ResNet18 frame classifier & screening pipeline.

Strictly follows user protocol:
- ImageNet-pretrained ResNet18, 384 px, batch 16, AMP off, cuda (cpu fallback).
- AdamW lr 3e-4, class-weighted BCE, up to 12 epochs with early stopping on val AUC.
- Augmentation: small rotation/scale, brightness/contrast, CLAHE; no flips, no color jitter.
- Target: any lesion. Run 2 seeds (42, 1337) and ensemble both.
- Save weights, run_config.json, preds_val.json, preds_test.json.
- Scoring variants selected on VAL:
  a) Classifier alone
  b) Detector frame score (max box score, run1 with 544_padded)
  c) Weighted ensemble of a) and b) (weight tuned on val)
  d) Smoothed variants over neighbouring frames in the same video
- Video-level screening: mean of top-k frame scores (k tuned on val), evaluated on test videos.
- 500-iteration patient cluster bootstrap 95% CIs.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.frame_classifier import (
    CADICAFrameDataset,
    build_resnet18,
    compute_binary_metrics,
    evaluate_model_on_loader,
    find_optimal_threshold,
    get_transforms,
    patient_bootstrap_ci,
    temporal_smooth_scores,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_frame_classifier")


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split_items(
    yolo_dir: Path = ROOT / "outputs" / "yolo_dataset",
    frames_index_path: Path = ROOT / "data_index" / "frames_index.csv",
) -> dict[str, list[dict[str, Any]]]:
    """Load image paths and labels for train, val, and test splits."""
    fi = pd.read_csv(frames_index_path).set_index("frame_id")
    split_items: dict[str, list[dict[str, Any]]] = {}

    for split in ["train", "val", "test"]:
        img_dir = yolo_dir / "images" / split
        lbl_dir = yolo_dir / "labels" / split
        items = []

        for img_path in sorted(img_dir.glob("*.png")):
            fid = img_path.stem
            lbl_path = lbl_dir / f"{fid}.txt"

            # Label: 1 if label file has non-zero size, 0 if empty
            is_pos = 1.0 if (lbl_path.exists() and lbl_path.stat().st_size > 0) else 0.0

            # Patient ID & video ID from frames_index
            if fid in fi.index:
                row = fi.loc[fid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                pid = str(row["patient_id"])
                vid = str(row["video_id"])
            else:
                parts = fid.split("_")
                pid = parts[0]
                vid = parts[1]

            items.append({
                "frame_id": fid,
                "image_path": str(img_path),
                "label": is_pos,
                "patient_id": pid,
                "video_id": vid,
            })

        split_items[split] = items
        pos_cnt = sum(1 for it in items if it["label"] == 1.0)
        neg_cnt = len(items) - pos_cnt
        logger.info("Loaded %s: %d frames (pos=%d, neg=%d)", split, len(items), pos_cnt, neg_cnt)

    return split_items


def train_single_seed(
    seed: int,
    split_items: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    device: torch.device,
    max_epochs: int = 12,
    patience: int = 3,
    batch_size: int = 16,
    lr: float = 3e-4,
    img_size: int = 384,
) -> tuple[dict[str, float], dict[str, float], Path]:
    """Train ResNet18 with one random seed, saving the best weights on val ROC-AUC."""
    set_seed(seed)
    logger.info("=== Starting training for seed %d ===", seed)

    train_tf, eval_tf = get_transforms(img_size=img_size)

    train_ds = CADICAFrameDataset(split_items["train"], transform=train_tf)
    val_ds = CADICAFrameDataset(split_items["val"], transform=eval_tf)
    test_ds = CADICAFrameDataset(split_items["test"], transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Class weighting for BCE: N_neg / N_pos
    train_labels = [it["label"] for it in split_items["train"]]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    logger.info("Class weight pos_weight = %.4f (pos=%d, neg=%d)", float(pos_weight.item()), n_pos, n_neg)

    model = build_resnet18(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = -1.0
    best_weights = None
    no_improve_epochs = 0

    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = weights_dir / f"seed_{seed}_best.pt"

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(labels)

        train_loss = running_loss / len(train_ds)

        # Validation evaluation
        val_probs, val_labels, _ = evaluate_model_on_loader(model, val_loader, device)
        val_auc = float(roc_auc_score(val_labels, val_probs))
        elapsed = time.time() - t0

        logger.info(
            "Seed %d | Epoch %2d/%2d | Train Loss: %.4f | Val AUC: %.4f | Time: %.1fs",
            seed, epoch, max_epochs, train_loss, val_auc, elapsed
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, best_weights_path)
            no_improve_epochs = 0
            logger.info("  -> Saved new best checkpoint (Val AUC: %.4f)", val_auc)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", patience)
                break

    # Load best model for predictions
    logger.info("Loading best checkpoint for seed %d (Val AUC: %.4f)...", seed, best_val_auc)
    model.load_state_dict(best_weights)

    val_probs, _, val_fids = evaluate_model_on_loader(model, val_loader, device)
    test_probs, _, test_fids = evaluate_model_on_loader(model, test_loader, device)

    val_preds_dict = dict(zip(val_fids, [float(p) for p in val_probs]))
    test_preds_dict = dict(zip(test_fids, [float(p) for p in test_probs]))

    return val_preds_dict, test_preds_dict, best_weights_path


def load_detector_frame_scores(
    preds_544_path: Path,
) -> dict[str, float]:
    """Extract frame scores as max box score from detector predictions JSON."""
    with open(preds_544_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    frame_scores = {}
    for fid, boxes in preds.items():
        if len(boxes) == 0:
            frame_scores[fid] = 0.0
        else:
            # Each box: [x, y, w, h, score, class_id]
            scores = [float(b[4]) for b in boxes if len(b) >= 5]
            frame_scores[fid] = max(scores) if scores else 0.0

    return frame_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate CADICA ResNet18 frame classifier.")
    parser.add_argument("--epochs", type=int, default=12, help="Maximum epochs per seed (default: 12)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (default: 3)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--img-size", type=int, default=384, help="Image size (default: 384)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337], help="Random seeds (default: 42 1337)")
    parser.add_argument("--output-dir", type=str, default="outputs/runs/exp_frame_resnet18", help="Output run directory")
    args = parser.parse_args()

    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # 1. Load data
    split_items = load_split_items()

    # 2. Train both seeds
    val_preds_seeds = []
    test_preds_seeds = []
    best_weight_paths = []

    start_time = time.time()
    for s in args.seeds:
        v_preds, t_preds, w_path = train_single_seed(
            seed=s,
            split_items=split_items,
            output_dir=run_dir,
            device=device,
            max_epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            img_size=args.img_size,
        )
        val_preds_seeds.append(v_preds)
        test_preds_seeds.append(t_preds)
        best_weight_paths.append(w_path)

    total_train_minutes = (time.time() - start_time) / 60.0

    # Save ensemble weights copy as best.pt
    primary_best_pt = run_dir / "weights" / "best.pt"
    if best_weight_paths:
        import shutil
        shutil.copy2(best_weight_paths[0], primary_best_pt)

    # Compute ensemble predictions: average of both seeds
    val_fids = [it["frame_id"] for it in split_items["val"]]
    test_fids = [it["frame_id"] for it in split_items["test"]]

    val_cls_scores = {
        fid: float(np.mean([seed_dict[fid] for seed_dict in val_preds_seeds]))
        for fid in val_fids
    }
    test_cls_scores = {
        fid: float(np.mean([seed_dict[fid] for seed_dict in test_preds_seeds]))
        for fid in test_fids
    }

    # Save ensemble frame predictions
    preds_val_file = run_dir / "preds_val.json"
    preds_test_file = run_dir / "preds_test.json"
    preds_val_file.write_text(json.dumps(val_cls_scores, indent=2), encoding="utf-8")
    preds_test_file.write_text(json.dumps(test_cls_scores, indent=2), encoding="utf-8")
    logger.info("Saved ensemble frame predictions to %s and %s", preds_val_file, preds_test_file)

    # Save run_config.json
    run_config = {
        "run_name": run_dir.name,
        "task": "classify_frame",
        "model": "torchvision_resnet18",
        "imgsz": args.img_size,
        "batch": args.batch_size,
        "lr": args.lr,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "patience": args.patience,
        "device": str(device),
        "total_train_minutes": round(total_train_minutes, 2),
        "fold": None,
        "split": "split_v1",
        "best_weights": [str(p) for p in best_weight_paths],
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    # 3. Load Detector frame scores (run1 with 544_padded)
    det_val_scores = load_detector_frame_scores(ROOT / "outputs" / "preds_val_544.json")
    det_test_scores = load_detector_frame_scores(ROOT / "outputs" / "preds_test_544.json")

    # Assemble DataFrames for evaluation
    def make_eval_df(items: list[dict[str, Any]], cls_scores: dict[str, float], det_scores: dict[str, float]) -> pd.DataFrame:
        rows = []
        for it in items:
            fid = it["frame_id"]
            rows.append({
                "frame_id": fid,
                "patient_id": it["patient_id"],
                "video_id": it["video_id"],
                "label": int(it["label"]),
                "score_cls": cls_scores.get(fid, 0.0),
                "score_det": det_scores.get(fid, 0.0),
            })
        return pd.DataFrame(rows)

    df_val = make_eval_df(split_items["val"], val_cls_scores, det_val_scores)
    df_test = make_eval_df(split_items["test"], test_cls_scores, det_test_scores)

    # 4. Tune ensemble weight on VAL: score_ens = w * score_cls + (1 - w) * score_det
    best_w = 0.5
    best_ens_auc = -1.0
    for w_cand in np.linspace(0.0, 1.0, 21):
        s_cand = w_cand * df_val["score_cls"] + (1.0 - w_cand) * df_val["score_det"]
        auc_cand = roc_auc_score(df_val["label"], s_cand)
        if auc_cand > best_ens_auc:
            best_ens_auc = auc_cand
            best_w = float(w_cand)

    logger.info("Selected optimal ensemble weight on VAL: w_cls=%.2f (VAL ROC-AUC=%.4f)", best_w, best_ens_auc)
    df_val["score_ens"] = best_w * df_val["score_cls"] + (1.0 - best_w) * df_val["score_det"]
    df_test["score_ens"] = best_w * df_test["score_cls"] + (1.0 - best_w) * df_test["score_det"]

    # 5. Apply temporal smoothing over neighbouring frames in the same video
    for col in ["score_cls", "score_det", "score_ens"]:
        smooth_col = f"{col}_smooth"
        df_val[smooth_col] = temporal_smooth_scores(df_val, score_col=col, window_size=3)
        df_test[smooth_col] = temporal_smooth_scores(df_test, score_col=col, window_size=3)

    # Variants dictionary
    variants = {
        "classifier_alone": ("score_cls", False),
        "detector_alone": ("score_det", False),
        "weighted_ensemble": ("score_ens", False),
        "classifier_smoothed": ("score_cls_smooth", True),
        "detector_smoothed": ("score_det_smooth", True),
        "ensemble_smoothed": ("score_ens_smooth", True),
    }

    # Frame-level results
    frame_results: dict[str, Any] = {}
    winner_variant = None
    best_val_balanced_acc = -1.0

    logger.info("=== FRAME-LEVEL EVALUATION & BOOTSTRAP ===")
    for var_name, (score_col, is_smooth) in variants.items():
        # Choose threshold on VAL (maximizing balanced accuracy)
        opt_th = find_optimal_threshold(df_val["label"].values, df_val[score_col].values, metric="balanced_accuracy")

        # Evaluate on VAL and TEST at opt_th
        val_metrics = compute_binary_metrics(df_val["label"].values, df_val[score_col].values, threshold=opt_th)
        test_metrics = compute_binary_metrics(df_test["label"].values, df_test[score_col].values, threshold=opt_th)

        # 500-iteration patient cluster bootstrap CIs
        val_cis = patient_bootstrap_ci(df_val, patient_col="patient_id", score_col=score_col, label_col="label", threshold=opt_th)
        test_cis = patient_bootstrap_ci(df_test, patient_col="patient_id", score_col=score_col, label_col="label", threshold=opt_th)

        val_metrics["ci_95"] = val_cis
        test_metrics["ci_95"] = test_cis

        frame_results[var_name] = {
            "val": val_metrics,
            "test": test_metrics,
            "threshold": opt_th,
        }

        logger.info(
            "Frame Variant '%s' (th=%.4f) | Val AUC=%.4f (BalAcc=%.4f) | Test AUC=%.4f (BalAcc=%.4f)",
            var_name, opt_th, val_metrics["roc_auc"], val_metrics["balanced_accuracy"],
            test_metrics["roc_auc"], test_metrics["balanced_accuracy"],
        )

        if val_metrics["balanced_accuracy"] > best_val_balanced_acc:
            best_val_balanced_acc = val_metrics["balanced_accuracy"]
            winner_variant = var_name

    logger.info("Winner variant on VAL: %s (VAL Balanced Accuracy: %.4f)", winner_variant, best_val_balanced_acc)

    # 6. Video-level task: video label = lesion video. Video score = mean of top-k frame scores over video's keyframes.
    logger.info("=== VIDEO-LEVEL EVALUATION ===")
    vi = pd.read_csv(ROOT / "data_index" / "videos_index.csv")

    def aggregate_video_scores(df_frames: pd.DataFrame, score_col: str, k: int) -> pd.DataFrame:
        v_rows = []
        for (pid, vid), group in df_frames.groupby(["patient_id", "video_id"]):
            scores = group[score_col].values
            # Top-k mean
            if len(scores) <= k:
                v_score = float(np.mean(scores))
            else:
                topk = np.partition(scores, -k)[-k:]
                v_score = float(np.mean(topk))

            # Video label from videos_index
            match_v = vi[(vi["patient_id"] == pid) & (vi["video_id"] == vid)]
            if not match_v.empty:
                v_label = 1 if bool(match_v.iloc[0]["is_lesion_video"]) else 0
            else:
                v_label = 1 if (group["label"].max() == 1) else 0

            v_rows.append({
                "patient_id": pid,
                "video_id": vid,
                "video_score": v_score,
                "label": v_label,
            })
        return pd.DataFrame(v_rows)

    # Tune top-k on VAL using winner frame variant
    win_score_col = variants[winner_variant][0]
    best_k = 1
    best_val_video_auc = -1.0
    for k_cand in [1, 2, 3, 5, 10]:
        df_v_val_cand = aggregate_video_scores(df_val, score_col=win_score_col, k=k_cand)
        try:
            auc_v = roc_auc_score(df_v_val_cand["label"], df_v_val_cand["video_score"])
        except Exception:
            auc_v = 0.5
        if auc_v > best_val_video_auc:
            best_val_video_auc = auc_v
            best_k = k_cand

    logger.info("Selected optimal top-k on VAL: k=%d (VAL Video ROC-AUC=%.4f)", best_k, best_val_video_auc)

    video_results: dict[str, Any] = {}
    for var_name, (score_col, _) in variants.items():
        v_val_df = aggregate_video_scores(df_val, score_col=score_col, k=best_k)
        v_test_df = aggregate_video_scores(df_test, score_col=score_col, k=best_k)

        opt_th_v = find_optimal_threshold(v_val_df["label"].values, v_val_df["video_score"].values, metric="balanced_accuracy")
        val_m_v = compute_binary_metrics(v_val_df["label"].values, v_val_df["video_score"].values, threshold=opt_th_v)
        test_m_v = compute_binary_metrics(v_test_df["label"].values, v_test_df["video_score"].values, threshold=opt_th_v)

        val_m_v["ci_95"] = patient_bootstrap_ci(v_val_df, patient_col="patient_id", score_col="video_score", label_col="label", threshold=opt_th_v)
        test_m_v["ci_95"] = patient_bootstrap_ci(v_test_df, patient_col="patient_id", score_col="video_score", label_col="label", threshold=opt_th_v)

        video_results[var_name] = {
            "val": val_m_v,
            "test": test_m_v,
            "threshold": opt_th_v,
            "k": best_k,
        }

        logger.info(
            "Video Variant '%s' (th=%.4f, k=%d) | Val AUC=%.4f (BalAcc=%.4f) | Test AUC=%.4f (BalAcc=%.4f)",
            var_name, opt_th_v, best_k, val_m_v["roc_auc"], val_m_v["balanced_accuracy"],
            test_m_v["roc_auc"], test_m_v["balanced_accuracy"],
        )

    # 7. Patient-level evaluation
    def aggregate_patient_scores(df_v: pd.DataFrame) -> pd.DataFrame:
        p_rows = []
        for pid, group in df_v.groupby("patient_id"):
            p_score = float(group["video_score"].max())
            p_label = int(group["label"].max())
            p_rows.append({"patient_id": pid, "patient_score": p_score, "label": p_label})
        return pd.DataFrame(p_rows)

    p_val_df = aggregate_patient_scores(aggregate_video_scores(df_val, score_col=win_score_col, k=best_k))
    p_test_df = aggregate_patient_scores(aggregate_video_scores(df_test, score_col=win_score_col, k=best_k))

    th_p = find_optimal_threshold(p_val_df["label"].values, p_val_df["patient_score"].values, metric="balanced_accuracy")
    p_val_metrics = compute_binary_metrics(p_val_df["label"].values, p_val_df["patient_score"].values, threshold=th_p)
    p_test_metrics = compute_binary_metrics(p_test_df["label"].values, p_test_df["patient_score"].values, threshold=th_p)

    patient_results = {
        "val": p_val_metrics,
        "test": p_test_metrics,
        "threshold": th_p,
    }

    # 8. Save comprehensive evaluation outputs in metrics.json format for audit compatibility
    eval_val_dir = ROOT / "outputs" / "eval" / f"{run_dir.name}_val"
    eval_test_dir = ROOT / "outputs" / "eval" / f"{run_dir.name}_test"
    eval_val_dir.mkdir(parents=True, exist_ok=True)
    eval_test_dir.mkdir(parents=True, exist_ok=True)

    win_val = frame_results[winner_variant]["val"]
    win_test = frame_results[winner_variant]["test"]

    val_metrics_json = {
        "run_name": run_dir.name,
        "split": "val",
        "winner_variant": winner_variant,
        "frame": {
            "roc_auc": win_val["roc_auc"],
            "pr_auc": win_val["pr_auc"],
            "balanced_accuracy": win_val["balanced_accuracy"],
            "accuracy": win_val["accuracy"],
            "sensitivity": win_val["sensitivity"],
            "specificity": win_val["specificity"],
            "f1": win_val["f1"],
            "precision": win_val["precision"],
            "confusion_matrix": win_val["confusion_matrix"],
            "positive_prevalence": win_val["positive_prevalence"],
            "majority_baseline": win_val["majority_baseline"],
            "conf_threshold": win_val["threshold"],
            "ci_95": win_val["ci_95"],
        },
        "video": video_results[winner_variant]["val"],
        "patient": patient_results["val"],
        "all_frame_variants": {k: v["val"] for k, v in frame_results.items()},
        "all_video_variants": {k: v["val"] for k, v in video_results.items()},
    }

    test_metrics_json = {
        "run_name": run_dir.name,
        "split": "test",
        "winner_variant": winner_variant,
        "frame": {
            "roc_auc": win_test["roc_auc"],
            "pr_auc": win_test["pr_auc"],
            "balanced_accuracy": win_test["balanced_accuracy"],
            "accuracy": win_test["accuracy"],
            "sensitivity": win_test["sensitivity"],
            "specificity": win_test["specificity"],
            "f1": win_test["f1"],
            "precision": win_test["precision"],
            "confusion_matrix": win_test["confusion_matrix"],
            "positive_prevalence": win_test["positive_prevalence"],
            "majority_baseline": win_test["majority_baseline"],
            "conf_threshold": win_test["threshold"],
            "ci_95": win_test["ci_95"],
        },
        "video": video_results[winner_variant]["test"],
        "patient": patient_results["test"],
        "all_frame_variants": {k: v["test"] for k, v in frame_results.items()},
        "all_video_variants": {k: v["test"] for k, v in video_results.items()},
    }

    (eval_val_dir / "metrics.json").write_text(json.dumps(val_metrics_json, indent=2), encoding="utf-8")
    (eval_test_dir / "metrics.json").write_text(json.dumps(test_metrics_json, indent=2), encoding="utf-8")

    # Also save overall summary
    summary_path = ROOT / "outputs" / "eval" / "screening_summary.json"
    full_summary = {
        "winner_variant": winner_variant,
        "best_w": best_w,
        "best_k": best_k,
        "frame_results": frame_results,
        "video_results": video_results,
        "patient_results": patient_results,
    }
    summary_path.write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
    logger.info("Saved metrics.json to %s and %s", eval_val_dir, eval_test_dir)
    logger.info("=== SCREENING PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
