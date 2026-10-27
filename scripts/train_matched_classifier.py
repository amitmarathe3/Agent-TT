"""Matched-negative frame classifier training & video screening evaluation.

Positives = annotated keyframes (from lesion videos)
Negatives = selected keyframes from NONLESION videos only (both sides are keyframes)

Recipe:
- Torchvision ResNet18 (ImageNet-pretrained), 384 px, batch 16, AMP off, CUDA.
- AdamW lr 3e-4, class-weighted BCE, up to 8 epochs with early stopping on val AUC (patience 3).
- Seed: 42.
- 500-iteration patient cluster bootstrap 95% CIs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_matched_classifier")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_matched_items() -> dict[str, list[dict]]:
    fi = pd.read_csv(ROOT / "data_index" / "frames_index.csv")
    sp = pd.read_csv(ROOT / "data_index" / "splits" / "split_v1.csv")
    p2s = dict(zip(sp["patient_id"], sp["split"]))
    fi["split"] = fi["patient_id"].map(p2s)
    
    # Selected keyframes only
    sel_kf = fi[fi["is_selected_keyframe"] == True].copy()
    
    # Positives: has_groundtruth == True (from lesion videos)
    # Negatives: has_lesion_video == False (from nonlesion videos)
    matched = sel_kf[sel_kf["has_groundtruth"] | (~sel_kf["has_lesion_video"])].copy()
    
    extracted_base = ROOT / "outputs" / "extracted_keyframes"
    yolo_base = ROOT / "outputs" / "yolo_dataset" / "images"
    
    split_items = {"train": [], "val": [], "test": []}
    
    for _, row in matched.iterrows():
        split = row["split"]
        fid = row["frame_id"]
        is_pos = 1.0 if row["has_groundtruth"] else 0.0
        
        # Check path: either in yolo_dataset or in extracted_keyframes
        yolo_path = yolo_base / split / f"{fid}.png"
        extracted_path = extracted_base / split / f"{fid}.png"
        
        if yolo_path.exists():
            img_path = str(yolo_path)
        elif extracted_path.exists():
            img_path = str(extracted_path)
        else:
            logger.warning("Frame image not found: %s", fid)
            continue
            
        split_items[split].append({
            "frame_id": fid,
            "image_path": img_path,
            "label": is_pos,
            "patient_id": str(row["patient_id"]),
            "video_id": str(row["video_id"]),
        })
        
    for s in ["train", "val", "test"]:
        n_pos = sum(1 for x in split_items[s] if x["label"] == 1.0)
        n_neg = len(split_items[s]) - n_pos
        logger.info("Matched %s items: total=%d, pos=%d, neg=%d", s, len(split_items[s]), n_pos, n_neg)
        if n_neg < 30:
            raise ValueError(f"Split {s} has under 30 negatives ({n_neg})!")
            
    return split_items


def main() -> None:
    seed = 42
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    split_items = prepare_matched_items()
    
    output_dir = ROOT / "outputs" / "runs" / "exp_frame_matched_negatives"
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    train_tf, eval_tf = get_transforms(img_size=384)
    train_ds = CADICAFrameDataset(split_items["train"], transform=train_tf)
    val_ds = CADICAFrameDataset(split_items["val"], transform=eval_tf)
    test_ds = CADICAFrameDataset(split_items["test"], transform=eval_tf)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    
    train_labels = [it["label"] for it in split_items["train"]]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    logger.info("Class weight pos_weight = %.4f", float(pos_weight.item()))
    
    model = build_resnet18(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_val_auc = -1.0
    best_weights_path = weights_dir / "best.pt"
    no_improve = 0
    max_epochs = 8
    patience = 3
    
    if not best_weights_path.exists():
        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            model.train()
            total_loss = 0.0
            
            for batch in train_loader:
                images = batch["image"].to(device)
                targets = batch["label"].to(device)
                optimizer.zero_grad()
                logits = model(images).squeeze(-1)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(targets)
                
            epoch_loss = total_loss / len(train_ds)
            val_preds, val_targets, _ = evaluate_model_on_loader(model, val_loader, device)
            val_auc = float(roc_auc_score(val_targets, val_preds))
            dt = time.time() - t0
            logger.info("Epoch %d/%d (%.1fs) - Train Loss: %.4f - Val ROC-AUC: %.4f", epoch, max_epochs, dt, epoch_loss, val_auc)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), best_weights_path)
                no_improve = 0
                logger.info("  --> Best val AUC improved to %.4f. Saved checkpoint.", best_val_auc)
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping triggered after %d epochs without improvement.", patience)
                    break
    else:
        logger.info("Found existing checkpoint %s. Skipping training loop.", best_weights_path)
                
    val_preds_path = output_dir / "preds_val.json"
    test_preds_path = output_dir / "preds_test.json"
    if val_preds_path.exists() and test_preds_path.exists():
        logger.info("Found existing predictions on disk. Loading...")
        preds_val_dict = json.load(open(val_preds_path))
        preds_test_dict = json.load(open(test_preds_path))
        val_preds = np.array([preds_val_dict[it["frame_id"]] for it in split_items["val"]])
        val_targets = np.array([it["label"] for it in split_items["val"]])
        test_preds = np.array([preds_test_dict[it["frame_id"]] for it in split_items["test"]])
        test_targets = np.array([it["label"] for it in split_items["test"]])
    else:
        # Load best checkpoint
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
        model.eval()
        
        val_preds, val_targets, _ = evaluate_model_on_loader(model, val_loader, device)
        test_preds, test_targets, _ = evaluate_model_on_loader(model, test_loader, device)
        
        # Save predictions
        preds_val_dict = {it["frame_id"]: float(p) for it, p in zip(split_items["val"], val_preds)}
        preds_test_dict = {it["frame_id"]: float(p) for it, p in zip(split_items["test"], test_preds)}
        with open(val_preds_path, "w") as f:
            json.dump(preds_val_dict, f, indent=2)
        with open(test_preds_path, "w") as f:
            json.dump(preds_test_dict, f, indent=2)
        
    # Threshold tuning on VAL only
    val_thresh = find_optimal_threshold(val_targets, val_preds, metric="balanced_accuracy")
    val_eval = compute_binary_metrics(val_targets, val_preds, threshold=val_thresh)
    logger.info("Optimal threshold chosen on VAL: %.4f (Val BalAcc: %.4f)", val_thresh, val_eval["balanced_accuracy"])
    
    val_metrics = compute_binary_metrics(val_targets, val_preds, threshold=val_thresh)
    test_metrics = compute_binary_metrics(test_targets, test_preds, threshold=val_thresh)
    
    # Bootstrap CIs (patient clustered)
    val_df = pd.DataFrame({
        "patient_id": [it["patient_id"] for it in split_items["val"]],
        "target": val_targets,
        "score": val_preds,
    })
    test_df = pd.DataFrame({
        "patient_id": [it["patient_id"] for it in split_items["test"]],
        "target": test_targets,
        "score": test_preds,
    })
    val_metrics["ci_95"] = patient_bootstrap_ci(val_df, patient_col="patient_id", score_col="score", label_col="target", threshold=val_thresh, n_bootstraps=500, seed=42)
    test_metrics["ci_95"] = patient_bootstrap_ci(test_df, patient_col="patient_id", score_col="score", label_col="target", threshold=val_thresh, n_bootstraps=500, seed=42)
    
    # Save frame metrics
    for sp_name, m_dict in [("val", val_metrics), ("test", test_metrics)]:
        eval_dir = ROOT / "outputs" / "eval" / f"exp_frame_matched_negatives_{sp_name}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_name": "exp_frame_matched_negatives",
            "split": sp_name,
            "threshold": round(val_thresh, 4),
            "frame": m_dict,
        }
        with open(eval_dir / "metrics.json", "w") as f:
            json.dump(payload, f, indent=2)
            
    print("\n================ MATCHED-NEGATIVE FRAME RESULTS ================")
    print(f"Optimal Val Threshold: {val_thresh:.4f}")
    print(f"VAL  ROC-AUC: {val_metrics['roc_auc']:.4f} {val_metrics['ci_95']['roc_auc']}, BalAcc: {val_metrics['balanced_accuracy']:.4f} {val_metrics['ci_95']['balanced_accuracy']}, Sens: {val_metrics['sensitivity']:.4f}, Spec: {val_metrics['specificity']:.4f}, Prev: {val_metrics['positive_prevalence']:.4f}, Maj: {val_metrics['majority_baseline']:.4f}")
    print(f"TEST ROC-AUC: {test_metrics['roc_auc']:.4f} {test_metrics['ci_95']['roc_auc']}, BalAcc: {test_metrics['balanced_accuracy']:.4f} {test_metrics['ci_95']['balanced_accuracy']}, Sens: {test_metrics['sensitivity']:.4f}, Spec: {test_metrics['specificity']:.4f}, Prev: {test_metrics['positive_prevalence']:.4f}, Maj: {test_metrics['majority_baseline']:.4f}")

    # ==================== ITEM 5: VIDEO-LEVEL REDO ====================
    # Restrict to SELECTED videos only; score each video from its selected keyframes only; label from lesionVideos.txt / nonlesionVideos.txt
    print("\n================ ITEM 5: VIDEO-LEVEL REDO (SELECTED VIDEOS ONLY) ================")
    video_results = {}
    for sp_name, s_items, s_preds in [("val", split_items["val"], val_preds), ("test", split_items["test"], test_preds)]:
        v_df = pd.DataFrame({
            "patient_id": [it["patient_id"] for it in s_items],
            "video_id": [it["video_id"] for it in s_items],
            "target": [it["label"] for it in s_items],
            "score": s_preds,
        })
        
        # Each selected video has multiple keyframes. Compute video score as mean of top-k (k<=10) keyframe scores
        v_agg = v_df.groupby(["patient_id", "video_id"]).agg(
            target=("target", "max"),
            score=("score", lambda s: float(np.mean(np.sort(s)[-min(len(s), 10):]))),
        ).reset_index()
        
        v_targets = v_agg["target"].values
        v_scores = v_agg["score"].values
        v_auc = float(roc_auc_score(v_targets, v_scores))
        
        if sp_name == "val":
            v_thresh = find_optimal_threshold(v_targets, v_scores, metric="balanced_accuracy")
        
        v_metrics = compute_binary_metrics(v_targets, v_scores, threshold=v_thresh)
        
        # Bootstrap
        v_ci = patient_bootstrap_ci(v_agg, patient_col="patient_id", score_col="score", label_col="target", threshold=v_thresh, n_bootstraps=500, seed=42)
        v_metrics["ci_95"] = v_ci
        v_metrics["n_videos"] = len(v_agg)
        v_metrics["n_lesion_videos"] = int(sum(v_targets))
        v_metrics["n_nonlesion_videos"] = len(v_agg) - int(sum(v_targets))
        
        video_results[sp_name] = v_metrics
        
        print(f"{sp_name.upper()} SELECTED VIDEOS (Total={len(v_agg)}, Lesion={int(sum(v_targets))}, Nonlesion={len(v_agg)-int(sum(v_targets))}):")
        print(f"  Threshold: {v_thresh:.4f}")
        print(f"  ROC-AUC:   {v_metrics['roc_auc']:.4f} {v_ci['roc_auc']}")
        print(f"  BalAcc:    {v_metrics['balanced_accuracy']:.4f} {v_ci['balanced_accuracy']}")
        print(f"  Sens:      {v_metrics['sensitivity']:.4f}, Spec: {v_metrics['specificity']:.4f}, F1: {v_metrics['f1']:.4f}")
        
    with open(ROOT / "outputs" / "eval" / "video_level_redo_selected.json", "w") as f:
        json.dump(video_results, f, indent=2)

if __name__ == "__main__":
    main()
