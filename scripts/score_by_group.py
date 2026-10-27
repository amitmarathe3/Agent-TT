"""Score-by-group analysis on the original ResNet18 classifier.

Analyzes original model predictions across:
1. Annotated keyframes vs unannotated selected keyframes
2. Annotated keyframes vs distant non-keyframes
3. Frame position within video (first third, middle third, last third)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent

def main() -> None:
    fi = pd.read_csv(ROOT / "data_index" / "frames_index.csv")
    sp = pd.read_csv(ROOT / "data_index" / "splits" / "split_v1.csv")
    p2s = dict(zip(sp["patient_id"], sp["split"]))
    fi["split"] = fi["patient_id"].map(p2s)
    
    # Extract frame number from frame_id (e.g. p10_v1_00016 -> 16)
    def get_frame_num(fid: str) -> int:
        return int(fid.split("_")[-1])
    
    fi["frame_num"] = fi["frame_id"].map(get_frame_num)
    
    # Video bounds (min and max frame number per video)
    v_bounds = fi.groupby(["patient_id", "video_id"])["frame_num"].agg(["min", "max", "count"]).reset_index()
    v_bounds["span"] = v_bounds["max"] - v_bounds["min"]
    fi = fi.merge(v_bounds, on=["patient_id", "video_id"], suffixes=("", "_v"))
    fi["rel_pos"] = np.where(fi["span"] > 0, (fi["frame_num"] - fi["min"]) / fi["span"], 0.5)
    
    fi = fi.drop_duplicates(subset=["frame_id"])
    fi_dict = fi.set_index("frame_id").to_dict(orient="index")
    
    matched_preds = json.load(open(ROOT / "outputs" / "runs" / "exp_frame_resnet18" / "preds_matched_negatives.json"))
    
    results = {}
    
    for split in ["val", "test"]:
        preds_file = ROOT / "outputs" / "runs" / "exp_frame_resnet18" / f"preds_{split}.json"
        preds = json.load(open(preds_file))
        
        # Combine canonical preds with matched negatives
        all_preds = dict(preds)
        for k, v in matched_preds.items():
            if k in fi_dict and fi_dict[k]["split"] == split:
                all_preds[k] = v
                
        df_rows = []
        for fid, score in all_preds.items():
            if fid not in fi_dict:
                continue
            info = fi_dict[fid]
            df_rows.append({
                "frame_id": fid,
                "score": score,
                "has_gt": bool(info["has_groundtruth"]),
                "is_keyframe": bool(info["is_selected_keyframe"]),
                "split_type": info["split_type"],
                "has_lesion_video": bool(info["has_lesion_video"]),
                "rel_pos": info["rel_pos"],
            })
            
        df = pd.DataFrame(df_rows)
        
        # Group 1: Annotated keyframes (has_gt=True, is_keyframe=True)
        # vs Unannotated selected keyframes (has_gt=False, is_keyframe=True, split_type=selected)
        g1_pos = df[df["has_gt"] & df["is_keyframe"]]
        g1_neg = df[(~df["has_gt"]) & df["is_keyframe"] & (df["split_type"] == "selected")]
        
        mean_score_ann_kf = float(g1_pos["score"].mean()) if len(g1_pos) else float("nan")
        mean_score_unann_kf = float(g1_neg["score"].mean()) if len(g1_neg) else float("nan")
        
        g1_df = pd.concat([
            pd.DataFrame({"score": g1_pos["score"], "label": 1}),
            pd.DataFrame({"score": g1_neg["score"], "label": 0}),
        ])
        auc_ann_vs_unann_kf = float(roc_auc_score(g1_df["label"], g1_df["score"])) if len(g1_df["label"].unique()) == 2 else float("nan")
        
        # Group 2: Annotated keyframes vs Distant non-keyframes (has_gt=False, is_keyframe=False)
        g2_neg = df[(~df["has_gt"]) & (~df["is_keyframe"])]
        mean_score_distant_non_kf = float(g2_neg["score"].mean()) if len(g2_neg) else float("nan")
        
        g2_df = pd.concat([
            pd.DataFrame({"score": g1_pos["score"], "label": 1}),
            pd.DataFrame({"score": g2_neg["score"], "label": 0}),
        ])
        auc_ann_vs_distant_non_kf = float(roc_auc_score(g2_df["label"], g2_df["score"])) if len(g2_df["label"].unique()) == 2 else float("nan")
        
        # Group 3: Position in video (first third [0, 0.333), middle [0.333, 0.667), last [0.667, 1.0])
        # Evaluated on canonical eval set (original preds)
        canonical_fids = set(preds.keys())
        can_df = df[df["frame_id"].isin(canonical_fids)].copy()
        
        def pos_bin(r):
            if r < 1/3:
                return "first_third"
            elif r < 2/3:
                return "middle_third"
            else:
                return "last_third"
                
        can_df["bin"] = can_df["rel_pos"].map(pos_bin)
        
        bin_results = {}
        for b in ["first_third", "middle_third", "last_third"]:
            b_df = can_df[can_df["bin"] == b]
            n_pos = int(b_df["has_gt"].sum())
            n_neg = len(b_df) - n_pos
            mean_pos = float(b_df[b_df["has_gt"]]["score"].mean()) if n_pos > 0 else float("nan")
            mean_neg = float(b_df[~b_df["has_gt"]]["score"].mean()) if n_neg > 0 else float("nan")
            b_auc = float(roc_auc_score(b_df["has_gt"].astype(int), b_df["score"])) if len(b_df["has_gt"].unique()) == 2 else float("nan")
            bin_results[b] = {
                "n_frames": len(b_df),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "mean_pos_score": round(mean_pos, 4) if not np.isnan(mean_pos) else None,
                "mean_neg_score": round(mean_neg, 4) if not np.isnan(mean_neg) else None,
                "roc_auc": round(b_auc, 4) if not np.isnan(b_auc) else None,
            }
            
        results[split] = {
            "annotated_vs_unannotated_selected_kf": {
                "n_annotated_kf": len(g1_pos),
                "mean_score_annotated_kf": round(mean_score_ann_kf, 4),
                "n_unannotated_kf": len(g1_neg),
                "mean_score_unannotated_kf": round(mean_score_unann_kf, 4),
                "roc_auc": round(auc_ann_vs_unann_kf, 4),
            },
            "annotated_vs_distant_non_kf": {
                "n_annotated_kf": len(g1_pos),
                "mean_score_annotated_kf": round(mean_score_ann_kf, 4),
                "n_distant_non_kf": len(g2_neg),
                "mean_score_distant_non_kf": round(mean_score_distant_non_kf, 4),
                "roc_auc": round(auc_ann_vs_distant_non_kf, 4),
            },
            "by_video_position": bin_results,
        }
        
    out_file = ROOT / "outputs" / "eval" / "score_by_group.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("=== SCORE-BY-GROUP RESULTS ===")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
