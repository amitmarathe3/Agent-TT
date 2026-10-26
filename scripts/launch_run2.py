#!/usr/bin/env python3
"""Launch improved training Run 2 for CADICA Agent-TT.

Changes vs Run 1 (exp_7class_yolo11s_oversample_run1):
  - imgsz          : 512  -> 640  (native YOLO11s resolution)
  - epochs         : 100  -> 150  (longer convergence window)
  - patience       : 25   -> 40   (avoid early stopping)
  - oversample mul : 3x   -> 5x   for p99, p100, p90_98
  - Predict using  : 544_padded  (proven 2.5x mAP improvement)

After training this script automatically:
  1. Predicts on val + test with 544_padded mode
  2. Evaluates both splits -> outputs/eval/<run>_val/ and _test/
  3. Updates leaderboard.csv
  4. Runs audit_project.py and prints result

Usage:
    python scripts/launch_run2.py [--smoke-test] [--skip-train]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = r"C:\Users\HP\AppData\Local\Programs\Python\Python314\python.exe"

RUN_NAME = "exp_7class_yolo11s_finetune_run2"
SCHEME   = "7class"
MODEL    = str(ROOT / "outputs" / "runs" / "exp_7class_yolo11s_oversample_run1" / "weights" / "best.pt")
IMGSZ    = 512          # GTX 1650 (4 GB) safe — 640 causes OOM
BATCH    = 8            # explicit batch (auto-batch OOMs at 640)
EPOCHS   = 25           # warm-start 25 epochs fine-tuning
PATIENCE = 20           # early stopping patience
OVERSAMPLE_MUL = 5      # repeat p99/p100/p90_98 5x instead of 3x

LB = ROOT / "outputs" / "experiments" / "leaderboard.csv"


def run(cmd: list[str], desc: str, cwd: Path = ROOT, timeout: int = 7200) -> int:
    print(f"\n>>> {desc}")
    print(f"    {' '.join(str(c) for c in cmd)}\n")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(cwd), timeout=timeout)
    elapsed = time.time() - t0
    status = "OK" if r.returncode == 0 else f"FAILED (rc={r.returncode})"
    print(f"\n    [{status}] {desc} finished in {elapsed/60:.1f} min")
    return r.returncode


def patch_trainer_multiplier(multiplier: int) -> None:
    """Temporarily patch prepare_oversampled_dataset multiplier default."""
    trainer = ROOT / "src" / "models" / "yolo_trainer.py"
    src = trainer.read_text(encoding="utf-8")
    new_src = re.sub(r"multiplier: int = \d+,", f"multiplier: int = {multiplier},", src, count=1)
    if new_src != src:
        trainer.write_text(new_src, encoding="utf-8")
        print(f"    [patch] oversampling multiplier set to {multiplier}x")
    else:
        print(f"    [warn] multiplier patch not applied")


def restore_trainer_multiplier() -> None:
    """Restore default multiplier=3 after training."""
    trainer = ROOT / "src" / "models" / "yolo_trainer.py"
    src = trainer.read_text(encoding="utf-8")
    new_src = re.sub(r"multiplier: int = \d+,", "multiplier: int = 3,", src, count=1)
    if new_src != src:
        trainer.write_text(new_src, encoding="utf-8")
        print("    [restore] multiplier restored to default 3x")


def append_leaderboard(run_name: str, metrics_v: dict, metrics_t: dict) -> None:
    """Append/update leaderboard.csv with run2 results."""
    import pandas as pd

    row = {
        "run_name"               : run_name,
        "scheme"                 : SCHEME,
        "model"                  : MODEL,
        "imgsz"                  : IMGSZ,
        "oversample"             : True,
        "fold"                   : "",
        "train_minutes"          : "",
        "val_P"                  : round(metrics_v.get("detection", {}).get("micro", {}).get("precision", float("nan")), 4),
        "val_R"                  : round(metrics_v.get("detection", {}).get("micro", {}).get("recall",    float("nan")), 4),
        "val_F1"                 : round(metrics_v.get("detection", {}).get("micro", {}).get("f1",        float("nan")), 4),
        "val_mAP50"              : round(metrics_v.get("map", {}).get("mAP_50",   float("nan")), 4),
        "val_mAP5095"            : round(metrics_v.get("map", {}).get("mAP_50_95", float("nan")), 4),
        "val_frame_sensitivity"  : round(metrics_v.get("frame", {}).get("sensitivity", float("nan")), 4),
        "val_frame_specificity"  : round(metrics_v.get("frame", {}).get("specificity", float("nan")), 4),
        "val_roc_auc"            : round(metrics_v.get("frame", {}).get("roc_auc",     float("nan")), 4),
        "test_P"                 : round(metrics_t.get("detection", {}).get("micro", {}).get("precision", float("nan")), 4),
        "test_R"                 : round(metrics_t.get("detection", {}).get("micro", {}).get("recall",    float("nan")), 4),
        "test_F1"                : round(metrics_t.get("detection", {}).get("micro", {}).get("f1",        float("nan")), 4),
        "test_mAP50"             : round(metrics_t.get("map", {}).get("mAP_50",   float("nan")), 4),
        "test_mAP5095"           : round(metrics_t.get("map", {}).get("mAP_50_95", float("nan")), 4),
        "test_frame_sensitivity" : round(metrics_t.get("frame", {}).get("sensitivity", float("nan")), 4),
        "test_frame_specificity" : round(metrics_t.get("frame", {}).get("specificity", float("nan")), 4),
        "test_roc_auc"           : round(metrics_t.get("frame", {}).get("roc_auc",     float("nan")), 4),
    }

    LB.parent.mkdir(parents=True, exist_ok=True)
    if LB.exists():
        df = pd.read_csv(LB)
        df = df[df["run_name"] != run_name]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(LB, index=False)
    print(f"    [leaderboard] Updated: {LB}")
    print(f"    val mAP@0.5={row['val_mAP50']}  test mAP@0.5={row['test_mAP50']}")
    print(f"    val AUC={row['val_roc_auc']}  test AUC={row['test_roc_auc']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="2 epochs, quick sanity test.")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training; just predict + evaluate existing weights.")
    args = parser.parse_args(argv)

    smoke = args.smoke_test
    epochs = 2 if smoke else EPOCHS
    run_name = RUN_NAME + ("_smoke" if smoke else "")

    print("=" * 70)
    print("  CADICA Agent-TT -- Improved Training Run 2")
    print(f"  Run name : {run_name}")
    print(f"  imgsz    : {IMGSZ}  batch: {BATCH}  epochs: {epochs}  patience: {PATIENCE}")
    print(f"  oversample multiplier: {OVERSAMPLE_MUL}x")
    print("  inference: 544_padded (proven 2.5x mAP vs 512-direct)")
    print("=" * 70)

    if not args.skip_train:
        rc = run(
            [
                PY, "scripts/train_yolo.py",
                "--scheme", SCHEME,
                "--model", MODEL,
                "--imgsz", str(IMGSZ),
                "--batch", str(BATCH),
                "--epochs", str(epochs),
                "--patience", str(PATIENCE),
                "--oversample-rare",
                "--oversample-multiplier", str(OVERSAMPLE_MUL),
                "--run-name", run_name,
                "--workers", "2",
            ] + (["--smoke-test"] if smoke else []),
            desc=f"Training {run_name}",
            timeout=28800,
        )

        if rc != 0:
            print(f"\n[FAIL] Training failed (rc={rc}). Aborting.")
            return rc

    best_pt = ROOT / "outputs" / "runs" / run_name / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"\n[FAIL] best.pt not found at {best_pt}. Aborting.")
        return 1

    # ---------------------------------------------------------------- predict
    for split in ("val", "test"):
        out_json = ROOT / "outputs" / "runs" / run_name / f"preds_{split}.json"
        rc = run(
            [
                PY, "scripts/predict_yolo.py",
                "--run-dir", str(ROOT / "outputs" / "runs" / run_name),
                "--split", split,
                "--output", str(out_json),
                "--preprocess", "544_padded",
                "--conf", "0.001",
                "--iou", "0.6",
                "--max-det", "50",
                "--imgsz", "512",
            ],
            desc=f"Predicting {split} (544_padded)",
            timeout=1200,
        )
        if rc != 0:
            print(f"\n[FAIL] Prediction ({split}) failed.")
            return rc

    # ---------------------------------------------------------------- evaluate
    for split in ("val", "test"):
        pred_json = ROOT / "outputs" / "runs" / run_name / f"preds_{split}.json"
        eval_name = f"{run_name}_{split}"
        rc = run(
            [
                PY, "scripts/evaluate.py",
                "--pred", str(pred_json),
                "--split", split,
                "--scheme", SCHEME,
                "--run-name", eval_name,
                "--bootstrap",
            ],
            desc=f"Evaluating {split}",
            timeout=1200,
        )
        if rc != 0:
            print(f"\n[FAIL] Evaluation ({split}) failed.")
            return rc

    # ---------------------------------------------------------------- leaderboard
    mv = ROOT / "outputs" / "eval" / f"{run_name}_val"  / "metrics.json"
    mt = ROOT / "outputs" / "eval" / f"{run_name}_test" / "metrics.json"
    if mv.exists() and mt.exists():
        metrics_v = json.loads(mv.read_text(encoding="utf-8"))
        metrics_t = json.loads(mt.read_text(encoding="utf-8"))
        try:
            append_leaderboard(run_name, metrics_v, metrics_t)
        except Exception as e:
            print(f"    [warn] leaderboard update failed: {e}")
    else:
        print("    [warn] metrics.json missing -- skipping leaderboard update")

    # ---------------------------------------------------------------- audit
    rc = run([PY, "scripts/audit_project.py"], desc="Final project audit", timeout=120)

    print("\n" + "=" * 70)
    print("  DONE. Review outputs/eval/<run>_val/metrics.json for results.")
    print("=" * 70)
    return rc


if __name__ == "__main__":
    sys.exit(main())
