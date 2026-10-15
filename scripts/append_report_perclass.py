#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

names = {0: "p0_20", 1: "p20_50", 2: "p50_70", 3: "p70_90", 4: "p90_98", 5: "p99", 6: "p100"}

for sp in ["val", "test"]:
    metrics_path = Path(f"outputs/eval/exp_7class_yolo11s_oversample_run1_{sp}/metrics.json")
    report_path = Path(f"outputs/eval/exp_7class_yolo11s_oversample_run1_{sp}/metrics.md")
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    det = m["detection"]["per_class"]
    ap = m["map"]["per_class_AP_50"]
    lines = []
    lines.append("")
    lines.append("## Per-Class Summary (GT / TP / FP / FN / P / R / F1 / AP@0.5)")
    lines.append("")
    lines.append("| Class | GT | TP | FP | FN | Precision | Recall | F1 | AP@0.5 | Low Support (<30 GT) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for cid in sorted(det, key=lambda x: int(x)):
        d = det[cid]
        gt = int(d["TP"]) + int(d["FN"])
        ap_key = str(cid) if str(cid) in ap else int(cid)
        low = "yes" if gt < 30 else ""
        lines.append(
            f"{d['name']} | {gt} | {int(d['TP'])} | {int(d['FP'])} | {int(d['FN'])} | "
            f"{float(d['precision']):.4f} | {float(d['recall']):.4f} | {float(d['f1']):.4f} | "
            f"{float(ap[ap_key]):.4f} | {low} |"
        )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- `best.pt` (checkpoint selected on val) was used for both val and test.")
    lines.append("- Frame specificity and ROC/PR-AUC are computed on sampled hard negatives, not natural clinical prevalence.")
    lines.append("- Test was inspected once earlier in a single-class diagnostic (mAP@0.5 ~0.025); no tuning was derived from it.")
    lines.append("")
    text = report_path.read_text(encoding="utf-8")
    if "## Caveats" in text and "best.pt (checkpoint selected on val)" in text:
        text = text.split("## Caveats", 1)[0].rstrip()
    text = text.rstrip() + "\n" + "\n".join(lines)
    report_path.write_text(text, encoding="utf-8")
    print("updated", report_path)
