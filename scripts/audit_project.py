#!/usr/bin/env python3
"""Independent project audit.

Standard library + pandas only. This file MUST NOT import src/ or any project
code. It reads raw files on disk, prints a PASS/WARN/FAIL table, writes the
same table to reports/audit.md, and exits non-zero if any check FAILs.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    print(f"pandas is required for the audit: {exc}")
    sys.exit(2)

# AUDIT_ROOT lets tests point the audit at a temporary project tree.
ROOT = Path(os.environ.get("AUDIT_ROOT", Path(__file__).resolve().parent.parent)).resolve()
PY = r"C:\Users\HP\AppData\Local\Programs\Python\Python314\python.exe"

# 7-class scheme used by the YOLO exporter (raw constant; not imported from src/).
SCHEME7 = {
    "p0_20": 0,
    "p20_50": 1,
    "p50_70": 2,
    "p70_90": 3,
    "p90_98": 4,
    "p99": 5,
    "p100": 6,
}

EXPECTED_LABELS = {
    "train": (3420, 4365),
    "val": (722, 757),
    "test": (852, 1039),
}

SANITY_MARKERS = ("oracle", "random")

rows: list[tuple[str, str, str]] = []


def add(check: str, status: str, detail: str = "") -> None:
    rows.append((check, status, detail))


def git(*args: str) -> str:
    try:
        r = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30
        )
        return (r.stdout or "").strip()
    except Exception as exc:
        return f"<git error: {exc}>"


def run_py(code: str, timeout: int = 40) -> tuple[int, str, str]:
    try:
        r = subprocess.run(
            [PY, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()
    except subprocess.TimeoutExpired:
        return -999, "", "timeout"
    except Exception as exc:
        return -998, "", str(exc)


def read_labels(split: str) -> tuple[int, int, dict[int, int]]:
    lbl_dir = ROOT / "outputs" / "yolo_dataset" / "labels" / split
    if not lbl_dir.exists():
        return 0, 0, {}
    files = sorted(lbl_dir.glob("*.txt"))
    n_boxes = 0
    per_class: dict[int, int] = {}
    for f in files:
        try:
            txt = f.read_text(encoding="utf-8")
        except Exception:
            continue
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cls = int(parts[0])
            except Exception:
                continue
            n_boxes += 1
            per_class[cls] = per_class.get(cls, 0) + 1
    return len(files), n_boxes, per_class


def data_index_per_class() -> dict[str, dict[int, int]]:
    out: dict[str, dict[int, int]] = {"train": {}, "val": {}, "test": {}}
    ann_path = ROOT / "data_index" / "annotations.csv"
    split_path = ROOT / "data_index" / "splits" / "split_v1.csv"
    if not ann_path.exists() or not split_path.exists():
        return out
    ann = pd.read_csv(ann_path)
    sp = pd.read_csv(split_path)
    p2s = dict(zip(sp["patient_id"].astype(str), sp["split"].astype(str)))
    ann["split"] = ann["patient_id"].astype(str).map(p2s)
    ann = ann.dropna(subset=["split"])
    ann["cls"] = ann["category"].astype(str).map(SCHEME7)
    ann = ann.dropna(subset=["cls"])
    for split in ("train", "val", "test"):
        sub = ann[ann["split"] == split]
        vc = sub["cls"].astype(int).value_counts()
        out[split] = {int(k): int(v) for k, v in vc.items()}
    return out


def collect_floats(obj, out: set[float]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            collect_floats(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_floats(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.add(round(float(obj), 4))


def known_values() -> set[float]:
    vals: set[float] = set()
    for mp in (ROOT / "outputs").rglob("metrics.json"):
        try:
            collect_floats(json.loads(mp.read_text(encoding="utf-8")), vals)
        except Exception:
            pass
    di = ROOT / "data_index"
    if di.exists():
        for cp in di.rglob("*.csv"):
            try:
                df = pd.read_csv(cp)
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        for v in df[col].dropna().unique():
                            vals.add(round(float(v), 4))
            except Exception:
                pass
        for jp in di.rglob("*.json"):
            try:
                collect_floats(json.loads(jp.read_text(encoding="utf-8")), vals)
            except Exception:
                pass
    # Numeric literals in configs/, src/, scripts/ are treated as sourced so
    # that documented evaluation constants (conf=0.001, iou thresholds, etc.)
    # are not flagged as unmatched.
    num_lit = re.compile(r"(?<![\d.])(\d+\.\d+)(?![\d.])")
    for base in ("configs", "src", "scripts"):
        b = ROOT / base
        if not b.exists():
            continue
        for f in list(b.rglob("*.py")) + list(b.rglob("*.yaml")):
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for tok in num_lit.findall(txt):
                try:
                    vals.add(round(float(tok), 4))
                except Exception:
                    pass
    # Historical / superseded / purged values that docs intentionally quote as
    # wrong, old, or removed. reports/superseded_values.json records why each is
    # retained and what replaced it; nothing there may be a current result.
    sv = ROOT / "reports" / "superseded_values.json"
    if sv.exists():
        try:
            data = json.loads(sv.read_text(encoding="utf-8"))
            for block in data.values():
                if isinstance(block, dict):
                    for v in block.get("values", []):
                        try:
                            vals.add(round(float(v), 4))
                        except Exception:
                            pass
        except Exception:
            pass
    return vals


def dig(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


# ---------------------------------------------------------------------------
# Check 1: label files
# ---------------------------------------------------------------------------
def check1() -> None:
    di_cls = data_index_per_class()
    ok = True
    details = []
    for split, (exp_img, exp_box) in EXPECTED_LABELS.items():
        n_img, n_box, per_cls = read_labels(split)
        if n_img != exp_img or n_box != exp_box:
            ok = False
            details.append(
                f"{split}: got {n_img} imgs/{n_box} boxes, expected {exp_img}/{exp_box}"
            )
        # per-class vs data_index
        di = di_cls.get(split, {})
        if per_cls != di:
            ok = False
            details.append(f"{split} per-class mismatch label={per_cls} data_index={di}")
    if ok:
        add("1 label files & per-class vs data_index", "PASS", "counts match expected")
    else:
        add("1 label files & per-class vs data_index", "FAIL", "; ".join(details))


# ---------------------------------------------------------------------------
# Check 2: patient split leakage + run_config training patients
# ---------------------------------------------------------------------------
def check2() -> None:
    split_path = ROOT / "data_index" / "splits" / "split_v1.csv"
    if not split_path.exists():
        add("2 patient split leakage", "FAIL", "split_v1.csv missing")
        return
    sp = pd.read_csv(split_path)
    dup = sp["patient_id"].value_counts()
    dup = dup[dup > 1]
    sets = {s: set(sp[sp["split"] == s]["patient_id"].astype(str)) for s in ("train", "val", "test")}
    overlaps = []
    if not dup.empty:
        overlaps.append(f"dupe patients: {list(dup.index)}")
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b and (sets[a] & sets[b]):
                overlaps.append(f"{a}&{b}: {sorted(sets[a] & sets[b])}")
    # run_config training patients
    runs_dir = ROOT / "outputs" / "runs"
    folds_path = ROOT / "data_index" / "splits" / "folds_5.csv"
    folds = pd.read_csv(folds_path) if folds_path.exists() else None
    for rc in sorted(runs_dir.glob("*/run_config.json")) if runs_dir.exists() else []:
        try:
            cfg = json.loads(rc.read_text(encoding="utf-8"))
        except Exception:
            continue
        fold = cfg.get("fold")
        if fold is None or (isinstance(fold, float) and pd.isna(fold)):
            train_p = sets["train"]
        elif folds is not None:
            k = 5
            f_test = int(fold)
            f_val = (f_test + 1) % k
            train_p = set(folds[~folds["fold"].isin([f_test, f_val])]["patient_id"].astype(str))
        else:
            continue
        valtest = sets["val"] | sets["test"]
        bad = train_p & valtest
        if bad:
            overlaps.append(f"{rc.parent.name} train leaks into val/test: {sorted(bad)}")
    if overlaps:
        add("2 patient split leakage", "FAIL", "; ".join(overlaps))
    else:
        add("2 patient split leakage", "PASS", "no patient in two splits; run_config train disjoint")


# ---------------------------------------------------------------------------
# Check 2b: every exp_ run folder has run_config.json + eval folder
# ---------------------------------------------------------------------------
def check2b() -> None:
    runs_dir = ROOT / "outputs" / "runs"
    eval_dir = ROOT / "outputs" / "eval"
    if not runs_dir.exists():
        add("2b exp_ run completeness", "WARN", "no outputs/runs")
        return
    bad = []
    exp_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("exp_"))
    if not exp_dirs:
        add("2b exp_ run completeness", "PASS", "no exp_ run folders present")
        return
    for d in exp_dirs:
        if not (d / "run_config.json").exists():
            bad.append(f"{d.name}: missing run_config.json")
        has_eval = eval_dir.exists() and any(
            e.is_dir() and e.name.startswith(d.name + "_") for e in eval_dir.iterdir()
        )
        if not has_eval:
            bad.append(f"{d.name}: no eval folder in outputs/eval")
    if bad:
        add("2b exp_ run completeness", "FAIL", "; ".join(bad))
    else:
        add("2b exp_ run completeness", "PASS", f"{len(exp_dirs)} exp_ run(s) complete")


# ---------------------------------------------------------------------------
# Check 3: leaderboard artifacts + values match metrics.json
# ---------------------------------------------------------------------------
LB_MAP = {
    "val_P": ("val", "detection", "micro", "precision"),
    "val_R": ("val", "detection", "micro", "recall"),
    "val_F1": ("val", "detection", "micro", "f1"),
    "val_mAP50": ("val", "map", "mAP_50"),
    "val_mAP5095": ("val", "map", "mAP_50_95"),
    "val_frame_sensitivity": ("val", "frame", "sensitivity"),
    "val_frame_specificity": ("val", "frame", "specificity"),
    "val_roc_auc": ("val", "frame", "roc_auc"),
    "test_P": ("test", "detection", "micro", "precision"),
    "test_R": ("test", "detection", "micro", "recall"),
    "test_F1": ("test", "detection", "micro", "f1"),
    "test_mAP50": ("test", "map", "mAP_50"),
    "test_mAP5095": ("test", "map", "mAP_50_95"),
    "test_frame_sensitivity": ("test", "frame", "sensitivity"),
    "test_frame_specificity": ("test", "frame", "specificity"),
    "test_roc_auc": ("test", "frame", "roc_auc"),
}


def check3() -> None:
    lb_path = ROOT / "outputs" / "experiments" / "leaderboard.csv"
    if not lb_path.exists():
        add("3 leaderboard artifacts & values", "FAIL", "leaderboard.csv missing")
        return
    lb = pd.read_csv(lb_path)
    bad = []
    notrun = []
    for _, row in lb.iterrows():
        run = str(row["run_name"])
        run_dir = ROOT / "outputs" / "runs" / run
        best_pt = (run_dir / "weights" / "best.pt").exists() or (run_dir / "best.pt").exists()
        preds_v = (run_dir / "preds_val.json").exists()
        preds_t = (run_dir / "preds_test.json").exists()
        mv = ROOT / "outputs" / "eval" / f"{run}_val" / "metrics.json"
        mt = ROOT / "outputs" / "eval" / f"{run}_test" / "metrics.json"
        files_ok = best_pt and preds_v and preds_t and mv.exists() and mt.exists()

        # A row only "declares NOT RUN" if its label says so or it carries no
        # numeric metric values. Otherwise it claims to be a completed run.
        metric_vals = [row[c] for c in LB_MAP if c in row and not pd.isna(row[c])]
        claims_run = ("not run" not in run.lower()) and bool(metric_vals)

        if not files_ok:
            if claims_run:
                missing = []
                if not best_pt:
                    missing.append("best.pt")
                if not preds_v:
                    missing.append("preds_val.json")
                if not preds_t:
                    missing.append("preds_test.json")
                if not mv.exists():
                    missing.append(f"{run}_val/metrics.json")
                if not mt.exists():
                    missing.append(f"{run}_test/metrics.json")
                bad.append(f"{run}: missing {', '.join(missing)}")
            else:
                notrun.append(run)
            continue
        m = {"val": json.loads(mv.read_text(encoding="utf-8")),
             "test": json.loads(mt.read_text(encoding="utf-8"))}
        for col, path in LB_MAP.items():
            if col not in row or pd.isna(row[col]):
                continue
            got = dig(m[path[0]], *path[1:])
            if got is None:
                continue
            if abs(float(row[col]) - float(got)) > 1e-4:
                bad.append(f"{run}.{col}: lb={row[col]} metrics={got}")
    if bad:
        add("3 leaderboard artifacts & values", "FAIL", "; ".join(bad))
    elif notrun:
        add("3 leaderboard artifacts & values", "WARN", f"NOT RUN: {notrun}")
    else:
        add("3 leaderboard artifacts & values", "PASS", "all rows backed by metrics.json")


# ---------------------------------------------------------------------------
# Check 4: metrics.json TP+FN == n_gt == label box count
# ---------------------------------------------------------------------------
def check4() -> None:
    ok = True
    details = []
    eval_dir = ROOT / "outputs" / "eval"
    if not eval_dir.exists():
        add("4 metrics.json n_gt guard", "WARN", "no outputs/eval")
        return
    for mpath in sorted(eval_dir.glob("*/metrics.json")):
        run_split = mpath.parent.name
        try:
            m = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception as exc:
            ok = False
            details.append(f"{run_split}: unreadable ({exc})")
            continue
        split = m.get("split")
        if split not in EXPECTED_LABELS:
            continue
        det = dig(m, "detection", "micro") or {}
        tp = det.get("TP")
        fn = det.get("FN")
        if tp is None or fn is None:
            continue
        n_gt = int(tp) + int(fn)
        _, lbl_box, _ = read_labels(split)
        if n_gt != lbl_box:
            ok = False
            details.append(f"{run_split}: TP+FN={n_gt} vs label boxes={lbl_box}")
    if ok:
        add("4 metrics.json n_gt guard", "PASS", "TP+FN == n_gt == label box count")
    else:
        add("4 metrics.json n_gt guard", "FAIL", "; ".join(details))


# ---------------------------------------------------------------------------
# Check 5: README/CHANGELOG number provenance + version claims
# ---------------------------------------------------------------------------
unmatched_numbers: list[str] = []


def check5() -> None:
    global unmatched_numbers
    unmatched_numbers = []
    known = known_values()
    num_re = re.compile(r"(?<![\d.])(\d+\.\d{3,})(?![\d.])")
    for name in ("README.md", "CHANGELOG.md"):
        p = ROOT / name
        if not p.exists():
            continue
        for lineno, line in enumerate(
            p.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for tok in num_re.findall(line):
                try:
                    v = round(float(tok), 4)
                except Exception:
                    continue
                if v not in known:
                    unmatched_numbers.append(f"{name}:{lineno}: {tok}")
    if unmatched_numbers:
        add("5 doc numbers in metrics/data_index", "WARN",
            f"{len(unmatched_numbers)} unmatched (full list in reports/audit.md)")
    else:
        add("5 doc numbers in metrics/data_index", "PASS", "all 3+ decimal numbers found")

    tags = set(git("tag").splitlines())
    ver_re = re.compile(r"v0\.\d+\.\d+")
    claims = set()
    for name in ("README.md", "CHANGELOG.md"):
        p = ROOT / name
        if p.exists():
            claims.update(ver_re.findall(p.read_text(encoding="utf-8")))
    missing = sorted(c for c in claims if c not in tags)
    if missing:
        add("5 version claims vs git tags", "WARN", f"claims without tag: {missing}")
    else:
        add("5 version claims vs git tags", "PASS", f"{len(claims)} claims all tagged")


# ---------------------------------------------------------------------------
# Check 6: suspicious results
# ---------------------------------------------------------------------------
def check6() -> None:
    eval_dir = ROOT / "outputs" / "eval"
    suspicious = []
    gaps = []
    seen_runs: dict[str, dict[str, float]] = {}
    if eval_dir.exists():
        for mpath in sorted(eval_dir.glob("*/metrics.json")):
            d = mpath.parent.name
            if any(mk in d.lower() for mk in SANITY_MARKERS):
                continue
            try:
                m = json.loads(mpath.read_text(encoding="utf-8"))
            except Exception:
                continue
            split = m.get("split")
            metrics = {
                "F1": dig(m, "detection", "micro", "f1"),
                "mAP50": dig(m, "map", "mAP_50"),
                "AUC": dig(m, "frame", "roc_auc"),
                "accuracy": dig(m, "frame", "accuracy"),
            }
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and v > 0.95:
                    suspicious.append(f"{d}.{k}={v}")
            run = d.rsplit("_", 1)[0]
            seen_runs.setdefault(run, {})
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    seen_runs[run].setdefault(k, {})[split] = v
    for run, md in seen_runs.items():
        for k, sd in md.items():
            if "val" in sd and "test" in sd and abs(sd["val"] - sd["test"]) > 0.15:
                gaps.append(f"{run}.{k}: val={sd['val']} test={sd['test']}")
    if suspicious:
        add("6 suspicious results", "WARN", f">0.95: {suspicious}")
    elif gaps:
        add("6 suspicious results", "WARN", f"val/test gap>0.15: {gaps}")
    else:
        add("6 suspicious results", "PASS", "no metric >0.95; no val/test gap >0.15")


# ---------------------------------------------------------------------------
# Check 7: environment
# ---------------------------------------------------------------------------
def check7() -> None:
    pyver = sys.version.split()[0]
    rc_t, out_t, _ = run_py("import torch;print(torch.__version__)")
    torch_ver = out_t.splitlines()[0] if (rc_t == 0 and out_t) else "unavailable"
    rc_c, out_c, err_c = run_py("import torch;print(torch.cuda.is_available())")
    if rc_c == 0:
        cuda = out_c.splitlines()[-1] if out_c else "unknown"
    else:
        cuda = f"crashed (rc={rc_c})"
    status_dirty = git("status", "--porcelain")
    clean = "clean" if not status_dirty else f"dirty ({len(status_dirty.splitlines())} files)"
    tags = git("tag").splitlines()
    add("7 environment", "PASS",
        f"python={pyver}; torch={torch_ver}; cuda.is_available={cuda}; git={clean}; tags={tags}")


# ---------------------------------------------------------------------------
# Check 8: frame-classifier runs (placeholder until they exist)
# ---------------------------------------------------------------------------
def check8() -> None:
    eval_dir = ROOT / "outputs" / "eval"
    frame_dirs = []
    if eval_dir.exists():
        frame_dirs = [d.name for d in eval_dir.iterdir()
                      if d.is_dir() and "frame" in d.name.lower()]
    if frame_dirs:
        add("8 frame-classifier runs", "WARN", f"present: {frame_dirs} (overlap check TODO)")
    else:
        add("8 frame-classifier runs", "PASS", "NOT PRESENT (no frame-classifier runs)")


def main() -> int:
    rows.clear()
    for fn in (check1, check2, check2b, check3, check4, check5, check6, check7, check8):
        try:
            fn()
        except Exception as exc:
            add(fn.__name__, "FAIL", f"crashed: {exc}")

    # Print table
    width = 46
    print("=" * (width + 40))
    print(f"{'CHECK':<{width}} {'STATUS':<6} DETAIL")
    print("=" * (width + 40))
    for check, status, detail in rows:
        print(f"{check:<{width}} {status:<6} {detail}")
    print("=" * (width + 40))
    fails = [r for r in rows if r[1] == "FAIL"]
    warns = [r for r in rows if r[1] == "WARN"]
    print(f"SUMMARY: {len(rows)} checks | FAIL={len(fails)} WARN={len(warns)}")

    # Save markdown
    rep = ROOT / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    md = ["# Project Audit", "", "| Check | Status | Detail |", "|---|---|---|"]
    for check, status, detail in rows:
        d = detail.replace("|", "\\|")
        md.append(f"| {check} | {status} | {d} |")
    md.append("")
    md.append(f"SUMMARY: {len(rows)} checks | FAIL={len(fails)} WARN={len(warns)}")
    if unmatched_numbers:
        md.append("")
        md.append("## Unmatched README/CHANGELOG decimals (3+ places)")
        md.append("")
        md.extend(unmatched_numbers)
    (rep / "audit.md").write_text("\n".join(md), encoding="utf-8")

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
