#!/usr/bin/env python3
"""Self-test for scripts/audit_project.py.

Builds a minimal temporary project, verifies check 3 (leaderboard artifacts)
passes when best.pt is present, then deletes best.pt and asserts the audit
exits non-zero because the run claims numbers but is missing its weights.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent / "scripts" / "audit_project.py"
RUN = "exp_selftest_run"


def _parse_check(output: str, needle: str) -> tuple[str, str]:
    """Return (status, detail) for the table row whose CHECK contains needle."""
    for line in output.splitlines():
        if needle in line:
            # Table format: "<CHECK padded to width> <STATUS> <detail>".
            parts = line.split()
            for i, tok in enumerate(parts):
                if tok in ("PASS", "WARN", "FAIL"):
                    return tok, " ".join(parts[i + 1:])
            return "", ""
    return "", ""


def _build_project(root: Path) -> None:
    run_dir = root / "outputs" / "runs" / RUN
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"dummy-weights")
    (run_dir / "preds_val.json").write_text("[]", encoding="utf-8")
    (run_dir / "preds_test.json").write_text("[]", encoding="utf-8")

    for split in ("val", "test"):
        ed = root / "outputs" / "eval" / f"{RUN}_{split}"
        ed.mkdir(parents=True)
        metrics = {
            "split": split,
            "detection": {"micro": {"precision": 0.5, "recall": 0.5, "f1": 0.5,
                                    "TP": 1, "FN": 1}},
            "map": {"mAP_50": 0.5, "mAP_50_95": 0.5},
            "frame": {"sensitivity": 0.5, "specificity": 0.5, "roc_auc": 0.5},
        }
        (ed / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    exp = root / "outputs" / "experiments"
    exp.mkdir(parents=True, exist_ok=True)
    header = "run_name,val_F1,test_F1\n"
    (exp / "leaderboard.csv").write_text(
        header + f"{RUN},0.5,0.5\n", encoding="utf-8"
    )


def _run_audit(root: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["AUDIT_ROOT"] = str(root)
    return subprocess.run(
        [sys.executable, str(AUDIT)],
        capture_output=True, text=True, env=env, timeout=180,
    )


def test_audit_flags_missing_best_pt(tmp_path: Path) -> None:
    _build_project(tmp_path)

    # Baseline: best.pt present -> check 3 should PASS.
    before = _run_audit(tmp_path)
    status_before, detail_before = _parse_check(before.stdout, "3 leaderboard")
    assert status_before == "PASS", f"expected check3 PASS, got {status_before}: {detail_before}"

    # Delete best.pt -> the run claims numbers but has no weights -> check 3 FAIL.
    (tmp_path / "outputs" / "runs" / RUN / "weights" / "best.pt").unlink()
    after = _run_audit(tmp_path)
    status_after, detail_after = _parse_check(after.stdout, "3 leaderboard")

    assert after.returncode != 0, "audit must exit non-zero when best.pt is missing"
    assert status_after == "FAIL", f"expected check3 FAIL, got {status_after}: {detail_after}"
    assert "best.pt" in detail_after, f"detail should mention best.pt: {detail_after}"


if __name__ == "__main__":
    # Allow running directly for a quick smoke check.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_audit_flags_missing_best_pt(Path(td))
    print("test_audit_flags_missing_best_pt: PASS")
