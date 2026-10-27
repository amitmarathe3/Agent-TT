#!/usr/bin/env python3
"""Batch runner to generate patient reports for all 6 test patients."""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = r"C:\Users\HP\AppData\Local\Programs\Python\Python314\python.exe"

TEST_PATIENTS = ["p8", "p11", "p16", "p22", "p34", "p36"]


def main():
    img_dir = ROOT / "outputs" / "yolo_dataset" / "images" / "test"
    assert img_dir.exists(), f"Image dir not found: {img_dir}"

    for pid in TEST_PATIENTS:
        print(f"Generating report for {pid}...")
        out_dir = ROOT / "outputs" / "reports" / pid
        cmd = [
            PY,
            str(ROOT / "scripts" / "patient_report.py"),
            "--input", str(img_dir),
            "--patient-id", pid,
            "--out", str(out_dir),
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Error on {pid}: {res.stderr}")
            sys.exit(res.returncode)
        print(f"[DONE] {pid}")

    # Copy p11 to sample_report.html
    p11_html = ROOT / "outputs" / "reports" / "p11" / "report.html"
    sample_html = ROOT / "reports" / "sample_report.html"
    sample_html.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p11_html, sample_html)
    print(f"Copied {p11_html} to {sample_html}")
    print("All 6 test patient reports generated successfully.")


if __name__ == "__main__":
    main()
