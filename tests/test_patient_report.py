"""Unit tests for patient_report.py generator.

Verifies:
- 3 synthetic images and 1 blank image run without crashing.
- report.html and report.json are generated on disk.
- Neither file contains 'diseased' or 'healthy' clinical verdict wording.
- Both files contain the detector reliability box.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
PY = r"C:\Users\HP\AppData\Local\Programs\Python\Python314\python.exe"


def test_patient_report_synthetic_and_blank(tmp_path: Path) -> None:
    """Test report generation on 3 synthetic images and 1 blank image."""
    in_dir = tmp_path / "inputs"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "outputs" / "reports" / "testpt"

    # Create 3 synthetic angiography-like images
    for idx in range(1, 4):
        img = np.full((512, 512, 3), 128, dtype=np.uint8)
        # Add a bright simulated vessel track
        cv2.line(img, (50 * idx, 100), (200 + 50 * idx, 400), (220, 220, 220), 4)
        cv2.imwrite(str(in_dir / f"testpt_0000{idx}.png"), img)

    # Create 1 all-black blank frame
    blank_img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.imwrite(str(in_dir / "testpt_00004.png"), blank_img)

    # Run patient report CLI
    cmd = [
        PY,
        str(ROOT / "scripts" / "patient_report.py"),
        "--input", str(in_dir),
        "--patient-id", "testpt",
        "--out", str(out_dir),
    ]

    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, f"Script failed with stderr:\n{res.stderr}\nstdout:\n{res.stdout}"

    # Verify report files exist
    html_file = out_dir / "report.html"
    json_file = out_dir / "report.json"
    assert html_file.exists(), "report.html was not generated"
    assert json_file.exists(), "report.json was not generated"

    html_content = html_file.read_text(encoding="utf-8")
    json_content = json_file.read_text(encoding="utf-8")
    json_data = json.loads(json_content)

    # Verify no 'diseased' or 'healthy' wording
    for text, name in [(html_content, "report.html"), (json_content, "report.json")]:
        match_diseased = re.search(r"\bdiseased\b", text, re.IGNORECASE)
        match_healthy = re.search(r"\bhealthy\b", text, re.IGNORECASE)
        assert match_diseased is None, f"Forbidden word 'diseased' found in {name}: {match_diseased}"
        assert match_healthy is None, f"Forbidden word 'healthy' found in {name}: {match_healthy}"

    # Verify reliability box exists
    assert "Detector Reliability" in html_content, "Reliability box missing from report.html"
    assert "reliability" in json_data, "Reliability section missing from report.json"
    assert "interpretive_warning" in json_data["reliability"]
    assert "small fraction of true lesions" in json_data["reliability"]["interpretive_warning"]
