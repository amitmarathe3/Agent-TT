"""Unit tests for patient_report.py generator.

Verifies:
- 3 synthetic images and 1 blank image run without crashing.
- report.html and report.json are generated on disk.
- Neither file contains 'diseased', 'healthy', 'positive', or 'negative'.
- Both files contain the detector reliability box and the false-alarm sentence.
- Scans every generated report.html and report.json (including the 6 test-patient reports and sample_report.html).
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

FORBIDDEN_WORDS = ["diseased", "healthy", "positive", "negative"]
FA_SENTENCE_SUB = (
    "Candidates can appear in patients without disease, so their presence or number is not evidence of disease."
)


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

    # Verify no 'diseased', 'healthy', 'positive', or 'negative' wording
    for text, name in [(html_content, "report.html"), (json_content, "report.json")]:
        for word in FORBIDDEN_WORDS:
            match = re.search(rf"\b{word}\b", text, re.IGNORECASE)
            assert match is None, f"Forbidden word '{word}' found in {name}: {match}"

    # Verify reliability box exists
    assert "Detector Reliability" in html_content, "Reliability box missing from report.html"
    assert "reliability" in json_data, "Reliability section missing from report.json"
    assert "interpretive_warning" in json_data["reliability"]
    assert "small fraction of true lesions" in json_data["reliability"]["interpretive_warning"]

    # Verify false-alarm sentence
    assert "On held-out patients, about" in html_content, "False-alarm sentence missing from report.html"
    assert FA_SENTENCE_SUB in html_content, "False-alarm sentence missing from report.html"
    assert "On held-out patients, about" in json_content, "False-alarm sentence missing from report.json"
    assert FA_SENTENCE_SUB in json_content, "False-alarm sentence missing from report.json"


def test_all_generated_reports_scanned() -> None:
    """Scan every generated report.html and report.json across the repository."""
    reports_dir = ROOT / "outputs" / "reports"
    report_files: list[Path] = []
    if reports_dir.exists():
        report_files.extend(sorted(reports_dir.glob("*/report.html")))
        report_files.extend(sorted(reports_dir.glob("*/report.json")))
    sample_html = ROOT / "reports" / "sample_report.html"
    if sample_html.exists():
        report_files.append(sample_html)

    # Must find at least the 6 test patients (html + json = 12) + sample_report.html = 13
    assert len(report_files) >= 13, f"Expected at least 13 report files, found {len(report_files)}: {report_files}"

    for rpath in report_files:
        text = rpath.read_text(encoding="utf-8")

        # Check forbidden words (case-insensitive)
        for word in FORBIDDEN_WORDS:
            match = re.search(rf"\b{word}\b", text, re.IGNORECASE)
            assert match is None, f"Forbidden word '{word}' found in {rpath.relative_to(ROOT)}: {match}"

        # Check reliability box
        if rpath.suffix == ".html":
            assert "box-reliability" in text or "Detector Reliability" in text, (
                f"Reliability box missing in {rpath.relative_to(ROOT)}"
            )
        elif rpath.suffix == ".json":
            data = json.loads(text)
            assert "reliability" in data, f"'reliability' block missing in {rpath.relative_to(ROOT)}"

        # Check false-alarm sentence
        assert "On held-out patients, about" in text, (
            f"False-alarm prefix missing in {rpath.relative_to(ROOT)}"
        )
        assert FA_SENTENCE_SUB in text, (
            f"False-alarm sentence missing in {rpath.relative_to(ROOT)}"
        )
