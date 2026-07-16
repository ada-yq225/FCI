"""Regression tests for the separate Tennessee STAR application layer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from case_studies.tennessee_star.download_data import (
    GUIDE_PATH,
    STUDENT_PATH,
    sha256,
)
from case_studies.tennessee_star.report import render_report
from case_studies.tennessee_star.study import load_star, prepare_study

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "case_studies" / "tennessee_star" / "output"


def test_committed_star_files_match_documented_hashes() -> None:
    assert sha256(STUDENT_PATH) == (
        "2ffa578822eb30fcd6626fa7c5cc734a721fd05dd522403f0d643f89e891bac8"
    )
    assert sha256(GUIDE_PATH) == (
        "e51ff1d28d5af28c128196b3a133957f9f2cd872b1abe348f156022193550130"
    )


def test_star_preparation_builds_expected_independent_panels() -> None:
    study = prepare_study(load_star())

    assert study.raw_rows == 11_601
    assert study.kindergarten_rows == 6_325
    assert study.kindergarten_schools == 79
    assert study.panels["attrition"].data.shape == (5_744, 9)
    assert study.panels["longitudinal"].data.shape == (2_787, 9)
    assert study.panels["focused_treatment"].data.shape == (2_976, 8)

    for panel in study.panels.values():
        assert not panel.data.isna().any().any()
        assert all(np.issubdtype(dtype, np.integer) for dtype in panel.data.dtypes)
        assert len(panel.school_ids) == len(panel.data)


def test_committed_star_report_is_reproducible_from_summary_payload() -> None:
    payload = json.loads(
        (OUTPUT / "star_case_study_summary.json").read_text(encoding="utf-8")
    )
    html = render_report(payload)

    assert "separate from the algorithm package" in html
    assert "Randomized-arm reference" in html
    assert "Learned Partial Ancestral Graphs" in html
    assert "Researcher self-assessment" in html
    assert "K_Class &lt;-&gt; Grade3_Achievement" in html
    assert "does <strong>not</strong>" in html
    assert "doi.org/10.7910/DVN/SIWH9F" in html


def test_star_summary_contains_both_algorithms_for_every_panel() -> None:
    payload = json.loads(
        (OUTPUT / "star_case_study_summary.json").read_text(encoding="utf-8")
    )
    combinations = {(run["panel"], run["algorithm"]) for run in payload["runs"]}

    assert combinations == {
        ("attrition", "fci"),
        ("attrition", "fci_plus"),
        ("longitudinal", "fci"),
        ("longitudinal", "fci_plus"),
        ("focused_treatment", "fci"),
        ("focused_treatment", "fci_plus"),
    }
