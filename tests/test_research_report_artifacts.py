"""Contracts for the auditable research-report evidence artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "reports" / "research"


def test_research_dossier_has_required_primary_source_sections() -> None:
    text = (RESEARCH / "fci_fci_plus_source_dossier.md").read_text(
        encoding="utf-8"
    )

    for heading in (
        "## Spirtes, Glymour, and Scheines (2000)",
        "## Claassen, Mooij, and Heskes (2013)",
        "## Source-to-implementation mapping",
        "## Claims that require finite-sample qualification",
    ):
        assert heading in text


def test_software_landscape_separates_executed_and_documented_evidence() -> None:
    payload = json.loads(
        (RESEARCH / "software_landscape.json").read_text(encoding="utf-8")
    )

    assert payload["as_of"] == "2026-07-26"
    assert {row["tool"] for row in payload["tools"]} >= {
        "fci_engine",
        "pcalg",
        "causal-learn",
        "Tetrad",
    }
    assert all(
        row["evidence_kind"] in {"executed", "documentation"}
        for row in payload["tools"]
    )


def test_claim_matrix_requires_a_locator_for_every_report_claim() -> None:
    with (RESEARCH / "claim_evidence_matrix.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert {
        "claim_id",
        "claim",
        "source",
        "locator",
        "evidence_kind",
        "repository_symbol",
        "validation_artifact",
    } == set(rows[0])
    assert all(row["claim_id"] and row["source"] and row["locator"] for row in rows)
