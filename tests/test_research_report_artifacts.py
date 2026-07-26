"""Contracts for the auditable research-report evidence artifacts."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import reports.generate_software_comparison as software_comparison
from fci_engine.simulation import realistic_oracle_cases
from reports.generate_software_comparison import generate_software_comparison


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "reports" / "research"


def test_required_research_figures_exist_and_are_nonempty() -> None:
    for name in (
        "fci_fci_plus_workflow.pdf",
        "source_implementation_map.pdf",
        "software_benchmark_comparison.pdf",
        "software_feature_comparison.pdf",
    ):
        path = ROOT / "reports" / "figures" / name
        assert path.stat().st_size > 10_000


def test_research_dossier_has_required_primary_source_sections() -> None:
    text = (RESEARCH / "fci_fci_plus_source_dossier.md").read_text(encoding="utf-8")

    for heading in (
        "## Spirtes, Glymour, and Scheines (2000)",
        "## Claassen, Mooij, and Heskes (2013)",
        "## Source-to-implementation mapping",
        "## Claims that require finite-sample qualification",
    ):
        assert heading in text


def test_software_landscape_separates_executed_and_documented_evidence() -> None:
    landscape_path = RESEARCH / "software_landscape.json"
    payload = json.loads(landscape_path.read_text(encoding="utf-8"))

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
    required_keys = {
        "tool",
        "version",
        "as_of",
        "evidence_kind",
        "algorithms",
        "ci_tests",
        "audit_exports",
        "source_urls",
        "comparison_limits",
    }
    for row in payload["tools"]:
        assert required_keys <= set(row)
        assert row["as_of"] == payload["as_of"]
        assert all(row[key] not in ("", None) for key in required_keys)

    local_urls = [
        source_url
        for row in payload["tools"]
        for source_url in row["source_urls"]
        if source_url.startswith(".")
    ]
    assert local_urls == [
        "../../pyproject.toml",
        "../../README.md",
        "../../src/fci_engine/result.py",
        "../../src/fci_engine/discovery/fci.py",
        "../../src/fci_engine/discovery/fci_plus.py",
    ]
    assert all((landscape_path.parent / url).resolve().exists() for url in local_urls)


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
    assert len({row["claim_id"] for row in rows}) == len(rows)
    assert all(all(value.strip() for value in row.values()) for row in rows)
    assert {row["evidence_kind"] for row in rows} <= {
        "primary_source",
        "repository",
        "executed",
        "documentation",
    }

    repository_symbols = "\n".join(row["repository_symbol"] for row in rows)
    assert "FCIResult.assumption_warnings" not in repository_symbols
    assert "FCIResult.assumption_notes" in repository_symbols

    for row in rows:
        symbol_locator = row["repository_symbol"]
        if symbol_locator.startswith("external:"):
            continue
        for part in symbol_locator.split(";"):
            path_text = part.strip().split("::", maxsplit=1)[0]
            if "/" in path_text:
                assert (ROOT / path_text).exists(), (
                    f"{row['claim_id']} has an invalid repository path: {path_text}"
                )


def test_software_comparison_generator_writes_complete_tables(
    tmp_path: Path,
) -> None:
    outputs = generate_software_comparison(
        output_dir=tmp_path,
        repeats=1,
        samples=600,
        include_pcalg=False,
    )

    cases = pd.read_csv(outputs["cases"])
    summary = pd.read_csv(outputs["summary"])
    features = pd.read_csv(outputs["features"])
    assert all(b"\r\n" not in path.read_bytes() for path in outputs.values())

    expected_algorithms = {
        "fci_engine.fci",
        "fci_engine.fci_plus",
        "fci_engine.fci_plus.robust",
        "causal-learn.fci.fisherz",
    }
    assert expected_algorithms <= set(summary["algorithm"])
    assert {
        "mean_skeleton_f1",
        "mean_exact_edge_f1",
        "mean_semantic_edge_f1",
        "mean_endpoint_accuracy",
        "mean_elapsed_seconds",
    } <= set(summary.columns)

    profiles = cases.groupby("algorithm")["profile"].first().to_dict()
    assert profiles["fci_engine.fci"] == "spirtes_2000_paper"
    assert profiles["fci_engine.fci_plus"] == "claassen_2013_paper"
    assert profiles["fci_engine.fci_plus.robust"] == "practical_robust"
    effective_alpha = cases.groupby("algorithm")["effective_alpha"].first().to_dict()
    assert effective_alpha["fci_engine.fci_plus"] == 0.001
    assert effective_alpha["fci_engine.fci_plus.robust"] == 0.05
    assert (
        cases.loc[cases["algorithm"] == "fci_engine.fci_plus", "configuration"]
        .str.contains('"skeleton_stable": false', regex=False)
        .all()
    )
    assert (
        cases.loc[cases["algorithm"] == "fci_engine.fci_plus.robust", "configuration"]
        .str.contains('"alpha": "auto"', regex=False)
        .all()
    )
    assert set(cases["timing_scope"]) == {"in_process_algorithm_call"}

    assert set(features["tool"]) >= {
        "fci_engine",
        "pcalg",
        "causal-learn",
        "Tetrad",
    }
    assert not (cases["algorithm"] == "Tetrad").any()
    assert not (summary["algorithm"] == "Tetrad").any()


def test_software_comparison_preserves_external_failures_as_skips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = realistic_oracle_cases(n_repeats=1, n_samples=120)[0]
    monkeypatch.setattr(
        software_comparison,
        "realistic_oracle_cases",
        lambda *, n_repeats, n_samples: [case],
    )
    monkeypatch.setattr(
        software_comparison,
        "_software_versions",
        lambda *, include_pcalg: {
            "fci_engine": "0.1.0",
            "causal-learn": "test-version",
            "pcalg": "test-version",
        },
    )

    def fail_causal_learn(*args: object, **kwargs: object) -> None:
        raise RuntimeError("causal-learn runtime failure")

    def time_out_pcalg(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="Rscript", timeout=60)

    monkeypatch.setattr(
        software_comparison,
        "run_causal_learn_fci",
        fail_causal_learn,
    )
    monkeypatch.setattr(
        software_comparison,
        "run_pcalg_fci_plus",
        time_out_pcalg,
    )

    outputs = generate_software_comparison(
        output_dir=tmp_path,
        repeats=1,
        samples=120,
        include_pcalg=True,
    )

    assert all(path.exists() for path in outputs.values())
    cases = pd.read_csv(outputs["cases"])
    summary = pd.read_csv(outputs["summary"])
    external = cases[cases["tool"].isin({"causal-learn", "pcalg"})]
    assert set(external["status"]) == {"skipped"}
    assert (
        external["skipped_reason"]
        .str.contains(
            "external runner failed",
            regex=False,
        )
        .all()
    )
    assert (
        external.loc[external["tool"] == "causal-learn", "skipped_reason"]
        .str.contains("RuntimeError", regex=False)
        .all()
    )
    assert (
        external.loc[external["tool"] == "pcalg", "skipped_reason"]
        .str.contains("TimeoutExpired", regex=False)
        .all()
    )
    assert set(
        summary.loc[summary["tool"].isin({"causal-learn", "pcalg"}), "n_skipped"]
    ) == {1}
