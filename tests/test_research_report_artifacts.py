"""Contracts for the auditable research-report evidence artifacts."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import reports.generate_report_figures as report_figures
import reports.generate_software_comparison as software_comparison
from fci_engine.simulation import realistic_oracle_cases
from reports.generate_software_comparison import generate_software_comparison


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "reports" / "research"
ORACLE_SUMMARY = ROOT / "reports" / "data" / "oracle_validation_summary.csv"


def test_latex_report_contains_rewritten_research_structure() -> None:
    tex = (ROOT / "reports" / "fci_plus_star_report.tex").read_text(encoding="utf-8")
    for title in (
        r"\chapter{Standard FCI in Spirtes et al. (2000)}",
        r"\chapter{FCI+ in Claassen et al. (2013)}",
        r"\chapter{Independent Python Implementation}",
        r"\chapter{Comparison with Established Software}",
        r"\chapter{Causal Interpretation and Evidence Hierarchy}",
    ):
        assert title in tex

    required_claims = (
        "documentation-only comparison",
        "does not estimate a treatment effect",
        "This project independently implements paper-aligned FCI and FCI+",
        "not a universal accuracy or speed superiority claim",
        "N^{2(k+2)}",
        "N^{2(k+1)}",
        "Theorem~6.4",
        "Algorithm~2",
        "Figure~4(b)",
    )
    normalized_tex = " ".join(tex.split())
    for claim in required_claims:
        assert claim in normalized_tex

    traceability_rows = (
        "Complete start and PC-style adjacency search",
        "Standard FCI stage ordering",
        "Possible-D-SEP path criterion",
        "Ordered Possible-D-SEP separator search",
        "Reset and second collider phase",
        "Original orientation closure",
        "Complete modern orientation",
        "Augmented skeleton",
        "Candidate D-sep pattern",
        "Recursive hierarchy",
        "Paired endpoint bases bounded by",
        "Separator minimization and candidate revisit",
        "Figure~4(b) false-link removal",
        "Exact MAG m-separation oracle",
        "Stable deletion and max-",
        "Conservative and robust orientation",
        "Structured trace and exports",
    )
    assert len(traceability_rows) == 17
    for row in traceability_rows:
        assert row in normalized_tex


def test_report_uses_bounded_star_claims_and_precise_graph_language() -> None:
    tex = (ROOT / "reports" / "fci_plus_star_report.tex").read_text(encoding="utf-8")
    normalized_tex = " ".join(tex.split())

    forbidden = (
        "supports a beneficial causal effect of small classes",
        r"\subsection{Small classes improve early observed achievement}",
        r"\subsection{Early achievement strongly predicts later structure}",
        r"\subsection{Attrition is selective but not clearly caused by class arm}",
        "the most stable structural correlate",
        "each panel's complete PAG exactly",
        "returns the same complete PAG",
        "retains the adjacency with two unresolved circles",
    )
    for claim in forbidden:
        assert claim not in normalized_tex

    required = (
        "design-supported observed-arm contrast",
        "consistent with a beneficial early effect",
        "one of the most consistently retained structural correlates",
        "exact graph equality for every tested cyclic order",
        r"K\_Class\leftarrow\!\circ Grade3\_Achievement",
        r"K\_Class\circ\!\!-\!\!\circ Grade3\_Achievement",
    )
    for claim in required:
        assert claim in normalized_tex


def test_oracle_validation_summary_is_frozen_and_reproducible(
    tmp_path: Path,
) -> None:
    from reports.generate_oracle_validation_summary import (
        generate_oracle_validation_summary,
    )

    regenerated = tmp_path / ORACLE_SUMMARY.name
    generate_oracle_validation_summary(regenerated)
    assert regenerated.read_bytes() == ORACLE_SUMMARY.read_bytes()

    with ORACLE_SUMMARY.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert len({row["target_sha256"] for row in rows}) == 1
    assert len(rows[0]["target_sha256"]) == 64
    assert {row["status"] for row in rows} == {"completed"}
    by_key = {(row["regime"], row["algorithm"], row["n_samples"]): row for row in rows}
    assert (
        by_key[("finite_sample", "fci_plus", "5000")]["exact_target_recovered"]
        == "false"
    )

    assert by_key[("exact_oracle", "fci", "")]["ci_test_count"] == "102"
    assert by_key[("exact_oracle", "fci_plus", "")]["ci_test_count"] == "63"
    assert by_key[("finite_sample", "fci_plus", "5000")]["exact_edge_f1"] == "0.923077"
    assert by_key[("finite_sample", "fci_plus", "50000")]["exact_edge_f1"] == "1.000000"


def test_report_figures_keep_legend_and_source_boundary_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.loads(report_figures.SUMMARY_PATH.read_text(encoding="utf-8"))
    captured: dict[str, object] = {}

    def capture(figure: object, filename: str) -> None:
        captured[filename] = figure

    monkeypatch.setattr(report_figures, "_save", capture)
    report_figures.plot_bootstrap_stability(payload)
    bootstrap = captured["star_bootstrap_stability.pdf"]
    axis = bootstrap.axes[0]  # type: ignore[attr-defined]
    legend = axis.get_legend()
    assert legend is not None
    anchor = legend.get_bbox_to_anchor().transformed(axis.transAxes.inverted())
    assert anchor.x0 >= 1.0

    report_figures.plot_figure4_validation()
    figure4 = captured["figure4_validation.pdf"]
    figure4_title = figure4.axes[0].get_title()  # type: ignore[attr-defined]
    assert "Repository-derived oracle PAG" in figure4_title
    assert "published Figure 4(b) MAG" in figure4_title

    plt.close(bootstrap)  # type: ignore[arg-type]
    plt.close(figure4)  # type: ignore[arg-type]


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
    claim_rows = {row["claim_id"]: row for row in rows}
    assert {
        "ORACLE-FIG4-QUERIES",
        "ORACLE-FINITE-SAMPLE",
    } <= set(claim_rows)
    for claim_id in ("ORACLE-FIG4-QUERIES", "ORACLE-FINITE-SAMPLE"):
        assert (
            claim_rows[claim_id]["validation_artifact"]
            == "reports/data/oracle_validation_summary.csv"
        )

    for row in rows:
        symbol_locator = row["repository_symbol"]
        if symbol_locator.startswith("external:"):
            continue
        for part in symbol_locator.split(";"):
            path_text = part.strip().split("::", maxsplit=1)[0]
            if "/" in path_text:
                assert (
                    ROOT / path_text
                ).exists(), (
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
