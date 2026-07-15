"""Regression tests for the evidence-first advisor showcase."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from fci_engine import compare_pag_shapes, compare_pag_shapes_semantic
from fci_engine.metrics import BenchmarkResult


@pytest.fixture(scope="module")
def showcase():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "10_fci_plus_advisor_showcase.py"
    )
    spec = importlib.util.spec_from_file_location("advisor_showcase", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def context(showcase):
    return showcase.build_showcase_context(external_enabled=False)


def test_showcase_context_records_oracle_success_and_sample_limitation(context) -> None:
    sample_5k, sample_50k = context.sample_runs

    assert context.exact_pass
    assert context.exact_separator == ("U", "V", "Z")
    assert context.exact_separator_source == "fci_plus_dsep"
    assert ("X", "Y") not in context.learned_shape
    assert sample_5k.xy_present
    assert sample_5k.exact_f1 < 1.0
    assert sample_5k.standard_fci_exact_f1 == 1.0
    assert not sample_5k.standard_fci_xy_present
    assert sample_5k.standard_fci_ci_tests > sample_5k.ci_tests
    assert not sample_50k.xy_present
    assert sample_50k.exact_f1 == 1.0
    assert sample_50k.separator == ("U", "V", "Z")


def test_advisor_showcase_renders_paper_evidence_and_reproducibility(
    showcase,
    context,
) -> None:
    report = showcase.render_showcase([], context)

    assert "FCI+ Paper-aligned Validation" in report
    assert "Algorithm 2, line by line" in report
    assert "Figure 4(b): true target vs learned PAG" in report
    assert "deterministic m-separation—not sampled data" in report
    assert "N = 5,000" in report
    assert "Illustrative finite-sample miss" in report
    assert "N = 50,000" in report
    assert "Exact recovery" in report
    assert "Standard FCI, same data" in report
    assert "O(N^(2(k+2)))" in report
    assert "alpha=0.001" in report
    assert "Real Run" not in report
    assert "<script src=" not in report
    assert "<link rel=" not in report


def test_same_cohort_leaderboard_excludes_partial_algorithms(showcase) -> None:
    exact = compare_pag_shapes({}, {})
    semantic = compare_pag_shapes_semantic({}, {})

    def completed(case: str, algorithm: str) -> BenchmarkResult:
        return BenchmarkResult(
            case_name=case,
            algorithm=algorithm,
            comparison=exact,
            semantic_comparison=semantic,
            edges={},
            elapsed_time=0.01,
            ci_test_count=4,
        )

    results = [
        completed("case_a", "full"),
        completed("case_b", "full"),
        completed("case_a", "partial"),
        BenchmarkResult(
            case_name="case_b",
            algorithm="partial",
            comparison=None,
            semantic_comparison=None,
            edges={},
            elapsed_time=None,
            skipped_reason="not installed",
        ),
    ]

    included, excluded, case_ids = showcase.same_cohort_aggregates(results)
    report = showcase.render_leaderboard(results)

    assert case_ids == ("case_a", "case_b")
    assert [item.algorithm for item in included] == ["full"]
    assert excluded == [("partial", "coverage 1/2; missing case_b; not installed")]
    assert "2/2" in report
    assert "coverage 1/2" in report
