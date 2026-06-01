"""Regression tests for the generated visual benchmark report."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_report_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "08_visual_benchmark_report.py"
    )
    spec = importlib.util.spec_from_file_location("visual_benchmark_report", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pag_svg_embeds_click_explanation_metadata() -> None:
    report = _load_report_module()
    svg = report.render_pag_svg(
        {("X", "Y"): ("CIRCLE", "CIRCLE")},
        ["X", "Y"],
        "Learned",
        {("X", "Y"): ("CIRCLE", "ARROW")},
        "learned",
    )

    assert "class='graph-edge'" in svg
    assert "data-edge='X o-o Y'" in svg
    assert "data-kind='under_oriented'" in svg
    assert "data-expected='X o-&gt; Y'" in svg
    assert "data-actual='X o-o Y'" in svg
    assert "data-status='exact/under_oriented'" in svg
    assert "data-endpoint-meaning=" in svg
    assert "data-reasoning=" in svg
    assert "The circle at X is deliberately unresolved" in svg
    assert "less informative" in svg


def test_interaction_script_updates_explanation_panel() -> None:
    report = _load_report_module()
    script = report.render_interaction_script()

    assert 'document.querySelectorAll(".graph-edge")' in script
    assert "Selected edge explanation" in script
    assert ".explain-expected" in script
    assert ".explain-actual" in script
    assert ".explain-endpoint-meaning" in script
    assert ".explain-reasoning" in script
    assert ".explain-orientation-trace" in script
    assert "edge-modal-backdrop" in script


def test_edge_modal_contains_explanation_fields() -> None:
    report = _load_report_module()
    modal = report.render_edge_modal()

    assert "role='dialog'" in modal
    assert "Endpoint meaning" in modal
    assert "Reasoning" in modal
    assert "edge-modal-close" in modal


def test_aggregate_chart_uses_non_overlapping_html_grid() -> None:
    report = _load_report_module()

    aggregate = report.BenchmarkAggregate(
        algorithm="fci_engine.fci_plus",
        n_cases=2,
        skipped_cases=0,
        mean_exact_edge_f1=0.75,
        mean_semantic_edge_f1=0.8,
        mean_skeleton_f1=0.9,
        mean_endpoint_accuracy=0.85,
        mean_elapsed_time=0.0123,
        mean_ci_test_count=42.0,
    )
    html = report.render_aggregate_chart([aggregate])

    assert "aggregate-grid" in html
    assert "aggregate-score-value" in html
    assert "aggregate-bar-fill" in html
    assert "<svg" not in html
    assert "0.750" in html
    assert "mean CI tests: 42.0" in html
