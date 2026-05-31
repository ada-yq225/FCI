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
