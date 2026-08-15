"""Regression tests for the separate Tennessee STAR application layer."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from case_studies.tennessee_star.download_data import (
    GUIDE_PATH,
    STUDENT_PATH,
    sha256,
)
from case_studies.tennessee_star.pcalg_reference import load_pcalg_reference
from case_studies.tennessee_star.report import _render_pag_svg, render_report
from case_studies.tennessee_star.study import (
    _fit_panel,
    cyclic_order_audit,
    load_star,
    prepare_study,
    sensitivity_analysis,
)

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


def test_committed_pcalg_tables_load_into_validated_report_records() -> None:
    study = prepare_study(load_star())

    records, metadata = load_pcalg_reference(study)

    assert {record["panel"] for record in records} == set(study.panels)
    assert all(record["pag_edges"] for record in records)
    assert all(record["order_audit"]["rows"] for record in records)
    assert metadata["algorithm"] == "pcalg_fci_plus"


def test_committed_star_report_is_reproducible_from_summary_payload() -> None:
    payload = json.loads(
        (OUTPUT / "star_case_study_summary.json").read_text(encoding="utf-8")
    )
    html = render_report(payload)

    assert "separate from the algorithm package" in html
    assert "Randomized-arm reference" in html
    assert "Learned Partial Ancestral Graphs from three implementations" in html
    assert "R pcalg::fciPlus" in html
    assert "Where the three algorithms agree" in html
    assert "Researcher self-assessment" in html
    assert "K_Class &lt;-&gt; Grade3_Achievement" in html
    assert "A separate robust FCI+ application profile" in html
    assert "a universally correct causal graph" in html
    assert "doi.org/10.7910/DVN/SIWH9F" in html


def test_pag_renderer_separates_incident_edge_endpoints() -> None:
    payload = json.loads(
        (OUTPUT / "star_case_study_summary.json").read_text(encoding="utf-8")
    )
    for run_index, run in enumerate(payload["runs"]):
        svg = _render_pag_svg(run, f"routing-{run_index}")
        endpoints_by_node: dict[str, list[str]] = {}
        for node_x, node_y, start, end in re.findall(
            r'data-node-x="([^"]+)" data-node-y="([^"]+)" '
            r'data-start="([^"]+)" data-end="([^"]+)"',
            svg,
        ):
            endpoints_by_node.setdefault(node_x, []).append(start)
            endpoints_by_node.setdefault(node_y, []).append(end)

        assert '<path class="edge-line"' in svg
        assert all(
            len(points) == len(set(points)) for points in endpoints_by_node.values()
        )


def test_pag_renderer_routes_long_collinear_edges_around_middle_nodes() -> None:
    run = {
        "algorithm": "fci_plus",
        "node_names": ["Ethnicity", "Free_Lunch", "School_Context"],
        "pag_edges": [
            {
                "x": "Ethnicity",
                "y": "School_Context",
                "endpoint_x": "TAIL",
                "endpoint_y": "ARROW",
                "edge": "Ethnicity --> School_Context",
            }
        ],
    }

    svg = _render_pag_svg(run, "collinear-routing")

    path = re.search(r'<path class="edge-line" d="([^"]+)"', svg)
    assert path is not None
    assert " C " in path.group(1)


def test_star_summary_contains_validation_and_robust_runs_for_every_panel() -> None:
    payload = json.loads(
        (OUTPUT / "star_case_study_summary.json").read_text(encoding="utf-8")
    )
    combinations = {(run["panel"], run["algorithm"]) for run in payload["runs"]}

    assert combinations == {
        ("attrition", "fci"),
        ("attrition", "fci_plus"),
        ("attrition", "fci_plus_robust"),
        ("attrition", "pcalg_fci_plus"),
        ("longitudinal", "fci"),
        ("longitudinal", "fci_plus"),
        ("longitudinal", "fci_plus_robust"),
        ("longitudinal", "pcalg_fci_plus"),
        ("focused_treatment", "fci"),
        ("focused_treatment", "fci_plus"),
        ("focused_treatment", "fci_plus_robust"),
        ("focused_treatment", "pcalg_fci_plus"),
    }


def test_robust_application_profile_is_conservative_and_order_stable() -> None:
    data = np.tile(
        np.array(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 0, 2],
                [0, 1, 0],
                [1, 0, 1],
                [2, 1, 2],
            ]
        ),
        (20, 1),
    )
    frame = pd.DataFrame(
        data,
        columns=["K_Class", "Gender", "Grade3_Achievement"],
    )
    result = _fit_panel(
        frame,
        algorithm="fci_plus_robust",
        alpha=0.05,
        sparsity_bound=2,
    )
    audit = cyclic_order_audit(
        frame,
        algorithm="fci_plus_robust",
        alpha=0.05,
        sparsity_bound=2,
        baseline=result,
    )

    assert result.config.skeleton_stable is True
    assert result.config.orientation_strategy == "robust"
    assert result.config.conservative_colliders is True
    assert audit["exact_pag_match_rate"] == 1.0
    assert audit["minimum_skeleton_jaccard"] == 1.0


def test_star_sensitivity_varies_fci_plus_k_without_duplicating_fci() -> None:
    rows = sensitivity_analysis(
        load_star(),
        alphas=(0.01,),
        bin_counts=(3,),
        sparsity_bounds=(2, 3, 4),
    )

    assert len(rows) == 7
    fci_rows = [row for row in rows if row["algorithm"] == "fci"]
    assert len(fci_rows) == 1
    assert fci_rows[0]["sparsity_bound"] is None
    for algorithm in ("fci_plus", "fci_plus_robust"):
        assert {
            row["sparsity_bound"] for row in rows if row["algorithm"] == algorithm
        } == {2, 3, 4}


def test_star_three_algorithm_comparison_records_replication_and_disagreement() -> None:
    payload = json.loads(
        (OUTPUT / "star_case_study_summary.json").read_text(encoding="utf-8")
    )
    comparisons = payload["three_algorithm_comparisons"]

    focused = comparisons["focused_treatment"]
    assert focused["all_three_exact_pag"] is True
    assert focused["consensus_skeleton_edges"] == 10
    assert focused["target_edges"] == {
        "fci": "K_Class <-> Grade3_Achievement",
        "fci_plus": "K_Class <-> Grade3_Achievement",
        "pcalg_fci_plus": "K_Class <-> Grade3_Achievement",
    }

    attrition = comparisons["attrition"]
    assert attrition["all_three_exact_pag"] is False
    assert attrition["target_edges"]["fci"] == ("K_Achievement <-- Grade3_Observed")
    assert attrition["target_edges"]["fci_plus"] == (
        "K_Achievement <-o Grade3_Observed"
    )
    assert attrition["target_edges"]["pcalg_fci_plus"] == (
        "K_Achievement --> Grade3_Observed"
    )


def test_star_robust_application_improves_order_and_temporal_audits() -> None:
    payload = json.loads(
        (OUTPUT / "star_case_study_summary.json").read_text(encoding="utf-8")
    )
    comparisons = payload["robust_application_comparisons"]
    run_index = {(run["panel"], run["algorithm"]): run for run in payload["runs"]}

    for panel, comparison in comparisons.items():
        assert comparison["robust_exact_order_rate"] == 1.0
        assert comparison["robust_minimum_skeleton_jaccard"] == 1.0
        assert comparison["robust_temporal_flags"] == 0
        robust_run = run_index[(panel, "fci_plus_robust")]
        assert robust_run["order_audit"]["target_adjacency_rate"] == 1.0

    assert comparisons["longitudinal"]["paper_exact_order_rate"] < 1.0
    assert comparisons["longitudinal"]["paper_temporal_flags"] > 0
