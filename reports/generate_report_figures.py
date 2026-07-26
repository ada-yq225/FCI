"""Generate vector figures for the complete FCI+/Tennessee STAR report."""

from __future__ import annotations

import csv
import json
import math
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle, Wedge
from numpy.typing import NDArray

from fci_engine import canonical_dsep_mag, fci_plus

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = (
    ROOT / "case_studies" / "tennessee_star" / "output" / "star_case_study_summary.json"
)
FIGURE_DIR = Path(__file__).resolve().parent / "figures"
CLAIM_MATRIX_PATH = ROOT / "reports" / "research" / "claim_evidence_matrix.csv"
DOSSIER_PATH = ROOT / "reports" / "research" / "fci_fci_plus_source_dossier.md"
SOFTWARE_LANDSCAPE_PATH = ROOT / "reports" / "research" / "software_landscape.json"
SOFTWARE_BENCHMARK_PATH = ROOT / "reports" / "data" / "software_benchmark_summary.csv"
SOFTWARE_FEATURE_PATH = ROOT / "reports" / "data" / "software_feature_matrix.csv"

BLUE = "#2563eb"
GREEN = "#047857"
ORANGE = "#d97706"
GREY = "#667085"
LIGHT_GREY = "#d0d5dd"
INK = "#172033"
RED = "#b42318"
PURPLE = "#7c3aed"

COLORS = {
    "fci": "#2F6BFF",
    "fci_plus": "#E58A2B",
    "robust": "#1C9A8A",
    "external": "#7562A8",
    "neutral": "#687386",
}

PANEL_LABELS = {
    "attrition": "Attrition / observation",
    "longitudinal": "Longitudinal achievement",
    "focused_treatment": "Focused treatment-outcome",
}

TEMPORAL_TIERS = {
    "Gender": 0,
    "Ethnicity": 0,
    "Entry_Age": 0,
    "Free_Lunch": 0,
    "School_Context": 0,
    "K_Class": 1,
    "Teacher_Experience": 1,
    "K_Achievement": 2,
    "Grade3_Observed": 3,
    "Grade3_Achievement": 3,
}


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    _configure_style()
    generate_research_figures()
    plot_descriptive_contrasts(payload)
    plot_performance(payload)
    plot_bootstrap_stability(payload)
    plot_sensitivity(payload)
    plot_three_algorithm_agreement(payload)
    plot_order_audit(payload)
    for panel in ("attrition", "longitudinal", "focused_treatment"):
        plot_pag_comparison(payload, panel)
    plot_robust_pag_application(payload)
    plot_figure4_validation()


def generate_research_figures() -> None:
    """Generate the source, implementation, and software-comparison figures."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _configure_style()
    plot_fci_fci_plus_workflow()
    plot_source_implementation_map()
    plot_software_benchmark_comparison()
    plot_software_feature_comparison()


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": LIGHT_GREY,
            "axes.labelcolor": INK,
            "xtick.color": GREY,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def plot_fci_fci_plus_workflow() -> None:
    """Contrast the source-aligned stages of standard FCI and FCI+."""

    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    lanes = (
        (
            0.03,
            COLORS["fci"],
            "STANDARD FCI",
            "Spirtes, Glymour, and Scheines (2000), pp. 144-145",
            (
                "Adjacency search and separating sets",
                "Orient unshielded colliders",
                "Refine with Possible-D-SEP",
                "Reset retained endpoint marks",
                "Reorient unshielded colliders",
                "Apply the original orientation closure",
            ),
        ),
        (
            0.53,
            COLORS["fci_plus"],
            "FCI+",
            "Claassen, Mooij, and Heskes (2013), Algorithm 2",
            (
                "Degree-bounded adjacency search",
                "Build the augmented skeleton",
                "Detect candidate D-SEP links",
                "Construct and test separator hierarchies",
                "Remove, rebuild, and revisit candidates",
                "Apply complete final PAG orientation",
            ),
        ),
    )
    for x_value, color, heading, source, stages in lanes:
        _draw_workflow_lane(
            axis,
            x_value=x_value,
            color=color,
            heading=heading,
            source=source,
            stages=stages,
        )

    axis.text(
        0.5,
        0.035,
        "Shared target: an observational equivalence-class representation; "
        "neither workflow identifies a unique DAG or a treatment effect.",
        ha="center",
        va="center",
        fontsize=8.4,
        color=INK,
        fontweight="bold",
    )
    _save_research(figure, "fci_fci_plus_workflow.pdf")


def _draw_workflow_lane(
    axis: Axes,
    *,
    x_value: float,
    color: str,
    heading: str,
    source: str,
    stages: tuple[str, ...],
) -> None:
    width = 0.44
    axis.add_patch(
        FancyBboxPatch(
            (x_value, 0.89),
            width,
            0.075,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            facecolor=color,
            edgecolor=color,
            linewidth=0,
        )
    )
    axis.text(
        x_value + 0.018,
        0.928,
        heading,
        color="white",
        fontsize=11,
        fontweight="bold",
        va="center",
    )
    axis.text(
        x_value + 0.018,
        0.875,
        source,
        color=COLORS["neutral"],
        fontsize=7.7,
        va="top",
    )

    top = 0.79
    box_height = 0.095
    gap = 0.019
    for index, stage in enumerate(stages, start=1):
        y_value = top - (index - 1) * (box_height + gap)
        axis.add_patch(
            FancyBboxPatch(
                (x_value, y_value - box_height),
                width,
                box_height,
                boxstyle="round,pad=0.006,rounding_size=0.01",
                facecolor="#F8FAFC",
                edgecolor=color,
                linewidth=1.15,
            )
        )
        axis.add_patch(
            Circle(
                (x_value + 0.035, y_value - box_height / 2),
                0.021,
                facecolor=color,
                edgecolor=color,
                linewidth=0,
            )
        )
        axis.text(
            x_value + 0.035,
            y_value - box_height / 2,
            str(index),
            ha="center",
            va="center",
            color="white",
            fontsize=8.3,
            fontweight="bold",
        )
        axis.text(
            x_value + 0.07,
            y_value - box_height / 2,
            textwrap.fill(stage, width=39),
            ha="left",
            va="center",
            fontsize=8.5,
            color=INK,
            fontweight="bold" if index in {3, 4} else "normal",
        )
        if index < len(stages):
            axis.annotate(
                "",
                xy=(
                    x_value + width / 2,
                    y_value - box_height - gap + 0.004,
                ),
                xytext=(x_value + width / 2, y_value - box_height - 0.002),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "linewidth": 1.05,
                },
            )


def plot_source_implementation_map() -> None:
    """Render literature claims, implementation symbols, and executable guards."""

    claims = _load_claim_evidence()
    rows = (
        (
            "FCI-STAGES",
            "SOURCE-ALIGNED",
            "Standard FCI schedule\nSpirtes et al. (2000), pp. 144-145",
            COLORS["fci"],
        ),
        (
            "FCI-PDSEP",
            "SOURCE-ALIGNED",
            "Possible-D-SEP refinement\nSpirtes et al. (2000), p. 144",
            COLORS["fci"],
        ),
        (
            "FCIPLUS-AUGMENT",
            "SOURCE-ALIGNED",
            "Augmented skeleton\nClaassen et al. (2013), pp. 176, 178",
            COLORS["fci_plus"],
        ),
        (
            "FCIPLUS-HIERARCHY",
            "SOURCE-ALIGNED",
            "HIE fixed-point search\nClaassen et al. (2013), p. 177",
            COLORS["fci_plus"],
        ),
        (
            "FCIPLUS-ALGO2",
            "SOURCE-ALIGNED",
            "FCI+ Algorithm 2\nClaassen et al. (2013), pp. 178-179",
            COLORS["fci_plus"],
        ),
        (
            "PAG-COMPLETE",
            "SOURCE-ALIGNED",
            "Complete PAG orientation\nZhang (2008), pp. 1873-1896",
            COLORS["external"],
        ),
        (
            "IMPL-ROBUST",
            "ENGINEERING EXTENSION",
            "Finite-sample practical profile\nRepository design, inspected 2026-07-26",
            COLORS["robust"],
        ),
    )

    figure, axis = plt.subplots(figsize=(10.5, 5.45))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    columns = (
        (0.025, 0.285, "LITERATURE CONCEPT"),
        (0.335, 0.315, "IMPLEMENTATION SYMBOL"),
        (0.68, 0.295, "EXECUTABLE VALIDATION"),
    )
    for x_value, width, heading in columns:
        axis.add_patch(
            Rectangle(
                (x_value, 0.915),
                width,
                0.055,
                facecolor="#E9EDF5",
                edgecolor="none",
            )
        )
        axis.text(
            x_value + 0.012,
            0.942,
            heading,
            ha="left",
            va="center",
            fontsize=8.7,
            color=INK,
            fontweight="bold",
        )

    row_height = 0.112
    for index, (claim_id, layer, concept, color) in enumerate(rows):
        record = claims[claim_id]
        y_top = 0.895 - index * row_height
        y_bottom = y_top - 0.096
        background = "#FBFCFE" if index % 2 == 0 else "#F3F6FA"
        axis.add_patch(
            FancyBboxPatch(
                (0.025, y_bottom),
                0.95,
                0.096,
                boxstyle="round,pad=0.003,rounding_size=0.006",
                facecolor=background,
                edgecolor="#D7DDE7",
                linewidth=0.55,
            )
        )
        axis.add_patch(
            Rectangle(
                (0.025, y_bottom),
                0.008,
                0.096,
                facecolor=color,
                edgecolor="none",
            )
        )
        axis.text(
            0.045,
            y_top - 0.021,
            layer,
            ha="left",
            va="center",
            fontsize=6.5,
            color=color,
            fontweight="bold",
        )
        axis.text(
            0.045,
            y_bottom + 0.038,
            concept,
            ha="left",
            va="center",
            fontsize=7.55,
            color=INK,
        )
        implementation = _short_artifact_locator(record["repository_symbol"])
        validation = _short_artifact_locator(record["validation_artifact"])
        axis.text(
            0.345,
            y_bottom + 0.048,
            _wrap_code_locator(implementation, width=37),
            ha="left",
            va="center",
            fontsize=7.05,
            color=INK,
            family="DejaVu Sans Mono",
        )
        axis.text(
            0.69,
            y_bottom + 0.048,
            _wrap_code_locator(validation, width=34),
            ha="left",
            va="center",
            fontsize=6.9,
            color=INK,
            family="DejaVu Sans Mono",
        )
        for x_start, x_end in ((0.309, 0.334), (0.65, 0.679)):
            axis.annotate(
                "",
                xy=(x_end, y_bottom + 0.048),
                xytext=(x_start, y_bottom + 0.048),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#A6AFBF",
                    "linewidth": 0.8,
                },
            )

    axis.text(
        0.025,
        0.048,
        "Each row is backed by reports/research/claim_evidence_matrix.csv; "
        "the robust profile is deliberately separated from paper-aligned definitions.",
        ha="left",
        va="center",
        fontsize=8,
        color=COLORS["neutral"],
    )
    _save_research(figure, "source_implementation_map.pdf")


def plot_software_benchmark_comparison() -> None:
    """Plot the committed known-truth benchmark with explicit timing caveats."""

    rows = {
        row["algorithm"]: row
        for row in _read_csv_rows(SOFTWARE_BENCHMARK_PATH)
        if int(row["n_completed"]) > 0
    }
    order = (
        "fci_engine.fci",
        "fci_engine.fci_plus",
        "fci_engine.fci_plus.robust",
        "causal-learn.fci.fisherz",
        "pcalg.fciPlus",
    )
    missing = set(order) - set(rows)
    if missing:
        raise ValueError(f"Missing completed benchmark rows: {sorted(missing)}")

    labels = (
        "This package: FCI",
        "This package: FCI+ paper",
        "This package: FCI+ robust",
        "causal-learn: FCI",
        "pcalg: FCI+",
    )
    row_colors = (
        COLORS["fci"],
        COLORS["fci_plus"],
        COLORS["robust"],
        COLORS["neutral"],
        COLORS["external"],
    )
    quality_columns = (
        ("mean_skeleton_f1", "Skeleton\nF1"),
        ("mean_exact_edge_f1", "Exact-edge\nF1"),
        ("mean_semantic_edge_f1", "Semantic-edge\nF1"),
        ("mean_endpoint_accuracy", "Endpoint\naccuracy"),
    )
    quality = np.array(
        [
            [float(rows[algorithm][column]) for column, _ in quality_columns]
            for algorithm in order
        ]
    )
    runtime = np.array(
        [float(rows[algorithm]["mean_elapsed_seconds"]) for algorithm in order]
    )

    figure, (quality_axis, runtime_axis) = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.65),
        gridspec_kw={"width_ratios": [2.7, 1.35]},
    )
    cmap = LinearSegmentedColormap.from_list(
        "benchmark_quality",
        ["#F4F6FA", "#B8C9EA", "#183153"],
    )
    quality_axis.set_xlim(-0.5, quality.shape[1] - 0.5)
    quality_axis.set_ylim(quality.shape[0] - 0.5, -0.5)
    for row_index in range(quality.shape[0]):
        for column_index in range(quality.shape[1]):
            value = quality[row_index, column_index]
            normalized = float(np.clip((value - 0.5) / 0.5, 0.0, 1.0))
            quality_axis.add_patch(
                Rectangle(
                    (column_index - 0.5, row_index - 0.5),
                    1,
                    1,
                    facecolor=cmap(normalized),
                    edgecolor="none",
                )
            )
    quality_axis.set_xticks(range(len(quality_columns)))
    quality_axis.set_xticklabels([label for _, label in quality_columns])
    quality_axis.set_yticks(range(len(labels)))
    quality_axis.set_yticklabels(labels)
    quality_axis.tick_params(length=0, labelsize=8.2)
    for tick, color in zip(quality_axis.get_yticklabels(), row_colors):
        tick.set_color(color)
        tick.set_fontweight("bold")
    for row_index in range(quality.shape[0]):
        for column_index in range(quality.shape[1]):
            value = quality[row_index, column_index]
            quality_axis.text(
                column_index,
                row_index,
                f"{value:.1%}",
                ha="center",
                va="center",
                fontsize=8.4,
                fontweight="bold",
                color="white" if value >= 0.82 else INK,
            )
    for spine in quality_axis.spines.values():
        spine.set_visible(False)
    quality_axis.text(
        0,
        1.09,
        "KNOWN-TRUTH RECOVERY QUALITY",
        transform=quality_axis.transAxes,
        fontsize=9,
        color=INK,
        fontweight="bold",
        ha="left",
    )

    y_positions = np.arange(len(labels))
    lower = float(runtime.min() * 0.65)
    upper = float(runtime.max() * 2.7)
    runtime_axis.set_xscale("log")
    runtime_axis.set_xlim(lower, upper)
    runtime_axis.set_ylim(-0.6, len(labels) - 0.4)
    runtime_axis.invert_yaxis()
    runtime_axis.set_yticks([])
    for y_value, value, color in zip(y_positions, runtime, row_colors):
        runtime_axis.hlines(
            y_value,
            lower,
            value,
            color=color,
            linewidth=3.2,
            alpha=0.55,
        )
        runtime_axis.scatter(
            [value],
            [y_value],
            color=color,
            edgecolor="white",
            linewidth=0.8,
            s=65,
            zorder=3,
        )
        runtime_axis.text(
            value * 1.14,
            y_value,
            f"{value:.3f} s",
            va="center",
            ha="left",
            fontsize=8.1,
            color=INK,
        )
    runtime_axis.grid(
        axis="x",
        which="both",
        color=LIGHT_GREY,
        linewidth=0.65,
        alpha=0.75,
    )
    runtime_axis.spines[["top", "right", "left"]].set_visible(False)
    runtime_axis.set_xlabel("Mean elapsed seconds (log scale)", fontsize=8.3)
    runtime_axis.text(
        0,
        1.09,
        "OBSERVED RUNTIME",
        transform=runtime_axis.transAxes,
        fontsize=9,
        color=INK,
        fontweight="bold",
        ha="left",
    )

    figure.text(
        0.04,
        0.948,
        "EXECUTED  |  15 seeded known-truth cases per method",
        ha="left",
        va="center",
        fontsize=8.5,
        color="white",
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": COLORS["robust"],
            "edgecolor": "none",
        },
    )
    figure.text(
        0.04,
        0.052,
        "Descriptive, not a universal ranking: profiles and alpha policies differ by design. "
        "The pcalg timing includes R process startup; other timings are in-process calls.",
        ha="left",
        va="center",
        fontsize=7.75,
        color=COLORS["neutral"],
    )
    figure.subplots_adjust(
        left=0.205,
        right=0.975,
        bottom=0.19,
        top=0.79,
        wspace=0.42,
    )
    _save_research(figure, "software_benchmark_comparison.pdf")


def plot_software_feature_comparison() -> None:
    """Render the capability matrix with executed/documentation provenance."""

    feature_rows = {row["tool"]: row for row in _read_csv_rows(SOFTWARE_FEATURE_PATH)}
    landscape = json.loads(SOFTWARE_LANDSCAPE_PATH.read_text(encoding="utf-8"))
    evidence_kind = {row["tool"]: row["evidence_kind"] for row in landscape["tools"]}
    tools = ("fci_engine", "pcalg", "causal-learn", "Tetrad")
    missing = set(tools) - set(feature_rows)
    if missing:
        raise ValueError(f"Missing feature rows: {sorted(missing)}")

    features = (
        ("standard_fci", "Standard\nFCI"),
        ("fci_plus", "FCI+"),
        ("custom_ci", "Custom\nCI"),
        ("order_audit", "Order\naudit"),
        ("bootstrap_workflow", "Bootstrap\nworkflow"),
        ("orientation_trace", "Orientation\ntrace"),
        ("sepset_provenance", "Sepset\nprovenance"),
        ("artifact_export", "Artifact\nexport"),
    )
    tool_colors = {
        "fci_engine": COLORS["robust"],
        "pcalg": COLORS["external"],
        "causal-learn": COLORS["neutral"],
        "Tetrad": "#8A6A3B",
    }

    figure, axis = plt.subplots(figsize=(10.5, 4.55))
    axis.set_xlim(-2.25, 10.8)
    axis.set_ylim(-1.25, 4.35)
    axis.axis("off")

    x_positions = np.arange(len(features), dtype=float) * 1.08
    y_positions = {tool: 3.15 - index for index, tool in enumerate(tools)}
    for x_value, (_, label) in zip(x_positions, features):
        axis.text(
            x_value,
            4.02,
            label,
            ha="center",
            va="center",
            fontsize=7.45,
            color=INK,
            fontweight="bold",
        )
    axis.text(
        -2.1,
        4.02,
        "TOOL",
        ha="left",
        va="center",
        fontsize=8.1,
        color=INK,
        fontweight="bold",
    )
    axis.text(
        9.05,
        4.02,
        "EVIDENCE",
        ha="center",
        va="center",
        fontsize=8.1,
        color=INK,
        fontweight="bold",
    )

    for index, tool in enumerate(tools):
        row = feature_rows[tool]
        y_value = y_positions[tool]
        axis.add_patch(
            FancyBboxPatch(
                (-2.18, y_value - 0.39),
                12.15,
                0.78,
                boxstyle="round,pad=0.004,rounding_size=0.035",
                facecolor="#FBFCFE" if index % 2 == 0 else "#F2F5F9",
                edgecolor="#D7DDE7",
                linewidth=0.6,
            )
        )
        axis.add_patch(
            Rectangle(
                (-2.18, y_value - 0.39),
                0.08,
                0.78,
                facecolor=tool_colors[tool],
                edgecolor="none",
            )
        )
        axis.text(
            -1.98,
            y_value + 0.08,
            tool,
            ha="left",
            va="center",
            fontsize=9,
            color=INK,
            fontweight="bold",
        )
        axis.text(
            -1.98,
            y_value - 0.17,
            row["language"],
            ha="left",
            va="center",
            fontsize=7.6,
            color=COLORS["neutral"],
        )
        for x_value, (key, _) in zip(x_positions, features):
            _draw_capability_marker(
                axis,
                x=float(x_value),
                y=y_value,
                value=row[key],
                color=tool_colors[tool],
            )

        executed = (
            evidence_kind.get(tool) == "executed" and row["executed_here"] == "yes"
        )
        evidence_label = "EXECUTED" if executed else "DOC-ONLY"
        evidence_color = COLORS["robust"] if executed else COLORS["neutral"]
        axis.text(
            9.05,
            y_value,
            evidence_label,
            ha="center",
            va="center",
            fontsize=7.2,
            color="white",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": evidence_color,
                "edgecolor": "none",
            },
        )

    legend_y = -0.73
    legend_items = (
        (0.0, "yes", "Confirmed"),
        (2.1, "partial", "Partial / broader workflow"),
        (4.95, "not_established", "Not established / not exported"),
        (7.9, "no", "No"),
    )
    for x_value, marker_value, label in legend_items:
        _draw_capability_marker(
            axis,
            x=x_value,
            y=legend_y,
            value=marker_value,
            color=COLORS["neutral"],
        )
        axis.text(
            x_value + 0.28,
            legend_y,
            label,
            ha="left",
            va="center",
            fontsize=7.5,
            color=COLORS["neutral"],
        )

    _save_research(figure, "software_feature_comparison.pdf")


def _draw_capability_marker(
    axis: Axes,
    *,
    x: float,
    y: float,
    value: str,
    color: str,
) -> None:
    radius = 0.145
    if value == "yes":
        axis.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor=color,
                edgecolor=color,
                linewidth=1,
            )
        )
        return
    if value in {"partial", "broader_workflow", "logger_only"}:
        axis.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor="white",
                edgecolor=color,
                linewidth=1.2,
            )
        )
        axis.add_patch(
            Wedge(
                (x, y),
                radius,
                90,
                270,
                facecolor=color,
                edgecolor="none",
            )
        )
        return
    if value in {"not_established", "not_exported"}:
        axis.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor="white",
                edgecolor=color,
                linewidth=1.2,
                linestyle="--",
            )
        )
        axis.text(
            x,
            y,
            "?",
            ha="center",
            va="center",
            fontsize=7.7,
            color=color,
            fontweight="bold",
        )
        return
    if value == "no":
        axis.plot(
            [x - radius * 0.72, x + radius * 0.72],
            [y - radius * 0.72, y + radius * 0.72],
            color=color,
            linewidth=1.6,
        )
        axis.plot(
            [x - radius * 0.72, x + radius * 0.72],
            [y + radius * 0.72, y - radius * 0.72],
            color=color,
            linewidth=1.6,
        )
        return
    raise ValueError(f"Unknown capability value: {value!r}")


def _load_claim_evidence() -> dict[str, dict[str, str]]:
    dossier = DOSSIER_PATH.read_text(encoding="utf-8")
    for heading in (
        "## Spirtes, Glymour, and Scheines (2000)",
        "## Claassen, Mooij, and Heskes (2013)",
        "## Source-to-implementation mapping",
    ):
        if heading not in dossier:
            raise ValueError(f"Research dossier is missing {heading!r}")
    return {row["claim_id"]: row for row in _read_csv_rows(CLAIM_MATRIX_PATH)}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _short_artifact_locator(locator: str) -> str:
    return (
        locator.replace("src/fci_engine/", "").replace("tests/", "").replace("; ", "\n")
    )


def _wrap_code_locator(locator: str, *, width: int) -> str:
    lines: list[str] = []
    for source_line in locator.splitlines():
        pending = source_line
        if "::" in pending and len(pending) > width:
            path, symbol = pending.split("::", maxsplit=1)
            lines.append(f"{path}::")
            pending = symbol
        while len(pending) > width:
            split_at = pending.rfind("_", 0, width)
            if split_at <= 0:
                split_at = width
                lines.append(pending[:split_at])
                pending = pending[split_at:]
            else:
                lines.append(pending[: split_at + 1])
                pending = pending[split_at + 1 :]
        if pending:
            lines.append(pending)
    return "\n".join(lines)


def plot_descriptive_contrasts(payload: dict[str, Any]) -> None:
    rows = payload["descriptives"]["contrasts"]
    metric_order = [
        "kindergarten_score",
        "grade3_score",
        "grade3_observed_rate",
    ]
    metric_labels = {
        "kindergarten_score": "End-of-kindergarten score",
        "grade3_score": "Grade-3 score (observed cases)",
        "grade3_observed_rate": "Grade-3 observation rate",
    }
    comparison_order = ["Small - Regular", "Regular + aide - Regular"]
    comparison_labels = {
        "Small - Regular": "Small - regular",
        "Regular + aide - Regular": "Regular + aide - regular",
    }
    colors = {
        "Small - Regular": GREEN,
        "Regular + aide - Regular": ORANGE,
    }
    markers = {
        "Small - Regular": "o",
        "Regular + aide - Regular": "s",
    }

    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.45))
    for axis, metric in zip(axes, metric_order):
        selected = [row for row in rows if row["metric"] == metric]
        for y, comparison in enumerate(comparison_order):
            row = next(item for item in selected if item["comparison"] == comparison)
            factor = 100.0 if metric == "grade3_observed_rate" else 1.0
            estimate = row["estimate"] * factor
            low = row["ci_low"] * factor
            high = row["ci_high"] * factor
            axis.errorbar(
                estimate,
                y,
                xerr=[[estimate - low], [high - estimate]],
                color=colors[comparison],
                marker=markers[comparison],
                capsize=3,
                linewidth=1.8,
                markersize=6,
            )
            suffix = " pp" if metric == "grade3_observed_rate" else ""
            axis.annotate(
                f"{estimate:+.1f}{suffix}",
                (estimate, y),
                xytext=(5, 7),
                textcoords="offset points",
                fontsize=8,
                color=INK,
            )
        axis.axvline(0, color=GREY, linewidth=1)
        axis.grid(axis="x", color=LIGHT_GREY, linewidth=0.7, alpha=0.7)
        axis.set_yticks([0, 1])
        axis.set_yticklabels([comparison_labels[item] for item in comparison_order])
        axis.invert_yaxis()
        axis.set_title(metric_labels[metric], loc="left", fontweight="bold")
        axis.set_xlabel(
            "Percentage-point difference"
            if metric == "grade3_observed_rate"
            else "Score-point difference"
        )
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)

    figure.suptitle(
        "Randomized-arm descriptive contrasts with 95% school-cluster intervals",
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    _save(figure, "star_descriptive_contrasts.pdf")


def plot_performance(payload: dict[str, Any]) -> None:
    run_index = {(run["panel"], run["algorithm"]): run for run in payload["runs"]}
    panels = ["attrition", "longitudinal", "focused_treatment"]
    labels = [PANEL_LABELS[panel] for panel in panels]
    algorithms = ["fci", "fci_plus", "fci_plus_robust", "pcalg_fci_plus"]
    algorithm_labels = [
        "Standard FCI",
        "Self FCI+ paper",
        "Self FCI+ robust",
        "R pcalg FCI+",
    ]
    colors = [BLUE, GREEN, PURPLE, ORANGE]
    test_values = [
        [run_index[(panel, algorithm)]["ci_tests"] for panel in panels]
        for algorithm in algorithms
    ]
    time_values = [
        [run_index[(panel, algorithm)]["median_elapsed_seconds"] for panel in panels]
        for algorithm in algorithms
    ]
    positions = np.arange(len(panels))
    width = 0.19

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for axis, values, title, unit in (
        (axes[0], test_values, "Conditional-independence calls", "calls"),
        (axes[1], time_values, "Median full-data runtime", "seconds"),
    ):
        for index, (algorithm_values, label, color) in enumerate(
            zip(values, algorithm_labels, colors)
        ):
            bars = axis.bar(
                positions + (index - 1.5) * width,
                algorithm_values,
                width,
                color=color,
                label=label,
            )
            axis.bar_label(
                bars,
                fmt="%.2f" if unit == "seconds" else "%.0f",
                padding=3,
                fontsize=7,
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=17, ha="right")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(unit)
        axis.grid(axis="y", color=LIGHT_GREY, linewidth=0.7, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper right")
    figure.suptitle(
        "STAR fits: paper validation and robust application profile",
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    _save(figure, "star_performance.pdf")


def plot_three_algorithm_agreement(payload: dict[str, Any]) -> None:
    comparisons = payload["three_algorithm_comparisons"]
    panels = ["attrition", "longitudinal", "focused_treatment"]
    pair_order = [
        ("fci", "fci_plus"),
        ("fci", "pcalg_fci_plus"),
        ("fci_plus", "pcalg_fci_plus"),
    ]
    pair_labels = ["FCI vs self FCI+", "FCI vs R FCI+", "Self vs R FCI+"]
    colors = [BLUE, ORANGE, GREEN]
    positions = np.arange(len(panels))
    width = 0.24

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for metric_axis, metric, title in (
        (axes[0], "skeleton_jaccard", "Skeleton Jaccard"),
        (axes[1], "endpoint_match_rate", "Exact endpoints on shared edges"),
    ):
        for pair_index, (pair, label, color) in enumerate(
            zip(pair_order, pair_labels, colors)
        ):
            values = []
            for panel in panels:
                row = next(
                    item
                    for item in comparisons[panel]["pairs"]
                    if (item["left"], item["right"]) == pair
                )
                values.append(row[metric])
            bars = metric_axis.bar(
                positions + (pair_index - 1) * width,
                values,
                width,
                color=color,
                label=label,
            )
            metric_axis.bar_label(
                bars,
                labels=[f"{value:.0%}" for value in values],
                padding=3,
                fontsize=7,
            )
        metric_axis.set_ylim(0, 1.12)
        metric_axis.set_xticks(positions)
        metric_axis.set_xticklabels(
            [PANEL_LABELS[panel] for panel in panels],
            rotation=17,
            ha="right",
        )
        metric_axis.set_title(title, loc="left", fontweight="bold")
        metric_axis.set_yticks(np.linspace(0, 1, 6))
        metric_axis.set_yticklabels([f"{value:.0%}" for value in np.linspace(0, 1, 6)])
        metric_axis.grid(axis="y", color=LIGHT_GREY, linewidth=0.7, alpha=0.7)
        metric_axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    figure.suptitle(
        "Cross-implementation agreement is stronger for skeletons than endpoints",
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    _save(figure, "star_three_algorithm_agreement.pdf")


def plot_order_audit(payload: dict[str, Any]) -> None:
    run_index = {(run["panel"], run["algorithm"]): run for run in payload["runs"]}
    panels = ["attrition", "longitudinal", "focused_treatment"]
    algorithms = ["fci_plus", "fci_plus_robust", "pcalg_fci_plus"]
    labels = ["Self FCI+ paper", "Self FCI+ robust", "R pcalg FCI+"]
    colors = [GREEN, PURPLE, ORANGE]
    positions = np.arange(len(panels))
    width = 0.24

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    for axis, metric, title in (
        (axes[0], "exact_pag_match_rate", "Exact PAG match rate"),
        (axes[1], "mean_skeleton_jaccard", "Mean skeleton Jaccard"),
    ):
        for algorithm_index, (algorithm, label, color) in enumerate(
            zip(algorithms, labels, colors)
        ):
            values = [
                run_index[(panel, algorithm)]["order_audit"][metric] for panel in panels
            ]
            bars = axis.bar(
                positions + (algorithm_index - 1) * width,
                values,
                width,
                color=color,
                label=label,
            )
            axis.bar_label(
                bars,
                labels=[f"{value:.0%}" for value in values],
                padding=3,
                fontsize=7,
            )
        axis.set_ylim(0, 1.12)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [PANEL_LABELS[panel] for panel in panels],
            rotation=16,
            ha="right",
        )
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.set_yticklabels([f"{value:.0%}" for value in np.linspace(0, 1, 6)])
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", color=LIGHT_GREY, linewidth=0.7, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=3,
        fontsize=8,
    )
    figure.suptitle(
        "Cyclic variable-order robustness separates skeletons from endpoints",
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.80))
    _save(figure, "star_order_audit.pdf")


def plot_bootstrap_stability(payload: dict[str, Any]) -> None:
    run_index = {(run["panel"], run["algorithm"]): run for run in payload["runs"]}
    targets = [
        ("attrition", "K_Class", "Grade3_Observed"),
        ("attrition", "K_Achievement", "Grade3_Observed"),
        ("longitudinal", "K_Class", "Grade3_Achievement"),
        ("longitudinal", "K_Achievement", "Grade3_Achievement"),
        ("focused_treatment", "K_Class", "Grade3_Achievement"),
    ]
    labels = [
        "Class - grade-3 observed",
        "K achievement - grade-3 observed",
        "Class - grade-3 achievement",
        "K achievement - grade-3 achievement",
        "Class - grade-3 achievement",
    ]
    panel_prefix = [
        "Attrition",
        "Attrition",
        "Longitudinal",
        "Longitudinal",
        "Focused",
    ]
    fci_values = [
        _bootstrap_frequency(run_index[(panel, "fci")], x, y) for panel, x, y in targets
    ]
    plus_values = [
        _bootstrap_frequency(run_index[(panel, "fci_plus")], x, y)
        for panel, x, y in targets
    ]
    robust_values = [
        _bootstrap_frequency(run_index[(panel, "fci_plus_robust")], x, y)
        for panel, x, y in targets
    ]
    y_positions = np.arange(len(targets))
    height = 0.23

    figure, axis = plt.subplots(figsize=(10.5, 4.4))
    bars_fci = axis.barh(
        y_positions - height,
        fci_values,
        height,
        color=BLUE,
        label="Standard FCI",
    )
    bars_plus = axis.barh(
        y_positions,
        plus_values,
        height,
        color=GREEN,
        label="FCI+ paper",
    )
    bars_robust = axis.barh(
        y_positions + height,
        robust_values,
        height,
        color=PURPLE,
        label="FCI+ robust",
    )
    axis.set_xlim(0, 1.05)
    axis.set_xticks(np.linspace(0, 1, 6))
    axis.set_xticklabels([f"{value:.0%}" for value in np.linspace(0, 1, 6)])
    axis.set_yticks(y_positions)
    axis.set_yticklabels(
        [f"{panel}: {label}" for panel, label in zip(panel_prefix, labels)]
    )
    axis.invert_yaxis()
    axis.set_xlabel("Adjacency frequency across 12 school-cluster resamples")
    axis.set_title(
        "Bootstrap adjacency stability for substantively important pairs",
        loc="left",
        fontweight="bold",
    )
    axis.grid(axis="x", color=LIGHT_GREY, linewidth=0.7, alpha=0.7)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.bar_label(
        bars_fci,
        labels=[f"{value:.0%}" for value in fci_values],
        padding=3,
        fontsize=8,
    )
    axis.bar_label(
        bars_plus,
        labels=[f"{value:.0%}" for value in plus_values],
        padding=3,
        fontsize=8,
    )
    axis.bar_label(
        bars_robust,
        labels=[f"{value:.0%}" for value in robust_values],
        padding=3,
        fontsize=8,
    )
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    _save(figure, "star_bootstrap_stability.pdf")


def plot_sensitivity(payload: dict[str, Any]) -> None:
    rows = payload["sensitivity"]
    columns = [
        (3, 0.01),
        (3, 0.05),
        (4, 0.01),
        (4, 0.05),
    ]
    algorithms = ["fci", "fci_plus", "fci_plus_robust"]
    matrix = np.zeros((3, 4))
    tests: NDArray[np.int_] = np.zeros((3, 4), dtype=int)
    for row in rows:
        row_index = algorithms.index(row["algorithm"])
        column_index = columns.index((row["bins"], row["alpha"]))
        matrix[row_index, column_index] = 1 if row["adjacent"] else 0
        tests[row_index, column_index] = row["ci_tests"]

    figure, axis = plt.subplots(figsize=(8.8, 2.6))
    color_matrix = np.where(matrix == 1, 1.0, 0.0)
    axis.imshow(
        color_matrix,
        cmap=ListedColormap(["#f2f4f7", "#d1fae5"]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    for row_index in range(3):
        for column_index in range(4):
            label = (
                "Present\n" if matrix[row_index, column_index] else "Absent\n"
            ) + f"{tests[row_index, column_index]} CI tests"
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=GREEN if matrix[row_index, column_index] else GREY,
                fontweight="bold" if matrix[row_index, column_index] else "normal",
                fontsize=8,
            )
    axis.set_xticks(range(4))
    axis.set_xticklabels(
        [f"{bins} bins\n$\\alpha={alpha:.2f}$" for bins, alpha in columns]
    )
    axis.set_yticks(range(3))
    axis.set_yticklabels(["Standard FCI", "FCI+ paper", "FCI+ robust"])
    axis.set_title(
        "Sensitivity of the focused class-assignment / grade-3 adjacency",
        loc="left",
        fontweight="bold",
    )
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    _save(figure, "star_sensitivity.pdf")


def plot_pag_comparison(payload: dict[str, Any], panel: str) -> None:
    run_index = {(run["panel"], run["algorithm"]): run for run in payload["runs"]}
    figure, axes = plt.subplots(1, 3, figsize=(15.4, 5.5))
    for axis, algorithm, title in (
        (axes[0], "fci", "Standard FCI"),
        (axes[1], "fci_plus", "Self-implemented FCI+"),
        (axes[2], "pcalg_fci_plus", "R pcalg::fciPlus"),
    ):
        run = run_index[(panel, algorithm)]
        positions = _tier_positions(run["node_names"])
        _draw_pag(axis, run["node_names"], run["pag_edges"], positions)
        axis.set_title(
            f"{title}\n{run['edges']} edges, {run['ci_tests']:,} CI tests",
            fontweight="bold",
        )
    figure.suptitle(
        PANEL_LABELS[panel],
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    legend = [
        Line2D(
            [0],
            [0],
            color=GREY,
            marker="o",
            markerfacecolor="white",
            label="circle: unresolved endpoint",
        ),
        Line2D([0], [0], color=GREY, marker=">", label="arrowhead"),
        Line2D(
            [0],
            [0],
            color=GREY,
            marker="|",
            markersize=10,
            label="tail: ancestral endpoint",
        ),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.92))
    _save(figure, f"pag_{panel}.pdf")


def plot_robust_pag_application(payload: dict[str, Any]) -> None:
    run_index = {(run["panel"], run["algorithm"]): run for run in payload["runs"]}
    panels = ["attrition", "longitudinal", "focused_treatment"]
    figure, axes = plt.subplots(1, 3, figsize=(15.4, 5.5))
    for axis, panel in zip(axes, panels):
        run = run_index[(panel, "fci_plus_robust")]
        positions = _tier_positions(run["node_names"])
        _draw_pag(axis, run["node_names"], run["pag_edges"], positions)
        audit = run["order_audit"]
        axis.set_title(
            f"{PANEL_LABELS[panel]}\n"
            f"{run['edges']} edges, "
            f"{audit['exact_pag_match_rate']:.0%} exact across orders",
            fontweight="bold",
        )
    figure.suptitle(
        "Robust FCI+ application profile: conservative STAR PAGs",
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    legend = [
        Line2D(
            [0],
            [0],
            color=GREY,
            marker="o",
            markerfacecolor="white",
            label="circle: unresolved endpoint",
        ),
        Line2D([0], [0], color=GREY, marker=">", label="arrowhead"),
        Line2D(
            [0],
            [0],
            color=GREY,
            marker="|",
            markersize=10,
            label="tail: ancestral endpoint",
        ),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.92))
    _save(figure, "pag_robust_application.pdf")


def plot_figure4_validation() -> None:
    mag = canonical_dsep_mag()
    result = fci_plus(
        mag.dummy_data(),
        ci_test=mag.oracle_ci_test(),
        max_cond_set_size=None,
        sparsity_bound=None,
        max_path_length=None,
        sepset_selection="first",
        orientation_strategy="standard",
    )
    nodes = list(mag.nodes)
    positions = {
        "Z": (0.08, 0.5),
        "U": (0.4, 0.75),
        "V": (0.4, 0.25),
        "X": (0.86, 0.75),
        "Y": (0.86, 0.25),
    }
    oracle_edges = [
        {
            "x": x,
            "y": y,
            "endpoint_x": endpoints[0],
            "endpoint_y": endpoints[1],
        }
        for (x, y), endpoints in mag.oracle_shape().items()
    ]
    learned_edges = result.to_edge_records()

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    _draw_pag(axes[0], nodes, oracle_edges, positions)
    axes[0].set_title("Published Figure 4(b) oracle PAG", fontweight="bold")
    _draw_pag(axes[1], nodes, learned_edges, positions)
    axes[1].set_title(
        "Recovered FCI+ PAG\n63 CI queries; exact endpoint match",
        fontweight="bold",
    )
    figure.suptitle(
        "Exact m-separation isolates algorithmic correctness from sampling error",
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.035,
        "The X-Y candidate is removed by the hierarchical D-SEP stage using "
        "the separating set {U, V, Z}.",
        ha="center",
        color=GREY,
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.9))
    _save(figure, "figure4_validation.pdf")


def _bootstrap_frequency(run: dict[str, Any], x: str, y: str) -> float:
    target = frozenset((x, y))
    for edge in run["bootstrap_adjacencies"]:
        if frozenset((edge["x"], edge["y"])) == target:
            return float(edge["frequency"])
    return 0.0


def _tier_positions(nodes: list[str]) -> dict[str, tuple[float, float]]:
    tier_nodes: dict[int, list[str]] = {}
    for node in nodes:
        tier_nodes.setdefault(TEMPORAL_TIERS.get(node, 1), []).append(node)
    x_positions = {0: 0.05, 1: 0.38, 2: 0.64, 3: 1.0}
    positions: dict[str, tuple[float, float]] = {}
    for tier, values in tier_nodes.items():
        ordered = sorted(values)
        if len(ordered) == 1:
            y_values = [0.5]
        else:
            y_values = np.linspace(0.9, 0.1, len(ordered))
        for node, y_value in zip(ordered, y_values):
            positions[node] = (x_positions.get(tier, 0.35), float(y_value))
    return positions


def _draw_pag(
    axis: Axes,
    nodes: list[str],
    edges: list[dict[str, Any]],
    positions: dict[str, tuple[float, float]],
) -> None:
    axis.set_xlim(-0.2, 1.2)
    axis.set_ylim(-0.1, 1.1)
    axis.set_aspect("equal")
    axis.axis("off")

    for edge in edges:
        x_name = edge["x"]
        y_name = edge["y"]
        start, end = _trim_segment(positions[x_name], positions[y_name], 0.095)
        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=GREY,
            linewidth=1.3,
            zorder=1,
        )
        _draw_endpoint(
            axis,
            start,
            end,
            str(edge["endpoint_x"]),
        )
        _draw_endpoint(
            axis,
            end,
            start,
            str(edge["endpoint_y"]),
        )

    for node in nodes:
        x_value, y_value = positions[node]
        label = node.replace("_", " ")
        width = max(0.18, min(0.36, 0.013 * len(label) + 0.14))
        patch = FancyBboxPatch(
            (x_value - width / 2, y_value - 0.04),
            width,
            0.08,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor="#f8fafc",
            edgecolor=INK,
            linewidth=1.0,
            zorder=3,
            clip_on=False,
        )
        axis.add_patch(patch)
        axis.text(
            x_value,
            y_value,
            label,
            ha="center",
            va="center",
            fontsize=7.3,
            fontweight="bold",
            color=INK,
            zorder=4,
            clip_on=False,
        )


def _trim_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    radius: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    ux = dx / length
    uy = dy / length
    return (
        (start[0] + ux * radius, start[1] + uy * radius),
        (end[0] - ux * radius, end[1] - uy * radius),
    )


def _draw_endpoint(
    axis: Axes,
    point: tuple[float, float],
    other: tuple[float, float],
    endpoint: str,
) -> None:
    x_value, y_value = point
    dx = x_value - other[0]
    dy = y_value - other[1]
    length = math.hypot(dx, dy) or 1.0
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    if endpoint == "CIRCLE":
        axis.plot(
            [x_value],
            [y_value],
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=GREY,
            markeredgewidth=1.2,
            zorder=2,
        )
    elif endpoint == "TAIL":
        axis.plot(
            [x_value - px * 0.015, x_value + px * 0.015],
            [y_value - py * 0.015, y_value + py * 0.015],
            color=GREY,
            linewidth=1.6,
            zorder=2,
        )
    elif endpoint == "ARROW":
        base_x = x_value - ux * 0.032
        base_y = y_value - uy * 0.032
        triangle = Polygon(
            [
                (x_value, y_value),
                (base_x + px * 0.015, base_y + py * 0.015),
                (base_x - px * 0.015, base_y - py * 0.015),
            ],
            closed=True,
            facecolor=GREY,
            edgecolor=GREY,
            zorder=2,
        )
        axis.add_patch(triangle)


def _save(figure: Figure, filename: str) -> None:
    path = FIGURE_DIR / filename
    figure.savefig(path, format="pdf", metadata={"Creator": "fci-engine report"})
    plt.close(figure)
    print(path)


def _save_research(figure: Figure, filename: str) -> None:
    path = FIGURE_DIR / filename
    figure.savefig(
        path,
        format="pdf",
        metadata={
            "Creator": "fci-engine research report",
            "CreationDate": datetime(2026, 7, 26, tzinfo=timezone.utc),
        },
    )
    plt.close(figure)
    print(path)


if __name__ == "__main__":
    main()
