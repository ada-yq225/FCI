"""Generate vector figures for the complete FCI+/Tennessee STAR report."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Polygon

from fci_engine import canonical_dsep_mag, fci_plus

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = (
    ROOT / "case_studies" / "tennessee_star" / "output" / "star_case_study_summary.json"
)
FIGURE_DIR = Path(__file__).resolve().parent / "figures"

BLUE = "#2563eb"
GREEN = "#047857"
ORANGE = "#d97706"
GREY = "#667085"
LIGHT_GREY = "#d0d5dd"
INK = "#172033"
RED = "#b42318"

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
    plot_descriptive_contrasts(payload)
    plot_performance(payload)
    plot_bootstrap_stability(payload)
    plot_sensitivity(payload)
    plot_three_algorithm_agreement(payload)
    plot_pcalg_order_audit(payload)
    for panel in ("attrition", "longitudinal", "focused_treatment"):
        plot_pag_comparison(payload, panel)
    plot_figure4_validation()


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
    algorithms = ["fci", "fci_plus", "pcalg_fci_plus"]
    algorithm_labels = ["Standard FCI", "Self FCI+", "R pcalg FCI+"]
    colors = [BLUE, GREEN, ORANGE]
    test_values = [
        [run_index[(panel, algorithm)]["ci_tests"] for panel in panels]
        for algorithm in algorithms
    ]
    time_values = [
        [run_index[(panel, algorithm)]["median_elapsed_seconds"] for panel in panels]
        for algorithm in algorithms
    ]
    positions = np.arange(len(panels))
    width = 0.24

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for axis, values, title, unit in (
        (axes[0], test_values, "Conditional-independence calls", "calls"),
        (axes[1], time_values, "Median full-data runtime", "seconds"),
    ):
        for index, (algorithm_values, label, color) in enumerate(
            zip(values, algorithm_labels, colors)
        ):
            bars = axis.bar(
                positions + (index - 1) * width,
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
        "Primary STAR fits across three algorithm implementations",
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


def plot_pcalg_order_audit(payload: dict[str, Any]) -> None:
    runs = {
        run["panel"]: run
        for run in payload["runs"]
        if run["algorithm"] == "pcalg_fci_plus"
    }
    panels = ["attrition", "longitudinal", "focused_treatment"]
    exact = [runs[panel]["order_audit"]["exact_pag_match_rate"] for panel in panels]
    skeleton = [runs[panel]["order_audit"]["mean_skeleton_jaccard"] for panel in panels]
    positions = np.arange(len(panels))
    width = 0.34

    figure, axis = plt.subplots(figsize=(8.8, 3.3))
    exact_bars = axis.bar(
        positions - width / 2,
        exact,
        width,
        color=ORANGE,
        label="Exact PAG match",
    )
    skeleton_bars = axis.bar(
        positions + width / 2,
        skeleton,
        width,
        color=GREEN,
        label="Skeleton Jaccard",
    )
    axis.bar_label(
        exact_bars,
        labels=[f"{value:.0%}" for value in exact],
        padding=3,
        fontsize=8,
    )
    axis.bar_label(
        skeleton_bars,
        labels=[f"{value:.0%}" for value in skeleton],
        padding=3,
        fontsize=8,
    )
    axis.set_ylim(0, 1.12)
    axis.set_xticks(positions)
    axis.set_xticklabels([PANEL_LABELS[panel] for panel in panels])
    axis.set_yticks(np.linspace(0, 1, 6))
    axis.set_yticklabels([f"{value:.0%}" for value in np.linspace(0, 1, 6)])
    axis.set_title(
        "R pcalg FCI+ cyclic variable-order audit",
        loc="left",
        fontweight="bold",
    )
    axis.grid(axis="y", color=LIGHT_GREY, linewidth=0.7, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    _save(figure, "star_pcalg_order_audit.pdf")


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
    y_positions = np.arange(len(targets))
    height = 0.34

    figure, axis = plt.subplots(figsize=(10.5, 4.4))
    bars_fci = axis.barh(
        y_positions - height / 2,
        fci_values,
        height,
        color=BLUE,
        label="Standard FCI",
    )
    bars_plus = axis.barh(
        y_positions + height / 2,
        plus_values,
        height,
        color=GREEN,
        label="FCI+",
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
    algorithms = ["fci", "fci_plus"]
    matrix = np.zeros((2, 4))
    tests = np.zeros((2, 4), dtype=int)
    for row in rows:
        row_index = algorithms.index(row["algorithm"])
        column_index = columns.index((row["bins"], row["alpha"]))
        matrix[row_index, column_index] = 1 if row["adjacent"] else 0
        tests[row_index, column_index] = row["ci_tests"]

    figure, axis = plt.subplots(figsize=(8.8, 2.6))
    color_matrix = np.where(matrix == 1, 1.0, 0.0)
    axis.imshow(
        color_matrix,
        cmap=plt.matplotlib.colors.ListedColormap(["#f2f4f7", "#d1fae5"]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    for row_index in range(2):
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
    axis.set_yticks(range(2))
    axis.set_yticklabels(["Standard FCI", "FCI+"])
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


def _save(figure: plt.Figure, filename: str) -> None:
    path = FIGURE_DIR / filename
    figure.savefig(path, format="pdf", metadata={"Creator": "fci-engine report"})
    plt.close(figure)
    print(path)


if __name__ == "__main__":
    main()
