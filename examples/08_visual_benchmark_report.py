"""Generate a visual HTML report for realistic FCI oracle benchmarks."""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path
from typing import Optional

from fci_engine.metrics import (
    BenchmarkAggregate,
    BenchmarkResult,
    aggregate_benchmark_results,
    explain_pag_differences,
    run_oracle_benchmark,
)
from fci_engine.metrics.accuracy import Shape
from fci_engine.simulation import (
    OracleCase,
    default_oracle_cases,
    realistic_oracle_cases,
)

REPORT_PATH = Path(__file__).with_name("realistic_benchmark_report.html")
COLORS = {
    "fci_engine.fci": "#2563eb",
    "fci_engine.fci_plus": "#059669",
    "fci_engine.fci_plus.robust": "#047857",
    "fci_engine.fci.kernel": "#7c3aed",
    "fci_engine.fci_plus.kernel": "#a855f7",
    "fci_engine.fci_plus.kernel.robust": "#6d28d9",
    "causal-learn.fci.fisherz": "#d97706",
    "causal-learn.fci.kci": "#f59e0b",
    "pcalg.fciPlus": "#dc2626",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--no-external", action="store_true")
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    cases = [
        *default_oracle_cases(),
        *realistic_oracle_cases(n_repeats=args.repeats, n_samples=args.samples),
    ]
    results = run_oracle_benchmark(
        cases,
        include_causal_learn=not args.no_external,
        include_pcalg=not args.no_external,
    )
    html_text = render_report(cases, results)
    args.output.write_text(html_text, encoding="utf-8")
    print(f"Wrote {args.output}")


def render_report(cases: list[OracleCase], results: list[BenchmarkResult]) -> str:
    aggregates = aggregate_benchmark_results(results)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FCI Realistic Benchmark Report</title>
<style>
  :root {{
    color-scheme: light;
    --ink: #101827;
    --muted: #667085;
    --line: #d0d5dd;
    --panel: #ffffff;
    --band: #f3f6fa;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
      "Segoe UI", sans-serif;
    background: var(--band);
    color: var(--ink);
  }}
  header {{
    padding: 28px 36px 20px;
    background: #ffffff;
    border-bottom: 1px solid var(--line);
  }}
  h1 {{ margin: 0 0 8px; font-size: 28px; line-height: 1.15; }}
  h2 {{ margin: 0 0 14px; font-size: 18px; }}
  p {{ margin: 0; color: var(--muted); line-height: 1.5; }}
  main {{
    width: min(1680px, calc(100vw - 36px));
    margin: 24px auto 40px;
  }}
  section {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    margin-bottom: 18px;
    padding: 18px;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 18px;
  }}
  .graph-gallery {{
    display: grid;
    grid-template-columns: 1fr;
    gap: 18px;
  }}
  .case-card {{
    border-top: 1px solid #eaecf0;
    padding-top: 16px;
  }}
  .case-card:first-child {{
    border-top: 0;
    padding-top: 0;
  }}
  .graph-row {{
    display: grid;
    grid-template-columns: repeat(3, minmax(460px, 1fr));
    gap: 16px;
    align-items: start;
    min-width: 1420px;
  }}
  .graph-strip {{
    width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    padding-bottom: 6px;
  }}
  .graph-panel {{
    min-width: 0;
  }}
  .graph-svg {{
    display: block;
    width: 100%;
    height: auto;
  }}
  .graph-edge {{
    cursor: pointer;
    outline: none;
  }}
  .edge-hit {{
    stroke: transparent;
    stroke-width: 18;
    pointer-events: stroke;
  }}
  .edge-line,
  .edge-endpoint {{
    pointer-events: none;
    transition: stroke 120ms ease, stroke-width 120ms ease, fill 120ms ease;
  }}
  .graph-edge:hover .edge-line,
  .graph-edge:focus .edge-line {{
    stroke-width: 3.8;
  }}
  .graph-edge.is-selected .edge-line {{
    stroke: #0f172a;
    stroke-width: 4.2;
  }}
  .graph-edge.is-selected .edge-endpoint {{
    stroke: #0f172a;
    fill: #0f172a;
  }}
  .graph-edge.is-selected circle.edge-endpoint {{
    fill: #ffffff;
  }}
  .edge-explainer {{
    margin-top: 12px;
    border: 1px solid #d0d5dd;
    border-radius: 8px;
    background: #fbfcfe;
    padding: 12px;
  }}
  .edge-explainer-title {{
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 8px;
  }}
  .edge-explainer-grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
  }}
  .edge-explainer-item {{
    min-width: 0;
  }}
  .edge-explainer-label {{
    color: var(--muted);
    font-size: 11px;
    font-weight: 700;
    margin-bottom: 3px;
    text-transform: uppercase;
  }}
  .edge-explainer-value {{
    font-size: 13px;
    line-height: 1.45;
    overflow-wrap: anywhere;
  }}
  .edge-explainer-item.wide {{
    grid-column: 1 / -1;
  }}
  .edge-modal-backdrop[hidden] {{
    display: none;
  }}
  .edge-modal-backdrop {{
    position: fixed;
    inset: 0;
    z-index: 50;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
    background: rgba(16, 24, 39, 0.48);
  }}
  .edge-modal {{
    width: min(820px, 100%);
    max-height: calc(100vh - 48px);
    overflow: auto;
    border: 1px solid #d0d5dd;
    border-radius: 8px;
    background: #ffffff;
    box-shadow: 0 24px 70px rgba(16, 24, 39, 0.24);
    padding: 16px;
  }}
  .edge-modal-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 12px;
  }}
  .edge-modal-title {{
    font-size: 16px;
    font-weight: 800;
  }}
  .edge-modal-close {{
    border: 1px solid #d0d5dd;
    border-radius: 6px;
    background: #ffffff;
    color: #101827;
    cursor: pointer;
    font: inherit;
    font-size: 13px;
    font-weight: 700;
    padding: 6px 10px;
  }}
  .aggregate-grid {{
    display: grid;
    grid-template-columns: minmax(240px, 1.2fr) repeat(4, minmax(190px, 1fr));
    gap: 0;
    min-width: 1120px;
    border: 1px solid #eaecf0;
    border-radius: 8px;
    overflow: hidden;
  }}
  .aggregate-header,
  .aggregate-cell {{
    padding: 10px 12px;
    border-bottom: 1px solid #eaecf0;
    background: #ffffff;
    min-width: 0;
  }}
  .aggregate-header {{
    color: #475467;
    font-size: 12px;
    font-weight: 800;
    background: #f9fafb;
  }}
  .aggregate-name {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 800;
    overflow-wrap: anywhere;
  }}
  .aggregate-meta {{
    color: var(--muted);
    font-size: 12px;
    line-height: 1.45;
    margin-top: 6px;
  }}
  .aggregate-score {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 7px;
  }}
  .aggregate-score-label {{
    color: var(--muted);
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
  }}
  .aggregate-score-value {{
    color: #344054;
    font-size: 13px;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
  }}
  .aggregate-bar {{
    height: 9px;
    border-radius: 999px;
    background: #eef2f6;
    overflow: hidden;
  }}
  .aggregate-bar-fill {{
    height: 100%;
    border-radius: 999px;
  }}
  .case-title {{
    font-weight: 700;
    margin-bottom: 6px;
  }}
  .caption {{
    color: var(--muted);
    font-size: 13px;
    margin-top: 6px;
  }}
  .diff-table {{
    margin-top: 12px;
    font-size: 12px;
  }}
  .diff-table th, .diff-table td {{
    white-space: normal;
    vertical-align: top;
  }}
  .diff-empty {{
    margin-top: 10px;
    color: #16a34a;
    font-size: 13px;
    font-weight: 700;
  }}
  .diff-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
    gap: 14px;
    margin-top: 12px;
  }}
  .diff-title {{
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 6px;
  }}
  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin: 2px 0 14px;
    color: var(--muted);
    font-size: 13px;
  }}
  .legend-item {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }}
  .legend-line {{
    width: 24px;
    height: 3px;
    border-radius: 999px;
    background: currentColor;
  }}
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 10px;
    margin-bottom: 14px;
  }}
  .metric-card {{
    border: 1px solid #eaecf0;
    border-radius: 8px;
    padding: 10px;
    background: #fbfcfe;
  }}
  .metric-label {{
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 4px;
  }}
  .metric-value {{
    font-size: 18px;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
  }}
  .winner {{
    display: inline-flex;
    border-radius: 999px;
    padding: 3px 8px;
    font-size: 12px;
    font-weight: 800;
  }}
  .winner-engine {{ color: #047857; background: #ecfdf3; }}
  .winner-r {{ color: #b42318; background: #fff1f0; }}
  .winner-tie {{ color: #475467; background: #f2f4f7; }}
  .delta-pos {{ color: #047857; font-weight: 800; }}
  .delta-neg {{ color: #b42318; font-weight: 800; }}
  .delta-zero {{ color: #475467; font-weight: 800; }}
  .chart-scroll,
  .table-scroll {{
    width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }}
  .table-scroll {{
    border: 1px solid #eaecf0;
    border-radius: 8px;
  }}
  table {{
    width: 100%;
    min-width: 980px;
    border-collapse: collapse;
    font-size: 13px;
  }}
  .diff-table table {{
    min-width: 760px;
  }}
  th, td {{
    text-align: left;
    border-bottom: 1px solid #eaecf0;
    padding: 8px 10px;
    white-space: nowrap;
  }}
  th {{ color: #475467; font-weight: 700; background: #f9fafb; }}
  .score {{ font-variant-numeric: tabular-nums; }}
  .chip {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    min-width: max-content;
  }}
  .dot {{
    width: 9px;
    height: 9px;
    border-radius: 999px;
    background: currentColor;
  }}
  svg text {{ font-family: inherit; }}
  @media (max-width: 1220px) {{
    main {{ width: min(100vw - 24px, 900px); }}
    .grid, .diff-row, .summary-grid, .edge-explainer-grid {{
      grid-template-columns: 1fr;
    }}
  }}
</style>
</head>
<body>
<header>
  <h1>FCI+ vs R pcalg Oracle Benchmark</h1>
  <p>{len(cases)} hand-written oracle cases. The primary comparison is
  fci_engine FCI+ robust strategy against R pcalg::fciPlus; causal-learn
  remains in the lower score table as context, not as ground truth.</p>
</header>
<main>
  <section>
    <h2>Head-To-Head: fci_engine FCI+ Robust vs R pcalg::fciPlus</h2>
    {render_pcalg_head_to_head(cases, results)}
  </section>
  <section>
    <h2>Aggregate Accuracy Context</h2>
    {render_aggregate_chart(aggregates)}
  </section>
  <section>
    <h2>Per-Case Scores</h2>
    {render_score_table(results)}
  </section>
  <section>
    <h2>True Graph vs FCI+ vs R pcalg</h2>
    {render_graph_gallery(cases, results)}
  </section>
</main>
{render_edge_modal()}
{render_interaction_script()}
</body>
</html>
"""


def render_interaction_script() -> str:
    return """<script>
(function () {
  function setText(root, selector, value) {
    var node = root.querySelector(selector);
    if (node) {
      node.textContent = value || "";
    }
  }

  function fillExplanation(root, edgeNode) {
    var trace = edgeNode.dataset.rules || "No orientation trace";
    var reason = edgeNode.dataset.reason || "";
    if (reason && reason.indexOf("No per-rule reason") !== 0) {
      trace = trace + " - " + reason;
    }
    setText(root, ".edge-explainer-title", "Selected edge explanation");
    setText(root, ".edge-modal-title", "Selected edge explanation");
    setText(root, ".explain-edge", edgeNode.dataset.edge);
    setText(root, ".explain-status", edgeNode.dataset.kind);
    setText(root, ".explain-endpoint-check", edgeNode.dataset.status);
    setText(root, ".explain-expected", edgeNode.dataset.expected);
    setText(root, ".explain-actual", edgeNode.dataset.actual);
    setText(root, ".explain-endpoint-meaning", edgeNode.dataset.endpointMeaning);
    setText(root, ".explain-reasoning", edgeNode.dataset.reasoning);
    setText(root, ".explain-orientation-trace", trace);
    setText(root, ".explain-summary", edgeNode.dataset.explanation);
  }

  function openModal(edgeNode) {
    var modal = document.querySelector(".edge-modal-backdrop");
    if (!modal) {
      return;
    }
    fillExplanation(modal, edgeNode);
    modal.hidden = false;
    var closeButton = modal.querySelector(".edge-modal-close");
    if (closeButton) {
      closeButton.focus();
    }
  }

  function closeModal() {
    var modal = document.querySelector(".edge-modal-backdrop");
    if (modal) {
      modal.hidden = true;
    }
  }

  function explainEdge(edgeNode) {
    var caseCard = edgeNode.closest(".case-card");
    if (!caseCard) {
      return;
    }
    caseCard.querySelectorAll(".graph-edge.is-selected").forEach(function (node) {
      node.classList.remove("is-selected");
    });
    edgeNode.classList.add("is-selected");

    var panel = caseCard.querySelector(".edge-explainer");
    if (!panel) {
      return;
    }
    fillExplanation(panel, edgeNode);
    openModal(edgeNode);
  }

  document.querySelectorAll(".graph-edge").forEach(function (edgeNode) {
    edgeNode.addEventListener("click", function () {
      explainEdge(edgeNode);
    });
    edgeNode.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        explainEdge(edgeNode);
      }
    });
  });

  document.querySelectorAll(".edge-modal-close").forEach(function (button) {
    button.addEventListener("click", closeModal);
  });
  document.querySelectorAll(".edge-modal-backdrop").forEach(function (modal) {
    modal.addEventListener("click", function (event) {
      if (event.target === modal) {
        closeModal();
      }
    });
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      closeModal();
    }
  });
}());
</script>"""


def render_aggregate_chart(aggregates: list[BenchmarkAggregate]) -> str:
    metric_labels = (
        ("Exact F1", "mean_exact_edge_f1"),
        ("Semantic F1", "mean_semantic_edge_f1"),
        ("Skeleton F1", "mean_skeleton_f1"),
        ("Endpoint Acc", "mean_endpoint_accuracy"),
    )
    cells = ["<div class='aggregate-header'>Algorithm</div>"]
    cells.extend(
        f"<div class='aggregate-header'>{_esc(label)}</div>"
        for label, _attribute in metric_labels
    )

    for aggregate in aggregates:
        color = _color(aggregate.algorithm)
        cells.append(
            "<div class='aggregate-cell'>"
            f"<div class='aggregate-name'><span class='dot' style='background:{color}'></span>"
            f"<span>{_esc(aggregate.algorithm)}</span></div>"
            f"{_aggregate_meta(aggregate)}"
            "</div>"
        )
        for label, attribute in metric_labels:
            value = float(getattr(aggregate, attribute))
            cells.append(_aggregate_metric_cell(label, value, color))

    return (
        "<div class='chart-scroll'>"
        "<div class='aggregate-grid' role='table' "
        "aria-label='Aggregate benchmark scores'>"
        f"{''.join(cells)}"
        "</div></div>"
    )


def _aggregate_metric_cell(label: str, value: float, color: str) -> str:
    width = max(0.0, min(100.0, value * 100.0))
    return (
        "<div class='aggregate-cell'>"
        "<div class='aggregate-score'>"
        f"<span class='aggregate-score-label'>{_esc(label)}</span>"
        f"<span class='aggregate-score-value'>{value:.3f}</span>"
        "</div>"
        "<div class='aggregate-bar'>"
        f"<div class='aggregate-bar-fill' style='width:{width:.1f}%; "
        f"background:{color}'></div>"
        "</div>"
        "</div>"
    )


def _aggregate_meta(aggregate: BenchmarkAggregate) -> str:
    parts = []
    if aggregate.mean_ci_test_count is not None:
        parts.append(f"mean CI tests: {aggregate.mean_ci_test_count:.1f}")
    if aggregate.mean_elapsed_time is not None:
        parts.append(f"mean time: {aggregate.mean_elapsed_time:.4f}s")
    if not parts:
        return ""
    return f"<div class='aggregate-meta'>{_esc(' | '.join(parts))}</div>"


def render_pcalg_head_to_head(
    cases: list[OracleCase],
    results: list[BenchmarkResult],
) -> str:
    rows = []
    compared = 0
    engine_wins = 0
    r_wins = 0
    ties = 0
    semantic_deltas = []
    exact_deltas = []

    for case in cases:
        engine = _preferred_engine_result(case, results)
        pcalg = _pcalg_result(case, results)
        if engine is None:
            rows.append(
                _head_to_head_skip_row(case.name, "fci_engine FCI+ result missing")
            )
            continue
        if pcalg is None:
            rows.append(_head_to_head_skip_row(case.name, "R pcalg result missing"))
            continue
        if pcalg.skipped:
            rows.append(
                _head_to_head_skip_row(
                    case.name,
                    f"R pcalg skipped: {pcalg.skipped_reason or 'skipped'}",
                )
            )
            continue
        if engine.skipped:
            rows.append(
                _head_to_head_skip_row(
                    case.name,
                    f"fci_engine skipped: {engine.skipped_reason or 'skipped'}",
                )
            )
            continue

        assert engine.comparison is not None
        assert engine.semantic_comparison is not None
        assert pcalg.comparison is not None
        assert pcalg.semantic_comparison is not None

        compared += 1
        semantic_delta = (
            engine.semantic_comparison.semantic_edge_f1
            - pcalg.semantic_comparison.semantic_edge_f1
        )
        exact_delta = engine.comparison.exact_edge_f1 - pcalg.comparison.exact_edge_f1
        skeleton_delta = engine.comparison.skeleton_f1 - pcalg.comparison.skeleton_f1
        endpoint_delta = (
            engine.comparison.endpoint_accuracy - pcalg.comparison.endpoint_accuracy
        )
        semantic_deltas.append(semantic_delta)
        exact_deltas.append(exact_delta)
        winner = _head_to_head_winner(engine, pcalg)
        if winner == "fci_engine":
            engine_wins += 1
        elif winner == "r_pcalg":
            r_wins += 1
        else:
            ties += 1

        rows.append(
            "<tr>"
            f"<td>{_esc(case.name)}</td>"
            f"<td>{_winner_chip(winner)}</td>"
            f"<td class='score'>{engine.semantic_comparison.semantic_edge_f1:.3f}</td>"
            f"<td class='score'>{pcalg.semantic_comparison.semantic_edge_f1:.3f}</td>"
            f"<td class='score'>{_delta_span(semantic_delta)}</td>"
            f"<td class='score'>{engine.comparison.exact_edge_f1:.3f}</td>"
            f"<td class='score'>{pcalg.comparison.exact_edge_f1:.3f}</td>"
            f"<td class='score'>{_delta_span(exact_delta)}</td>"
            f"<td class='score'>{_delta_span(skeleton_delta)}</td>"
            f"<td class='score'>{_delta_span(endpoint_delta)}</td>"
            "</tr>"
        )

    return (
        f"{_head_to_head_summary(compared, engine_wins, ties, r_wins, semantic_deltas, exact_deltas)}"
        "<div class='table-scroll'><table>"
        "<thead><tr>"
        "<th>Case</th><th>Winner</th>"
        "<th>Our Semantic F1</th><th>R Semantic F1</th><th>Semantic Delta</th>"
        "<th>Our Exact F1</th><th>R Exact F1</th><th>Exact Delta</th>"
        "<th>Skeleton Delta</th><th>Endpoint Delta</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _head_to_head_skip_row(case_name: str, reason: str) -> str:
    return (
        "<tr>"
        f"<td>{_esc(case_name)}</td>"
        "<td><span class='winner winner-tie'>unavailable</span></td>"
        f"<td colspan='8'>{_esc(reason)}</td>"
        "</tr>"
    )


def _head_to_head_summary(
    compared: int,
    engine_wins: int,
    ties: int,
    r_wins: int,
    semantic_deltas: list[float],
    exact_deltas: list[float],
) -> str:
    mean_semantic_delta = _mean(semantic_deltas)
    mean_exact_delta = _mean(exact_deltas)
    return (
        "<div class='summary-grid'>"
        f"{_metric_card('Compared cases', str(compared))}"
        f"{_metric_card('fci_engine wins', str(engine_wins), 'delta-pos')}"
        f"{_metric_card('Ties', str(ties), 'delta-zero')}"
        f"{_metric_card('R pcalg wins', str(r_wins), 'delta-neg')}"
        f"{_metric_card('Mean semantic delta', _signed(mean_semantic_delta), _delta_class(mean_semantic_delta))}"
        f"{_metric_card('Mean exact delta', _signed(mean_exact_delta), _delta_class(mean_exact_delta))}"
        "</div>"
    )


def _metric_card(label: str, value: str, value_class: str = "") -> str:
    return (
        "<div class='metric-card'>"
        f"<div class='metric-label'>{_esc(label)}</div>"
        f"<div class='metric-value {value_class}'>{_esc(value)}</div>"
        "</div>"
    )


def _head_to_head_winner(engine: BenchmarkResult, pcalg: BenchmarkResult) -> str:
    assert engine.comparison is not None
    assert engine.semantic_comparison is not None
    assert pcalg.comparison is not None
    assert pcalg.semantic_comparison is not None
    engine_score = (
        engine.semantic_comparison.semantic_edge_f1,
        engine.comparison.exact_edge_f1,
        engine.comparison.endpoint_accuracy,
        engine.comparison.skeleton_f1,
    )
    pcalg_score = (
        pcalg.semantic_comparison.semantic_edge_f1,
        pcalg.comparison.exact_edge_f1,
        pcalg.comparison.endpoint_accuracy,
        pcalg.comparison.skeleton_f1,
    )
    if all(
        abs(left - right) <= 1e-12 for left, right in zip(engine_score, pcalg_score)
    ):
        return "tie"
    return "fci_engine" if engine_score > pcalg_score else "r_pcalg"


def _winner_chip(winner: str) -> str:
    labels = {
        "fci_engine": ("fci_engine", "winner-engine"),
        "r_pcalg": ("R pcalg", "winner-r"),
        "tie": ("tie", "winner-tie"),
    }
    label, class_name = labels.get(winner, ("unknown", "winner-tie"))
    return f"<span class='winner {class_name}'>{_esc(label)}</span>"


def _delta_span(value: float) -> str:
    return f"<span class='{_delta_class(value)}'>{_signed(value)}</span>"


def _delta_class(value: float) -> str:
    if value > 1e-12:
        return "delta-pos"
    if value < -1e-12:
        return "delta-neg"
    return "delta-zero"


def _signed(value: float) -> str:
    if abs(value) <= 1e-12:
        value = 0.0
    return f"{value:+.3f}"


def render_score_table(results: list[BenchmarkResult]) -> str:
    rows = []
    for result in results:
        if result.skipped:
            rows.append(
                "<tr>"
                f"<td>{_esc(result.case_name)}</td>"
                f"<td>{_algorithm_chip(result.algorithm)}</td>"
                f"<td colspan='8'>{_esc(result.skipped_reason or 'skipped')}</td>"
                "</tr>"
            )
            continue
        assert result.comparison is not None
        assert result.semantic_comparison is not None
        comparison = result.comparison
        semantic = result.semantic_comparison
        rows.append(
            "<tr>"
            f"<td>{_esc(result.case_name)}</td>"
            f"<td>{_algorithm_chip(result.algorithm)}</td>"
            f"<td class='score'>{comparison.exact_edge_f1:.3f}</td>"
            f"<td class='score'>{semantic.semantic_edge_f1:.3f}</td>"
            f"<td class='score'>{comparison.skeleton_f1:.3f}</td>"
            f"<td class='score'>{comparison.endpoint_accuracy:.3f}</td>"
            f"<td class='score'>{semantic.compatible_endpoint_accuracy:.3f}</td>"
            f"<td class='score'>{comparison.false_positive_edges}</td>"
            f"<td class='score'>{comparison.false_negative_edges}</td>"
            f"<td class='score'>{'' if result.ci_test_count is None else result.ci_test_count}</td>"
            "</tr>"
        )
    return (
        "<div class='table-scroll'><table>"
        "<thead><tr><th>Case</th><th>Algorithm</th><th>Exact F1</th>"
        "<th>Semantic F1</th><th>Skeleton F1</th><th>Endpoint Acc</th>"
        "<th>Compatible Endpoint Acc</th><th>FP</th><th>FN</th>"
        "<th>CI Tests</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_graph_gallery(
    cases: list[OracleCase],
    results: list[BenchmarkResult],
) -> str:
    cards = []
    for case in cases:
        learned = _preferred_engine_result(case, results)
        if learned is None:
            continue
        pcalg = _pcalg_result(case, results)
        oracle_svg = render_pag_svg(
            case.oracle_shape,
            list(case.data.columns),
            "Oracle PAG",
            learned.edges,
            "oracle",
            case.name,
            learned.orientation_trace,
        )
        learned_svg = render_pag_svg(
            learned.edges,
            list(case.data.columns),
            learned.algorithm,
            case.oracle_shape,
            "learned",
            case.name,
            learned.orientation_trace,
        )
        pcalg_svg = render_result_svg(case, pcalg, "R pcalg::fciPlus")
        cards.append(
            "<div class='case-card'>"
            f"<div class='case-title'>{_esc(case.name)}</div>"
            f"<p>{_esc(case.notes)}</p>"
            "<div class='graph-strip' style='margin-top:14px'>"
            "<div class='graph-row'>"
            "<div class='graph-panel'>"
            f"{oracle_svg}"
            "<div class='caption'>Hand-written expected PAG</div>"
            "</div>"
            "<div class='graph-panel'>"
            f"{learned_svg}"
            f"<div class='caption'>Learned output: {_esc(learned.algorithm)}</div>"
            "</div>"
            "<div class='graph-panel'>"
            f"{pcalg_svg}"
            f"<div class='caption'>{render_result_caption(pcalg)}</div>"
            "</div>"
            "</div>"
            "</div>"
            f"{render_edge_explainer(case.name)}"
            "<div class='diff-row'>"
            f"{render_difference_table(case, learned, 'FCI+ edge differences')}"
            f"{render_difference_table(case, pcalg, 'R pcalg edge differences')}"
            "</div>"
            "</div>"
        )
    legend = (
        "<div class='legend'>"
        "<span class='legend-item' style='color:#16a34a'><span class='legend-line'></span>exact match</span>"
        "<span class='legend-item' style='color:#d97706'><span class='legend-line'></span>endpoint differs</span>"
        "<span class='legend-item' style='color:#dc2626'><span class='legend-line'></span>missing or extra edge</span>"
        "<span class='legend-item' style='color:#475467'><span class='legend-line'></span>not compared</span>"
        "<span class='legend-item'>Click any edge to inspect expected endpoints, actual endpoints, and rule trace.</span>"
        "</div>"
    )
    return legend + "<div class='graph-gallery'>" + "".join(cards) + "</div>"


def render_result_svg(
    case: OracleCase,
    result: Optional[BenchmarkResult],
    title: str,
) -> str:
    if result is None:
        return render_placeholder_svg(title, "No pcalg result was produced.")
    if result.skipped:
        return render_placeholder_svg(title, result.skipped_reason or "Skipped")
    return render_pag_svg(
        result.edges,
        list(case.data.columns),
        result.algorithm,
        case.oracle_shape,
        "learned",
        case.name,
        result.orientation_trace,
    )


def render_edge_explainer(case_name: str) -> str:
    return (
        f"<div class='edge-explainer' data-edge-panel='{_esc(case_name)}'>"
        "<div class='edge-explainer-title'>Click an edge to explain this result.</div>"
        "<div class='edge-explainer-grid'>"
        f"{_explainer_item('Edge', 'Select an edge')}"
        f"{_explainer_item('Status', 'No edge selected')}"
        f"{_explainer_item('Endpoint check', 'No edge selected')}"
        f"{_explainer_item('Expected', 'No edge selected')}"
        f"{_explainer_item('Actual', 'No edge selected')}"
        f"{_explainer_item('Endpoint meaning', 'No edge selected', wide=True)}"
        f"{_explainer_item('Reasoning', 'No edge selected', wide=True)}"
        f"{_explainer_item('Orientation trace', 'No edge selected')}"
        "</div>"
        "<div class='caption explain-summary'>The panel explains the selected edge "
        "relative to the oracle PAG used by this benchmark case.</div>"
        "</div>"
    )


def render_edge_modal() -> str:
    return (
        "<div class='edge-modal-backdrop' hidden>"
        "<div class='edge-modal' role='dialog' aria-modal='true' "
        "aria-labelledby='edge-modal-title'>"
        "<div class='edge-modal-header'>"
        "<div id='edge-modal-title' class='edge-modal-title'>"
        "Selected edge explanation</div>"
        "<button type='button' class='edge-modal-close'>Close</button>"
        "</div>"
        "<div class='edge-explainer-grid'>"
        f"{_explainer_item('Edge', 'Select an edge')}"
        f"{_explainer_item('Status', 'No edge selected')}"
        f"{_explainer_item('Endpoint check', 'No edge selected')}"
        f"{_explainer_item('Expected', 'No edge selected')}"
        f"{_explainer_item('Actual', 'No edge selected')}"
        f"{_explainer_item('Endpoint meaning', 'No edge selected', wide=True)}"
        f"{_explainer_item('Reasoning', 'No edge selected', wide=True)}"
        f"{_explainer_item('Orientation trace', 'No edge selected', wide=True)}"
        "</div>"
        "<div class='caption explain-summary'>Select an edge to load the "
        "explanation.</div>"
        "</div>"
        "</div>"
    )


def _explainer_item(label: str, value: str, wide: bool = False) -> str:
    class_name = "explain-" + label.lower().replace(" ", "-")
    wide_class = " wide" if wide else ""
    return (
        f"<div class='edge-explainer-item{wide_class}'>"
        f"<div class='edge-explainer-label'>{_esc(label)}</div>"
        f"<div class='edge-explainer-value {class_name}'>{_esc(value)}</div>"
        "</div>"
    )


def render_result_caption(result: Optional[BenchmarkResult]) -> str:
    if result is None:
        return "R pcalg output unavailable"
    if result.skipped:
        return f"R pcalg skipped: {_esc(result.skipped_reason or 'skipped')}"
    return f"R package output: {_esc(result.algorithm)}"


def render_placeholder_svg(
    title: str,
    message: str,
    width: int = 720,
    height: int = 520,
) -> str:
    return (
        f'<svg class="graph-svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{_esc(title)} unavailable">'
        '<rect width="100%" height="100%" rx="8" fill="#ffffff" stroke="#d0d5dd"/>'
        f'<text x="16" y="26" font-size="15" font-weight="800" '
        f'fill="#101827">{_esc(title)}</text>'
        f'<text x="{width / 2:.1f}" y="{height / 2:.1f}" text-anchor="middle" '
        'font-size="13" fill="#667085">'
        f"{_esc(message)}</text>"
        "</svg>"
    )


def render_difference_table(
    case: OracleCase,
    learned: Optional[BenchmarkResult],
    title: str,
) -> str:
    if learned is None:
        return (
            "<div>"
            f"<div class='diff-title'>{_esc(title)}</div>"
            "<div class='caption'>No result was produced.</div>"
            "</div>"
        )
    if learned.skipped:
        return (
            "<div>"
            f"<div class='diff-title'>{_esc(title)}</div>"
            f"<div class='caption'>{_esc(learned.skipped_reason or 'Skipped')}</div>"
            "</div>"
        )
    differences = explain_pag_differences(
        case.oracle_shape,
        learned.edges,
        learned.orientation_trace,
    )
    if not differences:
        return (
            "<div>"
            f"<div class='diff-title'>{_esc(title)}</div>"
            "<div class='diff-empty'>No edge-level differences for this output.</div>"
            "</div>"
        )

    rows = []
    for diff in differences[:12]:
        rules = ", ".join(
            sorted({str(event.get("rule", "")) for event in diff.orientation_events})
        )
        if not rules:
            rules = "-"
        rows.append(
            "<tr>"
            f"<td>{_esc(diff.kind)}</td>"
            f"<td>{_esc(diff.edge[0])}-{_esc(diff.edge[1])}</td>"
            f"<td>{_esc(_edge_text(diff.edge, diff.expected))}</td>"
            f"<td>{_esc(_edge_text(diff.edge, diff.actual))}</td>"
            f"<td>{_esc('/'.join(diff.endpoint_status))}</td>"
            f"<td>{_esc(rules)}</td>"
            "</tr>"
        )
    more = (
        ""
        if len(differences) <= 12
        else f"<p class='caption'>+{len(differences) - 12} more differences omitted.</p>"
    )
    return (
        "<div class='diff-table'>"
        f"<div class='diff-title'>{_esc(title)}</div>"
        "<div class='table-scroll'><table><thead><tr><th>Kind</th><th>Edge</th><th>Expected</th>"
        "<th>Actual</th><th>Endpoint Status</th><th>Orientation Rules</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>{more}</div>"
    )


def render_pag_svg(
    shape: Shape,
    nodes: list[str],
    title: str,
    reference_shape: Optional[Shape] = None,
    comparison_side: str = "learned",
    case_name: str = "",
    orientation_trace: Optional[list[object]] = None,
    width: int = 720,
    height: int = 520,
) -> str:
    positions = _circle_layout(nodes, width, height)
    normalized_shape = _normalized_shape(shape)
    normalized_reference = (
        _normalized_shape(reference_shape) if reference_shape is not None else None
    )
    edge_parts = []
    for (x, y), endpoints in sorted(shape.items()):
        edge = (str(x), str(y))
        endpoint_x = _endpoint_name(endpoints[0])
        endpoint_y = _endpoint_name(endpoints[1])
        status = _edge_status(
            edge,
            normalized_shape,
            normalized_reference,
            comparison_side=comparison_side,
        )
        metadata = _edge_metadata(
            edge,
            normalized_shape,
            normalized_reference,
            comparison_side,
            orientation_trace,
        )
        stroke, stroke_width, dasharray = _edge_style(status)
        x1, y1 = positions[edge[0]]
        x2, y2 = positions[edge[1]]
        start, end = _trimmed_line((x1, y1), (x2, y2), 48)
        edge_parts.append(
            "<g class='graph-edge' tabindex='0' role='button' "
            f"aria-label='{_esc(metadata['aria_label'])}' "
            f"data-case='{_esc(case_name)}' "
            f"data-edge='{_esc(metadata['edge'])}' "
            f"data-kind='{_esc(metadata['kind'])}' "
            f"data-status='{_esc(metadata['endpoint_status'])}' "
            f"data-expected='{_esc(metadata['expected'])}' "
            f"data-actual='{_esc(metadata['actual'])}' "
            f"data-rules='{_esc(metadata['rules'])}' "
            f"data-explanation='{_esc(metadata['explanation'])}' "
            f"data-endpoint-meaning='{_esc(metadata['endpoint_meaning'])}' "
            f"data-reasoning='{_esc(metadata['reasoning'])}' "
            f"data-reason='{_esc(metadata['reason'])}'>"
            f"<title>{_esc(metadata['aria_label'])}</title>"
            f'<line class="edge-hit" x1="{start[0]:.1f}" y1="{start[1]:.1f}" '
            f'x2="{end[0]:.1f}" y2="{end[1]:.1f}"/>'
            f'<line class="edge-line" x1="{start[0]:.1f}" y1="{start[1]:.1f}" '
            f'x2="{end[0]:.1f}" y2="{end[1]:.1f}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"{dasharray}/>'
            f"{_endpoint_svg(endpoint_x, start, end, stroke)}"
            f"{_endpoint_svg(endpoint_y, end, start, stroke)}"
            "</g>"
        )

    node_parts = []
    for node in nodes:
        x, y = positions[node]
        node_parts.append(_node_svg(node, x, y))

    return (
        f'<svg class="graph-svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{_esc(title)}">'
        '<rect width="100%" height="100%" rx="8" fill="#ffffff" stroke="#d0d5dd"/>'
        f'<text x="16" y="26" font-size="15" font-weight="800" '
        f'fill="#101827">{_esc(title)}</text>'
        f"{''.join(edge_parts)}"
        f"{''.join(node_parts)}"
        "</svg>"
    )


def _edge_metadata(
    edge: tuple[str, str],
    shape: dict[tuple[str, str], tuple[str, str]],
    reference_shape: Optional[dict[tuple[str, str], tuple[str, str]]],
    comparison_side: str,
    orientation_trace: Optional[list[object]],
) -> dict[str, str]:
    displayed = shape[edge]
    if reference_shape is None:
        expected = None
        actual = displayed
        kind = "not_compared"
        endpoint_status = "not compared"
    elif comparison_side == "oracle":
        expected = displayed
        actual = reference_shape.get(edge)
        if actual is None:
            kind = "missing_edge"
            endpoint_status = "missing/missing"
        elif actual == expected:
            kind = "exact_match"
            endpoint_status = "exact/exact"
        else:
            endpoint_status_tuple = _endpoint_statuses(expected, actual)
            kind = _difference_kind(endpoint_status_tuple)
            endpoint_status = "/".join(endpoint_status_tuple)
    else:
        expected = reference_shape.get(edge)
        actual = displayed
        if expected is None:
            kind = "extra_edge"
            endpoint_status = "extra/extra"
        elif actual == expected:
            kind = "exact_match"
            endpoint_status = "exact/exact"
        else:
            endpoint_status_tuple = _endpoint_statuses(expected, actual)
            kind = _difference_kind(endpoint_status_tuple)
            endpoint_status = "/".join(endpoint_status_tuple)

    events = _trace_events_for_edge(orientation_trace, edge)
    rules = _event_rule_text(events)
    reason = _event_reason_text(events)
    explanation = _edge_explanation(kind, rules)
    endpoint_meaning = _endpoint_meaning(edge, displayed)
    reasoning = _edge_reasoning(
        edge,
        displayed,
        expected,
        actual,
        kind,
        endpoint_status,
        rules,
        reason,
        comparison_side,
    )
    edge_text = _edge_text(edge, displayed)
    return {
        "edge": edge_text,
        "kind": kind,
        "endpoint_status": endpoint_status,
        "expected": _edge_text(edge, expected),
        "actual": _edge_text(edge, actual),
        "rules": rules,
        "reason": reason,
        "explanation": explanation,
        "endpoint_meaning": endpoint_meaning,
        "reasoning": reasoning,
        "aria_label": f"{edge_text}. {kind}. {explanation}",
    }


def _endpoint_statuses(
    expected: tuple[str, str],
    actual: tuple[str, str],
) -> tuple[str, str]:
    return (
        _endpoint_semantic_status(expected[0], actual[0]),
        _endpoint_semantic_status(expected[1], actual[1]),
    )


def _endpoint_semantic_status(expected: str, actual: str) -> str:
    if expected == actual:
        return "exact"
    if expected == "CIRCLE" and actual in {"ARROW", "TAIL"}:
        return "over_oriented"
    if actual == "CIRCLE" and expected in {"ARROW", "TAIL"}:
        return "under_oriented"
    return "contradicted"


def _difference_kind(statuses: tuple[str, str]) -> str:
    if "contradicted" in statuses:
        return "endpoint_conflict"
    if "over_oriented" in statuses and "under_oriented" in statuses:
        return "mixed_endpoint_difference"
    if "over_oriented" in statuses:
        return "over_oriented"
    if "under_oriented" in statuses:
        return "under_oriented"
    return "exact_match"


def _edge_explanation(kind: str, rules: str) -> str:
    explanations = {
        "not_compared": "No oracle/reference comparison is attached to this panel.",
        "exact_match": "The edge and both endpoint marks match the oracle/reference PAG.",
        "missing_edge": "This oracle edge is absent from the compared learned output.",
        "extra_edge": "This learned edge is not present in the oracle PAG.",
        "under_oriented": (
            "The learned edge keeps at least one circle where the oracle has a "
            "definite tail or arrowhead."
        ),
        "over_oriented": (
            "The learned edge resolves at least one oracle circle into a definite "
            "endpoint; this is stronger than the oracle endpoint."
        ),
        "mixed_endpoint_difference": (
            "One endpoint is stronger than the oracle and another endpoint is "
            "weaker than the oracle."
        ),
        "endpoint_conflict": "At least one endpoint contradicts the oracle endpoint.",
    }
    explanation = explanations.get(kind, "This edge differs from the oracle/reference.")
    if rules and rules != "No orientation trace":
        return f"{explanation} Recorded rule evidence: {rules}."
    return explanation


def _endpoint_meaning(edge: tuple[str, str], endpoints: tuple[str, str]) -> str:
    x, y = edge
    relation = _relation_meaning(edge, endpoints)
    endpoint_parts = [
        _single_endpoint_meaning(x, y, endpoints[0]),
        _single_endpoint_meaning(y, x, endpoints[1]),
    ]
    return relation + " " + " ".join(endpoint_parts)


def _relation_meaning(edge: tuple[str, str], endpoints: tuple[str, str]) -> str:
    x, y = edge
    if endpoints == ("CIRCLE", "CIRCLE"):
        return (
            f"{x} o-o {y} means FCI kept an adjacency but did not identify either "
            "ancestral direction from the available conditional independence "
            "constraints."
        )
    if endpoints == ("CIRCLE", "ARROW"):
        return (
            f"{x} o-> {y} means the endpoint at {y} is fixed as an arrowhead, "
            f"so the PAG rules rule out {y} being an ancestor of {x}; the "
            f"endpoint at {x} remains unidentified."
        )
    if endpoints == ("ARROW", "CIRCLE"):
        return (
            f"{x} <-o {y} means the endpoint at {x} is fixed as an arrowhead, "
            f"so the PAG rules rule out {x} being an ancestor of {y}; the "
            f"endpoint at {y} remains unidentified."
        )
    if endpoints == ("TAIL", "ARROW"):
        return (
            f"{x} --> {y} means the learned PAG supports {x} as an ancestor or "
            f"direct cause candidate of {y}, and rules out {y} as an ancestor "
            f"of {x}."
        )
    if endpoints == ("ARROW", "TAIL"):
        return (
            f"{x} <-- {y} means the learned PAG supports {y} as an ancestor or "
            f"direct cause candidate of {x}, and rules out {x} as an ancestor "
            f"of {y}."
        )
    if endpoints == ("ARROW", "ARROW"):
        return (
            f"{x} <-> {y} means both endpoints are arrowheads. In a PAG this "
            "usually represents dependence compatible with latent confounding "
            "or an equivalent ancestral-graph constraint, rather than a direct "
            "observed arrow in both directions."
        )
    if endpoints == ("TAIL", "TAIL"):
        return (
            f"{x} --- {y} is an undirected tail-tail edge. In PAG terminology "
            "this is typically associated with selection-bias style constraints "
            "or background knowledge."
        )
    return (
        f"{_edge_text(edge, endpoints)} is a mixed PAG endpoint pattern; read "
        "each endpoint independently."
    )


def _single_endpoint_meaning(node: str, other: str, endpoint: str) -> str:
    if endpoint == "ARROW":
        return (
            f"The arrowhead at {node} says {node} is not an ancestor of {other} "
            "in every graph represented by this PAG."
        )
    if endpoint == "TAIL":
        return (
            f"The tail at {node} points the edge out of {node}, supporting "
            f"{node} as an ancestor or cause candidate of {other}."
        )
    if endpoint == "CIRCLE":
        return (
            f"The circle at {node} is deliberately unresolved: the data and "
            "orientation rules did not justify replacing it with a tail or "
            "arrowhead."
        )
    return f"The endpoint at {node} is not active on this edge."


def _edge_reasoning(
    edge: tuple[str, str],
    displayed: tuple[str, str],
    expected: Optional[tuple[str, str]],
    actual: Optional[tuple[str, str]],
    kind: str,
    endpoint_status: str,
    rules: str,
    reason: str,
    comparison_side: str,
) -> str:
    edge_text = _edge_text(edge, displayed)
    if comparison_side == "oracle":
        panel_text = (
            f"This click is on the oracle edge {edge_text}. The learned/reference "
            f"output for the same pair is {_edge_text(edge, actual)}."
        )
    else:
        panel_text = (
            f"This click is on the learned edge {edge_text}. The oracle/reference "
            f"edge for the same pair is {_edge_text(edge, expected)}."
        )

    comparison_text = _comparison_reasoning(kind, endpoint_status)
    rule_text = _rule_reasoning(rules, reason, comparison_side)
    return " ".join([panel_text, comparison_text, rule_text])


def _comparison_reasoning(kind: str, endpoint_status: str) -> str:
    if kind == "exact_match":
        return (
            "The benchmark marks this edge as an exact match because both "
            "endpoint symbols agree with the oracle/reference edge."
        )
    if kind == "missing_edge":
        return (
            "The benchmark marks this as missing because the oracle has the "
            "edge, but the compared output did not retain an adjacency for "
            "this pair."
        )
    if kind == "extra_edge":
        return (
            "The benchmark marks this as extra because the compared output "
            "retained an adjacency that is absent from the oracle PAG."
        )
    if kind == "under_oriented":
        return (
            f"The endpoint check is {endpoint_status}: at least one endpoint "
            "is still a circle even though the oracle/reference has a definite "
            "tail or arrowhead. This is conservative but less informative."
        )
    if kind == "over_oriented":
        return (
            f"The endpoint check is {endpoint_status}: at least one oracle "
            "circle was replaced by a definite endpoint. This is more specific "
            "than the oracle and should be inspected carefully."
        )
    if kind == "mixed_endpoint_difference":
        return (
            f"The endpoint check is {endpoint_status}: one endpoint is more "
            "specific than the oracle while another is less specific."
        )
    if kind == "endpoint_conflict":
        return (
            f"The endpoint check is {endpoint_status}: at least one endpoint "
            "points to a different ancestral claim than the oracle/reference."
        )
    return "This edge is not compared against an oracle/reference edge."


def _rule_reasoning(rules: str, reason: str, comparison_side: str) -> str:
    if not rules or rules == "No orientation trace":
        if comparison_side == "learned":
            return (
                "No internal orientation event is attached to this edge in the "
                "report. For fci_engine this usually means the edge survived "
                "skeleton/D-SEP search but its endpoints were not changed by a "
                "recorded orientation rule; for external R pcalg output, the "
                "rule-level trace is not exposed to this report."
            )
        return (
            "The oracle panel is hand-written for the benchmark, so it explains "
            "the target endpoint meaning rather than an algorithmic derivation."
        )

    rule_names = [rule.strip() for rule in rules.split(",") if rule.strip()]
    rule_explanations = [_single_rule_reasoning(rule) for rule in rule_names]
    unique_explanations = []
    for explanation in rule_explanations:
        if explanation not in unique_explanations:
            unique_explanations.append(explanation)
    text = " ".join(unique_explanations)
    if reason and not reason.startswith("No per-rule reason"):
        text += f" The concrete recorded trigger was: {reason}."
    return text


def _single_rule_reasoning(rule: str) -> str:
    if rule.startswith("orient_unshielded_colliders"):
        return (
            "The collider rule placed arrowheads into the middle node of an "
            "unshielded triple because the separating-set evidence did not "
            "support treating that middle node as a non-collider."
        )
    if rule == "R1":
        return (
            "Rule R1 oriented a tail to avoid introducing a new unshielded "
            "collider that would contradict the recorded separating sets."
        )
    if rule == "R2":
        return (
            "Rule R2 propagated ancestry along an existing directed or partially "
            "directed path, so the endpoint had to agree with that ancestral "
            "constraint."
        )
    if rule == "R3":
        return (
            "Rule R3 used a triangle-style PAG pattern to orient an endpoint "
            "that was forced by two adjacent constraints."
        )
    if rule == "R4":
        return (
            "Rule R4 used a discriminating path: the surrounding path structure "
            "made the collider/non-collider status of a triple identifiable."
        )
    if rule in {"R5", "R6", "R7"}:
        return (
            f"Rule {rule} is one of the selection-bias orientation rules; it "
            "propagates tail information through uncovered circle-path patterns."
        )
    if rule in {"R8", "R9", "R10"}:
        return (
            f"Rule {rule} prevents an orientation that would create an invalid "
            "ancestral cycle or conflict with an uncovered potentially directed "
            "path."
        )
    return f"{rule} changed an endpoint according to the PAG orientation rules."


def _trace_events_for_edge(
    orientation_trace: Optional[list[object]],
    edge: tuple[str, str],
) -> list[object]:
    if not orientation_trace:
        return []
    edge_key = frozenset(edge)
    return [
        event
        for event in orientation_trace
        if frozenset(str(node) for node in _event_value(event, "edge", ())) == edge_key
    ]


def _event_rule_text(events: list[object]) -> str:
    rules = sorted(
        {
            str(_event_value(event, "rule", "")).strip()
            for event in events
            if str(_event_value(event, "rule", "")).strip()
        }
    )
    if not rules:
        return "No orientation trace"
    return ", ".join(rules)


def _event_reason_text(events: list[object]) -> str:
    reasons = []
    for event in events:
        reason = str(_event_value(event, "reason", "")).strip()
        if reason and reason not in reasons:
            reasons.append(reason)
        if len(reasons) >= 3:
            break
    if not reasons:
        return "No per-rule reason was recorded for this edge."
    return " | ".join(reasons)


def _event_value(event: object, name: str, default: object) -> object:
    if isinstance(event, dict):
        return event.get(name, default)
    return getattr(event, name, default)


def _preferred_engine_result(
    case: OracleCase,
    results: list[BenchmarkResult],
) -> Optional[BenchmarkResult]:
    preferred = (
        "fci_engine.fci_plus.kernel.robust"
        if case.use_kernel_ci
        else "fci_engine.fci_plus.robust"
    )
    for result in results:
        if result.case_name == case.name and result.algorithm == preferred:
            return result
    fallback = (
        "fci_engine.fci_plus.kernel" if case.use_kernel_ci else "fci_engine.fci_plus"
    )
    for result in results:
        if result.case_name == case.name and result.algorithm == fallback:
            return result
    return None


def _pcalg_result(
    case: OracleCase,
    results: list[BenchmarkResult],
) -> Optional[BenchmarkResult]:
    for result in results:
        if result.case_name == case.name and result.algorithm == "pcalg.fciPlus":
            return result
    return None


def _circle_layout(
    nodes: list[str],
    width: int,
    height: int,
) -> dict[str, tuple[float, float]]:
    center_x = width / 2
    center_y = height / 2 + 20
    radius = min(width, height) * 0.38
    positions = {}
    for index, node in enumerate(nodes):
        angle = -math.pi / 2 + 2 * math.pi * index / max(len(nodes), 1)
        positions[node] = (
            center_x + radius * math.cos(angle),
            center_y + radius * math.sin(angle),
        )
    return positions


def _node_svg(node: str, x: float, y: float) -> str:
    lines = _label_lines(str(node))
    max_len = max(len(line) for line in lines)
    box_width = min(122, max(70, 8 * max_len + 20))
    box_height = 26 + 14 * len(lines)
    top = y - box_height / 2
    left = x - box_width / 2
    text_start = y - (len(lines) - 1) * 7 + 4
    text_lines = []
    for index, line in enumerate(lines):
        text_lines.append(
            f'<tspan x="{x:.1f}" y="{text_start + index * 14:.1f}">'
            f"{_esc(line)}</tspan>"
        )
    return (
        f"<g><title>{_esc(node)}</title>"
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{box_width:.1f}" '
        f'height="{box_height:.1f}" rx="7" fill="#f8fafc" '
        'stroke="#344054" stroke-width="1.4"/>'
        f'<text x="{x:.1f}" y="{text_start:.1f}" text-anchor="middle" '
        'font-size="11" font-weight="700" fill="#101827">'
        f"{''.join(text_lines)}</text></g>"
    )


def _label_lines(label: str) -> list[str]:
    words = _split_label_words(label)
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= 13:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) == 1:
            break
    if current and len(lines) < 2:
        lines.append(current)
    if not lines:
        lines = [label]
    if len(words) > 0 and " ".join(words) != " ".join(lines):
        lines[-1] = _ellipsis(lines[-1], 13)
    return [_ellipsis(line, 13) for line in lines[:2]]


def _split_label_words(label: str) -> list[str]:
    normalized = label.replace("_", " ").replace("-", " ")
    words: list[str] = []
    for raw_word in normalized.split():
        current = ""
        for index, char in enumerate(raw_word):
            if (
                index > 0
                and char.isupper()
                and (raw_word[index - 1].islower() or raw_word[index - 1].isdigit())
            ):
                if current:
                    words.append(current)
                current = char
            else:
                current += char
        if current:
            words.append(current)
    return words or [label]


def _ellipsis(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def _trimmed_line(
    start: tuple[float, float],
    end: tuple[float, float],
    radius: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = math.hypot(dx, dy)
    if distance == 0:
        return start, end
    ux = dx / distance
    uy = dy / distance
    return (
        (start[0] + ux * radius, start[1] + uy * radius),
        (end[0] - ux * radius, end[1] - uy * radius),
    )


def _endpoint_svg(
    endpoint: str,
    point: tuple[float, float],
    toward: tuple[float, float],
    color: str = "#475467",
) -> str:
    dx = point[0] - toward[0]
    dy = point[1] - toward[1]
    distance = math.hypot(dx, dy)
    if distance == 0:
        return ""
    ux = dx / distance
    uy = dy / distance
    px = -uy
    py = ux
    if endpoint == "CIRCLE":
        return (
            f'<circle class="edge-endpoint" cx="{point[0]:.1f}" cy="{point[1]:.1f}" r="5" '
            f'fill="#ffffff" stroke="{color}" stroke-width="1.8"/>'
        )
    if endpoint == "ARROW":
        tip = point
        base_x = point[0] - ux * 13
        base_y = point[1] - uy * 13
        p1 = (base_x + px * 5, base_y + py * 5)
        p2 = (base_x - px * 5, base_y - py * 5)
        return (
            f'<polygon class="edge-endpoint" points="{tip[0]:.1f},{tip[1]:.1f} '
            f'{p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" '
            f'fill="{color}"/>'
        )
    if endpoint == "TAIL":
        p1 = (point[0] + px * 7, point[1] + py * 7)
        p2 = (point[0] - px * 7, point[1] - py * 7)
        return (
            f'<line class="edge-endpoint" x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" '
            f'x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
            f'stroke="{color}" stroke-width="2.4"/>'
        )
    return ""


def _normalized_shape(
    shape: Optional[Shape],
) -> dict[tuple[str, str], tuple[str, str]]:
    if shape is None:
        return {}
    normalized = {}
    for (x, y), endpoints in shape.items():
        normalized[(str(x), str(y))] = (
            _endpoint_name(endpoints[0]),
            _endpoint_name(endpoints[1]),
        )
    return normalized


def _edge_status(
    edge: tuple[str, str],
    shape: dict[tuple[str, str], tuple[str, str]],
    reference_shape: Optional[dict[tuple[str, str], tuple[str, str]]],
    comparison_side: str,
) -> str:
    if reference_shape is None:
        return "neutral"
    if edge not in reference_shape:
        return "missing" if comparison_side == "oracle" else "extra"
    if shape[edge] == reference_shape[edge]:
        return "match"
    return "endpoint_diff"


def _edge_style(status: str) -> tuple[str, str, str]:
    styles = {
        "match": ("#16a34a", "2.2", ""),
        "endpoint_diff": ("#d97706", "2.6", ""),
        "missing": ("#dc2626", "2.8", ' stroke-dasharray="7 5"'),
        "extra": ("#dc2626", "2.8", ' stroke-dasharray="7 5"'),
        "neutral": ("#475467", "1.8", ""),
    }
    return styles.get(status, styles["neutral"])


def _algorithm_chip(algorithm: str) -> str:
    color = _color(algorithm)
    return (
        f'<span class="chip" style="color:{color}">'
        '<span class="dot"></span>'
        f'<span style="color:#101827">{_esc(algorithm)}</span>'
        "</span>"
    )


def _color(algorithm: str) -> str:
    return COLORS.get(algorithm, "#475467")


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _endpoint_name(endpoint: object) -> str:
    return getattr(endpoint, "name", str(endpoint)).upper()


def _edge_text(
    edge: tuple[str, str],
    endpoints: Optional[tuple[str, str]],
) -> str:
    if endpoints is None:
        return "missing"
    return f"{edge[0]} {_edge_mark(endpoints)} {edge[1]}"


def _edge_mark(endpoints: tuple[str, str]) -> str:
    left, right = endpoints
    left_marks = {
        "ARROW": "<",
        "TAIL": "-",
        "CIRCLE": "o",
        "NONE": " ",
    }
    right_marks = {
        "ARROW": ">",
        "TAIL": "-",
        "CIRCLE": "o",
        "NONE": " ",
    }
    return f"{left_marks.get(left, '?')}-{right_marks.get(right, '?')}"


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


if __name__ == "__main__":
    main()
