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
from fci_engine.simulation import OracleCase, default_oracle_cases, realistic_oracle_cases


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
    selected_cases = _selected_cases(cases)
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
  .chart-scroll svg {{
    min-width: 1120px;
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
    .grid, .diff-row, .summary-grid {{ grid-template-columns: 1fr; }}
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
    {render_graph_gallery(selected_cases, results)}
  </section>
</main>
</body>
</html>
"""


def render_aggregate_chart(aggregates: list[BenchmarkAggregate]) -> str:
    row_height = 74
    width = 1120
    height = 36 + row_height * len(aggregates)
    metric_width = 210
    label_width = 250
    svg = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        'role="img" aria-label="Aggregate benchmark scores">'
    ]
    svg.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    svg.append('<text x="10" y="22" font-size="13" fill="#667085">Algorithm</text>')
    for index, label in enumerate(
        ("Exact F1", "Semantic F1", "Skeleton F1", "Endpoint Acc")
    ):
        x = label_width + index * metric_width
        svg.append(f'<text x="{x}" y="22" font-size="13" fill="#667085">{label}</text>')

    for row, aggregate in enumerate(aggregates):
        y = 42 + row * row_height
        color = _color(aggregate.algorithm)
        svg.append(f'<line x1="0" y1="{y - 14}" x2="{width}" y2="{y - 14}" stroke="#eaecf0"/>')
        svg.append(f'<circle cx="16" cy="{y + 14}" r="5" fill="{color}"/>')
        svg.append(
            f'<text x="30" y="{y + 18}" font-size="14" font-weight="700" '
            f'fill="#101827">{_esc(aggregate.algorithm)}</text>'
        )
        for index, value in enumerate(
            (
                aggregate.mean_exact_edge_f1,
                aggregate.mean_semantic_edge_f1,
                aggregate.mean_skeleton_f1,
                aggregate.mean_endpoint_accuracy,
            )
        ):
            x = label_width + index * metric_width
            bar_width = 170
            filled = bar_width * value
            svg.append(
                f'<rect x="{x}" y="{y}" width="{bar_width}" height="16" '
                'rx="4" fill="#eef2f6"/>'
            )
            svg.append(
                f'<rect x="{x}" y="{y}" width="{filled:.2f}" height="16" '
                f'rx="4" fill="{color}"/>'
            )
            svg.append(
                f'<text x="{x + bar_width + 10}" y="{y + 13}" font-size="13" '
                f'fill="#344054">{value:.3f}</text>'
            )
        if aggregate.mean_ci_test_count is not None:
            svg.append(
                f'<text x="{label_width}" y="{y + 42}" font-size="12" '
                f'fill="#667085">mean CI tests: {aggregate.mean_ci_test_count:.1f}</text>'
            )
        if aggregate.mean_elapsed_time is not None:
            svg.append(
                f'<text x="{label_width + 190}" y="{y + 42}" font-size="12" '
                f'fill="#667085">mean time: {aggregate.mean_elapsed_time:.4f}s</text>'
            )
    svg.append("</svg>")
    return "<div class='chart-scroll'>" + "\n".join(svg) + "</div>"


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
        exact_delta = (
            engine.comparison.exact_edge_f1 - pcalg.comparison.exact_edge_f1
        )
        skeleton_delta = (
            engine.comparison.skeleton_f1 - pcalg.comparison.skeleton_f1
        )
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
        "<th>Our Semantic F1</th><th>R Semantic F1</th><th>Semantic Δ</th>"
        "<th>Our Exact F1</th><th>R Exact F1</th><th>Exact Δ</th>"
        "<th>Skeleton Δ</th><th>Endpoint Δ</th>"
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
        f"{_metric_card('Mean semantic Δ', _signed(mean_semantic_delta), _delta_class(mean_semantic_delta))}"
        f"{_metric_card('Mean exact Δ', _signed(mean_exact_delta), _delta_class(mean_exact_delta))}"
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
    if all(abs(left - right) <= 1e-12 for left, right in zip(engine_score, pcalg_score)):
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
        cards.append(
            "<div class='case-card'>"
            f"<div class='case-title'>{_esc(case.name)}</div>"
            f"<p>{_esc(case.notes)}</p>"
            "<div class='graph-strip' style='margin-top:14px'>"
            "<div class='graph-row'>"
            "<div class='graph-panel'>"
            f"{render_pag_svg(case.oracle_shape, list(case.data.columns), 'Oracle PAG', learned.edges, 'oracle')}"
            "<div class='caption'>Hand-written expected PAG</div>"
            "</div>"
            "<div class='graph-panel'>"
            f"{render_pag_svg(learned.edges, list(case.data.columns), learned.algorithm, case.oracle_shape, 'learned')}"
            f"<div class='caption'>Learned output: {_esc(learned.algorithm)}</div>"
            "</div>"
            "<div class='graph-panel'>"
            f"{render_result_svg(case, pcalg, 'R pcalg::fciPlus')}"
            f"<div class='caption'>{render_result_caption(pcalg)}</div>"
            "</div>"
            "</div>"
            "</div>"
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
        f'{_esc(message)}</text>'
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
    more = "" if len(differences) <= 12 else f"<p class='caption'>+{len(differences) - 12} more differences omitted.</p>"
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
        endpoint_x, endpoint_y = _endpoint_name(endpoints[0]), _endpoint_name(endpoints[1])
        status = _edge_status(
            (x, y),
            normalized_shape,
            normalized_reference,
            comparison_side=comparison_side,
        )
        stroke, stroke_width, dasharray = _edge_style(status)
        x1, y1 = positions[x]
        x2, y2 = positions[y]
        start, end = _trimmed_line((x1, y1), (x2, y2), 48)
        edge_parts.append(
            f'<line x1="{start[0]:.1f}" y1="{start[1]:.1f}" '
            f'x2="{end[0]:.1f}" y2="{end[1]:.1f}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"{dasharray}/>'
        )
        edge_parts.append(_endpoint_svg(endpoint_x, start, end, stroke))
        edge_parts.append(_endpoint_svg(endpoint_y, end, start, stroke))

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


def _selected_cases(cases: list[OracleCase]) -> list[OracleCase]:
    wanted = [
        "latent_medical",
        "nonlinear_common_cause",
        "finance_risk_r1",
        "enterprise_monitoring_r1",
    ]
    by_name = {case.name: case for case in cases}
    return [by_name[name] for name in wanted if name in by_name]


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
        "fci_engine.fci_plus.kernel"
        if case.use_kernel_ci
        else "fci_engine.fci_plus"
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
    if max_chars <= 1:
        return text[:max_chars]
    return text[: max_chars - 1] + "…"


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
            f'<circle cx="{point[0]:.1f}" cy="{point[1]:.1f}" r="5" '
            f'fill="#ffffff" stroke="{color}" stroke-width="1.8"/>'
        )
    if endpoint == "ARROW":
        tip = point
        base_x = point[0] - ux * 13
        base_y = point[1] - uy * 13
        p1 = (base_x + px * 5, base_y + py * 5)
        p2 = (base_x - px * 5, base_y - py * 5)
        return (
            f'<polygon points="{tip[0]:.1f},{tip[1]:.1f} '
            f'{p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" '
            f'fill="{color}"/>'
        )
    if endpoint == "TAIL":
        p1 = (point[0] + px * 7, point[1] + py * 7)
        p2 = (point[0] - px * 7, point[1] - py * 7)
        return (
            f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" '
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
