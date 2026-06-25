"""Generate an FCI+ versus pcalg::fciPlus comparison report."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

from fci_engine import (
    default_oracle_cases,
    realistic_oracle_cases,
    run_pcalg_comparison_benchmark,
)
from fci_engine.metrics import BenchmarkResult, aggregate_benchmark_results

HTML_PATH = Path(__file__).with_name("pcalg_fci_plus_comparison.html")
CSV_PATH = Path(__file__).with_name("pcalg_fci_plus_comparison.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--realistic", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--html", type=Path, default=HTML_PATH)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    args = parser.parse_args()

    cases = default_oracle_cases()
    if args.realistic:
        cases = [
            *cases,
            *realistic_oracle_cases(
                n_repeats=args.repeats,
                n_samples=args.samples,
            ),
        ]

    results = run_pcalg_comparison_benchmark(cases)
    args.html.write_text(render_report(results), encoding="utf-8")
    write_csv(args.csv, results)
    print(f"Wrote {args.html}")
    print(f"Wrote {args.csv}")


def write_csv(path: Path, results: list[BenchmarkResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "algorithm",
                "skipped",
                "skip_reason",
                "exact_edge_f1",
                "semantic_edge_f1",
                "skeleton_f1",
                "endpoint_accuracy",
                "elapsed_time",
                "ci_test_count",
            ]
        )
        for result in results:
            writer.writerow(_result_row(result))


def render_report(results: list[BenchmarkResult]) -> str:
    aggregates = aggregate_benchmark_results(results)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FCI+ pcalg Comparison</title>
<style>
body {{
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f6f7f9;
  color: #101828;
}}
header {{
  padding: 28px 36px;
  background: #ffffff;
  border-bottom: 1px solid #d0d5dd;
}}
main {{ width: min(1280px, calc(100vw - 36px)); margin: 24px auto 40px; }}
section {{
  background: #ffffff;
  border: 1px solid #d0d5dd;
  border-radius: 8px;
  padding: 18px;
  margin-bottom: 18px;
}}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
h2 {{ margin: 0 0 14px; font-size: 18px; }}
p {{ margin: 0; color: #667085; line-height: 1.5; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ padding: 9px 10px; border-bottom: 1px solid #eaecf0; text-align: left; }}
th {{ color: #344054; background: #f9fafb; font-weight: 700; }}
td.num {{ font-variant-numeric: tabular-nums; }}
.skip {{ color: #b42318; }}
</style>
</head>
<body>
<header>
  <h1>FCI+ versus pcalg::fciPlus</h1>
  <p>Focused comparison of the local Python FCI+ implementation, robust FCI+, and R pcalg::fciPlus.</p>
</header>
<main>
  <section>
    <h2>Aggregate Scores</h2>
    {_aggregate_table(aggregates)}
  </section>
  <section>
    <h2>Case Results</h2>
    {_results_table(results)}
  </section>
</main>
</body>
</html>
"""


def _aggregate_table(aggregates: object) -> str:
    rows = []
    for item in aggregates:
        rows.append(
            "<tr>"
            f"<td>{_esc(item.algorithm)}</td>"
            f"<td class='num'>{item.n_cases}</td>"
            f"<td class='num'>{item.skipped_cases}</td>"
            f"<td class='num'>{item.mean_exact_edge_f1:.3f}</td>"
            f"<td class='num'>{item.mean_semantic_edge_f1:.3f}</td>"
            f"<td class='num'>{item.mean_skeleton_f1:.3f}</td>"
            f"<td class='num'>{item.mean_endpoint_accuracy:.3f}</td>"
            f"<td class='num'>{_fmt(item.mean_elapsed_time)}</td>"
            f"<td class='num'>{_fmt(item.mean_ci_test_count)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Algorithm</th><th>Cases</th><th>Skipped</th>"
        "<th>Exact F1</th><th>Semantic F1</th><th>Skeleton F1</th>"
        "<th>Endpoint Acc.</th><th>Time</th><th>CI Tests</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _results_table(results: list[BenchmarkResult]) -> str:
    rows = []
    for result in results:
        if result.skipped:
            rows.append(
                "<tr>"
                f"<td>{_esc(result.case_name)}</td>"
                f"<td>{_esc(result.algorithm)}</td>"
                "<td class='skip' colspan='7'>"
                f"skipped: {_esc(result.skipped_reason or '')}</td>"
                "</tr>"
            )
            continue
        assert result.comparison is not None
        assert result.semantic_comparison is not None
        rows.append(
            "<tr>"
            f"<td>{_esc(result.case_name)}</td>"
            f"<td>{_esc(result.algorithm)}</td>"
            f"<td class='num'>{result.comparison.exact_edge_f1:.3f}</td>"
            f"<td class='num'>{result.semantic_comparison.semantic_edge_f1:.3f}</td>"
            f"<td class='num'>{result.comparison.skeleton_f1:.3f}</td>"
            f"<td class='num'>{result.comparison.endpoint_accuracy:.3f}</td>"
            f"<td class='num'>{_fmt(result.elapsed_time)}</td>"
            f"<td class='num'>{_fmt(result.ci_test_count)}</td>"
            f"<td>{len(result.edges)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Case</th><th>Algorithm</th><th>Exact F1</th>"
        "<th>Semantic F1</th><th>Skeleton F1</th><th>Endpoint Acc.</th>"
        "<th>Time</th><th>CI Tests</th><th>Edges</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _result_row(result: BenchmarkResult) -> list[object]:
    if result.skipped:
        return [
            result.case_name,
            result.algorithm,
            True,
            result.skipped_reason,
            "",
            "",
            "",
            "",
            "",
            "",
        ]
    assert result.comparison is not None
    assert result.semantic_comparison is not None
    return [
        result.case_name,
        result.algorithm,
        False,
        "",
        result.comparison.exact_edge_f1,
        result.semantic_comparison.semantic_edge_f1,
        result.comparison.skeleton_f1,
        result.comparison.endpoint_accuracy,
        result.elapsed_time,
        result.ci_test_count,
    ]


def _fmt(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
