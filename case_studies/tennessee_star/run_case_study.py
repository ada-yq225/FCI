"""Run the complete Tennessee STAR application and write report artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from case_studies.tennessee_star.pcalg_reference import (
    load_pcalg_reference,
    refresh_pcalg_reference,
)
from case_studies.tennessee_star.report import payload_json, render_report
from case_studies.tennessee_star.study import (
    OUTPUT_DIR,
    build_summary_payload,
    descriptive_statistics,
    load_star,
    prepare_study,
    run_discovery_suite,
    save_processed_panels,
    sensitivity_analysis,
)


def run(
    *,
    output_directory: Path = OUTPUT_DIR,
    alpha: float = 0.05,
    sparsity_bound: int = 3,
    benchmark_repeats: int = 3,
    bootstraps: int = 12,
    descriptive_bootstraps: int = 1000,
    n_jobs: int = 4,
    refresh_pcalg: bool = False,
    rscript: Path | None = None,
    pcalg_order_audit: bool = True,
) -> dict[str, Path]:
    """Execute the application and return generated artifact paths."""

    frame = load_star()
    study = prepare_study(frame)
    save_processed_panels(study)
    descriptives = descriptive_statistics(
        study,
        bootstrap_repeats=descriptive_bootstraps,
    )
    records = run_discovery_suite(
        study,
        alpha=alpha,
        sparsity_bound=sparsity_bound,
        benchmark_repeats=benchmark_repeats,
        bootstraps=bootstraps,
        n_jobs=n_jobs,
    )
    sensitivity = sensitivity_analysis(
        frame,
        sparsity_bound=sparsity_bound,
    )
    external_records: list[dict[str, Any]] = []
    external_metadata: dict[str, Any] | None = None
    if refresh_pcalg:
        refresh_pcalg_reference(
            output_directory=output_directory,
            rscript=rscript,
            alpha=alpha,
            benchmark_repeats=benchmark_repeats,
            order_audit=pcalg_order_audit,
        )
    external_records, external_metadata = load_pcalg_reference(
        study,
        output_directory=output_directory,
        expected_alpha=alpha,
    )

    payload = build_summary_payload(
        study,
        records,
        descriptives,
        sensitivity,
        alpha=alpha,
        sparsity_bound=sparsity_bound,
        bootstraps=bootstraps,
        external_records=external_records,
        external_metadata=external_metadata,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": output_directory / "star_case_study_report.html",
        "summary": output_directory / "star_case_study_summary.json",
        "benchmark": output_directory / "star_benchmark.csv",
        "pag_edges": output_directory / "star_pag_edges.csv",
        "bootstrap": output_directory / "star_bootstrap_adjacencies.csv",
        "sensitivity": output_directory / "star_sensitivity.csv",
        "contrasts": output_directory / "star_descriptive_contrasts.csv",
    }
    paths["report"].write_text(render_report(payload), encoding="utf-8")
    paths["summary"].write_text(payload_json(payload), encoding="utf-8")
    _benchmark_frame(payload).to_csv(paths["benchmark"], index=False)
    _pag_edge_frame(payload).to_csv(paths["pag_edges"], index=False)
    _bootstrap_frame(payload).to_csv(paths["bootstrap"], index=False)
    pd.DataFrame(payload["sensitivity"]).to_csv(
        paths["sensitivity"],
        index=False,
    )
    pd.DataFrame(payload["descriptives"]["contrasts"]).to_csv(
        paths["contrasts"],
        index=False,
    )
    return paths


def _benchmark_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for run_record in payload["runs"]:
        rows.append(
            {
                "panel": run_record["panel"],
                "algorithm": run_record["algorithm"],
                "samples": run_record["samples"],
                "nodes": run_record["nodes"],
                "edges": run_record["edges"],
                "ci_tests": run_record["ci_tests"],
                "median_elapsed_seconds": run_record["median_elapsed_seconds"],
                "temporal_flag_count": len(run_record["temporal_flags"]),
            }
        )
    return pd.DataFrame(rows)


def _pag_edge_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for run_record in payload["runs"]:
        for edge in run_record["pag_edges"]:
            rows.append(
                {
                    "panel": run_record["panel"],
                    "algorithm": run_record["algorithm"],
                    **edge,
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for run_record in payload["runs"]:
        for edge in run_record["bootstrap_adjacencies"]:
            rows.append(
                {
                    "panel": run_record["panel"],
                    "algorithm": run_record["algorithm"],
                    **edge,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--sparsity-bound", type=int, default=3)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--bootstraps", type=int, default=12)
    parser.add_argument("--descriptive-bootstraps", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--refresh-pcalg", action="store_true")
    parser.add_argument("--rscript", type=Path)
    parser.add_argument("--skip-pcalg-order-audit", action="store_true")
    args = parser.parse_args()

    paths = run(
        output_directory=args.output_directory,
        alpha=args.alpha,
        sparsity_bound=args.sparsity_bound,
        benchmark_repeats=args.benchmark_repeats,
        bootstraps=args.bootstraps,
        descriptive_bootstraps=args.descriptive_bootstraps,
        n_jobs=args.n_jobs,
        refresh_pcalg=args.refresh_pcalg,
        rscript=args.rscript,
        pcalg_order_audit=not args.skip_pcalg_order_audit,
    )
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")


if __name__ == "__main__":
    main()
