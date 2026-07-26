"""Generate matched known-truth FCI software comparison tables.

The executed benchmark and the documentation-derived capability matrix are
deliberately separate.  In particular, Tetrad is represented only in the
feature matrix because it was not executed in this environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from statistics import fmean
from typing import Any, Literal, Optional, Union

from fci_engine.api import fci, fci_plus
from fci_engine.config import FCIConfig, FCIPlusConfig
from fci_engine.metrics.benchmark import (
    BenchmarkResult,
    run_causal_learn_fci,
    run_fci_engine,
    run_pcalg_fci_plus,
)
from fci_engine.result import FCIResult
from fci_engine.simulation import realistic_oracle_cases
from fci_engine.simulation.oracle_cases import OracleCase


ROOT = Path(__file__).resolve().parents[1]
LANDSCAPE_PATH = ROOT / "reports" / "research" / "software_landscape.json"

CASE_COLUMNS = (
    "case",
    "scenario",
    "repeat",
    "n_samples",
    "n_variables",
    "oracle_edges",
    "algorithm",
    "algorithm_family",
    "tool",
    "software_version",
    "profile",
    "configuration",
    "ci_test_method",
    "alpha_policy",
    "effective_alpha",
    "timing_scope",
    "status",
    "skipped_reason",
    "skeleton_f1",
    "exact_edge_f1",
    "semantic_edge_f1",
    "endpoint_accuracy",
    "elapsed_seconds",
    "ci_test_count",
    "cache_hits",
    "learned_edges",
)

SUMMARY_COLUMNS = (
    "algorithm",
    "algorithm_family",
    "tool",
    "software_version",
    "profile",
    "configuration",
    "ci_test_method",
    "alpha_policy",
    "effective_alpha",
    "timing_scope",
    "n_requested",
    "n_completed",
    "n_skipped",
    "skip_reasons",
    "mean_skeleton_f1",
    "mean_exact_edge_f1",
    "mean_semantic_edge_f1",
    "mean_endpoint_accuracy",
    "mean_elapsed_seconds",
    "mean_ci_test_count",
)

FEATURE_COLUMNS = (
    "tool",
    "language",
    "standard_fci",
    "fci_plus",
    "latent_confounding",
    "selection_bias",
    "custom_ci",
    "order_audit",
    "bootstrap_workflow",
    "orientation_trace",
    "sepset_provenance",
    "artifact_export",
    "executed_here",
    "evidence_date",
)


def generate_software_comparison(
    *,
    output_dir: Path,
    repeats: int = 3,
    samples: int = 2_500,
    include_pcalg: bool = True,
) -> dict[str, Path]:
    """Run matched known-truth cases and write long-form and aggregate CSVs."""

    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if samples < 4:
        raise ValueError("samples must be at least 4.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = realistic_oracle_cases(n_repeats=repeats, n_samples=samples)
    versions = _software_versions(include_pcalg=include_pcalg)

    rows: list[dict[str, Any]] = []
    for case in cases:
        for result, metadata_row in _run_case(
            case,
            include_pcalg=include_pcalg,
            versions=versions,
        ):
            rows.append(_case_row(case, result, metadata_row))

    cases_path = output_dir / "software_benchmark_cases.csv"
    summary_path = output_dir / "software_benchmark_summary.csv"
    features_path = output_dir / "software_feature_matrix.csv"
    _write_csv(cases_path, rows, CASE_COLUMNS)
    _write_csv(summary_path, _summary_rows(rows), SUMMARY_COLUMNS)
    _write_csv(
        features_path,
        _feature_rows(LANDSCAPE_PATH),
        FEATURE_COLUMNS,
    )
    return {
        "cases": cases_path,
        "summary": summary_path,
        "features": features_path,
    }


def _run_case(
    case: OracleCase,
    *,
    include_pcalg: bool,
    versions: dict[str, str],
) -> Iterable[tuple[BenchmarkResult, dict[str, str]]]:
    paper_fci = FCIConfig.paper(alpha=case.alpha)
    if case.sparsity_bound is None:
        raise ValueError(f"{case.name} has no sparsity_bound for paper FCI+.")
    paper_fci_plus = FCIPlusConfig.paper(
        k=case.sparsity_bound,
        alpha=case.alpha,
    )
    robust_fci_plus = FCIPlusConfig.practical(
        max_cond_set_size=case.max_cond_set_size,
        sparsity_bound=case.sparsity_bound,
        max_path_length=case.max_path_length,
    )

    local_runs: tuple[tuple[str, str, FCIConfig, Callable[..., FCIResult]], ...] = (
        (
            "fci_engine.fci",
            "spirtes_2000_paper",
            paper_fci,
            fci,
        ),
        (
            "fci_engine.fci_plus",
            "claassen_2013_paper",
            paper_fci_plus,
            fci_plus,
        ),
        (
            "fci_engine.fci_plus.robust",
            "practical_robust",
            robust_fci_plus,
            fci_plus,
        ),
    )
    for algorithm, profile, config, function in local_runs:
        result = _run_local_config(case, function, algorithm, config)
        yield (
            result,
            _algorithm_metadata(
                algorithm=algorithm,
                family="FCI+" if "fci_plus" in algorithm else "FCI",
                tool="fci_engine",
                version=versions["fci_engine"],
                profile=profile,
                config=config,
                ci_test_method="fisher_z",
                effective_alpha=_effective_alpha(config.alpha, len(case.data)),
                timing_scope="in_process_algorithm_call",
            ),
        )

    causal_result = _safe_external_run(
        case,
        "causal-learn.fci.fisherz",
        lambda: run_causal_learn_fci(case, method="fisherz"),
    )
    causal_config = {
        "alpha": case.alpha,
        "depth": (-1 if case.max_cond_set_size is None else case.max_cond_set_size),
        "max_path_length": (
            -1 if case.max_path_length is None else case.max_path_length
        ),
    }
    yield (
        causal_result,
        _algorithm_metadata(
            algorithm="causal-learn.fci.fisherz",
            family="FCI",
            tool="causal-learn",
            version=versions["causal-learn"],
            profile="documented_fci_api",
            config=causal_config,
            ci_test_method="fisherz",
            effective_alpha=case.alpha,
            timing_scope="in_process_algorithm_call",
        ),
    )

    if include_pcalg:
        pcalg_result = _safe_external_run(
            case,
            "pcalg.fciPlus",
            lambda: run_pcalg_fci_plus(case),
        )
        yield (
            pcalg_result,
            _algorithm_metadata(
                algorithm="pcalg.fciPlus",
                family="FCI+",
                tool="pcalg",
                version=versions["pcalg"],
                profile="pcalg_fciPlus_default",
                config={
                    "alpha": case.alpha,
                    "indepTest": "gaussCItest",
                    "selectionBias": True,
                },
                ci_test_method="gaussCItest",
                effective_alpha=case.alpha,
                timing_scope="external_R_process_including_startup",
            ),
        )


def _run_local_config(
    case: OracleCase,
    function: Callable[..., FCIResult],
    algorithm: str,
    config: FCIConfig,
) -> BenchmarkResult:
    def configured(data: object, **_: Any) -> FCIResult:
        return function(data, config=config)

    return run_fci_engine(case, configured, algorithm)


def _safe_external_run(
    case: OracleCase,
    algorithm: str,
    runner: Callable[[], BenchmarkResult],
) -> BenchmarkResult:
    try:
        return runner()
    except Exception as exc:
        reason = f"external runner failed: {type(exc).__name__}: {exc}"
        return BenchmarkResult(
            case_name=case.name,
            algorithm=algorithm,
            comparison=None,
            semantic_comparison=None,
            edges={},
            elapsed_time=None,
            skipped_reason=reason.replace("\n", " ")[:500],
        )


def _algorithm_metadata(
    *,
    algorithm: str,
    family: str,
    tool: str,
    version: str,
    profile: str,
    config: Union[FCIConfig, dict[str, Any]],
    ci_test_method: str,
    effective_alpha: float,
    timing_scope: str,
) -> dict[str, str]:
    configuration: dict[str, Any]
    if isinstance(config, FCIConfig):
        configuration = asdict(config)
        configuration["ci_test"] = (
            None if config.ci_test is None else type(config.ci_test).__name__
        )
        configuration["background_knowledge"] = (
            None
            if config.background_knowledge is None
            else type(config.background_knowledge).__name__
        )
    else:
        configuration = config
    return {
        "algorithm": algorithm,
        "algorithm_family": family,
        "tool": tool,
        "software_version": version,
        "profile": profile,
        "configuration": json.dumps(
            configuration,
            sort_keys=True,
            separators=(", ", ": "),
        ),
        "ci_test_method": ci_test_method,
        "alpha_policy": (
            "sample_size_auto"
            if isinstance(config, FCIConfig) and config.alpha == "auto"
            else "fixed"
        ),
        "effective_alpha": str(effective_alpha),
        "timing_scope": timing_scope,
    }


def _case_row(
    case: OracleCase,
    result: BenchmarkResult,
    metadata_row: dict[str, str],
) -> dict[str, Any]:
    comparison = result.comparison
    semantic = result.semantic_comparison
    scenario, repeat = _split_case_name(case.name)
    return {
        "case": case.name,
        "scenario": scenario,
        "repeat": repeat,
        "n_samples": len(case.data),
        "n_variables": len(case.data.columns),
        "oracle_edges": len(case.oracle_shape),
        **metadata_row,
        "status": "skipped" if result.skipped else "completed",
        "skipped_reason": result.skipped_reason or "",
        "skeleton_f1": (None if comparison is None else comparison.skeleton_f1),
        "exact_edge_f1": (None if comparison is None else comparison.exact_edge_f1),
        "semantic_edge_f1": (None if semantic is None else semantic.semantic_edge_f1),
        "endpoint_accuracy": (
            None if comparison is None else comparison.endpoint_accuracy
        ),
        "elapsed_seconds": result.elapsed_time,
        "ci_test_count": result.ci_test_count,
        "cache_hits": result.cache_hits,
        "learned_edges": _serialize_edges(result.edges),
    }


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    algorithms = sorted({str(row["algorithm"]) for row in rows})
    for algorithm in algorithms:
        group = [row for row in rows if row["algorithm"] == algorithm]
        completed = [row for row in group if row["status"] == "completed"]
        first = group[0]
        reasons = sorted(
            {str(row["skipped_reason"]) for row in group if row["skipped_reason"]}
        )
        summaries.append(
            {
                key: first[key]
                for key in (
                    "algorithm",
                    "algorithm_family",
                    "tool",
                    "software_version",
                    "profile",
                    "configuration",
                    "ci_test_method",
                    "alpha_policy",
                    "effective_alpha",
                    "timing_scope",
                )
            }
            | {
                "n_requested": len(group),
                "n_completed": len(completed),
                "n_skipped": len(group) - len(completed),
                "skip_reasons": " | ".join(reasons),
                "mean_skeleton_f1": _mean_or_none(
                    row["skeleton_f1"] for row in completed
                ),
                "mean_exact_edge_f1": _mean_or_none(
                    row["exact_edge_f1"] for row in completed
                ),
                "mean_semantic_edge_f1": _mean_or_none(
                    row["semantic_edge_f1"] for row in completed
                ),
                "mean_endpoint_accuracy": _mean_or_none(
                    row["endpoint_accuracy"] for row in completed
                ),
                "mean_elapsed_seconds": _mean_or_none(
                    row["elapsed_seconds"] for row in completed
                ),
                "mean_ci_test_count": _mean_or_none(
                    row["ci_test_count"] for row in completed
                ),
            }
        )
    return summaries


def _feature_rows(landscape_path: Path) -> list[dict[str, Any]]:
    landscape = json.loads(landscape_path.read_text(encoding="utf-8"))
    detail = {
        "fci_engine": {
            "order_audit": "yes",
            "bootstrap_workflow": "yes",
            "orientation_trace": "yes",
            "sepset_provenance": "yes",
        },
        "pcalg": {
            "order_audit": "partial",
            "bootstrap_workflow": "not_established",
            "orientation_trace": "not_established",
            "sepset_provenance": "partial",
        },
        "causal-learn": {
            "order_audit": "not_established",
            "bootstrap_workflow": "not_established",
            "orientation_trace": "not_exported",
            "sepset_provenance": "not_exported",
        },
        "Tetrad": {
            "order_audit": "partial",
            "bootstrap_workflow": "broader_workflow",
            "orientation_trace": "logger_only",
            "sepset_provenance": "yes",
        },
    }
    rows: list[dict[str, Any]] = []
    for tool in landscape["tools"]:
        algorithms = {str(name).upper() for name in tool["algorithms"]}
        rows.append(
            {
                "tool": tool["tool"],
                "language": tool["language"],
                "standard_fci": _yes_no("FCI" in algorithms),
                "fci_plus": _yes_no("FCI+" in algorithms),
                "latent_confounding": _yes_no(tool["latent_confounding"]),
                "selection_bias": _yes_no(tool["selection_bias"]),
                "custom_ci": _yes_no(tool["custom_ci"]),
                **detail[tool["tool"]],
                "artifact_export": _yes_no(tool["audit_exports"]),
                "executed_here": _yes_no(tool["evidence_kind"] == "executed"),
                "evidence_date": tool["as_of"],
            }
        )
    return rows


def _software_versions(*, include_pcalg: bool) -> dict[str, str]:
    return {
        "fci_engine": _distribution_version("fci-engine", fallback="0.1.0"),
        "causal-learn": _distribution_version(
            "causal-learn",
            fallback="not installed",
        ),
        "pcalg": (
            _pcalg_version()
            if include_pcalg
            else "not requested (verified landscape: 2.7-12)"
        ),
    }


def _distribution_version(distribution: str, *, fallback: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return fallback


def _pcalg_version() -> str:
    rscript = shutil.which("Rscript")
    if rscript is None:
        for candidate in (
            Path("/opt/homebrew/bin/Rscript"),
            Path("/usr/local/bin/Rscript"),
            Path("/Library/Frameworks/R.framework/Resources/bin/Rscript"),
        ):
            if candidate.exists():
                rscript = str(candidate)
                break
    if rscript is None:
        return "not installed"
    try:
        completed = subprocess.run(
            [
                rscript,
                "-e",
                "cat(as.character(packageVersion('pcalg')))",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "version unavailable"
    if completed.returncode != 0:
        return "not installed"
    return completed.stdout.strip() or "version unavailable"


def _split_case_name(case_name: str) -> tuple[str, int]:
    stem, separator, repeat_text = case_name.rpartition("_r")
    if separator and repeat_text.isdigit():
        return stem, int(repeat_text)
    return case_name, 1


def _effective_alpha(
    alpha: Union[float, Literal["auto"]],
    n_samples: int,
) -> float:
    if alpha != "auto":
        return float(alpha)
    if n_samples < 1_000:
        return 0.05
    if n_samples < 5_000:
        return 0.01
    return 0.001


def _serialize_edges(
    edges: dict[tuple[str, str], tuple[str, str]],
) -> str:
    payload = [
        {
            "x": x,
            "y": y,
            "endpoint_x": endpoint_x,
            "endpoint_y": endpoint_y,
        }
        for (x, y), (endpoint_x, endpoint_y) in sorted(edges.items())
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _mean_or_none(
    values: Iterable[Optional[Union[float, int]]],
) -> Optional[float]:
    numeric = [float(value) for value in values if value is not None]
    return None if not numeric else fmean(numeric)


def _yes_no(value: object) -> str:
    return "yes" if bool(value) else "no"


def _write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: tuple[str, ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--samples", type=int, default=2_500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "data",
    )
    parser.add_argument(
        "--no-pcalg",
        action="store_true",
        help="Write local and causal-learn rows without executing R pcalg.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = generate_software_comparison(
        output_dir=args.output_dir,
        repeats=args.repeats,
        samples=args.samples,
        include_pcalg=not args.no_pcalg,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
