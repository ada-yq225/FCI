"""Oracle benchmark runners for FCI implementations."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional, Union

from fci_engine.ci import CITest, KernelCITest
from fci_engine.api import fci, fci_plus
from fci_engine.diagnostics import OrientationEvent
from fci_engine.metrics.accuracy import (
    NormalizedShape,
    PAGComparison,
    PAGESemanticComparison,
    compare_pag_shapes,
    compare_pag_shapes_semantic,
    shape_from_pag,
)
from fci_engine.simulation.oracle_cases import OracleCase


@dataclass(frozen=True)
class BenchmarkResult:
    """One algorithm result on one oracle case."""

    case_name: str
    algorithm: str
    comparison: Optional[PAGComparison]
    semantic_comparison: Optional[PAGESemanticComparison]
    edges: NormalizedShape
    elapsed_time: Optional[float]
    ci_test_count: Optional[int] = None
    cache_hits: Optional[int] = None
    orientation_trace: Optional[list[OrientationEvent]] = None
    skipped_reason: Optional[str] = None

    @property
    def skipped(self) -> bool:
        return self.skipped_reason is not None

    def summary(self) -> str:
        """Return a compact one-line benchmark summary."""

        if self.skipped:
            return f"{self.case_name:24s} {self.algorithm:18s} skipped: {self.skipped_reason}"
        assert self.comparison is not None
        assert self.semantic_comparison is not None
        return (
            f"{self.case_name:24s} {self.algorithm:18s} "
            f"{self.comparison.summary()} "
            f"{self.semantic_comparison.summary()} "
            f"time={self.elapsed_time:.4f}s ci={self.ci_test_count}"
        )


@dataclass(frozen=True)
class BenchmarkAggregate:
    """Aggregate benchmark scores for one algorithm."""

    algorithm: str
    n_cases: int
    skipped_cases: int
    mean_exact_edge_f1: float
    mean_semantic_edge_f1: float
    mean_skeleton_f1: float
    mean_endpoint_accuracy: float
    mean_elapsed_time: Optional[float]
    mean_ci_test_count: Optional[float]

    def summary(self) -> str:
        """Return one formatted leaderboard line."""

        elapsed = "NA" if self.mean_elapsed_time is None else f"{self.mean_elapsed_time:.4f}s"
        ci_tests = (
            "NA"
            if self.mean_ci_test_count is None
            else f"{self.mean_ci_test_count:.1f}"
        )
        return (
            f"{self.algorithm:26s} "
            f"exact_f1={self.mean_exact_edge_f1:.3f} "
            f"semantic_f1={self.mean_semantic_edge_f1:.3f} "
            f"skeleton_f1={self.mean_skeleton_f1:.3f} "
            f"endpoint_acc={self.mean_endpoint_accuracy:.3f} "
            f"time={elapsed} ci={ci_tests} "
            f"cases={self.n_cases} skipped={self.skipped_cases}"
        )


def run_oracle_benchmark(
    cases: list[OracleCase],
    include_causal_learn: bool = True,
    include_pcalg: bool = True,
    include_kernel_ci: bool = True,
) -> list[BenchmarkResult]:
    """Run fci_engine, optional causal-learn, and optional pcalg benchmarks."""

    results: list[BenchmarkResult] = []
    for case in cases:
        results.append(run_fci_engine(case, fci, "fci_engine.fci"))
        results.append(run_fci_engine(case, fci_plus, "fci_engine.fci_plus"))
        results.append(
            run_fci_engine(
                case,
                fci_plus,
                "fci_engine.fci_plus.leaf",
                orientation_strategy="leaf",
            )
        )
        if include_kernel_ci and case.use_kernel_ci:
            results.append(
                run_fci_engine(
                    case,
                    fci,
                    "fci_engine.fci.kernel",
                    ci_test=KernelCITest(
                        alpha=case.alpha,
                        n_permutations=99,
                        random_state=0,
                    ),
                )
            )
            results.append(
                run_fci_engine(
                    case,
                    fci_plus,
                    "fci_engine.fci_plus.kernel",
                    ci_test=KernelCITest(
                        alpha=case.alpha,
                        n_permutations=99,
                        random_state=0,
                    ),
                )
            )
            results.append(
                run_fci_engine(
                    case,
                    fci_plus,
                    "fci_engine.fci_plus.kernel.leaf",
                    ci_test=KernelCITest(
                        alpha=case.alpha,
                        n_permutations=99,
                        random_state=0,
                    ),
                    orientation_strategy="leaf",
                )
            )
        if include_causal_learn:
            results.append(run_causal_learn_fci(case))
            if include_kernel_ci and case.use_kernel_ci:
                results.append(run_causal_learn_fci(case, method="kci"))
        if include_pcalg:
            results.append(run_pcalg_fci_plus(case))
    return results


def run_fci_engine(
    case: OracleCase,
    algorithm: Callable[..., object],
    algorithm_name: str,
    ci_test: Optional[CITest] = None,
    **algorithm_kwargs: object,
) -> BenchmarkResult:
    """Run one fci_engine algorithm on an oracle case."""

    start = perf_counter()
    result = algorithm(
        case.data,
        alpha=case.alpha,
        ci_test=ci_test,
        max_cond_set_size=case.max_cond_set_size,
        max_path_length=case.max_path_length,
        **algorithm_kwargs,
    )
    elapsed = perf_counter() - start
    edges = shape_from_pag(result.graph)
    return BenchmarkResult(
        case_name=case.name,
        algorithm=algorithm_name,
        comparison=compare_pag_shapes(case.oracle_shape, edges),
        semantic_comparison=compare_pag_shapes_semantic(case.oracle_shape, edges),
        edges=edges,
        elapsed_time=elapsed,
        ci_test_count=result.ci_test_count,
        cache_hits=result.cache_hits,
        orientation_trace=result.orientation_trace,
    )


def run_causal_learn_fci(case: OracleCase, method: str = "fisherz") -> BenchmarkResult:
    """Run causal-learn FCI if it is installed."""

    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))
    try:
        with redirect_stderr(StringIO()):
            from causallearn.search.ConstraintBased.FCI import fci as causal_learn_fci
    except ImportError as exc:
        return _skipped(case, f"causal-learn.fci.{method}", f"not installed: {exc}")

    labels = list(case.data.columns)
    label_order = {label: index for index, label in enumerate(labels)}
    start = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with redirect_stdout(StringIO()):
            with redirect_stderr(StringIO()):
                graph, _ = causal_learn_fci(
                    case.data.to_numpy(),
                    independence_test_method=method,
                    alpha=case.alpha,
                    depth=(
                        -1
                        if case.max_cond_set_size is None
                        else case.max_cond_set_size
                    ),
                    max_path_length=(
                        -1 if case.max_path_length is None else case.max_path_length
                    ),
                    verbose=False,
                    show_progress=False,
                    node_names=labels,
                )
    elapsed = perf_counter() - start
    edges = _shape_from_causal_learn_graph(graph, label_order)
    return BenchmarkResult(
        case_name=case.name,
        algorithm=f"causal-learn.fci.{method}",
        comparison=compare_pag_shapes(case.oracle_shape, edges),
        semantic_comparison=compare_pag_shapes_semantic(case.oracle_shape, edges),
        edges=edges,
        elapsed_time=elapsed,
    )


def run_pcalg_fci_plus(case: OracleCase, timeout: int = 60) -> BenchmarkResult:
    """Run R pcalg::fciPlus if Rscript and pcalg are available."""

    rscript = _find_rscript()
    if rscript is None:
        return _skipped(case, "pcalg.fciPlus", "Rscript not available")

    with tempfile.TemporaryDirectory(prefix="fci_engine_pcalg_") as tmpdir:
        tmp_path = Path(tmpdir)
        data_path = tmp_path / "data.csv"
        script_path = tmp_path / "run_pcalg.R"
        case.data.to_csv(data_path, index=False)
        script_path.write_text(_pcalg_script(data_path, case), encoding="utf-8")

        start = perf_counter()
        completed = subprocess.run(
            [rscript, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        elapsed = perf_counter() - start

    if completed.returncode != 0:
        reason = completed.stderr.strip() or completed.stdout.strip()
        return _skipped(case, "pcalg.fciPlus", reason[:240])

    edges = _parse_pcalg_edges(completed.stdout)
    return BenchmarkResult(
        case_name=case.name,
        algorithm="pcalg.fciPlus",
        comparison=compare_pag_shapes(case.oracle_shape, edges),
        semantic_comparison=compare_pag_shapes_semantic(case.oracle_shape, edges),
        edges=edges,
        elapsed_time=elapsed,
    )


def _find_rscript() -> Optional[str]:
    """Return an Rscript executable path from PATH or common local installs."""

    rscript = shutil.which("Rscript")
    if rscript is not None:
        return rscript

    for candidate in (
        Path("/opt/homebrew/bin/Rscript"),
        Path("/usr/local/bin/Rscript"),
        Path("/Library/Frameworks/R.framework/Resources/bin/Rscript"),
    ):
        if candidate.exists():
            return str(candidate)
    return None


def format_benchmark_results(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as a human-readable table."""

    return "\n".join(result.summary() for result in results)


def aggregate_benchmark_results(
    results: list[BenchmarkResult],
) -> list[BenchmarkAggregate]:
    """Aggregate benchmark results by algorithm."""

    algorithms = sorted({result.algorithm for result in results})
    aggregates: list[BenchmarkAggregate] = []
    for algorithm in algorithms:
        group = [result for result in results if result.algorithm == algorithm]
        completed = [result for result in group if not result.skipped]
        comparisons = [
            result.comparison for result in completed if result.comparison is not None
        ]
        semantic_comparisons = [
            result.semantic_comparison
            for result in completed
            if result.semantic_comparison is not None
        ]
        elapsed_times = [
            result.elapsed_time for result in completed if result.elapsed_time is not None
        ]
        ci_counts = [
            result.ci_test_count for result in completed if result.ci_test_count is not None
        ]
        aggregates.append(
            BenchmarkAggregate(
                algorithm=algorithm,
                n_cases=len(completed),
                skipped_cases=len(group) - len(completed),
                mean_exact_edge_f1=_mean(
                    comparison.exact_edge_f1 for comparison in comparisons
                ),
                mean_semantic_edge_f1=_mean(
                    comparison.semantic_edge_f1
                    for comparison in semantic_comparisons
                ),
                mean_skeleton_f1=_mean(
                    comparison.skeleton_f1 for comparison in comparisons
                ),
                mean_endpoint_accuracy=_mean(
                    comparison.endpoint_accuracy for comparison in comparisons
                ),
                mean_elapsed_time=_optional_mean(elapsed_times),
                mean_ci_test_count=_optional_mean(ci_counts),
            )
        )

    return sorted(
        aggregates,
        key=lambda item: (
            item.mean_exact_edge_f1,
            item.mean_semantic_edge_f1,
            item.mean_skeleton_f1,
            item.mean_endpoint_accuracy,
            -(item.mean_elapsed_time or float("inf")),
        ),
        reverse=True,
    )


def format_benchmark_leaderboard(results: list[BenchmarkResult]) -> str:
    """Format an aggregate benchmark leaderboard."""

    return "\n".join(
        aggregate.summary() for aggregate in aggregate_benchmark_results(results)
    )


def _shape_from_causal_learn_graph(
    graph: object,
    label_order: dict[str, int],
) -> NormalizedShape:
    shape: NormalizedShape = {}
    for edge in graph.get_graph_edges():
        node1 = edge.get_node1().get_name()
        node2 = edge.get_node2().get_name()
        endpoint1 = str(edge.get_endpoint1()).upper()
        endpoint2 = str(edge.get_endpoint2()).upper()
        if label_order[node1] <= label_order[node2]:
            shape[(node1, node2)] = (endpoint1, endpoint2)
        else:
            shape[(node2, node1)] = (endpoint2, endpoint1)
    return shape


def _pcalg_script(data_path: Path, case: OracleCase) -> str:
    labels = ",".join(f'"{label}"' for label in case.data.columns)
    return f"""
suppressPackageStartupMessages(library(pcalg))
data <- read.csv("{data_path}")
labels <- c({labels})
suffStat <- list(C = cor(data), n = nrow(data))
fit <- fciPlus(
  suffStat = suffStat,
  indepTest = gaussCItest,
  alpha = {case.alpha},
  labels = labels,
  p = length(labels),
  verbose = FALSE,
  selectionBias = TRUE
)
amat <- fit@amat
cat("PCALG_EDGE_BEGIN\\n")
for (i in seq_len(ncol(amat) - 1)) {{
  for (j in seq((i + 1), ncol(amat))) {{
    if (amat[i, j] != 0 || amat[j, i] != 0) {{
      cat(labels[i], ",", labels[j], ",", amat[j, i], ",", amat[i, j], "\\n", sep = "")
    }}
  }}
}}
cat("PCALG_EDGE_END\\n")
"""


def _parse_pcalg_edges(output: str) -> NormalizedShape:
    in_edges = False
    shape: NormalizedShape = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line == "PCALG_EDGE_BEGIN":
            in_edges = True
            continue
        if line == "PCALG_EDGE_END":
            break
        if not in_edges or not line:
            continue
        row = next(csv.reader([line]))
        if len(row) != 4:
            continue
        x, y, endpoint_x, endpoint_y = row
        shape[(x, y)] = (_pcalg_endpoint(endpoint_x), _pcalg_endpoint(endpoint_y))
    return shape


def _pcalg_endpoint(mark: str) -> str:
    marks = {
        "1": "CIRCLE",
        "2": "ARROW",
        "3": "TAIL",
    }
    return marks.get(mark, "NONE")


def _skipped(case: OracleCase, algorithm: str, reason: str) -> BenchmarkResult:
    return BenchmarkResult(
        case_name=case.name,
        algorithm=algorithm,
        comparison=None,
        semantic_comparison=None,
        edges={},
        elapsed_time=None,
        skipped_reason=reason,
    )


def _mean(values: object) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def _optional_mean(values: list[Union[float, int]]) -> Optional[float]:
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)
