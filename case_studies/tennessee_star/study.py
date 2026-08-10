"""Reproducible Tennessee STAR application built on the public FCI API.

Nothing in this module is imported by ``fci_engine``. It is deliberately an
application layer: data coding, cohort definitions, domain audits, resampling,
and reporting choices live here rather than inside the algorithm package.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from fci_engine import FCIResult, fci, fci_plus
from fci_engine.ci import GSquareTest

from case_studies.tennessee_star.download_data import (
    DATASET_DOI,
    GUIDE_PATH,
    STUDENT_PATH,
)

AlgorithmName = Literal["fci", "fci_plus", "fci_plus_robust"]

HERE = Path(__file__).resolve().parent
PROCESSED_DIR = HERE / "data" / "processed"
OUTPUT_DIR = HERE / "output"

CLASS_LABELS = {
    1: "Small",
    2: "Regular",
    3: "Regular + aide",
}
GENDER_LABELS = {
    1: "Male",
    2: "Female",
}
ETHNICITY_LABELS = {
    1: "White",
    2: "Black",
    3: "Other",
}
FREE_LUNCH_LABELS = {
    1: "Free/reduced",
    2: "Non-free",
}
SCHOOL_LABELS = {
    1: "Inner city",
    2: "Suburban",
    3: "Rural",
    4: "Urban",
}

PANEL_DESCRIPTIONS = {
    "attrition": (
        "Kindergarten entrants with observed kindergarten achievement and "
        "covariates. The endpoint is whether both grade-3 reading and math "
        "scores are observed."
    ),
    "longitudinal": (
        "Complete cases with kindergarten and grade-3 achievement. "
        "Kindergarten achievement is retained to study achievement persistence."
    ),
    "focused_treatment": (
        "Complete grade-3 cases without kindergarten achievement in the node "
        "set. This focused specification avoids allowing an early outcome to "
        "act as a separator between randomized class assignment and grade-3 "
        "achievement."
    ),
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


@dataclass(frozen=True)
class PreparedPanel:
    """Encoded discrete panel and non-model metadata used for resampling."""

    name: str
    data: pd.DataFrame
    school_ids: np.ndarray[Any, np.dtype[np.int64]]
    category_labels: dict[str, dict[int, str]]
    description: str


@dataclass(frozen=True)
class PreparedStudy:
    """Raw cohort plus the three application panels."""

    raw_rows: int
    kindergarten_rows: int
    kindergarten_schools: int
    cohort: pd.DataFrame
    panels: dict[str, PreparedPanel]


@dataclass(frozen=True)
class DiscoveryRecord:
    """One fitted algorithm and its application-level diagnostics."""

    panel: str
    algorithm: AlgorithmName
    result: FCIResult
    elapsed_runs: tuple[float, ...]
    median_elapsed: float
    bootstrap_adjacencies: dict[tuple[str, str], float]
    temporal_flags: tuple[str, ...]
    order_audit: dict[str, Any]

    @property
    def edge_count(self) -> int:
        return len(self.result.edges)

    @property
    def ci_test_count(self) -> int:
        return self.result.ci_test_count


def load_star(path: Path = STUDENT_PATH) -> pd.DataFrame:
    """Load the official Harvard Dataverse tabular export."""

    if not path.exists():
        raise FileNotFoundError(
            f"STAR data not found at {path}. Run "
            "`python -m case_studies.tennessee_star.download_data` first."
        )
    frame = pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    if frame.shape != (11_601, 379):
        raise ValueError(f"Unexpected STAR data shape: {frame.shape}.")
    return frame


def prepare_study(
    frame: pd.DataFrame,
    achievement_bins: int = 4,
) -> PreparedStudy:
    """Prepare independent application panels from kindergarten entrants."""

    if achievement_bins < 2:
        raise ValueError("achievement_bins must be at least 2.")

    cohort = cast(
        pd.DataFrame,
        frame.loc[frame["gkclasstype"].notna()].copy(),
    )
    cohort["K_Class"] = cohort["gkclasstype"].map(CLASS_LABELS)
    cohort["Gender"] = cohort["gender"].map(GENDER_LABELS)
    cohort["Ethnicity"] = cohort["race"].map(
        lambda value: (
            ETHNICITY_LABELS.get(1 if value == 1 else 2 if value == 2 else 3)
            if pd.notna(value)
            else np.nan
        )
    )
    cohort["Free_Lunch"] = cohort["gkfreelunch"].map(FREE_LUNCH_LABELS)
    cohort["School_Context"] = cohort["gksurban"].map(SCHOOL_LABELS)
    cohort["Teacher_Experience"] = pd.cut(
        cohort["gktyears"],
        bins=[-1, 5, 10, 20, np.inf],
        labels=["0-5", "6-10", "11-20", "21+"],
    )

    entry_month = 1985 * 12 + 9
    birth_month = cohort["birthyear"] * 12 + cohort["birthmonth"]
    cohort["Entry_Age_Months"] = entry_month - birth_month
    cohort["Entry_Age"] = _quantile_labels(
        cohort["Entry_Age_Months"],
        bins=achievement_bins,
        prefix="Age",
    )

    cohort["K_Score"] = cohort["gktreadss"] + cohort["gktmathss"]
    cohort["Grade3_Score"] = cohort["g3treadss"] + cohort["g3tmathss"]
    cohort["K_Achievement"] = _quantile_labels(
        cohort["K_Score"],
        bins=achievement_bins,
        prefix="K",
    )
    cohort["Grade3_Achievement"] = _quantile_labels(
        cohort["Grade3_Score"],
        bins=achievement_bins,
        prefix="G3",
    )
    cohort["Grade3_Observed"] = np.where(
        cohort[["g3treadss", "g3tmathss"]].notna().all(axis=1),
        "Observed",
        "Not observed",
    )

    common = [
        "K_Class",
        "Gender",
        "Ethnicity",
        "Free_Lunch",
        "School_Context",
        "Entry_Age",
        "Teacher_Experience",
    ]
    panel_columns = {
        "attrition": [*common, "K_Achievement", "Grade3_Observed"],
        "longitudinal": [
            *common,
            "K_Achievement",
            "Grade3_Achievement",
        ],
        "focused_treatment": [*common, "Grade3_Achievement"],
    }

    panels = {
        name: _encode_panel(
            cohort,
            name=name,
            columns=columns,
        )
        for name, columns in panel_columns.items()
    }
    return PreparedStudy(
        raw_rows=len(frame),
        kindergarten_rows=len(cohort),
        kindergarten_schools=int(cohort["gkschid"].nunique()),
        cohort=cohort,
        panels=panels,
    )


def _quantile_labels(
    values: pd.Series,
    bins: int,
    prefix: str,
) -> pd.Series:
    labels = [f"{prefix} Q{index + 1}" for index in range(bins)]
    return cast(
        pd.Series,
        pd.qcut(values, q=bins, labels=labels, duplicates="drop"),
    )


def _encode_panel(
    cohort: pd.DataFrame,
    name: str,
    columns: list[str],
) -> PreparedPanel:
    selected = cohort.loc[:, ["gkschid", *columns]].dropna().copy()
    labels: dict[str, dict[int, str]] = {}
    encoded = pd.DataFrame(index=selected.index)
    for column in columns:
        codes, uniques = pd.factorize(selected[column], sort=True)
        encoded[column] = codes.astype(int)
        labels[column] = {int(index): str(value) for index, value in enumerate(uniques)}
    return PreparedPanel(
        name=name,
        data=encoded.reset_index(drop=True),
        school_ids=selected["gkschid"].astype(int).to_numpy(),
        category_labels=labels,
        description=PANEL_DESCRIPTIONS[name],
    )


def save_processed_panels(study: PreparedStudy) -> dict[str, Path]:
    """Save the exact numeric tables supplied to FCI/FCI+."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, panel in study.panels.items():
        path = PROCESSED_DIR / f"{name}_panel.csv"
        panel.data.to_csv(path, index=False)
        paths[name] = path
    return paths


def run_discovery_suite(
    study: PreparedStudy,
    *,
    alpha: float = 0.05,
    sparsity_bound: int = 3,
    benchmark_repeats: int = 3,
    bootstraps: int = 100,
    n_jobs: int = 4,
    random_state: int = 20260716,
    order_audit: bool = True,
) -> list[DiscoveryRecord]:
    """Fit validation and robust application profiles to every panel."""

    if benchmark_repeats <= 0:
        raise ValueError("benchmark_repeats must be positive.")
    if bootstraps < 0:
        raise ValueError("bootstraps cannot be negative.")

    records: list[DiscoveryRecord] = []
    algorithms: tuple[AlgorithmName, ...] = (
        "fci",
        "fci_plus",
        "fci_plus_robust",
    )
    for panel_index, panel in enumerate(study.panels.values()):
        for algorithm_index, algorithm in enumerate(algorithms):
            results = [
                _fit_panel(
                    panel.data,
                    algorithm=algorithm,
                    alpha=alpha,
                    sparsity_bound=sparsity_bound,
                )
                for _ in range(benchmark_repeats)
            ]
            first = results[0]
            elapsed = tuple(result.elapsed_time for result in results)
            frequencies = (
                cluster_bootstrap_adjacencies(
                    panel,
                    algorithm=algorithm,
                    alpha=alpha,
                    sparsity_bound=sparsity_bound,
                    n_bootstraps=bootstraps,
                    n_jobs=n_jobs,
                    random_state=(
                        random_state + 1000 * panel_index + 100 * algorithm_index
                    ),
                )
                if bootstraps
                else {}
            )
            records.append(
                DiscoveryRecord(
                    panel=panel.name,
                    algorithm=algorithm,
                    result=first,
                    elapsed_runs=elapsed,
                    median_elapsed=float(median(elapsed)),
                    bootstrap_adjacencies=frequencies,
                    temporal_flags=tuple(temporal_orientation_flags(first)),
                    order_audit=(
                        cyclic_order_audit(
                            panel.data,
                            algorithm=algorithm,
                            alpha=alpha,
                            sparsity_bound=sparsity_bound,
                            baseline=first,
                        )
                        if order_audit
                        else {}
                    ),
                )
            )
    return records


def _fit_panel(
    data: pd.DataFrame,
    *,
    algorithm: AlgorithmName,
    alpha: float,
    sparsity_bound: int,
) -> FCIResult:
    ci_test = GSquareTest(alpha=alpha)
    if algorithm == "fci":
        return fci(
            data,
            profile="paper",
            alpha=alpha,
            ci_test=ci_test,
        )
    if algorithm == "fci_plus":
        return fci_plus(
            data,
            profile="paper",
            k=sparsity_bound,
            alpha=alpha,
            ci_test=ci_test,
        )
    return fci_plus(
        data,
        profile="practical",
        max_cond_set_size=sparsity_bound,
        sparsity_bound=sparsity_bound,
        alpha=alpha,
        ci_test=ci_test,
    )


def cluster_bootstrap_adjacencies(
    panel: PreparedPanel,
    *,
    algorithm: AlgorithmName,
    alpha: float,
    sparsity_bound: int,
    n_bootstraps: int,
    n_jobs: int,
    random_state: int,
) -> dict[tuple[str, str], float]:
    """Estimate adjacency stability by resampling kindergarten schools."""

    if n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be positive.")
    if n_jobs <= 0:
        raise ValueError("n_jobs must be positive.")

    unique_schools = np.unique(panel.school_ids)
    school_rows = {
        school: np.flatnonzero(panel.school_ids == school) for school in unique_schools
    }
    rng = np.random.default_rng(random_state)
    samples = [
        rng.choice(unique_schools, size=len(unique_schools), replace=True)
        for _ in range(n_bootstraps)
    ]

    def run(sampled_schools: np.ndarray[Any, np.dtype[Any]]) -> set[tuple[str, str]]:
        row_indices = np.concatenate(
            [school_rows[int(school)] for school in sampled_schools]
        )
        sampled = panel.data.iloc[row_indices].reset_index(drop=True)
        result = _fit_panel(
            sampled,
            algorithm=algorithm,
            alpha=alpha,
            sparsity_bound=sparsity_bound,
        )
        return {_edge_key(x, y) for x, y in result.edges}

    if n_jobs == 1:
        replicate_edges = [run(sample) for sample in samples]
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            replicate_edges = list(executor.map(run, samples))

    counts: Counter[tuple[str, str]] = Counter()
    for edges in replicate_edges:
        counts.update(edges)
    return {edge: count / n_bootstraps for edge, count in sorted(counts.items())}


def descriptive_statistics(
    study: PreparedStudy,
    *,
    bootstrap_repeats: int = 1000,
    random_state: int = 91,
) -> dict[str, Any]:
    """Return class-arm outcomes and school-cluster bootstrap intervals."""

    cohort = study.cohort
    arms = ["Regular", "Regular + aide", "Small"]
    metrics: dict[str, tuple[str, Callable[[pd.Series], float]]] = {
        "kindergarten_score": ("K_Score", lambda series: float(series.mean())),
        "grade3_score": ("Grade3_Score", lambda series: float(series.mean())),
        "grade3_observed_rate": (
            "Grade3_Observed",
            lambda series: float((series == "Observed").mean()),
        ),
    }

    summaries: dict[str, dict[str, Any]] = {}
    for metric, (column, function) in metrics.items():
        arm_values = {}
        for arm in arms:
            values = cohort.loc[cohort["K_Class"] == arm, column].dropna()
            arm_values[arm] = {
                "n": int(len(values)),
                "estimate": function(values),
            }
        summaries[metric] = arm_values

    bootstrap = _cluster_bootstrap_arm_metrics(
        cohort,
        arms=arms,
        metrics=metrics,
        repeats=bootstrap_repeats,
        random_state=random_state,
    )
    for metric, arm_results in bootstrap["arms"].items():
        for arm, interval in arm_results.items():
            summaries[metric][arm]["ci_low"] = interval[0]
            summaries[metric][arm]["ci_high"] = interval[1]

    contrasts = []
    for metric, contrast_results in bootstrap["contrasts"].items():
        for comparison, values in contrast_results.items():
            estimate = (
                summaries[metric][comparison]["estimate"]
                - summaries[metric]["Regular"]["estimate"]
            )
            contrasts.append(
                {
                    "metric": metric,
                    "comparison": f"{comparison} - Regular",
                    "estimate": estimate,
                    "ci_low": values[0],
                    "ci_high": values[1],
                }
            )

    return {
        "arms": summaries,
        "contrasts": contrasts,
        "bootstrap_unit": "kindergarten school",
        "bootstrap_repeats": bootstrap_repeats,
    }


def _cluster_bootstrap_arm_metrics(
    cohort: pd.DataFrame,
    *,
    arms: list[str],
    metrics: dict[str, tuple[str, Callable[[pd.Series], float]]],
    repeats: int,
    random_state: int,
) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    schools = np.array(sorted(cohort["gkschid"].dropna().unique()), dtype=int)
    blocks = {school: cohort.loc[cohort["gkschid"] == school] for school in schools}
    arm_draws: dict[str, dict[str, list[float]]] = {
        metric: {arm: [] for arm in arms} for metric in metrics
    }
    contrast_draws: dict[str, dict[str, list[float]]] = {
        metric: {arm: [] for arm in arms if arm != "Regular"} for metric in metrics
    }

    rng = np.random.default_rng(random_state)
    for _ in range(repeats):
        chosen = rng.choice(schools, size=len(schools), replace=True)
        sample = pd.concat([blocks[int(school)] for school in chosen])
        estimates: dict[str, dict[str, float]] = {}
        for metric, (column, function) in metrics.items():
            estimates[metric] = {}
            for arm in arms:
                values = sample.loc[sample["K_Class"] == arm, column].dropna()
                value = function(values)
                estimates[metric][arm] = value
                arm_draws[metric][arm].append(value)
            for arm in arms:
                if arm == "Regular":
                    continue
                contrast_draws[metric][arm].append(
                    estimates[metric][arm] - estimates[metric]["Regular"]
                )

    return {
        "arms": {
            metric: {
                arm: _percentile_interval(draws) for arm, draws in arm_results.items()
            }
            for metric, arm_results in arm_draws.items()
        },
        "contrasts": {
            metric: {
                arm: _percentile_interval(draws)
                for arm, draws in metric_results.items()
            }
            for metric, metric_results in contrast_draws.items()
        },
    }


def _percentile_interval(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    quantiles = np.asarray(np.quantile(array, [0.025, 0.975]), dtype=float)
    return float(quantiles[0]), float(quantiles[1])


def sensitivity_analysis(
    frame: pd.DataFrame,
    *,
    alphas: tuple[float, ...] = (0.01, 0.05),
    bin_counts: tuple[int, ...] = (3, 4),
    sparsity_bounds: tuple[int, ...] = (2, 3, 4),
) -> list[dict[str, Any]]:
    """Test the focused target across alpha, binning, and FCI+ degree bounds.

    Standard FCI has no FCI+ sparsity bound, so it is fitted once per
    ``(bins, alpha)`` cell. Both FCI+ profiles are fitted at every supplied
    bound. The bound is an analysis setting: this sensitivity grid does not
    assert that the true STAR MAG has maximum degree ``k``.
    """

    if not sparsity_bounds:
        raise ValueError("sparsity_bounds must contain at least one value.")
    if any(bound < 0 for bound in sparsity_bounds):
        raise ValueError("sparsity_bounds must be non-negative.")

    rows = []
    for bins in bin_counts:
        panel = prepare_study(frame, achievement_bins=bins).panels["focused_treatment"]
        for alpha in alphas:
            configurations: list[tuple[AlgorithmName, int | None]] = [("fci", None)]
            fci_plus_algorithms: tuple[AlgorithmName, ...] = (
                "fci_plus",
                "fci_plus_robust",
            )
            configurations.extend(
                (algorithm, bound)
                for algorithm in fci_plus_algorithms
                for bound in sparsity_bounds
            )
            for algorithm, bound in configurations:
                result = _fit_panel(
                    panel.data,
                    algorithm=algorithm,
                    alpha=alpha,
                    sparsity_bound=(max(sparsity_bounds) if bound is None else bound),
                )
                edge = edge_for_pair(
                    result,
                    "K_Class",
                    "Grade3_Achievement",
                )
                rows.append(
                    {
                        "bins": bins,
                        "alpha": alpha,
                        "algorithm": algorithm,
                        "sparsity_bound": bound,
                        "adjacent": edge is not None,
                        "edge": edge,
                        "edge_count": len(result.edges),
                        "ci_tests": result.ci_test_count,
                    }
                )
    return rows


def edge_for_pair(
    result: FCIResult,
    x: str,
    y: str,
) -> str | None:
    """Return the learned edge representation for an unordered node pair."""

    if result.graph.is_adjacent(x, y):
        return result.graph.edge_repr(x, y)
    return None


def temporal_orientation_flags(result: FCIResult) -> list[str]:
    """Flag directed arrows that contradict the known measurement ordering."""

    flags = []
    for x, y in result.edges:
        if result.graph.is_directed_edge(x, y):
            source, target = x, y
        elif result.graph.is_directed_edge(y, x):
            source, target = y, x
        else:
            continue
        source_tier = TEMPORAL_TIERS.get(source)
        target_tier = TEMPORAL_TIERS.get(target)
        if (
            source_tier is not None
            and target_tier is not None
            and source_tier > target_tier
        ):
            flags.append(f"{source} --> {target}")
    return sorted(flags)


def skeleton_jaccard(left: FCIResult, right: FCIResult) -> float:
    """Return skeleton Jaccard agreement between two fitted PAGs."""

    left_edges = {_edge_key(x, y) for x, y in left.edges}
    right_edges = {_edge_key(x, y) for x, y in right.edges}
    union = left_edges | right_edges
    return 1.0 if not union else len(left_edges & right_edges) / len(union)


def cyclic_order_audit(
    data: pd.DataFrame,
    *,
    algorithm: AlgorithmName,
    alpha: float,
    sparsity_bound: int,
    baseline: FCIResult | None = None,
) -> dict[str, Any]:
    """Refit every cyclic column shift and quantify order sensitivity."""

    columns = list(data.columns)
    if not columns:
        raise ValueError("Order audit requires at least one column.")
    reference = baseline or _fit_panel(
        data,
        algorithm=algorithm,
        alpha=alpha,
        sparsity_bound=sparsity_bound,
    )
    reference_shape = _pag_shape(reference)
    if "Grade3_Observed" in columns:
        target = ("K_Achievement", "Grade3_Observed")
    elif "K_Achievement" in columns:
        target = ("K_Achievement", "Grade3_Achievement")
    else:
        target = ("K_Class", "Grade3_Achievement")
    rows: list[dict[str, Any]] = []
    for shift in range(len(columns)):
        ordering = columns[shift:] + columns[:shift]
        result = (
            reference
            if shift == 0 and tuple(reference.nodes) == tuple(columns)
            else _fit_panel(
                data.loc[:, ordering],
                algorithm=algorithm,
                alpha=alpha,
                sparsity_bound=sparsity_bound,
            )
        )
        rows.append(
            {
                "ordering": f"cyclic_shift_{shift}",
                "first_node": ordering[0],
                "exact_pag_match": _pag_shape(result) == reference_shape,
                "skeleton_jaccard": skeleton_jaccard(reference, result),
                "edges": len(result.edges),
                "ci_tests": result.ci_test_count,
                "elapsed_seconds": result.elapsed_time,
                "target_edge": edge_for_pair(result, *target),
                "temporal_flags": temporal_orientation_flags(result),
            }
        )
    exact_matches = sum(bool(row["exact_pag_match"]) for row in rows)
    target_matches = sum(row["target_edge"] is not None for row in rows)
    temporal_failures = sum(bool(row["temporal_flags"]) for row in rows)
    return {
        "exact_pag_match_rate": exact_matches / len(rows),
        "mean_skeleton_jaccard": float(
            np.mean([float(row["skeleton_jaccard"]) for row in rows])
        ),
        "minimum_skeleton_jaccard": min(float(row["skeleton_jaccard"]) for row in rows),
        "target_adjacency_rate": target_matches / len(rows),
        "temporal_reversal_rate": temporal_failures / len(rows),
        "ci_test_range": [
            min(int(row["ci_tests"]) for row in rows),
            max(int(row["ci_tests"]) for row in rows),
        ],
        "rows": rows,
    }


def _pag_shape(result: FCIResult) -> dict[tuple[str, str], tuple[str, str]]:
    return {
        _edge_key(str(edge["x"]), str(edge["y"])): _canonical_endpoints(edge)
        for edge in result.to_edge_records()
    }


def _edge_key(x: str, y: str) -> tuple[str, str]:
    left, right = str(x), str(y)
    return (left, right) if left <= right else (right, left)


def records_by_panel(
    records: list[DiscoveryRecord],
) -> dict[str, dict[AlgorithmName, DiscoveryRecord]]:
    """Index records by panel and algorithm."""

    indexed: dict[str, dict[AlgorithmName, DiscoveryRecord]] = {}
    for record in records:
        indexed.setdefault(record.panel, {})[record.algorithm] = record
    return indexed


def build_summary_payload(
    study: PreparedStudy,
    records: list[DiscoveryRecord],
    descriptives: dict[str, Any],
    sensitivity: list[dict[str, Any]],
    *,
    alpha: float,
    sparsity_bound: int,
    bootstraps: int,
    external_records: list[dict[str, Any]] | None = None,
    external_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON/report data contract for the completed case study."""

    indexed = records_by_panel(records)
    comparisons = {}
    for panel, algorithms in indexed.items():
        comparisons[panel] = {
            "skeleton_jaccard": skeleton_jaccard(
                algorithms["fci"].result,
                algorithms["fci_plus"].result,
            ),
            "fci_plus_runtime_ratio": (
                algorithms["fci_plus"].median_elapsed / algorithms["fci"].median_elapsed
            ),
            "fci_plus_ci_test_ratio": (
                algorithms["fci_plus"].ci_test_count / algorithms["fci"].ci_test_count
            ),
        }

    run_rows = []
    for record in records:
        run_rows.append(
            {
                "panel": record.panel,
                "algorithm": record.algorithm,
                "samples": record.result.n_samples,
                "nodes": len(record.result.nodes),
                "node_names": list(record.result.nodes),
                "edges": record.edge_count,
                "ci_tests": record.ci_test_count,
                "median_elapsed_seconds": record.median_elapsed,
                "elapsed_runs_seconds": list(record.elapsed_runs),
                "temporal_flags": list(record.temporal_flags),
                "pag_edges": record.result.to_edge_records(),
                "bootstrap_adjacencies": [
                    {
                        "x": edge[0],
                        "y": edge[1],
                        "frequency": frequency,
                    }
                    for edge, frequency in record.bootstrap_adjacencies.items()
                ],
                "assumption_notes": record.result.assumption_notes(),
                "order_audit": record.order_audit,
            }
        )
    if external_records:
        run_rows.extend(external_records)

    return {
        "source": {
            "title": "Tennessee Student/Teacher Achievement Ratio (STAR)",
            "doi": DATASET_DOI,
            "student_file": str(STUDENT_PATH),
            "user_guide": str(GUIDE_PATH),
            "license": "CC0 1.0",
        },
        "configuration": {
            "ci_test": "G-square on discretized variables",
            "alpha": alpha,
            "fci_profile": "paper",
            "fci_plus_profile": "paper",
            "fci_plus_k": sparsity_bound,
            "robust_application_profile": "practical",
            "robust_application_options": {
                "stable_skeleton": True,
                "sepset_selection": "max_pvalue",
                "conservative_colliders": True,
                "orientation_strategy": "robust",
                "possible_dsep_stage": False,
                "max_cond_set_size": sparsity_bound,
                "sparsity_bound": sparsity_bound,
            },
            "cluster_bootstraps": bootstraps,
            "external_reference": external_metadata,
        },
        "cohorts": {
            "raw_rows": study.raw_rows,
            "kindergarten_rows": study.kindergarten_rows,
            "kindergarten_schools": study.kindergarten_schools,
            "panels": {
                name: {
                    "rows": len(panel.data),
                    "nodes": len(panel.data.columns),
                    "description": panel.description,
                }
                for name, panel in study.panels.items()
            },
        },
        "descriptives": descriptives,
        "runs": run_rows,
        "comparisons": comparisons,
        "three_algorithm_comparisons": _three_algorithm_comparisons(run_rows),
        "robust_application_comparisons": _robust_application_comparisons(run_rows),
        "sensitivity": sensitivity,
    }


def _three_algorithm_comparisons(
    run_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    algorithms = ("fci", "fci_plus", "pcalg_fci_plus")
    indexed = {(str(run["panel"]), str(run["algorithm"])): run for run in run_rows}
    output: dict[str, Any] = {}
    for panel in PANEL_DESCRIPTIONS:
        if any((panel, algorithm) not in indexed for algorithm in algorithms):
            continue
        panel_runs = {
            algorithm: indexed[(panel, algorithm)] for algorithm in algorithms
        }
        skeletons = {
            algorithm: {
                _edge_key(str(edge["x"]), str(edge["y"])) for edge in run["pag_edges"]
            }
            for algorithm, run in panel_runs.items()
        }
        endpoint_shapes = {
            algorithm: {
                _edge_key(str(edge["x"]), str(edge["y"])): _canonical_endpoints(edge)
                for edge in run["pag_edges"]
            }
            for algorithm, run in panel_runs.items()
        }
        pairs = []
        for left, right in (
            ("fci", "fci_plus"),
            ("fci", "pcalg_fci_plus"),
            ("fci_plus", "pcalg_fci_plus"),
        ):
            union = skeletons[left] | skeletons[right]
            common = skeletons[left] & skeletons[right]
            exact_matches = sum(
                endpoint_shapes[left][edge] == endpoint_shapes[right][edge]
                for edge in common
            )
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "skeleton_jaccard": (
                        1.0 if not union else len(common) / len(union)
                    ),
                    "common_edges": len(common),
                    "exact_endpoint_matches": exact_matches,
                    "endpoint_match_rate": (
                        1.0 if not common else exact_matches / len(common)
                    ),
                }
            )

        consensus_skeleton = set.intersection(*skeletons.values())
        unanimous_endpoint_edges = {
            edge
            for edge in consensus_skeleton
            if len({endpoint_shapes[algorithm][edge] for algorithm in algorithms}) == 1
        }
        output[panel] = {
            "pairs": pairs,
            "consensus_skeleton_edges": len(consensus_skeleton),
            "unanimous_endpoint_edges": len(unanimous_endpoint_edges),
            "union_skeleton_edges": len(set.union(*skeletons.values())),
            "all_three_exact_pag": all(
                endpoint_shapes[algorithm] == endpoint_shapes[algorithms[0]]
                for algorithm in algorithms[1:]
            ),
            "target_edges": _target_edge_comparison(panel_runs, panel),
        }
    return output


def _robust_application_comparisons(
    run_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    indexed = {(str(run["panel"]), str(run["algorithm"])): run for run in run_rows}
    output: dict[str, Any] = {}
    for panel in PANEL_DESCRIPTIONS:
        paper = indexed.get((panel, "fci_plus"))
        robust = indexed.get((panel, "fci_plus_robust"))
        if paper is None or robust is None:
            continue
        paper_skeleton = {
            _edge_key(str(edge["x"]), str(edge["y"])) for edge in paper["pag_edges"]
        }
        robust_skeleton = {
            _edge_key(str(edge["x"]), str(edge["y"])) for edge in robust["pag_edges"]
        }
        union = paper_skeleton | robust_skeleton
        common = paper_skeleton & robust_skeleton
        target = {
            "attrition": ("K_Achievement", "Grade3_Observed"),
            "longitudinal": ("K_Achievement", "Grade3_Achievement"),
            "focused_treatment": ("K_Class", "Grade3_Achievement"),
        }[panel]
        target_key = _edge_key(*target)

        def target_edge(run: dict[str, Any]) -> str | None:
            return next(
                (
                    str(edge["edge"])
                    for edge in run["pag_edges"]
                    if _edge_key(str(edge["x"]), str(edge["y"])) == target_key
                ),
                None,
            )

        output[panel] = {
            "paper_target_edge": target_edge(paper),
            "robust_target_edge": target_edge(robust),
            "skeleton_jaccard": 1.0 if not union else len(common) / len(union),
            "paper_exact_order_rate": paper["order_audit"].get("exact_pag_match_rate"),
            "robust_exact_order_rate": robust["order_audit"].get(
                "exact_pag_match_rate"
            ),
            "paper_minimum_skeleton_jaccard": paper["order_audit"].get(
                "minimum_skeleton_jaccard"
            ),
            "robust_minimum_skeleton_jaccard": robust["order_audit"].get(
                "minimum_skeleton_jaccard"
            ),
            "paper_temporal_flags": len(paper["temporal_flags"]),
            "robust_temporal_flags": len(robust["temporal_flags"]),
            "ci_test_ratio": robust["ci_tests"] / paper["ci_tests"],
            "runtime_ratio": (
                robust["median_elapsed_seconds"] / paper["median_elapsed_seconds"]
            ),
        }
    return output


def _canonical_endpoints(edge: dict[str, Any]) -> tuple[str, str]:
    x = str(edge["x"])
    y = str(edge["y"])
    endpoint_x = str(edge["endpoint_x"])
    endpoint_y = str(edge["endpoint_y"])
    if x <= y:
        return endpoint_x, endpoint_y
    return endpoint_y, endpoint_x


def _target_edge_comparison(
    panel_runs: dict[str, dict[str, Any]],
    panel: str,
) -> dict[str, str | None]:
    target = {
        "attrition": ("K_Achievement", "Grade3_Observed"),
        "longitudinal": ("K_Achievement", "Grade3_Achievement"),
        "focused_treatment": ("K_Class", "Grade3_Achievement"),
    }[panel]
    target_key = _edge_key(*target)
    return {
        algorithm: next(
            (
                str(edge["edge"])
                for edge in run["pag_edges"]
                if _edge_key(str(edge["x"]), str(edge["y"])) == target_key
            ),
            None,
        )
        for algorithm, run in panel_runs.items()
    }
