"""Bootstrap stability diagnostics for FCI outputs."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from math import ceil
from typing import Any, Literal, Optional, overload

import numpy as np
import pandas as pd
from typing_extensions import Unpack

from fci_engine.config import (
    FCIOptions,
    FCIPlusConfig,
    FCIPlusPaperOptions,
)
from fci_engine.discovery.base import BaseFCIEstimator
from fci_engine.discovery.fci import FCI
from fci_engine.discovery.fci_plus import FCIPlus
from fci_engine.graph import PAG
from fci_engine.result import FCIResult
from fci_engine.types import Array


def bootstrap_edge_frequencies(
    data: object,
    n_bootstraps: int = 20,
    sample_fraction: float = 1.0,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    **fci_kwargs: Unpack[FCIOptions],
) -> dict[str, float]:
    """Return exact PAG edge representation frequencies over bootstrap runs."""

    if n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be positive.")
    if sample_fraction <= 0.0:
        raise ValueError("sample_fraction must be positive.")

    counts: Counter[str] = Counter()
    results = _bootstrap_results(
        FCI,
        data,
        n_bootstraps=n_bootstraps,
        sample_fraction=sample_fraction,
        random_state=random_state,
        n_jobs=n_jobs,
        fci_kwargs=dict(fci_kwargs),
    )
    for result in results:
        for x, y in result.graph.edges():
            counts[result.graph.edge_repr(x, y)] += 1

    return {
        edge: count / n_bootstraps
        for edge, count in sorted(counts.items(), key=lambda item: item[0])
    }


def bootstrap_adjacency_frequencies(
    data: object,
    n_bootstraps: int = 20,
    sample_fraction: float = 1.0,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    **fci_kwargs: Unpack[FCIOptions],
) -> dict[tuple[str, str], float]:
    """Return unordered adjacency frequencies over bootstrap FCI runs."""

    if n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be positive.")
    if sample_fraction <= 0.0:
        raise ValueError("sample_fraction must be positive.")

    counts: Counter[tuple[str, str]] = Counter()
    results = _bootstrap_results(
        FCI,
        data,
        n_bootstraps=n_bootstraps,
        sample_fraction=sample_fraction,
        random_state=random_state,
        n_jobs=n_jobs,
        fci_kwargs=dict(fci_kwargs),
    )
    for result in results:
        for x, y in result.graph.edges():
            counts[(str(x), str(y))] += 1

    return {
        edge: count / n_bootstraps
        for edge, count in sorted(counts.items(), key=lambda item: item[0])
    }


def stable_fci(
    data: object,
    n_bootstraps: int = 20,
    edge_threshold: float = 0.5,
    sample_fraction: float = 1.0,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    **fci_kwargs: Unpack[FCIOptions],
) -> FCIResult:
    """Run FCI and remove edges with low bootstrap adjacency frequency.

    This is a stability-selection wrapper around standard FCI. It intentionally
    does not change the FCI orientation rules; it only filters the final graph's
    skeleton by bootstrap support.
    """

    return _stable_discovery(
        FCI,
        data,
        n_bootstraps=n_bootstraps,
        edge_threshold=edge_threshold,
        sample_fraction=sample_fraction,
        random_state=random_state,
        n_jobs=n_jobs,
        **fci_kwargs,
    )


@overload
def stable_fci_plus(
    data: object,
    n_bootstraps: int = 20,
    edge_threshold: float = 0.5,
    sample_fraction: float = 1.0,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    profile: None = None,
    **fci_kwargs: Unpack[FCIOptions],
) -> FCIResult: ...


@overload
def stable_fci_plus(
    data: object,
    n_bootstraps: int = 20,
    edge_threshold: float = 0.5,
    sample_fraction: float = 1.0,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    profile: Literal["practical"] = "practical",
    **fci_kwargs: Unpack[FCIOptions],
) -> FCIResult: ...


@overload
def stable_fci_plus(
    data: object,
    n_bootstraps: int = 20,
    edge_threshold: float = 0.5,
    sample_fraction: float = 1.0,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    profile: Literal["paper", "paper_aligned"] = "paper",
    **fci_kwargs: Unpack[FCIPlusPaperOptions],
) -> FCIResult: ...


def stable_fci_plus(
    data: object,
    n_bootstraps: int = 20,
    edge_threshold: float = 0.5,
    sample_fraction: float = 1.0,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    profile: Optional[str] = None,
    **fci_kwargs: Any,
) -> FCIResult:
    """Run FCI+ and remove edges with low bootstrap adjacency frequency.

    This is the same stability-selection wrapper as ``stable_fci`` but uses the
    FCI+ sparse hierarchical D-SEP pipeline for both the full-data fit and each
    bootstrap replicate.
    """

    if profile is not None:
        config = FCIPlusConfig.from_profile(profile, **fci_kwargs)
        fci_kwargs = {"config": config}

    return _stable_discovery(
        FCIPlus,
        data,
        n_bootstraps=n_bootstraps,
        edge_threshold=edge_threshold,
        sample_fraction=sample_fraction,
        random_state=random_state,
        n_jobs=n_jobs,
        **fci_kwargs,
    )


def _stable_discovery(
    estimator_cls: type[BaseFCIEstimator],
    data: object,
    n_bootstraps: int,
    edge_threshold: float,
    sample_fraction: float,
    random_state: Optional[int],
    n_jobs: int,
    **fci_kwargs: Any,
) -> FCIResult:
    if not 0.0 <= edge_threshold <= 1.0:
        raise ValueError("edge_threshold must be between 0 and 1.")

    result = estimator_cls(**fci_kwargs).fit(data)
    frequencies = _bootstrap_adjacency_frequencies(
        estimator_cls,
        data,
        n_bootstraps=n_bootstraps,
        sample_fraction=sample_fraction,
        random_state=random_state,
        n_jobs=n_jobs,
        **fci_kwargs,
    )

    graph = result.graph.copy()
    for x, y in list(graph.edges()):
        if frequencies.get((str(x), str(y)), 0.0) < edge_threshold:
            graph.remove_edge(x, y)

    return replace(
        result,
        graph=graph,
        bootstrap_edge_frequencies=_edge_repr_frequencies(graph, frequencies),
    )


def _bootstrap_adjacency_frequencies(
    estimator_cls: type[BaseFCIEstimator],
    data: object,
    n_bootstraps: int = 20,
    sample_fraction: float = 1.0,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    **fci_kwargs: Any,
) -> dict[tuple[str, str], float]:
    if n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be positive.")
    if sample_fraction <= 0.0:
        raise ValueError("sample_fraction must be positive.")

    counts: Counter[tuple[str, str]] = Counter()
    results = _bootstrap_results(
        estimator_cls,
        data,
        n_bootstraps=n_bootstraps,
        sample_fraction=sample_fraction,
        random_state=random_state,
        n_jobs=n_jobs,
        fci_kwargs=fci_kwargs,
    )
    for result in results:
        for x, y in result.graph.edges():
            counts[(str(x), str(y))] += 1

    return {
        edge: count / n_bootstraps
        for edge, count in sorted(counts.items(), key=lambda item: item[0])
    }


def _bootstrap_results(
    estimator_cls: type[BaseFCIEstimator],
    data: object,
    *,
    n_bootstraps: int,
    sample_fraction: float,
    random_state: Optional[int],
    n_jobs: int,
    fci_kwargs: dict[str, Any],
) -> list[FCIResult]:
    if n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be positive.")
    if sample_fraction <= 0.0:
        raise ValueError("sample_fraction must be positive.")
    if n_jobs <= 0:
        raise ValueError("n_jobs must be positive.")

    n_samples = _n_rows(data)
    if n_samples == 0:
        raise ValueError("Cannot bootstrap an empty dataset.")

    bootstrap_size = max(1, int(ceil(n_samples * sample_fraction)))
    rng = np.random.default_rng(random_state)
    index_sets = [
        rng.integers(0, n_samples, size=bootstrap_size) for _ in range(n_bootstraps)
    ]

    def fit_one(indices: Array) -> FCIResult:
        kwargs = fci_kwargs if n_jobs == 1 else deepcopy(fci_kwargs)
        sample = _sample_rows(data, indices)
        return estimator_cls(**kwargs).fit(sample)

    if n_jobs == 1:
        return [fit_one(indices) for indices in index_sets]
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        return list(executor.map(fit_one, index_sets))


def _edge_repr_frequencies(
    graph: PAG,
    adjacency_frequencies: dict[tuple[str, str], float],
) -> dict[str, float]:
    return {
        graph.edge_repr(x, y): adjacency_frequencies.get((str(x), str(y)), 0.0)
        for x, y in graph.edges()
    }


def _n_rows(data: object) -> int:
    if isinstance(data, pd.DataFrame):
        return int(data.shape[0])
    array = np.asarray(data)
    if array.ndim == 0:
        raise ValueError("data must be row-oriented.")
    return int(array.shape[0])


def _sample_rows(data: object, indices: Array) -> object:
    if isinstance(data, pd.DataFrame):
        return data.iloc[indices].reset_index(drop=True)
    return np.asarray(data)[indices]
