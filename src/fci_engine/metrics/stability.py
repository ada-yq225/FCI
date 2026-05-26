"""Bootstrap stability diagnostics for FCI outputs."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from math import ceil
from typing import Optional

import numpy as np
import pandas as pd

from fci_engine.discovery.fci import FCI
from fci_engine.result import FCIResult


def bootstrap_edge_frequencies(
    data: object,
    n_bootstraps: int = 20,
    sample_fraction: float = 1.0,
    random_state: Optional[int] = None,
    **fci_kwargs: object,
) -> dict[str, float]:
    """Return exact PAG edge representation frequencies over bootstrap runs."""

    if n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be positive.")
    if sample_fraction <= 0.0:
        raise ValueError("sample_fraction must be positive.")

    n_samples = _n_rows(data)
    if n_samples == 0:
        raise ValueError("Cannot bootstrap an empty dataset.")

    bootstrap_size = max(1, int(ceil(n_samples * sample_fraction)))
    rng = np.random.default_rng(random_state)
    counts: Counter[str] = Counter()

    for _ in range(n_bootstraps):
        indices = rng.integers(0, n_samples, size=bootstrap_size)
        sample = _sample_rows(data, indices)
        result = FCI(**fci_kwargs).fit(sample)
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
    **fci_kwargs: object,
) -> dict[tuple[str, str], float]:
    """Return unordered adjacency frequencies over bootstrap FCI runs."""

    if n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be positive.")
    if sample_fraction <= 0.0:
        raise ValueError("sample_fraction must be positive.")

    n_samples = _n_rows(data)
    if n_samples == 0:
        raise ValueError("Cannot bootstrap an empty dataset.")

    bootstrap_size = max(1, int(ceil(n_samples * sample_fraction)))
    rng = np.random.default_rng(random_state)
    counts: Counter[tuple[str, str]] = Counter()

    for _ in range(n_bootstraps):
        indices = rng.integers(0, n_samples, size=bootstrap_size)
        sample = _sample_rows(data, indices)
        result = FCI(**fci_kwargs).fit(sample)
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
    **fci_kwargs: object,
) -> FCIResult:
    """Run FCI and remove edges with low bootstrap adjacency frequency.

    This is a stability-selection wrapper around standard FCI. It intentionally
    does not change the FCI orientation rules; it only filters the final graph's
    skeleton by bootstrap support.
    """

    if not 0.0 <= edge_threshold <= 1.0:
        raise ValueError("edge_threshold must be between 0 and 1.")

    result = FCI(**fci_kwargs).fit(data)
    frequencies = bootstrap_adjacency_frequencies(
        data,
        n_bootstraps=n_bootstraps,
        sample_fraction=sample_fraction,
        random_state=random_state,
        **fci_kwargs,
    )

    graph = result.graph.copy()
    for x, y in list(graph.edges()):
        if frequencies.get((str(x), str(y)), 0.0) < edge_threshold:
            graph.remove_edge(x, y)

    return replace(
        result,
        graph=graph,
        bootstrap_edge_frequencies={
            f"{x} -- {y}": frequency for (x, y), frequency in frequencies.items()
        },
    )


def _n_rows(data: object) -> int:
    if isinstance(data, pd.DataFrame):
        return int(data.shape[0])
    array = np.asarray(data)
    if array.ndim == 0:
        raise ValueError("data must be row-oriented.")
    return int(array.shape[0])


def _sample_rows(data: object, indices: np.ndarray) -> object:
    if isinstance(data, pd.DataFrame):
        return data.iloc[indices].reset_index(drop=True)
    return np.asarray(data)[indices]
