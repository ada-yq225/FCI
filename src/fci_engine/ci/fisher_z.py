"""Fisher-Z conditional independence test for Gaussian data."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import norm

from fci_engine.ci.base import CITest, CITestResult


class FisherZTest(CITest):
    """Fisher-Z test based on partial correlations.

    The input data must be a two-dimensional NumPy array. Variables are
    referenced by integer column indices.
    """

    method = "fisher_z"

    def test(
        self,
        data: np.ndarray,
        x: int,
        y: int,
        cond_set: Sequence[int] = (),
    ) -> CITestResult:
        data = self._validate_data(data)
        n_samples, n_features = data.shape
        cond_tuple = tuple(cond_set)
        self._validate_indices(n_features, x, y, cond_tuple)

        partial_corr = self._partial_correlation(data, x, y, cond_tuple)
        partial_corr = self._clip_correlation(partial_corr)

        effective_n = n_samples - len(cond_tuple) - 3
        if effective_n <= 0:
            statistic = 0.0
            p_value = 1.0
        else:
            statistic = float(np.sqrt(effective_n) * np.arctanh(partial_corr))
            p_value = float(2.0 * norm.sf(abs(statistic)))

        return CITestResult(
            independent=p_value > self.alpha,
            p_value=p_value,
            statistic=statistic,
            method=self.method,
            n_samples=n_samples,
        )

    @staticmethod
    def _validate_data(data: np.ndarray) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            raise TypeError("FisherZTest expects data to be a numpy.ndarray.")
        if data.ndim != 2:
            raise ValueError("FisherZTest expects a two-dimensional data array.")
        if data.shape[0] < 4:
            raise ValueError("FisherZTest requires at least four samples.")
        return np.asarray(data, dtype=float)

    @staticmethod
    def _validate_indices(
        n_features: int,
        x: int,
        y: int,
        cond_set: tuple[int, ...],
    ) -> None:
        indices = (x, y, *cond_set)
        if len(set(indices)) != len(indices):
            raise ValueError("x, y, and cond_set must refer to distinct columns.")
        for index in indices:
            if not isinstance(index, int):
                raise TypeError("FisherZTest variable indices must be integers.")
            if index < 0 or index >= n_features:
                raise IndexError(f"Column index out of bounds: {index}.")

    @classmethod
    def _partial_correlation(
        cls,
        data: np.ndarray,
        x: int,
        y: int,
        cond_set: tuple[int, ...],
    ) -> float:
        variables = (x, y, *cond_set)
        corr = np.corrcoef(data[:, variables], rowvar=False)
        if corr.shape == ():
            return 0.0
        if not np.all(np.isfinite(corr)):
            raise ValueError("Cannot compute correlations for non-finite data.")

        if not cond_set:
            return float(corr[0, 1])

        precision = cls._safe_inverse(corr)
        denom = precision[0, 0] * precision[1, 1]
        if denom <= 0.0:
            return 0.0
        return float(-precision[0, 1] / np.sqrt(denom))

    @staticmethod
    def _safe_inverse(matrix: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(matrix)

    @staticmethod
    def _clip_correlation(correlation: float) -> float:
        eps = np.finfo(float).eps
        return float(np.clip(correlation, -1.0 + eps, 1.0 - eps))
