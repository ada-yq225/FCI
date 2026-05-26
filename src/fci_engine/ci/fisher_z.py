"""Fisher-Z conditional independence test for Gaussian data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
        data: object,
        x: int,
        y: int,
        cond_set: Sequence[int] = (),
    ) -> CITestResult:
        corr, n_samples, n_features = self._correlation_matrix(data)
        cond_tuple = tuple(cond_set)
        self._validate_indices(n_features, x, y, cond_tuple)

        partial_corr = self._partial_correlation_from_corr(corr, x, y, cond_tuple)
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

    @classmethod
    def _correlation_matrix(cls, data: object) -> tuple[np.ndarray, int, int]:
        if isinstance(data, Mapping):
            return cls._correlation_from_sufficient_stats(data)
        data = cls._validate_data(data)
        corr = np.corrcoef(data, rowvar=False)
        if not np.all(np.isfinite(corr)):
            raise ValueError("Cannot compute correlations for non-finite data.")
        return np.asarray(corr, dtype=float), data.shape[0], data.shape[1]

    @staticmethod
    def _validate_data(data: object) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            raise TypeError(
                "FisherZTest expects data to be a numpy.ndarray or "
                "sufficient-statistics mapping."
            )
        if data.ndim != 2:
            raise ValueError("FisherZTest expects a two-dimensional data array.")
        if data.shape[0] < 4:
            raise ValueError("FisherZTest requires at least four samples.")
        return np.asarray(data, dtype=float)

    @classmethod
    def _correlation_from_sufficient_stats(
        cls,
        stats: Mapping[str, object],
    ) -> tuple[np.ndarray, int, int]:
        n_samples = cls._extract_n_samples(stats)
        if "correlation" in stats:
            corr = np.asarray(stats["correlation"], dtype=float)
        elif "corr" in stats:
            corr = np.asarray(stats["corr"], dtype=float)
        elif "covariance" in stats:
            corr = cls._covariance_to_correlation(
                np.asarray(stats["covariance"], dtype=float)
            )
        elif "cov" in stats:
            corr = cls._covariance_to_correlation(np.asarray(stats["cov"], dtype=float))
        else:
            raise ValueError(
                "Sufficient statistics must include 'correlation'/'corr' or "
                "'covariance'/'cov'."
            )

        cls._validate_correlation_matrix(corr)
        return corr, n_samples, corr.shape[0]

    @staticmethod
    def _extract_n_samples(stats: Mapping[str, object]) -> int:
        if "n_samples" in stats:
            n_samples = stats["n_samples"]
        elif "n" in stats:
            n_samples = stats["n"]
        else:
            raise ValueError("Sufficient statistics must include 'n_samples' or 'n'.")
        if not isinstance(n_samples, (int, np.integer)):
            raise TypeError("n_samples must be an integer.")
        if int(n_samples) < 4:
            raise ValueError("FisherZTest requires at least four samples.")
        return int(n_samples)

    @classmethod
    def _covariance_to_correlation(cls, covariance: np.ndarray) -> np.ndarray:
        cls._validate_square_matrix(covariance, "covariance")
        diagonal = np.diag(covariance)
        if np.any(diagonal <= 0.0):
            raise ValueError("Covariance diagonal entries must be positive.")
        scale = np.sqrt(diagonal)
        return covariance / np.outer(scale, scale)

    @classmethod
    def _validate_correlation_matrix(cls, corr: np.ndarray) -> None:
        cls._validate_square_matrix(corr, "correlation")
        if not np.allclose(corr, corr.T, atol=1e-8):
            raise ValueError("Correlation matrix must be symmetric.")
        if not np.allclose(np.diag(corr), 1.0, atol=1e-6):
            raise ValueError("Correlation matrix diagonal must be all ones.")

    @staticmethod
    def _validate_square_matrix(matrix: np.ndarray, name: str) -> None:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} matrix must be square.")
        if matrix.shape[0] < 2:
            raise ValueError(f"{name} matrix must contain at least two variables.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} matrix must contain only finite values.")

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

    @classmethod
    def _partial_correlation_from_corr(
        cls,
        corr: np.ndarray,
        x: int,
        y: int,
        cond_set: tuple[int, ...],
    ) -> float:
        variables = (x, y, *cond_set)
        sub_corr = corr[np.ix_(variables, variables)]
        if not cond_set:
            return float(sub_corr[0, 1])

        precision = cls._safe_inverse(sub_corr)
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
