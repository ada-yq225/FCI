"""Kernel-based conditional independence tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np
from scipy import stats

from fci_engine.ci.base import CITest, CITestResult


class KernelCITest(CITest):
    """RBF kernel conditional independence test.

    Empty conditioning sets use an unconditional HSIC permutation test. Non-empty
    conditioning sets use a kernel conditional independence statistic: RBF Gram
    matrices for ``x`` and ``y`` are residualized in feature space with a kernel
    ridge residual maker built from the conditioning variables. A Gamma
    approximation is then used for the conditional null distribution.
    """

    method = "kernel_ci"

    def __init__(
        self,
        alpha: float = 0.05,
        n_permutations: int = 200,
        gamma: Optional[float] = None,
        regularization: float = 1e-3,
        eigenvalue_threshold: float = 1e-5,
        random_state: Optional[int] = 0,
    ) -> None:
        super().__init__(alpha=alpha)
        if n_permutations <= 0:
            raise ValueError("n_permutations must be positive.")
        if regularization <= 0.0:
            raise ValueError("regularization must be positive.")
        if eigenvalue_threshold <= 0.0:
            raise ValueError("eigenvalue_threshold must be positive.")
        self.n_permutations = n_permutations
        self.gamma = gamma
        self.regularization = regularization
        self.eigenvalue_threshold = eigenvalue_threshold
        self.random_state = random_state

    def test(
        self,
        data: np.ndarray,
        x: int,
        y: int,
        cond_set: Sequence[int] = (),
    ) -> CITestResult:
        array = self._validate_data(data)
        cond_tuple = tuple(cond_set)
        self._validate_indices(array.shape[1], x, y, cond_tuple)

        x_values = array[:, [x]]
        y_values = array[:, [y]]
        if cond_tuple:
            z_values = array[:, cond_tuple]
            statistic, p_value = self._conditional_kci_test(
                x_values,
                y_values,
                z_values,
            )
        else:
            statistic, p_value = self._hsic_permutation_test(x_values, y_values)

        return CITestResult(
            independent=p_value > self.alpha,
            p_value=p_value,
            statistic=statistic,
            method=self.method,
            n_samples=array.shape[0],
        )

    @staticmethod
    def _validate_data(data: np.ndarray) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            raise TypeError("KernelCITest expects data to be a numpy.ndarray.")
        if data.ndim != 2:
            raise ValueError("KernelCITest expects a two-dimensional data array.")
        if data.shape[0] < 5:
            raise ValueError("KernelCITest requires at least five samples.")
        array = np.asarray(data, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError("KernelCITest requires finite numeric data.")
        return array

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
                raise TypeError("KernelCITest variable indices must be integers.")
            if index < 0 or index >= n_features:
                raise IndexError(f"Column index out of bounds: {index}.")

    def _hsic_permutation_test(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
    ) -> tuple[float, float]:
        rng = np.random.default_rng(self.random_state)
        k = _center_kernel(_rbf_kernel(x_values, self.gamma))
        l = _center_kernel(_rbf_kernel(y_values, self.gamma))
        statistic = _hsic_statistic(k, l)
        exceedances = 0

        for _ in range(self.n_permutations):
            permutation = rng.permutation(y_values.shape[0])
            permuted_l = l[np.ix_(permutation, permutation)]
            if _hsic_statistic(k, permuted_l) >= statistic:
                exceedances += 1

        p_value = (exceedances + 1.0) / (self.n_permutations + 1.0)
        return float(statistic), float(p_value)

    def _conditional_kci_test(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        z_values: np.ndarray,
    ) -> tuple[float, float]:
        """Compute a kernel conditional independence statistic.

        The conditioning kernel defines a feature-space residual maker
        ``R = lambda * (Kz + lambda I)^(-1)``. Applying ``R K R`` removes the
        component of each centered Gram matrix explained by the conditioning
        variables. The Gamma approximation uses the leading eigen-components of
        both residualized Gram matrices.
        """

        xz_values = np.column_stack(
            [_standardize(x_values), 0.5 * _standardize(z_values)]
        )
        x_kernel = _center_kernel(_rbf_kernel(xz_values, self.gamma))
        y_kernel = _center_kernel(_rbf_kernel(_standardize(y_values), self.gamma))
        z_kernel = _center_kernel(_rbf_kernel(_standardize(z_values), self.gamma))

        residual = _kernel_residual_maker(z_kernel, self.regularization)
        x_residual_kernel = _symmetrize(
            _safe_matmul(_safe_matmul(residual, x_kernel), residual)
        )
        y_residual_kernel = _symmetrize(
            _safe_matmul(_safe_matmul(residual, y_kernel), residual)
        )
        statistic = float(np.sum(x_residual_kernel * y_residual_kernel))
        p_value = _gamma_p_value(
            statistic,
            x_residual_kernel,
            y_residual_kernel,
            self.eigenvalue_threshold,
        )
        return statistic, p_value


def _rbf_kernel(values: np.ndarray, gamma: Optional[float]) -> np.ndarray:
    distances = _squared_distances(values)
    if gamma is None:
        gamma = _median_gamma(distances)
    return np.exp(-gamma * distances)


def _squared_distances(values: np.ndarray) -> np.ndarray:
    norms = np.sum(values * values, axis=1, keepdims=True)
    distances = norms + norms.T - 2.0 * _safe_matmul(values, values.T)
    return np.maximum(distances, 0.0)


def _median_gamma(distances: np.ndarray) -> float:
    upper = distances[np.triu_indices_from(distances, k=1)]
    positive = upper[upper > 0.0]
    if positive.size == 0:
        return 1.0
    median_distance = float(np.median(positive))
    if median_distance <= 0.0:
        return 1.0
    return 1.0 / median_distance


def _center_kernel(kernel: np.ndarray) -> np.ndarray:
    row_mean = kernel.mean(axis=1, keepdims=True)
    col_mean = kernel.mean(axis=0, keepdims=True)
    grand_mean = kernel.mean()
    return kernel - row_mean - col_mean + grand_mean


def _hsic_statistic(k: np.ndarray, l: np.ndarray) -> float:
    n_samples = k.shape[0]
    return float(np.sum(k * l) / ((n_samples - 1) ** 2))


def _standardize(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, ddof=1, keepdims=True)
    scale[scale == 0.0] = 1.0
    return centered / scale


def _kernel_residual_maker(kernel: np.ndarray, regularization: float) -> np.ndarray:
    n_samples = kernel.shape[0]
    system = kernel + regularization * np.eye(n_samples)
    try:
        residual = regularization * np.linalg.solve(system, np.eye(n_samples))
    except np.linalg.LinAlgError:
        residual = regularization * np.linalg.pinv(system)
    return _symmetrize(residual)


def _gamma_p_value(
    statistic: float,
    x_kernel: np.ndarray,
    y_kernel: np.ndarray,
    eigenvalue_threshold: float,
) -> float:
    uu_product = _eigen_component_product(
        x_kernel,
        y_kernel,
        eigenvalue_threshold,
    )
    if uu_product.size == 0:
        return 1.0

    mean = float(np.trace(uu_product))
    variance = float(2.0 * np.trace(_safe_matmul(uu_product, uu_product)))
    if mean <= 0.0 or variance <= 0.0 or not np.isfinite(mean + variance):
        return 1.0

    shape = mean * mean / variance
    scale = variance / mean
    if shape <= 0.0 or scale <= 0.0:
        return 1.0
    return float(stats.gamma.sf(statistic, shape, scale=scale))


def _eigen_component_product(
    x_kernel: np.ndarray,
    y_kernel: np.ndarray,
    eigenvalue_threshold: float,
) -> np.ndarray:
    x_values, x_vectors = np.linalg.eigh(_symmetrize(x_kernel))
    y_values, y_vectors = np.linalg.eigh(_symmetrize(y_kernel))
    x_values, x_vectors = _leading_positive_eigensystem(
        x_values,
        x_vectors,
        eigenvalue_threshold,
    )
    y_values, y_vectors = _leading_positive_eigensystem(
        y_values,
        y_vectors,
        eigenvalue_threshold,
    )
    if x_values.size == 0 or y_values.size == 0:
        return np.empty((0, 0))

    x_features = x_vectors * np.sqrt(x_values)
    y_features = y_vectors * np.sqrt(y_values)
    n_samples = x_kernel.shape[0]
    n_features = x_features.shape[1] * y_features.shape[1]
    joint_features = np.empty((n_samples, n_features))
    column = 0
    for x_index in range(x_features.shape[1]):
        for y_index in range(y_features.shape[1]):
            joint_features[:, column] = (
                x_features[:, x_index] * y_features[:, y_index]
            )
            column += 1

    if n_features > n_samples:
        return _safe_matmul(joint_features, joint_features.T)
    return _safe_matmul(joint_features.T, joint_features)


def _leading_positive_eigensystem(
    values: np.ndarray,
    vectors: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    if values.size == 0 or values[0] <= 0.0:
        return np.empty(0), np.empty((vectors.shape[0], 0))
    keep = values > values[0] * threshold
    return values[keep], vectors[:, keep]


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _safe_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        product = left @ right
    if not np.all(np.isfinite(product)):
        raise FloatingPointError(
            "Kernel matrix multiplication produced non-finite values."
        )
    return product
