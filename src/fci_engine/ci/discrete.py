"""Discrete conditional independence tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import chi2

from fci_engine.ci.base import CITest, CITestResult
from fci_engine.types import Array


class ChiSquareTest(CITest):
    """Pearson chi-square CI test for discrete variables."""

    method = "chi_square"

    def test(
        self,
        data: Array,
        x: int,
        y: int,
        cond_set: Sequence[int] = (),
    ) -> CITestResult:
        encoded = _validate_and_encode_discrete_data(data)
        _validate_indices(encoded.shape[1], x, y, tuple(cond_set))
        statistic, dof = _conditional_table_statistic(
            encoded,
            x,
            y,
            tuple(cond_set),
            statistic_name=self.method,
        )
        p_value = float(chi2.sf(statistic, dof)) if dof > 0 else 1.0
        return CITestResult(
            independent=p_value > self.alpha,
            p_value=p_value,
            statistic=float(statistic),
            method=self.method,
            n_samples=encoded.shape[0],
        )


class GSquareTest(CITest):
    """Likelihood-ratio G-square CI test for discrete variables."""

    method = "g_square"

    def test(
        self,
        data: Array,
        x: int,
        y: int,
        cond_set: Sequence[int] = (),
    ) -> CITestResult:
        encoded = _validate_and_encode_discrete_data(data)
        _validate_indices(encoded.shape[1], x, y, tuple(cond_set))
        statistic, dof = _conditional_table_statistic(
            encoded,
            x,
            y,
            tuple(cond_set),
            statistic_name=self.method,
        )
        p_value = float(chi2.sf(statistic, dof)) if dof > 0 else 1.0
        return CITestResult(
            independent=p_value > self.alpha,
            p_value=p_value,
            statistic=float(statistic),
            method=self.method,
            n_samples=encoded.shape[0],
        )


def _validate_and_encode_discrete_data(data: object) -> Array:
    if isinstance(data, pd.DataFrame):
        array = data.to_numpy()
    else:
        array = np.asarray(data)

    if array.ndim != 2:
        raise ValueError("Discrete CI tests expect a two-dimensional data array.")
    if array.shape[0] == 0:
        raise ValueError("Discrete CI tests require at least one sample.")

    encoded_columns = []
    for column_index in range(array.shape[1]):
        column = array[:, column_index]
        if pd.isna(column).any():
            raise ValueError("Discrete CI tests do not accept missing values.")
        encoded, _ = pd.factorize(column, sort=True)
        encoded_columns.append(encoded)

    return cast(
        Array,
        np.column_stack(encoded_columns).astype(int, copy=False),
    )


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
            raise TypeError("Discrete CI variable indices must be integers.")
        if index < 0 or index >= n_features:
            raise IndexError(f"Column index out of bounds: {index}.")


def _conditional_table_statistic(
    data: Array,
    x: int,
    y: int,
    cond_set: tuple[int, ...],
    statistic_name: str,
) -> tuple[float, int]:
    if not cond_set:
        return _table_statistic(
            _contingency_table(data[:, x], data[:, y]), statistic_name
        )

    total_statistic = 0.0
    total_dof = 0
    cond_data = data[:, cond_set]
    _, inverse = np.unique(cond_data, axis=0, return_inverse=True)
    for state_index in range(int(inverse.max()) + 1):
        mask = inverse == state_index
        if mask.sum() == 0:
            continue
        statistic, dof = _table_statistic(
            _contingency_table(data[mask, x], data[mask, y]),
            statistic_name,
        )
        total_statistic += statistic
        total_dof += dof
    return total_statistic, total_dof


def _contingency_table(x_values: Array, y_values: Array) -> Array:
    n_x = int(x_values.max()) + 1
    n_y = int(y_values.max()) + 1
    table: Array = np.zeros((n_x, n_y), dtype=float)
    np.add.at(table, (x_values, y_values), 1.0)
    return table


def _table_statistic(table: Array, statistic_name: str) -> tuple[float, int]:
    row_nonzero = table.sum(axis=1) > 0
    col_nonzero = table.sum(axis=0) > 0
    compact = table[np.ix_(row_nonzero, col_nonzero)]
    if compact.shape[0] < 2 or compact.shape[1] < 2:
        return 0.0, 0

    total = compact.sum()
    expected = np.outer(compact.sum(axis=1), compact.sum(axis=0)) / total
    dof = (compact.shape[0] - 1) * (compact.shape[1] - 1)
    valid = expected > 0.0

    if statistic_name == "chi_square":
        statistic = float(
            np.sum(((compact[valid] - expected[valid]) ** 2) / expected[valid])
        )
    elif statistic_name == "g_square":
        observed_positive = compact > 0.0
        valid = valid & observed_positive
        statistic = float(
            2.0 * np.sum(compact[valid] * np.log(compact[valid] / expected[valid]))
        )
    else:
        raise ValueError(f"Unknown discrete CI statistic: {statistic_name!r}.")

    return statistic, int(dof)
