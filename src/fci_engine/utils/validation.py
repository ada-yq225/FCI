"""Input validation and normalization utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from fci_engine.types import Array


def validate_numeric_data(
    data: Any,
    *,
    allow_nan: bool = False,
) -> tuple[Array, list[str]]:
    """Return numeric data as ``ndarray`` plus variable names.

    DataFrame columns are preserved as variable names. NumPy array columns are
    named ``X0``, ``X1``, and so on.
    """

    return validate_tabular_data(
        data,
        require_numeric=True,
        allow_nan=allow_nan,
    )


def validate_tabular_data(
    data: Any,
    *,
    require_numeric: bool,
    allow_nan: bool = False,
) -> tuple[Array, list[str]]:
    """Normalize supported tabular input without aliasing caller-owned data.

    The returned array is read-only. Discovery treats a dataset as immutable
    for the lifetime of a run so CI-result and correlation caches cannot become
    stale after an in-place input mutation.
    """

    if isinstance(data, pd.DataFrame):
        return _validate_dataframe(
            data,
            require_numeric=require_numeric,
            allow_nan=allow_nan,
        )
    if isinstance(data, np.ndarray):
        return _validate_array(
            data,
            require_numeric=require_numeric,
            allow_nan=allow_nan,
        )
    raise TypeError("data must be a pandas.DataFrame or numpy.ndarray.")


def _validate_dataframe(
    data: pd.DataFrame,
    *,
    require_numeric: bool,
    allow_nan: bool,
) -> tuple[Array, list[str]]:
    if data.ndim != 2:
        raise ValueError("DataFrame input must be two-dimensional.")
    if data.shape[0] == 0:
        raise ValueError("DataFrame input must contain at least one row.")
    if data.shape[1] == 0:
        raise ValueError("DataFrame input must contain at least one column.")

    names = [str(column) for column in data.columns]
    _validate_unique_names(names)
    if require_numeric:
        non_numeric = [
            name
            for name, dtype in zip(names, data.dtypes)
            if not is_numeric_dtype(dtype)
        ]
        if non_numeric:
            columns = ", ".join(non_numeric)
            raise TypeError(
                "The configured CI test requires numeric DataFrame columns; "
                f"non-numeric columns: {columns}."
            )
        array = data.to_numpy(dtype=float, copy=True)
        _validate_numeric_array_values(array, allow_nan=allow_nan)
    else:
        array = data.to_numpy(copy=True)
    array.setflags(write=False)
    return array, names


def _validate_array(
    data: Array,
    *,
    require_numeric: bool,
    allow_nan: bool,
) -> tuple[Array, list[str]]:
    if data.ndim != 2:
        raise ValueError("ndarray input must be two-dimensional.")
    if data.shape[0] == 0:
        raise ValueError("ndarray input must contain at least one row.")
    if data.shape[1] == 0:
        raise ValueError("ndarray input must contain at least one column.")

    if require_numeric:
        try:
            array = np.array(data, dtype=float, copy=True)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "ndarray input must be numeric for the configured CI test."
            ) from exc
        _validate_numeric_array_values(array, allow_nan=allow_nan)
    else:
        array = np.array(data, copy=True)
    array.setflags(write=False)
    names = [f"X{i}" for i in range(array.shape[1])]
    return array, names


def _validate_numeric_array_values(data: Array, *, allow_nan: bool) -> None:
    if allow_nan:
        if np.any(np.isinf(data)):
            raise ValueError("data must not contain infinite values.")
        return

    if not np.all(np.isfinite(data)):
        raise ValueError("data must contain only finite numeric values.")


def _validate_unique_names(names: list[str]) -> None:
    if len(set(names)) != len(names):
        raise ValueError(
            "DataFrame column names must be unique after conversion to strings."
        )
