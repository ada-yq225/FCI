"""Input validation and normalization utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def validate_numeric_data(data: Any) -> tuple[np.ndarray, list[str]]:
    """Return numeric data as ``ndarray`` plus variable names.

    DataFrame columns are preserved as variable names. NumPy array columns are
    named ``X0``, ``X1``, and so on.
    """

    if isinstance(data, pd.DataFrame):
        return _validate_dataframe(data)
    if isinstance(data, np.ndarray):
        return _validate_array(data)
    raise TypeError("data must be a pandas.DataFrame or numpy.ndarray.")


def _validate_dataframe(data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    if data.ndim != 2:
        raise ValueError("DataFrame input must be two-dimensional.")
    if data.shape[1] == 0:
        raise ValueError("DataFrame input must contain at least one column.")

    non_numeric = [
        str(column) for column in data.columns if not is_numeric_dtype(data[column])
    ]
    if non_numeric:
        columns = ", ".join(non_numeric)
        raise TypeError(
            "Fisher-Z requires numeric DataFrame columns; "
            f"non-numeric columns: {columns}."
        )

    array = data.to_numpy(dtype=float, copy=True)
    _validate_numeric_array_values(array)
    return array, [str(column) for column in data.columns]


def _validate_array(data: np.ndarray) -> tuple[np.ndarray, list[str]]:
    if data.ndim != 2:
        raise ValueError("ndarray input must be two-dimensional.")
    if data.shape[1] == 0:
        raise ValueError("ndarray input must contain at least one column.")

    try:
        array = np.asarray(data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("ndarray input must be numeric for Fisher-Z.") from exc

    _validate_numeric_array_values(array)
    names = [f"X{i}" for i in range(array.shape[1])]
    return array, names


def _validate_numeric_array_values(data: np.ndarray) -> None:
    if not np.all(np.isfinite(data)):
        raise ValueError("data must contain only finite numeric values.")
