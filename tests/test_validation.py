import numpy as np
import pandas as pd
import pytest

from fci_engine.utils.validation import validate_numeric_data


def test_dataframe_variable_names_are_preserved() -> None:
    frame = pd.DataFrame(
        {
            "load": [1.0, 2.0, 3.0],
            "latency": [2.5, 3.5, 4.5],
        }
    )

    data, names = validate_numeric_data(frame)

    assert names == ["load", "latency"]
    assert data.dtype.kind == "f"
    assert data.shape == (3, 2)


def test_ndarray_variable_names_are_generated() -> None:
    array = np.ones((5, 3))

    data, names = validate_numeric_data(array)

    assert names == ["X0", "X1", "X2"]
    assert np.array_equal(data, array)
    assert data is not array
    assert not data.flags.writeable


def test_normalized_array_cannot_be_changed_through_caller_input() -> None:
    array = np.arange(12.0).reshape(4, 3)

    data, _ = validate_numeric_data(array)
    array[:] = -1.0

    assert np.array_equal(data, np.arange(12.0).reshape(4, 3))


def test_rejects_1d_array_with_clear_error() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        validate_numeric_data(np.ones(5))


def test_rejects_non_numeric_dataframe_columns_for_fisher_z() -> None:
    frame = pd.DataFrame(
        {
            "load": [1.0, 2.0, 3.0],
            "status": ["ok", "warn", "ok"],
        }
    )

    with pytest.raises(TypeError, match="non-numeric columns: status"):
        validate_numeric_data(frame)


@pytest.mark.parametrize(
    "data",
    [
        np.empty((0, 2)),
        pd.DataFrame(columns=["x", "y"], dtype=float),
    ],
)
def test_rejects_empty_datasets(data) -> None:
    with pytest.raises(ValueError, match="at least one row"):
        validate_numeric_data(data)


@pytest.mark.parametrize("columns", [["x", "x"], [1, "1"]])
def test_rejects_dataframe_column_names_that_are_not_unique_as_strings(
    columns,
) -> None:
    frame = pd.DataFrame(np.ones((5, 2)), columns=columns)

    with pytest.raises(ValueError, match="unique after conversion to strings"):
        validate_numeric_data(frame)


def test_rejects_non_finite_values() -> None:
    array = np.array([[1.0, 2.0], [np.nan, 3.0]])

    with pytest.raises(ValueError, match="finite"):
        validate_numeric_data(array)


def test_allow_nan_keeps_missing_values_but_rejects_infinity() -> None:
    array = np.array([[1.0, np.nan], [2.0, 3.0]])

    data, names = validate_numeric_data(array, allow_nan=True)

    assert names == ["X0", "X1"]
    assert np.isnan(data[0, 1])

    with pytest.raises(ValueError, match="infinite"):
        validate_numeric_data(
            np.array([[1.0, np.inf], [2.0, 3.0]]),
            allow_nan=True,
        )


def test_rejects_unsupported_input_type() -> None:
    with pytest.raises(TypeError, match="pandas.DataFrame or numpy.ndarray"):
        validate_numeric_data([[1.0, 2.0], [3.0, 4.0]])
