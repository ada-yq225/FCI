import numpy as np
import pandas as pd
import pytest

from fci_engine.ci import ChiSquareTest, GSquareTest


def test_chi_square_detects_independent_discrete_variables() -> None:
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2, size=800)
    y = rng.integers(0, 2, size=800)
    data = np.column_stack([x, y])

    result = ChiSquareTest(alpha=0.01).test(data, 0, 1, ())

    assert result.independent
    assert result.method == "chi_square"
    assert result.n_samples == 800


def test_g_square_detects_dependent_discrete_variables() -> None:
    rng = np.random.default_rng(2)
    x = rng.integers(0, 3, size=1000)
    noise = rng.random(1000) < 0.05
    y = np.where(noise, (x + 1) % 3, x)
    data = np.column_stack([x, y])

    result = GSquareTest(alpha=0.01).test(data, 0, 1, ())

    assert not result.independent
    assert result.p_value < 0.01
    assert result.method == "g_square"


def test_discrete_tests_accept_dataframe_categories() -> None:
    frame = pd.DataFrame(
        {
            "x": ["low", "high", "low", "high"] * 25,
            "y": ["yes", "yes", "no", "no"] * 25,
        }
    )

    result = ChiSquareTest(alpha=0.05).test(frame, 0, 1, ())

    assert result.n_samples == 100


def test_discrete_tests_reject_missing_values() -> None:
    data = np.array([[0, 1], [1, None]], dtype=object)

    with pytest.raises(ValueError, match="missing"):
        ChiSquareTest().test(data, 0, 1, ())
