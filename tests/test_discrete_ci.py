import numpy as np
import pandas as pd
import pytest

from fci_engine import fci_plus
from fci_engine.ci import CITestCache, ChiSquareTest, GSquareTest


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


def test_g_square_runs_end_to_end_on_discrete_common_cause() -> None:
    rng = np.random.default_rng(31)
    n_samples = 5000
    z = rng.integers(0, 2, size=n_samples)
    x = np.bitwise_xor(z, rng.random(n_samples) < 0.1).astype(int)
    y = np.bitwise_xor(z, rng.random(n_samples) < 0.1).astype(int)
    data = pd.DataFrame({"Z": z, "X": x, "Y": y})

    result = fci_plus(
        data,
        profile="paper",
        k=1,
        ci_test=GSquareTest(alpha=0.001),
        alpha=0.001,
    )

    assert result.graph.is_adjacent("Z", "X")
    assert result.graph.is_adjacent("Z", "Y")
    assert not result.graph.is_adjacent("X", "Y")
    assert any("expected cell counts" in note for note in result.assumption_notes())


def test_discrete_fci_pipeline_accepts_categorical_dataframe() -> None:
    rng = np.random.default_rng(32)
    n_samples = 2_000
    z_binary = rng.integers(0, 2, size=n_samples)
    x_binary = np.bitwise_xor(z_binary, rng.random(n_samples) < 0.05)
    y_binary = np.bitwise_xor(z_binary, rng.random(n_samples) < 0.05)
    data = pd.DataFrame(
        {
            "Z": np.where(z_binary, "present", "absent"),
            "X": np.where(x_binary, "high", "low"),
            "Y": np.where(y_binary, "yes", "no"),
        }
    )

    result = fci_plus(
        data,
        profile="paper",
        k=1,
        ci_test=GSquareTest(alpha=0.001),
        alpha=0.001,
    )

    assert result.graph.is_adjacent("Z", "X")
    assert result.graph.is_adjacent("Z", "Y")
    assert not result.graph.is_adjacent("X", "Y")


def test_ci_cache_preserves_categorical_data_capability() -> None:
    data = pd.DataFrame(
        {
            "X": ["a", "b"] * 100,
            "Y": ["yes", "no"] * 100,
        }
    )

    result = fci_plus(
        data,
        ci_test=CITestCache(GSquareTest(alpha=0.01)),
        max_cond_set_size=0,
    )

    assert result.nodes == ("X", "Y")
