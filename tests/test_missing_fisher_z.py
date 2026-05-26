import numpy as np
import pytest

from fci_engine.ci import MissingValueFisherZTest


def test_missing_value_fisher_z_uses_querywise_complete_cases() -> None:
    rng = np.random.default_rng(10)
    x = rng.normal(size=400)
    y = 0.8 * x + rng.normal(scale=0.3, size=400)
    z = rng.normal(size=400)
    data = np.column_stack([x, y, z])
    data[:25, 2] = np.nan

    result = MissingValueFisherZTest(alpha=0.01).test(data, 0, 1, [])

    assert not result.independent
    assert result.method == "mv_fisher_z"
    assert result.n_samples == 400


def test_missing_value_fisher_z_drops_rows_for_conditioning_variables() -> None:
    rng = np.random.default_rng(11)
    data = rng.normal(size=(300, 3))
    data[:40, 2] = np.nan

    result = MissingValueFisherZTest(alpha=0.01).test(data, 0, 1, [2])

    assert result.n_samples == 260
    assert np.isfinite(result.p_value)


def test_missing_value_fisher_z_requires_enough_complete_rows() -> None:
    data = np.array(
        [
            [1.0, 2.0],
            [np.nan, 2.0],
            [1.0, np.nan],
            [np.nan, np.nan],
        ]
    )

    with pytest.raises(ValueError, match="complete rows"):
        MissingValueFisherZTest().test(data, 0, 1, [])
