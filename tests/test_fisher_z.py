import numpy as np

from fci_engine.ci import FisherZTest


def test_independent_gaussian_variables_return_independent() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(1_000, 2))

    result = FisherZTest(alpha=0.01).test(data, 0, 1, [])

    assert result.independent
    assert result.p_value > 0.01
    assert result.method == "fisher_z"
    assert result.n_samples == 1_000
    assert result.statistic is not None


def test_correlated_gaussian_variables_return_dependent() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=1_000)
    y = 0.85 * x + rng.normal(scale=0.25, size=1_000)
    data = np.column_stack([x, y])

    result = FisherZTest(alpha=0.01).test(data, 0, 1, [])

    assert not result.independent
    assert result.p_value < 0.01


def test_chain_becomes_conditionally_independent_given_middle_node() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=2_000)
    y = 0.9 * x + rng.normal(scale=0.5, size=2_000)
    z = 0.9 * y + rng.normal(scale=0.5, size=2_000)
    data = np.column_stack([x, y, z])
    test = FisherZTest(alpha=0.01)

    marginal = test.test(data, 0, 2, [])
    conditional = test.test(data, 0, 2, [1])

    assert not marginal.independent
    assert conditional.independent


def test_singular_correlation_submatrix_is_handled() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=500)
    y = 0.7 * x + rng.normal(scale=0.4, size=500)
    duplicate_x = x.copy()
    data = np.column_stack([x, y, duplicate_x])

    result = FisherZTest(alpha=0.05).test(data, 0, 1, [2])

    assert result.method == "fisher_z"
    assert np.isfinite(result.p_value)
    assert result.statistic is not None
    assert np.isfinite(result.statistic)


def test_fisher_z_accepts_correlation_sufficient_statistics() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=1_200)
    y = 0.75 * x + rng.normal(scale=0.35, size=1_200)
    data = np.column_stack([x, y])
    corr = np.corrcoef(data, rowvar=False)
    test = FisherZTest(alpha=0.01)

    raw_result = test.test(data, 0, 1, [])
    stats_result = test.test({"correlation": corr, "n_samples": data.shape[0]}, 0, 1, [])

    assert raw_result.independent == stats_result.independent
    assert np.isclose(raw_result.p_value, stats_result.p_value)
    assert stats_result.n_samples == data.shape[0]


def test_fisher_z_accepts_covariance_sufficient_statistics() -> None:
    rng = np.random.default_rng(5)
    data = rng.normal(size=(900, 3))
    covariance = np.cov(data, rowvar=False)

    result = FisherZTest(alpha=0.01).test(
        {"covariance": covariance, "n_samples": data.shape[0]},
        0,
        1,
        [2],
    )

    assert result.method == "fisher_z"
    assert result.n_samples == data.shape[0]
    assert np.isfinite(result.p_value)
