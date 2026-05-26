import numpy as np

from fci_engine.ci import FisherZTest, KernelCITest


def test_kernel_ci_detects_nonlinear_dependence_missed_by_correlation() -> None:
    rng = np.random.default_rng(20)
    x = rng.uniform(-1.0, 1.0, size=160)
    y = x**2 + rng.normal(scale=0.05, size=160)
    data = np.column_stack([x, y])

    fisher_result = FisherZTest(alpha=0.05).test(data, 0, 1, [])
    kernel_result = KernelCITest(
        alpha=0.05,
        n_permutations=80,
        random_state=0,
    ).test(data, 0, 1, [])

    assert fisher_result.independent
    assert not kernel_result.independent
    assert kernel_result.method == "kernel_ci"


def test_kernel_ci_treats_independent_variables_as_independent() -> None:
    rng = np.random.default_rng(21)
    data = rng.normal(size=(140, 2))

    result = KernelCITest(
        alpha=0.01,
        n_permutations=80,
        random_state=0,
    ).test(data, 0, 1, [])

    assert result.independent


def test_kernel_ci_supports_residualized_conditioning_sets() -> None:
    rng = np.random.default_rng(22)
    z = rng.normal(size=160)
    x = 0.8 * z + rng.normal(scale=0.4, size=160)
    y = -0.7 * z + rng.normal(scale=0.4, size=160)
    data = np.column_stack([x, y, z])

    marginal = KernelCITest(
        alpha=0.05,
        n_permutations=60,
        random_state=1,
    ).test(data, 0, 1, [])
    conditional = KernelCITest(
        alpha=0.05,
        n_permutations=60,
        random_state=1,
    ).test(data, 0, 1, [2])

    assert not marginal.independent
    assert conditional.independent


def test_kernel_ci_conditions_on_nonlinear_common_cause() -> None:
    rng = np.random.default_rng(23)
    z = rng.uniform(-2.0, 2.0, size=180)
    shared_nonlinear_signal = z**2
    x = 0.4 * z + shared_nonlinear_signal + rng.normal(scale=0.25, size=180)
    y = -0.4 * z + shared_nonlinear_signal + rng.normal(scale=0.25, size=180)
    data = np.column_stack([x, y, z])

    marginal = KernelCITest(
        alpha=0.05,
        n_permutations=80,
        random_state=2,
    ).test(data, 0, 1, [])
    conditional = KernelCITest(
        alpha=0.05,
        n_permutations=80,
        random_state=2,
    ).test(data, 0, 1, [2])

    assert not marginal.independent
    assert conditional.independent
