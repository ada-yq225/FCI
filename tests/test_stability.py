import numpy as np
import pytest

from fci_engine import (
    bootstrap_adjacency_frequencies,
    bootstrap_edge_frequencies,
    stable_fci,
    stable_fci_plus,
)
from fci_engine.ci import CITest, CITestResult


class AlwaysDependentCITest(CITest):
    def test(
        self,
        data: np.ndarray,
        x: int,
        y: int,
        cond_set: tuple[int, ...],
    ) -> CITestResult:
        return CITestResult(
            independent=False,
            p_value=0.001,
            statistic=None,
            method="always_dependent",
            n_samples=data.shape[0],
        )


class FirstDependentThenIndependentCITest(CITest):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def test(
        self,
        data: np.ndarray,
        x: int,
        y: int,
        cond_set: tuple[int, ...],
    ) -> CITestResult:
        self.calls += 1
        independent = self.calls > 1
        return CITestResult(
            independent=independent,
            p_value=0.9 if independent else 0.001,
            statistic=None,
            method="stateful",
            n_samples=data.shape[0],
        )


def test_bootstrap_edge_frequencies_reports_exact_pag_edge_rates() -> None:
    data = np.random.default_rng(3).normal(size=(40, 2))

    frequencies = bootstrap_edge_frequencies(
        data,
        n_bootstraps=3,
        random_state=0,
        ci_test=AlwaysDependentCITest(),
        do_pdsep=False,
    )

    assert frequencies == {"X0 o-o X1": 1.0}


def test_bootstrap_adjacency_frequencies_reports_skeleton_rates() -> None:
    data = np.random.default_rng(4).normal(size=(40, 2))

    frequencies = bootstrap_adjacency_frequencies(
        data,
        n_bootstraps=3,
        random_state=0,
        ci_test=AlwaysDependentCITest(),
        do_pdsep=False,
    )

    assert frequencies == {("X0", "X1"): 1.0}


def test_bootstrap_adjacency_frequencies_supports_parallel_jobs() -> None:
    data = np.random.default_rng(41).normal(size=(40, 2))

    frequencies = bootstrap_adjacency_frequencies(
        data,
        n_bootstraps=4,
        random_state=0,
        n_jobs=2,
        ci_test=AlwaysDependentCITest(),
        do_pdsep=False,
    )

    assert frequencies == {("X0", "X1"): 1.0}


def test_bootstrap_rejects_nonpositive_parallelism() -> None:
    data = np.random.default_rng(42).normal(size=(40, 2))

    with pytest.raises(ValueError, match="n_jobs"):
        bootstrap_adjacency_frequencies(
            data,
            n_bootstraps=2,
            n_jobs=0,
        )


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_bootstrap_count_must_be_a_positive_integer(value) -> None:
    data = np.random.default_rng(43).normal(size=(40, 2))

    with pytest.raises(ValueError, match="positive integer"):
        bootstrap_adjacency_frequencies(data, n_bootstraps=value)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, True])
def test_bootstrap_sample_fraction_must_be_finite_and_positive(value) -> None:
    data = np.random.default_rng(44).normal(size=(40, 2))

    with pytest.raises(ValueError, match="finite and positive"):
        bootstrap_adjacency_frequencies(
            data,
            n_bootstraps=2,
            sample_fraction=value,
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, True])
def test_stable_fci_rejects_invalid_edge_threshold(value) -> None:
    data = np.random.default_rng(45).normal(size=(40, 2))

    with pytest.raises(ValueError, match="between 0 and 1"):
        stable_fci(data, n_bootstraps=2, edge_threshold=value)


def test_stable_fci_filters_low_frequency_edges() -> None:
    data = np.random.default_rng(5).normal(size=(60, 2))

    result = stable_fci(
        data,
        n_bootstraps=3,
        edge_threshold=0.5,
        random_state=0,
        ci_test=FirstDependentThenIndependentCITest(),
        do_pdsep=False,
    )

    assert result.graph.edges() == []
    assert result.bootstrap_edge_frequencies == {}


def test_stable_fci_records_frequency_for_kept_edge_representation() -> None:
    data = np.random.default_rng(6).normal(size=(60, 2))

    result = stable_fci(
        data,
        n_bootstraps=3,
        edge_threshold=0.5,
        random_state=0,
        ci_test=AlwaysDependentCITest(),
        do_pdsep=False,
    )

    assert result.graph.edge_repr("X0", "X1") == "X0 o-o X1"
    assert result.bootstrap_edge_frequencies == {"X0 o-o X1": 1.0}
    assert result.to_edge_records()[0]["bootstrap_frequency"] == 1.0


def test_stable_fci_plus_uses_fci_plus_pipeline() -> None:
    data = np.random.default_rng(7).normal(size=(60, 2))

    result = stable_fci_plus(
        data,
        n_bootstraps=3,
        edge_threshold=0.5,
        random_state=0,
        ci_test=AlwaysDependentCITest(),
        max_cond_set_size=1,
    )

    assert result.algorithm == "fci_plus"
    assert result.graph.edge_repr("X0", "X1") == "X0 o-o X1"
    assert result.bootstrap_edge_frequencies == {"X0 o-o X1": 1.0}


def test_stable_fci_plus_accepts_practical_profile() -> None:
    data = np.random.default_rng(8).normal(size=(60, 2))

    result = stable_fci_plus(
        data,
        profile="practical",
        n_bootstraps=2,
        edge_threshold=0.5,
        random_state=0,
        ci_test=AlwaysDependentCITest(),
        max_cond_set_size=1,
    )

    assert result.algorithm == "fci_plus"
    assert result.config.sparsity_bound == 1
    assert result.config.orientation_strategy == "robust"
