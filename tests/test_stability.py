import numpy as np

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
