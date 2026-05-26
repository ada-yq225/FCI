import numpy as np
import pandas as pd

import fci_engine.discovery.fci as fci_pipeline
from fci_engine import FCI, FCIResult, fci
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


def test_fci_fit_returns_result() -> None:
    data = np.random.default_rng(0).normal(size=(100, 3))

    result = FCI(ci_test=AlwaysDependentCITest()).fit(data)

    assert isinstance(result, FCIResult)
    assert result.graph.nodes == ("X0", "X1", "X2")
    assert result.ci_test_count > 0
    assert result.cache_hits >= 0


def test_function_api_fci_works() -> None:
    data = np.random.default_rng(1).normal(size=(80, 2))

    result = fci(data, ci_test=AlwaysDependentCITest())

    assert isinstance(result, FCIResult)
    assert result.graph.nodes == ("X0", "X1")


def test_result_summary_contains_useful_information() -> None:
    data = np.random.default_rng(2).normal(size=(80, 3))

    result = fci(data, ci_test=AlwaysDependentCITest(), do_pdsep=False)
    summary = result.summary()

    assert "FCIResult" in summary
    assert "nodes:" in summary
    assert "edges:" in summary
    assert "CI tests:" in summary
    assert "Possible-D-Sep: False" in summary


def test_do_pdsep_false_skips_pds_stage(monkeypatch) -> None:
    data = np.random.default_rng(3).normal(size=(80, 3))
    called = {"pdsep": False}

    def fail_if_called(*args, **kwargs):
        called["pdsep"] = True
        raise AssertionError("PDS refinement should be skipped.")

    monkeypatch.setattr(fci_pipeline, "refine_skeleton_with_pdsep", fail_if_called)

    result = FCI(ci_test=AlwaysDependentCITest(), do_pdsep=False).fit(data)

    assert isinstance(result, FCIResult)
    assert not called["pdsep"]


def test_dataframe_column_names_appear_in_graph_nodes() -> None:
    frame = pd.DataFrame(
        np.random.default_rng(4).normal(size=(100, 3)),
        columns=["load", "latency", "errors"],
    )

    estimator = FCI(ci_test=AlwaysDependentCITest())
    result = estimator.fit(frame)

    assert estimator.variable_names == ["load", "latency", "errors"]
    assert result.graph.nodes == ("load", "latency", "errors")
