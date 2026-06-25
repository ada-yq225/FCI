import json

import numpy as np
import pandas as pd

from fci_engine import EdgeExplanation, FCIConfig, fci, fci_plus
from fci_engine.ci import CITest, CITestResult


class OracleCITest(CITest):
    def test(
        self,
        data: np.ndarray,
        x: int,
        y: int,
        cond_set: tuple[int, ...],
    ) -> CITestResult:
        independent = frozenset((x, y)) == frozenset((0, 2)) and cond_set == (1,)
        return CITestResult(
            independent=independent,
            p_value=0.9 if independent else 0.001,
            statistic=None,
            method="oracle",
            n_samples=data.shape[0],
        )


def _chain_result():
    data = pd.DataFrame(
        np.zeros((30, 3)),
        columns=["X", "Y", "Z"],
    )
    return fci(data, ci_test=OracleCITest(), max_cond_set_size=1, do_pdsep=False)


def test_result_exposes_algorithm_name() -> None:
    data = np.zeros((30, 3))

    standard = fci(data, ci_test=OracleCITest(), max_cond_set_size=1)
    plus = fci_plus(data, ci_test=OracleCITest(), max_cond_set_size=1)

    assert standard.algorithm == "fci"
    assert plus.algorithm == "fci_plus"
    assert "- algorithm: fci" in standard.summary()
    assert "- algorithm: fci_plus" in plus.summary()
    assert standard.dsep_diagnostics is None
    assert plus.dsep_diagnostics is not None
    assert "candidate_edges_seen" in plus.dsep_diagnostics


def test_to_edge_records_and_dataframe() -> None:
    result = _chain_result()

    records = result.to_edge_records()
    frame = result.to_pandas_edges()

    assert records
    assert {"x", "y", "endpoint_x", "endpoint_y", "edge"} <= set(frame.columns)
    assert len(frame) == len(records)


def test_to_networkx_preserves_pag_endpoint_attributes() -> None:
    result = _chain_result()

    graph = result.to_networkx()

    assert set(graph.nodes) == {"X", "Y", "Z"}
    assert graph.number_of_edges() == len(result.graph.edges())
    for _, _, attributes in graph.edges(data=True):
        assert "endpoint_x" in attributes
        assert "endpoint_y" in attributes
        assert "edge" in attributes


def test_explain_edge_for_removed_edge_includes_sepset_and_ci_tests() -> None:
    result = _chain_result()

    explanation = result.explain_edge("X", "Z")

    assert isinstance(explanation, EdgeExplanation)
    assert not explanation.edge_exists
    assert explanation.sepset == ["Y"]
    assert explanation.sepset_source == "initial"
    assert explanation.ci_tests
    assert "X ... Z" in explanation.summary()


def test_result_json_round_trip_and_save(tmp_path) -> None:
    result = _chain_result()
    output_path = tmp_path / "result.json"

    payload = result.to_dict()
    result.save_json(output_path)
    loaded = json.loads(output_path.read_text())

    assert payload["algorithm"] == "fci"
    assert payload["nodes"] == ["X", "Y", "Z"]
    assert "edges" in payload
    assert len(payload["sepsets"]) == 1
    assert "ci_test_trace" in payload
    assert payload["config"]["orientation_strategy"] == "standard"
    assert payload["dsep_diagnostics"] is None
    assert loaded["algorithm"] == payload["algorithm"]


def test_fci_config_is_exported_from_top_level_package() -> None:
    assert FCIConfig(orientation_strategy="leaf").orientation_strategy == "leaf"


def test_robust_orientation_strategy_enables_conservative_colliders() -> None:
    data = pd.DataFrame(
        np.zeros((30, 3)),
        columns=["X", "Y", "Z"],
    )

    result = fci_plus(
        data,
        ci_test=OracleCITest(),
        max_cond_set_size=1,
        orientation_strategy="robust",
    )

    assert result.config.orientation_strategy == "robust"
    assert result.config.conservative_colliders is True
