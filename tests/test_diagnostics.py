import numpy as np

from fci_engine import FCI
from fci_engine.ci import CITest, CITestResult
from fci_engine.discovery.rules import apply_orientation_rules
from fci_engine.graph import Endpoint, PAG


class ColliderOracleCITest(CITest):
    def test(
        self,
        data: np.ndarray,
        x: int,
        y: int,
        cond_set: tuple[int, ...],
    ) -> CITestResult:
        independent = frozenset((x, y)) == frozenset((0, 2)) and not cond_set
        return CITestResult(
            independent=independent,
            p_value=0.9 if independent else 0.001,
            statistic=None,
            method="oracle",
            n_samples=data.shape[0],
        )


def test_fci_result_records_ci_trace_orientation_trace_and_sepset_sources() -> None:
    data = np.random.default_rng(0).normal(size=(50, 3))

    result = FCI(
        ci_test=ColliderOracleCITest(),
        do_pdsep=False,
    ).fit(data)

    assert result.ci_test_trace
    assert result.ci_test_trace[0].method == "oracle"
    assert result.ci_test_trace[0].x in {"X0", "X1", "X2"}
    assert result.sepsets[("X0", "X2")] == set()
    assert result.sepset_sources[("X0", "X2")] == "initial"
    assert any(
        event.rule == "orient_unshielded_colliders"
        and event.oriented_endpoint == "X1"
        for event in result.orientation_trace
    )
    assert "orientation events:" in result.summary()
    assert "CI trace events:" in result.summary()


def test_orientation_rule_trace_records_before_and_after_edges() -> None:
    graph = PAG(["A", "B", "C"])
    graph.add_edge("A", "B", Endpoint.TAIL, Endpoint.ARROW)
    graph.add_edge("B", "C", Endpoint.TAIL, Endpoint.ARROW)
    graph.add_circle_edge("A", "C")
    trace = []

    apply_orientation_rules(graph, {}, trace=trace)

    assert graph.edge_repr("A", "C") == "A --> C"
    assert [event.rule for event in trace] == ["R2", "R2"]
    assert trace[0].before_edge == "A o-o C"
    assert trace[-1].after_edge == "C <-- A"
