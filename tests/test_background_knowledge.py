import numpy as np
import pytest

from fci_engine import BackgroundKnowledge, fci
from fci_engine.ci import CITest, CITestResult
from fci_engine.graph import Endpoint, PAG
from fci_engine.knowledge import apply_background_knowledge


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


def test_background_knowledge_requires_directed_edge() -> None:
    knowledge = BackgroundKnowledge(required_edges={("X", "Y")})
    graph = PAG(["X", "Y"])
    graph.add_circle_edge("X", "Y")
    trace = []

    apply_background_knowledge(graph, knowledge, trace=trace)

    assert graph.edge_repr("X", "Y") == "X --> Y"
    assert [event.rule for event in trace] == [
        "background_knowledge",
        "background_knowledge",
    ]


def test_background_knowledge_forbids_direction_by_orienting_reverse() -> None:
    knowledge = BackgroundKnowledge(forbidden_edges={("X", "Y")})
    graph = PAG(["X", "Y"])
    graph.add_circle_edge("X", "Y")

    apply_background_knowledge(graph, knowledge)

    assert graph.edge_repr("X", "Y") == "X <-- Y"


def test_background_knowledge_rejects_contradictory_constraints() -> None:
    with pytest.raises(ValueError, match="both require and forbid"):
        BackgroundKnowledge(
            required_edges={("X", "Y")},
            forbidden_edges={("X", "Y")},
        )

    with pytest.raises(ValueError, match="both directions"):
        BackgroundKnowledge(
            required_edges={("X", "Y"), ("Y", "X")},
        )


def test_fci_applies_required_background_knowledge() -> None:
    data = np.random.default_rng(0).normal(size=(100, 2))
    knowledge = BackgroundKnowledge(required_edges={("X0", "X1")})

    result = fci(
        data,
        ci_test=AlwaysDependentCITest(),
        background_knowledge=knowledge,
        do_pdsep=False,
    )

    assert result.graph.edge_repr("X0", "X1") == "X0 --> X1"
    assert any(event.rule == "background_knowledge" for event in result.orientation_trace)


def test_fci_preserves_background_knowledge_through_orientation_rules() -> None:
    data = np.random.default_rng(1).normal(size=(100, 3))
    knowledge = BackgroundKnowledge(forbidden_edges={("X0", "X1")})

    result = fci(
        data,
        ci_test=AlwaysDependentCITest(),
        background_knowledge=knowledge,
        do_pdsep=False,
    )

    assert result.graph.get_endpoint("X1", "X0") is Endpoint.ARROW
    assert result.graph.get_endpoint("X0", "X1") is Endpoint.TAIL
