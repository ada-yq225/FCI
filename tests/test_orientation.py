import numpy as np

from fci_engine.ci import CITest, CITestResult
from fci_engine.discovery import (
    find_unshielded_triples,
    orient_unshielded_colliders_conservative,
    orient_unshielded_colliders,
    reset_endpoint_marks,
)
from fci_engine.graph import Endpoint, PAG


class OracleCITest(CITest):
    def __init__(self, independencies: set[tuple[frozenset[int], frozenset[int]]]):
        super().__init__(alpha=0.05)
        self.independencies = independencies

    def test(self, data, x, y, cond_set=()):
        key = (frozenset((x, y)), frozenset(cond_set))
        independent = key in self.independencies
        return CITestResult(
            independent=independent,
            p_value=0.9 if independent else 0.001,
            statistic=None,
            method="oracle",
            n_samples=len(data),
        )


def make_unshielded_triple_graph() -> PAG:
    graph = PAG(["X", "Z", "Y"])
    graph.add_circle_edge("X", "Z")
    graph.add_circle_edge("Z", "Y")
    return graph


def test_find_unshielded_triples_returns_centered_triple_once() -> None:
    graph = make_unshielded_triple_graph()

    assert find_unshielded_triples(graph) == [("X", "Z", "Y")]


def test_unshielded_triple_orients_as_collider_when_center_not_in_sepset() -> None:
    graph = make_unshielded_triple_graph()

    orient_unshielded_colliders(graph, {("X", "Y"): set()})

    assert graph.has_arrowhead("X", "Z")
    assert graph.has_arrowhead("Y", "Z")
    assert graph.edge_repr("X", "Z") == "X o-> Z"
    assert graph.edge_repr("Z", "Y") == "Z <-o Y"


def test_unshielded_triple_is_not_oriented_when_center_is_in_sepset() -> None:
    graph = make_unshielded_triple_graph()

    orient_unshielded_colliders(graph, {("X", "Y"): {"Z"}})

    assert graph.edge_repr("X", "Z") == "X o-o Z"
    assert graph.edge_repr("Z", "Y") == "Z o-o Y"


def test_collider_arrowheads_are_at_center_not_endpoints() -> None:
    graph = make_unshielded_triple_graph()

    orient_unshielded_colliders(graph, {("X", "Y"): set()})

    assert graph.get_endpoint("X", "Z") is Endpoint.ARROW
    assert graph.get_endpoint("Z", "X") is Endpoint.CIRCLE
    assert graph.get_endpoint("Y", "Z") is Endpoint.ARROW
    assert graph.get_endpoint("Z", "Y") is Endpoint.CIRCLE


def test_conservative_collider_orients_when_all_sepsets_exclude_center() -> None:
    graph = make_unshielded_triple_graph()
    oracle = OracleCITest({(frozenset((0, 2)), frozenset())})

    _, ambiguous = orient_unshielded_colliders_conservative(
        np.ones((20, 3)),
        graph,
        {("X", "Y"): set()},
        oracle,
        max_cond_set_size=1,
    )

    assert ambiguous == []
    assert graph.has_arrowhead("X", "Z")
    assert graph.has_arrowhead("Y", "Z")


def test_conservative_collider_leaves_ambiguous_triple_unoriented() -> None:
    graph = make_unshielded_triple_graph()
    oracle = OracleCITest(
        {
            (frozenset((0, 2)), frozenset()),
            (frozenset((0, 2)), frozenset((1,))),
        }
    )

    _, ambiguous = orient_unshielded_colliders_conservative(
        np.ones((20, 3)),
        graph,
        {("X", "Y"): set()},
        oracle,
        max_cond_set_size=1,
    )

    assert ambiguous == [("X", "Z", "Y")]
    assert graph.edge_repr("X", "Z") == "X o-o Z"
    assert graph.edge_repr("Z", "Y") == "Z o-o Y"


def test_conservative_collider_does_not_orient_definite_noncollider() -> None:
    graph = make_unshielded_triple_graph()
    oracle = OracleCITest({(frozenset((0, 2)), frozenset((1,)))})

    _, ambiguous = orient_unshielded_colliders_conservative(
        np.ones((20, 3)),
        graph,
        {("X", "Y"): {"Z"}},
        oracle,
        max_cond_set_size=1,
    )

    assert ambiguous == []
    assert graph.edge_repr("X", "Z") == "X o-o Z"
    assert graph.edge_repr("Z", "Y") == "Z o-o Y"


def test_existing_arrowheads_are_preserved() -> None:
    graph = PAG(["X", "Z", "Y"])
    graph.add_edge("X", "Z", Endpoint.CIRCLE, Endpoint.ARROW)
    graph.add_circle_edge("Z", "Y")

    orient_unshielded_colliders(graph, {("X", "Y"): set()})

    assert graph.get_endpoint("X", "Z") is Endpoint.ARROW
    assert graph.get_endpoint("Z", "X") is Endpoint.CIRCLE
    assert graph.get_endpoint("Y", "Z") is Endpoint.ARROW


def test_existing_tail_is_not_overwritten_with_arrowhead() -> None:
    graph = PAG(["X", "Z", "Y"])
    graph.add_edge("X", "Z", Endpoint.CIRCLE, Endpoint.TAIL)
    graph.add_circle_edge("Z", "Y")

    orient_unshielded_colliders(graph, {("X", "Y"): set()})

    assert graph.get_endpoint("X", "Z") is Endpoint.TAIL
    assert graph.get_endpoint("Y", "Z") is Endpoint.ARROW


def test_reset_endpoint_marks_preserves_skeleton_and_clears_orientations() -> None:
    graph = PAG(["A", "B", "C"])
    graph.add_edge("A", "B", Endpoint.TAIL, Endpoint.ARROW)
    graph.add_edge("B", "C", Endpoint.ARROW, Endpoint.ARROW)

    reset_endpoint_marks(graph)

    assert graph.edges() == [("A", "B"), ("B", "C")]
    assert graph.edge_repr("A", "B") == "A o-o B"
    assert graph.edge_repr("B", "C") == "B o-o C"
