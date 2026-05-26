from fci_engine.discovery.rules import (
    rule_orient_tail_along_directed_chain,
    rule_orient_tail_along_uncovered_pd_path,
    rule_orient_tail_with_two_directed_parents,
    rule_selection_bias_tail_from_noncollider,
    rule_selection_bias_tail_from_undirected,
    rule_uncovered_circle_path_selection_bias,
)
from fci_engine.graph import Endpoint, PAG


def test_r5_orients_uncovered_circle_path_to_undirected_edges() -> None:
    graph = PAG(["A", "B", "C", "D"])
    graph.add_circle_edge("A", "B")
    graph.add_circle_edge("A", "C")
    graph.add_circle_edge("C", "D")
    graph.add_circle_edge("D", "B")

    changed = rule_uncovered_circle_path_selection_bias(graph, {})

    assert changed
    assert graph.edge_repr("A", "B") == "A --- B"
    assert graph.edge_repr("A", "C") == "A --- C"
    assert graph.edge_repr("C", "D") == "C --- D"
    assert graph.edge_repr("D", "B") == "D --- B"


def test_r6_propagates_tail_from_undirected_edge() -> None:
    graph = PAG(["A", "B", "C"])
    graph.add_edge("A", "B", Endpoint.TAIL, Endpoint.TAIL)
    graph.add_edge("B", "C", Endpoint.CIRCLE, Endpoint.ARROW)

    changed = rule_selection_bias_tail_from_undirected(graph, {})

    assert changed
    assert graph.edge_repr("B", "C") == "B --> C"


def test_r7_propagates_tail_from_unshielded_noncollider() -> None:
    graph = PAG(["A", "B", "C"])
    graph.add_edge("A", "B", Endpoint.TAIL, Endpoint.CIRCLE)
    graph.add_edge("B", "C", Endpoint.CIRCLE, Endpoint.ARROW)

    changed = rule_selection_bias_tail_from_noncollider(graph, {})

    assert changed
    assert graph.edge_repr("B", "C") == "B --> C"


def test_r8_orients_tail_along_directed_chain() -> None:
    graph = PAG(["A", "B", "C"])
    graph.add_edge("A", "B", Endpoint.CIRCLE, Endpoint.ARROW)
    graph.add_edge("B", "C", Endpoint.TAIL, Endpoint.ARROW)
    graph.add_edge("A", "C", Endpoint.CIRCLE, Endpoint.ARROW)

    changed = rule_orient_tail_along_directed_chain(graph, {})

    assert changed
    assert graph.edge_repr("A", "C") == "A --> C"


def test_r9_orients_tail_from_uncovered_possibly_directed_path() -> None:
    graph = PAG(["A", "B", "D", "C"])
    graph.add_edge("A", "C", Endpoint.CIRCLE, Endpoint.ARROW)
    graph.add_circle_edge("A", "B")
    graph.add_circle_edge("B", "D")
    graph.add_edge("D", "C", Endpoint.CIRCLE, Endpoint.ARROW)

    changed = rule_orient_tail_along_uncovered_pd_path(graph, {})

    assert changed
    assert graph.edge_repr("A", "C") == "A --> C"


def test_r10_orients_tail_from_two_parent_paths() -> None:
    graph = PAG(["A", "M", "N", "B", "D", "C"])
    graph.add_edge("A", "C", Endpoint.CIRCLE, Endpoint.ARROW)
    graph.add_circle_edge("A", "M")
    graph.add_circle_edge("M", "B")
    graph.add_circle_edge("A", "N")
    graph.add_circle_edge("N", "D")
    graph.add_edge("B", "C", Endpoint.TAIL, Endpoint.ARROW)
    graph.add_edge("D", "C", Endpoint.TAIL, Endpoint.ARROW)

    changed = rule_orient_tail_with_two_directed_parents(graph, {})

    assert changed
    assert graph.edge_repr("A", "C") == "A --> C"
