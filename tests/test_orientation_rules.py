import pytest

from fci_engine.discovery.orientation import (
    definite_noncollider,
    has_directed_path,
    is_unshielded_triple,
)
from fci_engine.discovery.rules import (
    apply_orientation_rules,
    rule_avoid_directed_cycles,
    rule_avoid_new_unshielded_colliders,
    rule_discriminating_paths,
    rule_propagate_arrowheads,
    rule_propagate_arrowheads_along_directed_paths,
)
from fci_engine.graph import Endpoint, PAG


def test_is_unshielded_triple_helper() -> None:
    graph = PAG(["X", "Z", "Y"])
    graph.add_circle_edge("X", "Z")
    graph.add_circle_edge("Z", "Y")

    assert is_unshielded_triple(graph, "X", "Z", "Y")

    graph.add_circle_edge("X", "Y")
    assert not is_unshielded_triple(graph, "X", "Z", "Y")


def test_has_directed_path_helper_respects_excluded_edge() -> None:
    graph = PAG(["A", "B", "C"])
    graph.add_edge("A", "B", Endpoint.TAIL, Endpoint.ARROW)
    graph.add_edge("B", "C", Endpoint.TAIL, Endpoint.ARROW)
    graph.add_edge("A", "C", Endpoint.TAIL, Endpoint.ARROW)

    assert has_directed_path(graph, "A", "C")
    assert has_directed_path(graph, "A", "C", excluded_edge=("A", "C"))
    assert not has_directed_path(graph, "C", "A")


def test_definite_noncollider_helper_uses_tail_or_sepset() -> None:
    graph = PAG(["X", "Z", "Y"])
    graph.add_edge("X", "Z", Endpoint.CIRCLE, Endpoint.TAIL)
    graph.add_circle_edge("Z", "Y")

    assert definite_noncollider(graph, "X", "Z", "Y")

    graph = PAG(["X", "Z", "Y"])
    graph.add_circle_edge("X", "Z")
    graph.add_circle_edge("Z", "Y")
    assert definite_noncollider(graph, "X", "Z", "Y", {("X", "Y"): {"Z"}})


def test_rule_avoid_new_unshielded_colliders_orients_tail_at_middle() -> None:
    graph = PAG(["X", "Z", "Y"])
    graph.add_edge("X", "Z", Endpoint.CIRCLE, Endpoint.ARROW)
    graph.add_circle_edge("Z", "Y")

    changed = rule_avoid_new_unshielded_colliders(graph, {})

    assert changed
    assert graph.edge_repr("X", "Z") == "X o-> Z"
    assert graph.edge_repr("Z", "Y") == "Z --o Y"


def test_rule_propagate_arrowheads_local_r2_pattern() -> None:
    graph = PAG(["A", "B", "C"])
    graph.add_edge("A", "B", Endpoint.TAIL, Endpoint.ARROW)
    graph.add_edge("B", "C", Endpoint.TAIL, Endpoint.ARROW)
    graph.add_circle_edge("A", "C")

    changed = rule_propagate_arrowheads(graph, {})

    assert changed
    assert graph.edge_repr("A", "C") == "A o-> C"


def test_rule_propagate_arrowheads_along_directed_paths() -> None:
    graph = PAG(["A", "B", "C", "D"])
    graph.add_edge("A", "B", Endpoint.TAIL, Endpoint.ARROW)
    graph.add_edge("B", "C", Endpoint.TAIL, Endpoint.ARROW)
    graph.add_edge("C", "D", Endpoint.TAIL, Endpoint.ARROW)
    graph.add_circle_edge("A", "D")

    changed = rule_propagate_arrowheads_along_directed_paths(graph, {})

    assert changed
    assert graph.edge_repr("A", "D") == "A o-> D"


def test_rule_avoid_directed_cycles_orients_tail_on_source() -> None:
    graph = PAG(["A", "B", "C"])
    graph.add_edge("A", "B", Endpoint.TAIL, Endpoint.ARROW)
    graph.add_edge("B", "C", Endpoint.TAIL, Endpoint.ARROW)
    graph.add_edge("A", "C", Endpoint.CIRCLE, Endpoint.ARROW)

    changed = rule_avoid_directed_cycles(graph, {})

    assert changed
    assert graph.edge_repr("A", "C") == "A --> C"


def test_apply_orientation_rules_converges_iteratively() -> None:
    graph = PAG(["A", "B", "C"])
    graph.add_edge("A", "B", Endpoint.TAIL, Endpoint.ARROW)
    graph.add_edge("B", "C", Endpoint.TAIL, Endpoint.ARROW)
    graph.add_circle_edge("A", "C")

    result = apply_orientation_rules(graph, {})

    assert result is graph
    assert graph.edge_repr("A", "C") == "A --> C"


def test_existing_arrowheads_are_preserved_by_rules() -> None:
    graph = PAG(["X", "Z", "Y"])
    graph.add_edge("X", "Z", Endpoint.ARROW, Endpoint.ARROW)
    graph.add_circle_edge("Z", "Y")

    apply_orientation_rules(graph, {})

    assert graph.edge_repr("X", "Z") == "X <-> Z"
    assert graph.edge_repr("Z", "Y") == "Z --o Y"


def test_rules_do_not_create_endpoint_contradictions() -> None:
    graph = PAG(["A", "B", "C"])
    graph.add_edge("A", "B", Endpoint.TAIL, Endpoint.ARROW)
    graph.add_edge("B", "C", Endpoint.TAIL, Endpoint.ARROW)
    graph.add_edge("A", "C", Endpoint.ARROW, Endpoint.ARROW)

    apply_orientation_rules(graph, {})

    assert graph.edge_repr("A", "C") == "A <-> C"
    assert graph.get_endpoint("A", "C") is Endpoint.ARROW
    assert graph.get_endpoint("C", "A") is Endpoint.ARROW


def test_rule_discriminating_paths_is_explicit_todo() -> None:
    graph = PAG(["A", "B"])
    graph.add_circle_edge("A", "B")

    assert not rule_discriminating_paths(graph, {})


@pytest.mark.skip(
    reason=(
        "TODO: standard FCI R4 discriminating path orientation needs a complete "
        "discriminating-path search and endpoint update policy."
    )
)
def test_discriminating_path_rule_r4_orients_endpoint() -> None:
    raise AssertionError("R4 discriminating path orientation is not implemented yet.")
