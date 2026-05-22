from fci_engine.graph import Endpoint, PAG


def test_basic_endpoint_queries() -> None:
    pag = PAG(["X", "Y", "Z"])
    pag.add_edge("X", "Y", Endpoint.TAIL, Endpoint.ARROW)
    pag.add_edge("Y", "Z", Endpoint.ARROW, Endpoint.ARROW)

    assert pag.has_tail("Y", "X")
    assert pag.has_arrowhead("X", "Y")
    assert pag.is_directed_edge("X", "Y")
    assert not pag.is_directed_edge("Y", "X")
    assert pag.is_bidirected_edge("Y", "Z")


def test_possible_ancestor_follows_possibly_directed_paths() -> None:
    pag = PAG(["X", "Y", "Z"])
    pag.add_edge("X", "Y", Endpoint.CIRCLE, Endpoint.ARROW)
    pag.add_circle_edge("Y", "Z")

    assert pag.is_possible_ancestor("X", "Z")
    assert not pag.is_possible_ancestor("Y", "X")


def test_definite_ancestor_uses_only_directed_edges() -> None:
    pag = PAG(["X", "Y", "Z", "W"])
    pag.add_edge("X", "Y", Endpoint.TAIL, Endpoint.ARROW)
    pag.add_edge("Y", "Z", Endpoint.TAIL, Endpoint.ARROW)
    pag.add_edge("X", "W", Endpoint.CIRCLE, Endpoint.ARROW)

    assert pag.is_definite_ancestor("X", "Z")
    assert not pag.is_definite_ancestor("Z", "X")
    assert not pag.is_definite_ancestor("X", "W")


def test_cause_queries_return_nodes_in_graph_order() -> None:
    pag = PAG(["A", "B", "C", "D"])
    pag.add_edge("A", "C", Endpoint.TAIL, Endpoint.ARROW)
    pag.add_edge("B", "C", Endpoint.CIRCLE, Endpoint.ARROW)
    pag.add_edge("D", "C", Endpoint.ARROW, Endpoint.ARROW)

    assert pag.definite_causes("C") == ["A"]
    assert pag.possible_causes("C") == ["A", "B"]
