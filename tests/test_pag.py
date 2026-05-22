import pytest

from fci_engine.graph import Endpoint, PAG


def test_pag_initialization() -> None:
    pag = PAG(["X", "Y", "Z"])

    assert pag.nodes == ("X", "Y", "Z")
    assert pag.edges() == []
    assert pag.get_endpoint("X", "Y") is Endpoint.NONE
    assert pag.neighbors("X") == []


def test_pag_rejects_duplicate_nodes() -> None:
    with pytest.raises(ValueError, match="unique"):
        PAG(["X", "X"])


def test_add_and_remove_edge() -> None:
    pag = PAG(["X", "Y"])

    pag.add_circle_edge("X", "Y")

    assert pag.is_adjacent("X", "Y")
    assert pag.get_endpoint("X", "Y") is Endpoint.CIRCLE
    assert pag.get_endpoint("Y", "X") is Endpoint.CIRCLE
    assert pag.edges() == [("X", "Y")]

    pag.remove_edge("X", "Y")

    assert not pag.is_adjacent("X", "Y")
    assert pag.get_endpoint("X", "Y") is Endpoint.NONE
    assert pag.get_endpoint("Y", "X") is Endpoint.NONE


def test_neighbors_are_returned_in_node_order() -> None:
    pag = PAG(["X", "Y", "Z", "W"])
    pag.add_circle_edge("X", "Z")
    pag.add_circle_edge("X", "Y")

    assert pag.neighbors("X") == ["Y", "Z"]


def test_endpoint_setting_and_orientation_helpers() -> None:
    pag = PAG(["X", "Y"])
    pag.add_circle_edge("X", "Y")

    pag.set_endpoint("X", "Y", Endpoint.ARROW)
    assert pag.get_endpoint("X", "Y") is Endpoint.ARROW
    assert pag.edge_repr("X", "Y") == "X o-> Y"

    pag.orient_tail("Y", "X")
    assert pag.get_endpoint("Y", "X") is Endpoint.TAIL
    assert pag.edge_repr("X", "Y") == "X --> Y"

    pag.orient_arrowhead("Y", "X")
    assert pag.edge_repr("X", "Y") == "X <-> Y"


@pytest.mark.parametrize(
    ("endpoint_x", "endpoint_y", "expected"),
    [
        (Endpoint.CIRCLE, Endpoint.CIRCLE, "X o-o Y"),
        (Endpoint.CIRCLE, Endpoint.ARROW, "X o-> Y"),
        (Endpoint.TAIL, Endpoint.ARROW, "X --> Y"),
        (Endpoint.ARROW, Endpoint.ARROW, "X <-> Y"),
        (Endpoint.TAIL, Endpoint.TAIL, "X --- Y"),
    ],
)
def test_major_edge_representations(
    endpoint_x: Endpoint,
    endpoint_y: Endpoint,
    expected: str,
) -> None:
    pag = PAG(["X", "Y"])
    pag.add_edge("X", "Y", endpoint_x, endpoint_y)

    assert pag.edge_repr("X", "Y") == expected


def test_to_edge_list_includes_endpoints() -> None:
    pag = PAG(["X", "Y", "Z"])
    pag.add_edge("X", "Y", Endpoint.TAIL, Endpoint.ARROW)
    pag.add_circle_edge("Y", "Z")

    assert pag.to_edge_list() == [
        ("X", "Y", Endpoint.TAIL, Endpoint.ARROW),
        ("Y", "Z", Endpoint.CIRCLE, Endpoint.CIRCLE),
    ]


def test_copy_is_independent() -> None:
    pag = PAG(["X", "Y"])
    pag.add_circle_edge("X", "Y")

    copied = pag.copy()
    copied.orient_arrowhead("X", "Y")

    assert pag.edge_repr("X", "Y") == "X o-o Y"
    assert copied.edge_repr("X", "Y") == "X o-> Y"
