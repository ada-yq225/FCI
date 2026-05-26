from fci_engine.diagnostics import OrientationEvent
from fci_engine.graph import Endpoint
from fci_engine.metrics import (
    compare_pag_shapes_semantic,
    explain_pag_differences,
)


def test_semantic_comparison_distinguishes_over_under_and_conflict() -> None:
    expected = {
        ("A", "B"): ("CIRCLE", "ARROW"),
        ("C", "D"): ("TAIL", "ARROW"),
        ("E", "F"): ("TAIL", "ARROW"),
        ("G", "H"): ("ARROW", "ARROW"),
    }
    actual = {
        ("A", "B"): ("TAIL", "ARROW"),
        ("C", "D"): ("CIRCLE", "ARROW"),
        ("E", "F"): ("ARROW", "ARROW"),
        ("I", "J"): ("CIRCLE", "CIRCLE"),
    }

    comparison = compare_pag_shapes_semantic(expected, actual)

    assert comparison.compatible_edges == 2
    assert comparison.over_oriented_edges == 1
    assert comparison.under_oriented_edges == 1
    assert comparison.contradicted_edges == 1
    assert comparison.false_positive_edges == 1
    assert comparison.false_negative_edges == 1
    assert comparison.semantic_edge_f1 == 0.5


def test_explain_pag_differences_includes_orientation_events() -> None:
    expected = {("A", "B"): ("CIRCLE", "ARROW")}
    actual = {("A", "B"): ("TAIL", "ARROW")}
    trace = [
        OrientationEvent(
            rule="R8",
            edge=("A", "B"),
            oriented_endpoint="A",
            before=Endpoint.CIRCLE,
            after=Endpoint.TAIL,
            before_edge="A o-> B",
            after_edge="A --> B",
            iteration=2,
            reason="test trace",
        )
    ]

    differences = explain_pag_differences(expected, actual, trace)

    assert len(differences) == 1
    assert differences[0].kind == "over_oriented"
    assert differences[0].endpoint_status == ("over_oriented", "exact")
    assert differences[0].orientation_events[0]["rule"] == "R8"
