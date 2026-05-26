import pytest

from fci_engine.simulation import CausalGraphSpec


def test_causal_graph_spec_generates_latent_medical_pag_shape() -> None:
    spec = CausalGraphSpec(
        observed_nodes=["X1", "X2", "A", "B", "D"],
        latent_nodes=["H"],
        directed_edges=[
            ("X1", "A"),
            ("X2", "B"),
            ("H", "A"),
            ("H", "B"),
            ("A", "D"),
        ],
        definite_directed_edges=[("A", "D")],
    )

    assert spec.to_pag_shape() == {
        ("X1", "A"): ("CIRCLE", "ARROW"),
        ("X2", "B"): ("CIRCLE", "ARROW"),
        ("A", "B"): ("ARROW", "ARROW"),
        ("A", "D"): ("TAIL", "ARROW"),
    }


def test_causal_graph_spec_rejects_unknown_edges() -> None:
    with pytest.raises(ValueError, match="Unknown directed edge endpoint"):
        CausalGraphSpec(
            observed_nodes=["X", "Y"],
            directed_edges=[("X", "Z")],
        )
