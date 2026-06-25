import pytest

import numpy as np

from fci_engine.simulation import CausalGraphSpec, MAGSpec, canonical_dsep_mag


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


def test_mag_spec_m_separation_chain_and_collider_cases() -> None:
    chain = MAGSpec(
        nodes=["X", "Z", "Y"],
        directed_edges=[("X", "Z"), ("Z", "Y")],
    )
    collider = MAGSpec(
        nodes=["X", "Z", "Y"],
        directed_edges=[("X", "Z"), ("Y", "Z")],
    )

    assert not chain.is_m_separated("X", "Y")
    assert chain.is_m_separated("X", "Y", {"Z"})
    assert collider.is_m_separated("X", "Y")
    assert not collider.is_m_separated("X", "Y", {"Z"})


def test_mag_oracle_ci_test_uses_exact_m_separation() -> None:
    mag = MAGSpec(
        nodes=["X", "Z", "Y"],
        directed_edges=[("X", "Z"), ("Z", "Y")],
    )
    oracle = mag.oracle_ci_test()
    data = np.zeros((10, 3))

    dependent = oracle.test(data, 0, 2, ())
    independent = oracle.test(data, 0, 2, (1,))

    assert not dependent.independent
    assert independent.independent
    assert independent.method == "mag_oracle"


def test_mag_spec_returns_explicit_pag_shape() -> None:
    mag = MAGSpec(
        nodes=["X", "Y", "Z"],
        directed_edges=[("X", "Z")],
        bidirected_edges=[("X", "Y")],
        pag_shape={
            ("X", "Y"): ("ARROW", "ARROW"),
            ("X", "Z"): ("CIRCLE", "ARROW"),
        },
    )

    assert mag.oracle_shape() == {
        ("X", "Y"): ("ARROW", "ARROW"),
        ("X", "Z"): ("CIRCLE", "ARROW"),
    }


def test_mag_spec_implied_skeleton_shape_uses_m_separation() -> None:
    mag = MAGSpec(
        nodes=["X", "Z", "Y"],
        directed_edges=[("X", "Z"), ("Z", "Y")],
    )

    assert mag.implied_skeleton_shape() == {
        ("X", "Z"): ("TAIL", "ARROW"),
        ("Z", "Y"): ("TAIL", "ARROW"),
    }


def test_canonical_dsep_mag_requires_nonadjacent_dsep_node() -> None:
    mag = canonical_dsep_mag()

    assert not mag.is_m_separated("X", "Y", {"U", "V"})
    assert mag.is_m_separated("X", "Y", {"U", "Z", "V"})
