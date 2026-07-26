"""End-to-end regression tests from published FCI/FCI+ examples."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from fci_engine import (
    MAGSpec,
    canonical_dsep_mag,
    fci,
    fci_plus,
    sample_canonical_dsep_data,
    shape_from_pag,
    spirtes_latent_reference_mag,
    zhang_orientation_reference_mag,
)


REFERENCE_FACTORIES: tuple[Callable[[], MAGSpec], ...] = (
    canonical_dsep_mag,
    spirtes_latent_reference_mag,
    zhang_orientation_reference_mag,
)


@pytest.mark.parametrize("factory", REFERENCE_FACTORIES)
@pytest.mark.parametrize("algorithm", (fci, fci_plus))
def test_exact_oracle_recovers_published_complete_pag(factory, algorithm) -> None:
    mag = factory()

    result = algorithm(
        mag.dummy_data(),
        ci_test=mag.oracle_ci_test(),
        max_cond_set_size=None,
        sparsity_bound=None,
        max_path_length=None,
        sepset_selection="first",
        orientation_strategy="standard",
    )

    assert shape_from_pag(result.graph) == mag.oracle_shape()


def test_fci_plus_removes_figure4_link_only_in_hierarchical_dsep_stage() -> None:
    mag = canonical_dsep_mag()

    result = fci_plus(
        mag.dummy_data(),
        ci_test=mag.oracle_ci_test(),
        max_cond_set_size=None,
        sparsity_bound=None,
        max_path_length=None,
        sepset_selection="first",
        orientation_strategy="standard",
    )

    assert result.sepsets[("X", "Y")] == {"U", "V", "Z"}
    assert result.sepset_sources[("X", "Y")] == "fci_plus_dsep"
    assert result.dsep_diagnostics["candidate_edges_seen"] == 1
    assert result.dsep_diagnostics["edges_removed"] == 1


@pytest.mark.parametrize(
    "order",
    (
        ("Z", "U", "V", "X", "Y"),
        ("X", "Y", "U", "V", "Z"),
        ("Y", "V", "Z", "U", "X"),
        ("V", "X", "Z", "Y", "U"),
    ),
)
@pytest.mark.parametrize("algorithm", (fci, fci_plus))
def test_figure4_pag_is_invariant_to_variable_order(order, algorithm) -> None:
    mag = canonical_dsep_mag(order)

    result = algorithm(
        mag.dummy_data(),
        ci_test=mag.oracle_ci_test(),
        max_cond_set_size=None,
        sparsity_bound=None,
        max_path_length=None,
        sepset_selection="first",
        orientation_strategy="standard",
    )

    assert shape_from_pag(result.graph) == mag.oracle_shape()
    assert mag.is_m_separated("X", "Y", result.sepsets[("X", "Y")])


def test_large_sample_latent_sem_recovers_figure4_pag() -> None:
    data = sample_canonical_dsep_data(n_samples=50_000, seed=1)
    expected = canonical_dsep_mag().oracle_shape()

    result = fci_plus(
        data,
        alpha=0.001,
        max_cond_set_size=3,
        sparsity_bound=3,
        max_path_length=None,
        sepset_selection="first",
        orientation_strategy="standard",
    )

    assert shape_from_pag(result.graph) == expected
    assert result.sepset_sources[("X", "Y")] == "fci_plus_dsep"


def test_fci_plus_paper_profile_orients_selection_bias_cycle_with_r5() -> None:
    mag = MAGSpec(
        nodes=("A", "B", "C", "D"),
        undirected_edges=[
            ("A", "B"),
            ("B", "D"),
            ("D", "C"),
            ("C", "A"),
        ],
    )

    result = fci_plus(
        mag.dummy_data(),
        profile="paper",
        k=2,
        ci_test=mag.oracle_ci_test(),
    )

    assert all(result.graph.is_undirected_edge(x, y) for x, y in result.graph.edges())
    assert "R5" in {event.rule for event in result.orientation_trace}


def test_spirtes_paper_profile_stops_before_selection_bias_tail_completion() -> None:
    mag = MAGSpec(
        nodes=("A", "B", "C", "D"),
        undirected_edges=[
            ("A", "B"),
            ("B", "D"),
            ("D", "C"),
            ("C", "A"),
        ],
    )

    result = fci(
        mag.dummy_data(),
        profile="paper",
        ci_test=mag.oracle_ci_test(),
    )

    assert all(result.graph.is_circle_edge(x, y) for x, y in result.graph.edges())
    assert "R5" not in {event.rule for event in result.orientation_trace}
