import numpy as np
import pandas as pd

from fci_engine import FCIPlus, fci, fci_plus
from fci_engine.ci import CITest, CITestResult, MissingValueFisherZTest
from fci_engine.discovery.dsep import (
    _base_combinations_at_depth,
    build_augmented_skeleton,
    hierarchy,
    minimal_dsep,
    possible_dsep_links,
    refine_skeleton_with_fci_plus_dsep,
)
from fci_engine.graph import Endpoint, PAG


class OracleCITest(CITest):
    def __init__(
        self,
        names: list[str],
        independencies: set[tuple[str, str, frozenset[str]]],
        p_values=None,
    ):
        super().__init__(alpha=0.05)
        self.names = names
        self.independencies = independencies
        self.p_values = p_values or {}

    def test(self, data, x, y, cond_set=()):
        x_name = self.names[x]
        y_name = self.names[y]
        cond_names = frozenset(self.names[index] for index in cond_set)
        key = _oracle_key(x_name, y_name, cond_names)
        independent = key in self.independencies
        return CITestResult(
            independent=independent,
            p_value=self.p_values.get(key, 0.9 if independent else 0.001),
            statistic=None,
            method="oracle",
            n_samples=len(data),
        )


def test_hierarchy_recursively_adds_known_sepsets() -> None:
    sepsets = {
        ("A", "B"): {"Z1"},
        ("Z1", "C"): {"Z2"},
    }

    assert hierarchy({"A", "B", "C"}, sepsets) == {"A", "B", "C", "Z1", "Z2"}


def test_possible_dsep_links_detects_bidirected_witness_pattern() -> None:
    graph = PAG(["U", "X", "Y", "V"])
    graph.add_circle_edge("U", "X")
    graph.add_circle_edge("X", "Y")
    graph.add_circle_edge("Y", "V")

    assert possible_dsep_links(graph) == [("X", "Y")]


def test_possible_dsep_links_requires_not_against_arrowhead_paths() -> None:
    graph = PAG(["U", "X", "Y", "V"])
    graph.add_edge("U", "X", Endpoint.ARROW, Endpoint.CIRCLE)
    graph.add_circle_edge("X", "Y")
    graph.add_circle_edge("Y", "V")

    assert possible_dsep_links(graph) == []


def test_augmented_skeleton_orients_single_node_dependency_arrowheads() -> None:
    nodes = ["X", "Y", "S", "A"]
    graph = PAG(nodes)
    graph.add_circle_edge("X", "A")
    graph.add_circle_edge("S", "A")
    sepsets = {
        ("X", "Y"): {"S"},
        ("Y", "X"): {"S"},
    }
    oracle = OracleCITest(names=nodes, independencies=set())

    augmented = build_augmented_skeleton(
        graph,
        sepsets,
        np.zeros((20, len(nodes))),
        oracle,
    )

    assert augmented.get_endpoint("X", "A") is Endpoint.ARROW
    assert augmented.get_endpoint("S", "A") is Endpoint.ARROW


def test_minimal_dsep_rechecks_nodes_until_fixed_point() -> None:
    nodes = ["X", "Y", "A", "B", "C"]
    graph = PAG(nodes)
    graph.add_circle_edge("X", "Y")
    oracle = OracleCITest(
        nodes,
        {
            _oracle_key("X", "Y", frozenset({"A", "C"})),
            _oracle_key("X", "Y", frozenset({"C"})),
        },
    )

    minimized = minimal_dsep(
        np.zeros((20, len(nodes))),
        graph,
        "X",
        "Y",
        {"A", "B", "C"},
        oracle,
    )

    assert minimized == {"C"}


def test_fci_plus_dsep_refinement_uses_hierarchical_sepsets() -> None:
    nodes = ["U", "X", "Y", "V", "A", "B", "Z"]
    graph = PAG(nodes)
    for edge in [("U", "X"), ("X", "Y"), ("Y", "V"), ("X", "A"), ("Y", "B")]:
        graph.add_circle_edge(*edge)

    sepsets = {
        ("A", "B"): {"Z"},
        ("B", "A"): {"Z"},
    }
    oracle = OracleCITest(
        nodes,
        {
            _oracle_key("X", "Y", frozenset({"A", "B", "Z"})),
        },
    )
    data = np.zeros((20, len(nodes)))
    sources = {}

    refined, refined_sepsets = refine_skeleton_with_fci_plus_dsep(
        data,
        graph,
        sepsets,
        oracle,
        max_degree=1,
        sepset_sources=sources,
    )

    assert not refined.is_adjacent("X", "Y")
    assert refined_sepsets[("X", "Y")] == {"A", "B", "Z"}
    assert sources[("X", "Y")] == "fci_plus_dsep"


def test_fci_plus_dsep_selects_strongest_sepset_at_same_depth() -> None:
    nodes = ["U", "X", "Y", "V", "A", "B", "C", "D"]
    graph = PAG(nodes)
    for edge in [
        ("U", "X"),
        ("X", "Y"),
        ("Y", "V"),
        ("X", "A"),
        ("Y", "B"),
        ("X", "C"),
        ("Y", "D"),
    ]:
        graph.add_circle_edge(*edge)

    weak = _oracle_key("X", "Y", frozenset({"A", "B"}))
    strong = _oracle_key("X", "Y", frozenset({"C", "D"}))
    oracle = OracleCITest(
        nodes,
        {weak, strong},
        p_values={
            weak: 0.13,
            strong: 0.94,
        },
    )

    refined, refined_sepsets = refine_skeleton_with_fci_plus_dsep(
        np.zeros((20, len(nodes))),
        graph,
        {},
        oracle,
        max_degree=1,
        sepset_selection="max_pvalue",
    )

    assert not refined.is_adjacent("X", "Y")
    assert refined_sepsets[("X", "Y")] == {"C", "D"}
    assert refined_sepsets[("Y", "X")] == {"C", "D"}


def test_algorithm2_base_loop_enumerates_separate_endpoint_bases() -> None:
    graph = PAG(["X", "Y", "A", "B", "C", "D"])

    pairs = _base_combinations_at_depth(
        graph,
        base_x=["A", "C"],
        base_y=["B", "D"],
        depth=2,
        max_degree=1,
    )

    assert pairs == [
        (("A",), ("B",)),
        (("A",), ("D",)),
        (("C",), ("B",)),
        (("C",), ("D",)),
    ]


def test_fci_plus_sparsity_bound_is_separate_from_conditioning_cap() -> None:
    nodes = ["U", "X", "Y", "V", "A", "B", "C", "D"]
    graph = PAG(nodes)
    for edge in [
        ("U", "X"),
        ("X", "Y"),
        ("Y", "V"),
        ("X", "A"),
        ("Y", "B"),
        ("X", "C"),
        ("Y", "D"),
    ]:
        graph.add_circle_edge(*edge)

    separating = _oracle_key("X", "Y", frozenset({"A", "C", "B", "D"}))
    oracle = OracleCITest(nodes, {separating})

    refined, _ = refine_skeleton_with_fci_plus_dsep(
        np.zeros((20, len(nodes))),
        graph,
        {},
        oracle,
        max_degree=2,
    )

    assert not refined.is_adjacent("X", "Y")


def test_augmented_skeleton_only_orients_existing_candidate_edges() -> None:
    nodes = ["X", "Y", "S", "A"]
    graph = PAG(nodes)
    graph.add_circle_edge("S", "A")
    sepsets = {
        ("X", "Y"): {"S"},
        ("Y", "X"): {"S"},
    }
    oracle = OracleCITest(names=nodes, independencies=set())

    augmented = build_augmented_skeleton(
        graph,
        sepsets,
        np.zeros((20, len(nodes))),
        oracle,
    )

    assert not augmented.is_adjacent("X", "A")
    assert augmented.get_endpoint("S", "A") is Endpoint.ARROW


def test_fci_plus_public_api_returns_result() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=800)
    y = 0.8 * x + rng.normal(scale=0.4, size=800)
    z = 0.8 * y + rng.normal(scale=0.4, size=800)
    data = pd.DataFrame({"X": x, "Y": y, "Z": z})

    result = fci_plus(data, alpha=0.001, max_cond_set_size=2)

    assert result.graph.nodes == ("X", "Y", "Z")
    assert "FCIResult" in result.summary()


def test_fci_and_fci_plus_are_separate_entry_points() -> None:
    rng = np.random.default_rng(124)
    data = rng.normal(size=(500, 4))

    standard = fci(data, alpha=0.001, max_cond_set_size=1)
    plus = FCIPlus(alpha=0.001, max_cond_set_size=1).fit(data)

    assert standard.graph.nodes == plus.graph.nodes
    assert plus.config.do_pdsep is False


def test_fci_plus_result_records_explicit_sparsity_bound() -> None:
    rng = np.random.default_rng(126)
    data = rng.normal(size=(300, 4))

    result = fci_plus(
        data,
        alpha=0.001,
        max_cond_set_size=1,
        sparsity_bound=2,
    )

    assert result.config.max_cond_set_size == 1
    assert result.config.sparsity_bound == 2


def test_fci_plus_result_records_dsep_diagnostics() -> None:
    nodes = ["U", "X", "Y", "V", "A", "B", "Z"]
    graph = PAG(nodes)
    for edge in [("U", "X"), ("X", "Y"), ("Y", "V"), ("X", "A"), ("Y", "B")]:
        graph.add_circle_edge(*edge)

    sepsets = {
        ("A", "B"): {"Z"},
        ("B", "A"): {"Z"},
    }
    oracle = OracleCITest(
        nodes,
        {
            _oracle_key("X", "Y", frozenset({"A", "B", "Z"})),
        },
    )

    from fci_engine.diagnostics import DSEPDiagnostics

    diagnostics = DSEPDiagnostics()
    refine_skeleton_with_fci_plus_dsep(
        np.zeros((20, len(nodes))),
        graph,
        sepsets,
        oracle,
        max_degree=1,
        diagnostics=diagnostics,
    )

    assert diagnostics.candidate_edges_seen >= 1
    assert diagnostics.hierarchy_queries >= 1
    assert diagnostics.ci_tests >= 1
    assert diagnostics.edges_removed == 1
    assert diagnostics.max_conditioning_size == 3


def test_fci_plus_accepts_missing_values_with_missing_value_ci_test() -> None:
    rng = np.random.default_rng(125)
    x = rng.normal(size=160)
    y = 0.8 * x + rng.normal(scale=0.4, size=160)
    z = 0.7 * x + rng.normal(scale=0.4, size=160)
    data = np.column_stack([x, y, z])
    data[:20, 2] = np.nan

    result = fci_plus(
        data,
        ci_test=MissingValueFisherZTest(alpha=0.001),
        max_cond_set_size=1,
    )

    assert result.graph.nodes == ("X0", "X1", "X2")
    assert {event.method for event in result.ci_test_trace} == {"mv_fisher_z"}


def _oracle_key(
    x: str,
    y: str,
    cond_set: frozenset[str],
) -> tuple[str, str, frozenset[str]]:
    ordered = tuple(sorted((x, y)))
    return ordered[0], ordered[1], cond_set
