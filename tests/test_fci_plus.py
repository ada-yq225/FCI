import numpy as np
import pandas as pd

from fci_engine import FCIPlus, fci, fci_plus
from fci_engine.ci import CITest, CITestResult
from fci_engine.discovery.dsep import (
    build_augmented_skeleton,
    hierarchy,
    possible_dsep_links,
    refine_skeleton_with_fci_plus_dsep,
)
from fci_engine.graph import Endpoint, PAG


class OracleCITest(CITest):
    def __init__(
        self,
        names: list[str],
        independencies: set[tuple[str, str, frozenset[str]]],
    ):
        super().__init__(alpha=0.05)
        self.names = names
        self.independencies = independencies

    def test(self, data, x, y, cond_set=()):
        x_name = self.names[x]
        y_name = self.names[y]
        cond_names = frozenset(self.names[index] for index in cond_set)
        key = _oracle_key(x_name, y_name, cond_names)
        independent = key in self.independencies
        return CITestResult(
            independent=independent,
            p_value=0.9 if independent else 0.001,
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


def _oracle_key(
    x: str,
    y: str,
    cond_set: frozenset[str],
) -> tuple[str, str, frozenset[str]]:
    ordered = tuple(sorted((x, y)))
    return ordered[0], ordered[1], cond_set
