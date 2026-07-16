import numpy as np

from fci_engine.ci import CITest, CITestResult
from fci_engine.discovery.pdsep import possible_dsep, refine_skeleton_with_pdsep
from fci_engine.graph import Endpoint, PAG


class OracleCITest(CITest):
    def __init__(
        self,
        independencies: set[tuple[frozenset[int], frozenset[int]]],
        p_values=None,
    ) -> None:
        super().__init__(alpha=0.05)
        self.independencies = independencies
        self.p_values = p_values or {}

    def test(
        self,
        data: np.ndarray,
        x: int,
        y: int,
        cond_set: tuple[int, ...],
    ) -> CITestResult:
        key = (frozenset((x, y)), frozenset(cond_set))
        independent = key in self.independencies
        return CITestResult(
            independent=independent,
            p_value=self.p_values.get(key, 0.9 if independent else 0.001),
            statistic=None,
            method="oracle",
            n_samples=data.shape[0],
        )


def make_reachable_pds_graph() -> PAG:
    graph = PAG(["X", "Y", "A", "B", "C"])
    graph.add_circle_edge("X", "Y")
    graph.add_edge("X", "A", Endpoint.CIRCLE, Endpoint.ARROW)
    graph.add_edge("A", "B", Endpoint.ARROW, Endpoint.ARROW)
    graph.add_edge("B", "C", Endpoint.ARROW, Endpoint.CIRCLE)
    return graph


def test_possible_dsep_excludes_endpoints() -> None:
    graph = make_reachable_pds_graph()

    candidates = possible_dsep(graph, "X", "Y")

    assert "X" not in candidates
    assert "Y" not in candidates
    assert {"A", "B", "C"}.issubset(candidates)


def test_possible_dsep_does_not_simply_return_all_nodes_for_blocked_paths() -> None:
    graph = PAG(["X", "Y", "A", "B"])
    graph.add_circle_edge("X", "Y")
    graph.add_circle_edge("X", "A")
    graph.add_circle_edge("A", "B")

    assert possible_dsep(graph, "X", "Y") == {"A"}


def test_possible_dsep_max_path_length_limits_search() -> None:
    graph = make_reachable_pds_graph()

    assert possible_dsep(graph, "X", "Y", max_path_length=1) == {"A"}
    assert possible_dsep(graph, "X", "Y", max_path_length=2) == {"A", "B"}


def test_possible_dsep_handles_cyclic_shielded_paths() -> None:
    graph = PAG(["X", "Y", "A", "B", "C", "D"])
    graph.add_circle_edge("X", "Y")
    graph.add_edge("X", "A", Endpoint.CIRCLE, Endpoint.ARROW)
    graph.add_edge("A", "C", Endpoint.ARROW, Endpoint.ARROW)
    graph.add_edge("C", "D", Endpoint.ARROW, Endpoint.CIRCLE)
    graph.add_circle_edge("D", "A")
    graph.add_circle_edge("A", "B")

    assert possible_dsep(graph, "X", "Y", max_path_length=1) == {"A"}
    assert possible_dsep(graph, "X", "Y", max_path_length=3) == {
        "A",
        "C",
        "D",
    }


def test_possible_dsep_can_reach_candidates_through_other_endpoint() -> None:
    graph = PAG(["X", "Y", "A"])
    graph.add_edge("X", "Y", Endpoint.ARROW, Endpoint.ARROW)
    graph.add_edge("Y", "A", Endpoint.ARROW, Endpoint.ARROW)

    assert possible_dsep(graph, "X", "Y") == {"A"}


def test_possible_dsep_rejects_triangle_with_definite_noncollider_tail() -> None:
    graph = PAG(["X", "Y", "A", "B", "C"])
    graph.add_circle_edge("X", "Y")
    graph.add_edge("X", "A", Endpoint.CIRCLE, Endpoint.TAIL)
    graph.add_circle_edge("A", "B")
    graph.add_circle_edge("X", "B")
    graph.add_edge("B", "C", Endpoint.ARROW, Endpoint.ARROW)

    assert possible_dsep(graph, "X", "Y") == {"A", "B"}


def test_pdsep_refinement_removes_edge_with_pds_conditioning_set() -> None:
    data = np.ones((20, 5))
    graph = make_reachable_pds_graph()
    sepsets = {}
    oracle = OracleCITest({(frozenset((0, 1)), frozenset((3,)))})

    refined, updated_sepsets = refine_skeleton_with_pdsep(
        data,
        graph,
        sepsets,
        oracle,
        max_cond_set_size=1,
    )

    assert not refined.is_adjacent("X", "Y")
    assert updated_sepsets[("X", "Y")] == {"B"}
    assert updated_sepsets[("Y", "X")] == {"B"}


def test_pdsep_refinement_uses_candidates_from_both_edge_directions() -> None:
    data = np.ones((20, 3))
    graph = PAG(["X", "Y", "A"])
    graph.add_circle_edge("X", "Y")
    graph.add_circle_edge("Y", "A")
    sepsets = {}
    oracle = OracleCITest({(frozenset((0, 1)), frozenset((2,)))})

    refined, updated_sepsets = refine_skeleton_with_pdsep(
        data,
        graph,
        sepsets,
        oracle,
        max_cond_set_size=1,
    )

    assert possible_dsep(graph, "X", "Y") == set()
    assert possible_dsep(graph, "Y", "X") == {"A"}
    assert not refined.is_adjacent("X", "Y")
    assert updated_sepsets[("X", "Y")] == {"A"}
    assert updated_sepsets[("Y", "X")] == {"A"}


def test_pdsep_does_not_mix_the_two_directional_candidate_pools() -> None:
    data = np.ones((20, 4))
    graph = PAG(["X", "Y", "A", "B"])
    graph.add_circle_edge("X", "Y")
    graph.add_circle_edge("X", "A")
    graph.add_circle_edge("Y", "B")
    oracle = OracleCITest(
        {
            (
                frozenset((0, 1)),
                frozenset((2, 3)),
            )
        }
    )

    refined, updated_sepsets = refine_skeleton_with_pdsep(
        data,
        graph,
        {},
        oracle,
        max_cond_set_size=2,
        sepset_selection="first",
    )

    assert refined.is_adjacent("X", "Y")
    assert ("X", "Y") not in updated_sepsets


def test_pdsep_selects_strongest_sepset_at_same_depth() -> None:
    data = np.ones((20, 4))
    graph = PAG(["X", "Y", "A", "B"])
    graph.add_circle_edge("X", "Y")
    graph.add_circle_edge("X", "A")
    graph.add_circle_edge("X", "B")
    weak = (frozenset((0, 1)), frozenset((2,)))
    strong = (frozenset((0, 1)), frozenset((3,)))
    oracle = OracleCITest(
        {weak, strong},
        p_values={
            weak: 0.11,
            strong: 0.91,
        },
    )

    refined, updated_sepsets = refine_skeleton_with_pdsep(
        data,
        graph,
        {},
        oracle,
        max_cond_set_size=1,
        sepset_selection="max_pvalue",
    )

    assert not refined.is_adjacent("X", "Y")
    assert updated_sepsets[("X", "Y")] == {"B"}
    assert updated_sepsets[("Y", "X")] == {"B"}


def test_pdsep_refinement_updates_existing_sepsets() -> None:
    data = np.ones((20, 5))
    graph = make_reachable_pds_graph()
    sepsets = {("X", "Y"): {"A"}, ("Y", "X"): {"A"}}
    oracle = OracleCITest({(frozenset((0, 1)), frozenset((3,)))})

    _, updated_sepsets = refine_skeleton_with_pdsep(
        data,
        graph,
        sepsets,
        oracle,
        max_cond_set_size=1,
    )

    assert updated_sepsets[("X", "Y")] == {"B"}
    assert updated_sepsets[("Y", "X")] == {"B"}


def make_order_dependent_pds_graph() -> PAG:
    graph = PAG(["X", "Y", "A", "B"])
    graph.add_edge("X", "Y", Endpoint.ARROW, Endpoint.CIRCLE)
    graph.add_edge("A", "X", Endpoint.CIRCLE, Endpoint.ARROW)
    graph.add_circle_edge("A", "B")
    return graph


def test_stable_pdsep_uses_start_of_stage_candidate_snapshot() -> None:
    data = np.ones((20, 4))
    graph = make_order_dependent_pds_graph()
    oracle = OracleCITest(
        {
            (frozenset((0, 1)), frozenset()),
            (frozenset((2, 3)), frozenset((1,))),
        }
    )

    refined, updated_sepsets = refine_skeleton_with_pdsep(
        data,
        graph,
        {},
        oracle,
        max_cond_set_size=1,
    )

    assert not refined.is_adjacent("X", "Y")
    assert not refined.is_adjacent("A", "B")
    assert updated_sepsets[("A", "B")] == {"Y"}


def test_order_dependent_pdsep_kept_for_explicit_compatibility() -> None:
    data = np.ones((20, 4))
    graph = make_order_dependent_pds_graph()
    oracle = OracleCITest(
        {
            (frozenset((0, 1)), frozenset()),
            (frozenset((2, 3)), frozenset((1,))),
        }
    )

    refined, updated_sepsets = refine_skeleton_with_pdsep(
        data,
        graph,
        {},
        oracle,
        max_cond_set_size=1,
        stable=False,
    )

    assert not refined.is_adjacent("X", "Y")
    assert refined.is_adjacent("A", "B")
    assert ("A", "B") not in updated_sepsets
