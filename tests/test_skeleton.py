import numpy as np

from fci_engine.ci import CITest, CITestResult
from fci_engine.discovery import create_complete_pag, learn_initial_skeleton


class OracleCITest(CITest):
    def __init__(
        self,
        independencies: set[tuple[frozenset[int], frozenset[int]]],
        p_values=None,
    ) -> None:
        super().__init__(alpha=0.05)
        self.independencies = independencies
        self.p_values = p_values or {}
        self.calls: list[tuple[frozenset[int], frozenset[int]]] = []

    def test(
        self,
        data: np.ndarray,
        x: int,
        y: int,
        cond_set: tuple[int, ...],
    ) -> CITestResult:
        key = (frozenset((x, y)), frozenset(cond_set))
        self.calls.append(key)
        independent = key in self.independencies
        return CITestResult(
            independent=independent,
            p_value=self.p_values.get(key, 0.9 if independent else 0.001),
            statistic=None,
            method="oracle",
            n_samples=data.shape[0],
        )


def test_complete_pag_has_all_unordered_pairs() -> None:
    graph = create_complete_pag(["A", "B", "C", "D"])

    assert len(graph.edges()) == 6
    assert graph.edges() == [
        ("A", "B"),
        ("A", "C"),
        ("A", "D"),
        ("B", "C"),
        ("B", "D"),
        ("C", "D"),
    ]
    assert all(graph.is_circle_edge(x, y) for x, y in graph.edges())


def test_chain_removes_endpoints_when_conditioning_on_middle() -> None:
    data = np.ones((20, 3))
    graph = create_complete_pag(["X", "Y", "Z"])
    oracle = OracleCITest({(frozenset((0, 2)), frozenset((1,)))})

    learned, sepsets = learn_initial_skeleton(data, graph, oracle)

    assert learned.is_adjacent("X", "Y")
    assert learned.is_adjacent("Y", "Z")
    assert not learned.is_adjacent("X", "Z")
    assert sepsets[("X", "Z")] == {"Y"}
    assert sepsets[("Z", "X")] == {"Y"}


def test_fork_removes_endpoints_when_conditioning_on_middle() -> None:
    data = np.ones((20, 3))
    graph = create_complete_pag(["X", "Y", "Z"])
    oracle = OracleCITest({(frozenset((0, 2)), frozenset((1,)))})

    learned, sepsets = learn_initial_skeleton(data, graph, oracle)

    assert learned.is_adjacent("X", "Y")
    assert learned.is_adjacent("Y", "Z")
    assert not learned.is_adjacent("X", "Z")
    assert sepsets[("X", "Z")] == {"Y"}
    assert sepsets[("Z", "X")] == {"Y"}


def test_collider_removes_endpoints_marginally() -> None:
    data = np.ones((20, 3))
    graph = create_complete_pag(["X", "Y", "Z"])
    oracle = OracleCITest({(frozenset((0, 2)), frozenset())})

    learned, sepsets = learn_initial_skeleton(data, graph, oracle)

    assert learned.is_adjacent("X", "Y")
    assert learned.is_adjacent("Y", "Z")
    assert not learned.is_adjacent("X", "Z")
    assert sepsets[("X", "Z")] == set()
    assert sepsets[("Z", "X")] == set()
    assert (frozenset((0, 2)), frozenset((1,))) not in oracle.calls


def test_max_cond_set_size_is_respected() -> None:
    data = np.ones((20, 3))
    graph = create_complete_pag(["X", "Y", "Z"])
    oracle = OracleCITest({(frozenset((0, 2)), frozenset((1,)))})

    learned, sepsets = learn_initial_skeleton(data, graph, oracle, max_cond_set_size=0)

    assert learned.is_adjacent("X", "Z")
    assert sepsets == {}


def test_stable_skeleton_defers_removals_within_conditioning_depth() -> None:
    data = np.ones((20, 4))
    graph = create_complete_pag(["A", "B", "C", "D"])
    oracle = OracleCITest(
        {
            (frozenset((0, 2)), frozenset((1,))),
            (frozenset((0, 3)), frozenset((1,))),
            (frozenset((2, 3)), frozenset((0,))),
        }
    )

    learned, sepsets = learn_initial_skeleton(data, graph, oracle, stable=True)

    assert not learned.is_adjacent("A", "C")
    assert not learned.is_adjacent("A", "D")
    assert not learned.is_adjacent("C", "D")
    assert sepsets[("C", "D")] == {"A"}


def test_skeleton_selects_strongest_sepset_at_same_depth() -> None:
    data = np.ones((20, 4))
    graph = create_complete_pag(["X", "Y", "A", "B"])
    weak = (frozenset((0, 1)), frozenset((2,)))
    strong = (frozenset((0, 1)), frozenset((3,)))
    oracle = OracleCITest(
        {weak, strong},
        p_values={
            weak: 0.12,
            strong: 0.93,
        },
    )

    learned, sepsets = learn_initial_skeleton(
        data,
        graph,
        oracle,
        max_cond_set_size=1,
        sepset_selection="max_pvalue",
    )

    assert not learned.is_adjacent("X", "Y")
    assert sepsets[("X", "Y")] == {"B"}
    assert sepsets[("Y", "X")] == {"B"}


def test_skeleton_can_keep_first_sepset_for_compatibility() -> None:
    data = np.ones((20, 4))
    graph = create_complete_pag(["X", "Y", "A", "B"])
    weak = (frozenset((0, 1)), frozenset((2,)))
    strong = (frozenset((0, 1)), frozenset((3,)))
    oracle = OracleCITest(
        {weak, strong},
        p_values={
            weak: 0.12,
            strong: 0.93,
        },
    )

    learned, sepsets = learn_initial_skeleton(
        data,
        graph,
        oracle,
        max_cond_set_size=1,
        sepset_selection="first",
    )

    assert not learned.is_adjacent("X", "Y")
    assert sepsets[("X", "Y")] == {"A"}
    assert sepsets[("Y", "X")] == {"A"}


def test_order_dependent_skeleton_kept_for_explicit_compatibility() -> None:
    data = np.ones((20, 4))
    graph = create_complete_pag(["A", "B", "C", "D"])
    oracle = OracleCITest(
        {
            (frozenset((0, 2)), frozenset((1,))),
            (frozenset((0, 3)), frozenset((1,))),
            (frozenset((2, 3)), frozenset((0,))),
        }
    )

    learned, sepsets = learn_initial_skeleton(data, graph, oracle, stable=False)

    assert not learned.is_adjacent("A", "C")
    assert not learned.is_adjacent("A", "D")
    assert learned.is_adjacent("C", "D")
    assert ("C", "D") not in sepsets
