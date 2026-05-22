"""Initial PC-style skeleton discovery used by FCI."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from itertools import combinations
from typing import Optional

from fci_engine.ci import CITest
from fci_engine.graph import PAG
from fci_engine.utils.validation import validate_numeric_data


SepsetMap = dict[tuple[Hashable, Hashable], set[Hashable]]


def create_complete_pag(nodes: Sequence[Hashable]) -> PAG:
    """Create a complete PAG with circle-circle edges."""

    graph = PAG(nodes)
    for i, x in enumerate(graph.nodes):
        for y in graph.nodes[i + 1 :]:
            graph.add_circle_edge(x, y)
    return graph


def learn_initial_skeleton(
    data: object,
    graph: PAG,
    ci_test: CITest,
    max_cond_set_size: Optional[int] = None,
    verbose: bool = False,
) -> tuple[PAG, SepsetMap]:
    """Learn the initial undirected PAG skeleton using CI tests.

    This is the PC-style adjacency search used as FCI's first stage. It only
    removes edges and records separating sets; it does not orient colliders.
    """

    if max_cond_set_size is not None and max_cond_set_size < 0:
        raise ValueError("max_cond_set_size must be non-negative.")

    normalized_data, variable_names = validate_numeric_data(data)
    if normalized_data.shape[1] != len(graph.nodes):
        raise ValueError(
            "data column count must match the number of graph nodes "
            f"({normalized_data.shape[1]} != {len(graph.nodes)})."
        )

    graph_node_names = [str(node) for node in graph.nodes]
    if set(variable_names) == set(graph_node_names) and len(set(graph_node_names)) == len(
        graph_node_names
    ):
        column_index = {name: index for index, name in enumerate(variable_names)}
        normalized_data = normalized_data[
            :, [column_index[name] for name in graph_node_names]
        ]

    node_to_index = {node: index for index, node in enumerate(graph.nodes)}
    sepsets: SepsetMap = {}
    cond_size = 0

    while max_cond_set_size is None or cond_size <= max_cond_set_size:
        found_candidate = False

        for x, y in list(graph.edges()):
            if not graph.is_adjacent(x, y):
                continue

            candidate_neighbors = [node for node in graph.neighbors(x) if node != y]
            if len(candidate_neighbors) < cond_size:
                continue

            for cond_set in combinations(candidate_neighbors, cond_size):
                found_candidate = True
                result = ci_test.test(
                    normalized_data,
                    node_to_index[x],
                    node_to_index[y],
                    tuple(node_to_index[node] for node in cond_set),
                )
                if verbose:
                    status = "independent" if result.independent else "dependent"
                    print(f"CI({x}, {y} | {set(cond_set)}) -> {status}")

                if result.independent:
                    graph.remove_edge(x, y)
                    sepset = set(cond_set)
                    sepsets[(x, y)] = sepset
                    sepsets[(y, x)] = set(sepset)
                    break

        if not found_candidate:
            break
        cond_size += 1

    return graph, sepsets
