"""Initial PC-style skeleton discovery used by FCI."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from itertools import combinations
from typing import Optional

import numpy as np

from fci_engine.ci import CITest
from fci_engine.graph import PAG
from fci_engine.utils.validation import validate_numeric_data


SepsetMap = dict[tuple[Hashable, Hashable], set[Hashable]]
SepsetSourceMap = dict[tuple[Hashable, Hashable], str]


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
    sepset_sources: Optional[SepsetSourceMap] = None,
    stable: bool = True,
    sepset_selection: str = "max_pvalue",
) -> tuple[PAG, SepsetMap]:
    """Learn the initial undirected PAG skeleton using CI tests.

    This is the PC-style adjacency search used as FCI's first stage. It only
    removes edges and records separating sets; it does not orient colliders.
    When ``stable`` is true, adjacency sets are snapshotted at each conditioning
    depth and edge removals are applied after the depth is fully tested. This
    avoids order dependence within a depth level.
    """

    if max_cond_set_size is not None and max_cond_set_size < 0:
        raise ValueError("max_cond_set_size must be non-negative.")
    if sepset_selection not in {"first", "max_pvalue"}:
        raise ValueError("sepset_selection must be 'first' or 'max_pvalue'.")

    normalized_data, node_to_index = _prepare_data_for_graph(data, graph)
    sepsets: SepsetMap = {}
    cond_size = 0

    while max_cond_set_size is None or cond_size <= max_cond_set_size:
        found_candidate = False
        pending_removals: list[tuple[Hashable, Hashable, set[Hashable]]] = []
        adjacency_snapshot = {
            node: graph.neighbors(node) for node in graph.nodes
        } if stable else None

        for x, y in list(graph.edges()):
            if not graph.is_adjacent(x, y):
                continue

            if stable:
                assert adjacency_snapshot is not None
                candidate_sets = _conditioning_candidate_sets_from_adjacency(
                    adjacency_snapshot,
                    x,
                    y,
                )
            else:
                candidate_sets = _conditioning_candidate_sets(graph, x, y)

            if all(
                len(candidate_neighbors) < cond_size
                for candidate_neighbors in candidate_sets
            ):
                continue

            seen_conditioning_sets: set[frozenset[Hashable]] = set()
            edge_marked_for_removal = False
            best_separation: Optional[tuple[float, set[Hashable]]] = None
            for candidate_neighbors in candidate_sets:
                if len(candidate_neighbors) < cond_size:
                    continue

                for cond_set in combinations(candidate_neighbors, cond_size):
                    frozen_cond_set = frozenset(cond_set)
                    if frozen_cond_set in seen_conditioning_sets:
                        continue
                    seen_conditioning_sets.add(frozen_cond_set)
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
                        sepset = set(cond_set)
                        if sepset_selection == "max_pvalue":
                            if (
                                best_separation is None
                                or result.p_value > best_separation[0]
                            ):
                                best_separation = (result.p_value, sepset)
                            continue

                        edge_marked_for_removal = True
                        _queue_or_apply_removal(
                            graph,
                            sepsets,
                            pending_removals,
                            x,
                            y,
                            sepset,
                            sepset_sources,
                            stable=stable,
                        )
                        break
                if edge_marked_for_removal or not graph.is_adjacent(x, y):
                    break

            if best_separation is not None and graph.is_adjacent(x, y):
                _queue_or_apply_removal(
                    graph,
                    sepsets,
                    pending_removals,
                    x,
                    y,
                    best_separation[1],
                    sepset_sources,
                    stable=stable,
                )

        for x, y, sepset in pending_removals:
            if graph.is_adjacent(x, y):
                _remove_edge_with_sepset(
                    graph,
                    sepsets,
                    x,
                    y,
                    sepset,
                    sepset_sources,
                )

        if not found_candidate:
            break
        cond_size += 1

    return graph, sepsets


def _queue_or_apply_removal(
    graph: PAG,
    sepsets: SepsetMap,
    pending_removals: list[tuple[Hashable, Hashable, set[Hashable]]],
    x: Hashable,
    y: Hashable,
    sepset: set[Hashable],
    sepset_sources: Optional[SepsetSourceMap],
    stable: bool,
) -> None:
    if stable:
        pending_removals.append((x, y, sepset))
    else:
        _remove_edge_with_sepset(
            graph,
            sepsets,
            x,
            y,
            sepset,
            sepset_sources,
        )


def _conditioning_candidate_sets(
    graph: PAG,
    x: Hashable,
    y: Hashable,
) -> list[list[Hashable]]:
    """Return adjacency-based conditioning candidates from both edge endpoints."""

    return [
        [node for node in graph.neighbors(x) if node != y],
        [node for node in graph.neighbors(y) if node != x],
    ]


def _conditioning_candidate_sets_from_adjacency(
    adjacency: dict[Hashable, list[Hashable]],
    x: Hashable,
    y: Hashable,
) -> list[list[Hashable]]:
    return [
        [node for node in adjacency[x] if node != y],
        [node for node in adjacency[y] if node != x],
    ]


def _remove_edge_with_sepset(
    graph: PAG,
    sepsets: SepsetMap,
    x: Hashable,
    y: Hashable,
    sepset: set[Hashable],
    sepset_sources: Optional[SepsetSourceMap],
) -> None:
    graph.remove_edge(x, y)
    sepsets[(x, y)] = sepset
    sepsets[(y, x)] = set(sepset)
    if sepset_sources is not None:
        sepset_sources[(x, y)] = "initial"
        sepset_sources[(y, x)] = "initial"


def _prepare_data_for_graph(
    data: object,
    graph: PAG,
) -> tuple[np.ndarray, dict[Hashable, int]]:
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
    return normalized_data, node_to_index
