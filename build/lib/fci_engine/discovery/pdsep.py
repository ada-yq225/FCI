"""Possible-D-Sep refinement for standard FCI."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable
from itertools import combinations
from typing import Optional

from fci_engine.ci import CITest
from fci_engine.discovery.skeleton import (
    SepsetMap,
    SepsetSourceMap,
    _prepare_data_for_graph,
)
from fci_engine.graph import PAG


def possible_dsep(
    graph: PAG,
    x: Hashable,
    y: Hashable,
    max_path_length: Optional[int] = None,
) -> set[Hashable]:
    """Return conservative Possible-D-Sep candidates for ``x`` relative to ``y``.

    A node is reached through a path from ``x`` where each intermediate triple is
    either a collider on that path or is shielded by a triangle.
    """

    if max_path_length is not None and max_path_length < 0:
        raise ValueError("max_path_length must be non-negative.")
    if max_path_length == 0:
        return set()

    candidates: set[Hashable] = set()
    visited_paths: set[tuple[Hashable, ...]] = set()
    queue: deque[tuple[Hashable, ...]] = deque()

    for neighbor in graph.neighbors(x):
        if neighbor == y:
            continue
        path = (x, neighbor)
        queue.append(path)
        visited_paths.add(path)
        candidates.add(neighbor)

    while queue:
        path = queue.popleft()
        current = path[-1]
        path_length = len(path) - 1
        if max_path_length is not None and path_length >= max_path_length:
            continue

        previous = path[-2]
        for next_node in graph.neighbors(current):
            if next_node in path or next_node == y:
                continue
            if not _is_pds_step_allowed(graph, previous, current, next_node):
                continue

            next_path = (*path, next_node)
            if next_path in visited_paths:
                continue
            visited_paths.add(next_path)
            candidates.add(next_node)
            queue.append(next_path)

    candidates.discard(x)
    candidates.discard(y)
    return candidates


def refine_skeleton_with_pdsep(
    data: object,
    graph: PAG,
    sepsets: SepsetMap,
    ci_test: CITest,
    max_cond_set_size: Optional[int] = None,
    max_path_length: Optional[int] = None,
    verbose: bool = False,
    sepset_sources: Optional[SepsetSourceMap] = None,
) -> tuple[PAG, SepsetMap]:
    """Refine the current skeleton by searching Possible-D-Sep sets."""

    if max_cond_set_size is not None and max_cond_set_size < 0:
        raise ValueError("max_cond_set_size must be non-negative.")

    normalized_data, node_to_index = _prepare_data_for_graph(data, graph)

    for x, y in list(graph.edges()):
        if not graph.is_adjacent(x, y):
            continue

        candidates = possible_dsep(graph, x, y, max_path_length=max_path_length)
        candidate_nodes = [node for node in graph.nodes if node in candidates]
        max_size = len(candidate_nodes)
        if max_cond_set_size is not None:
            max_size = min(max_size, max_cond_set_size)

        removed = False
        for cond_size in range(max_size + 1):
            if len(candidate_nodes) < cond_size:
                continue

            for cond_set in combinations(candidate_nodes, cond_size):
                result = ci_test.test(
                    normalized_data,
                    node_to_index[x],
                    node_to_index[y],
                    tuple(node_to_index[node] for node in cond_set),
                )
                if verbose:
                    status = "independent" if result.independent else "dependent"
                    print(f"PDS-CI({x}, {y} | {set(cond_set)}) -> {status}")

                if result.independent:
                    graph.remove_edge(x, y)
                    sepset = set(cond_set)
                    sepsets[(x, y)] = sepset
                    sepsets[(y, x)] = set(sepset)
                    if sepset_sources is not None:
                        sepset_sources[(x, y)] = "pdsep"
                        sepset_sources[(y, x)] = "pdsep"
                    removed = True
                    break
            if removed:
                break

    return graph, sepsets


def _is_pds_step_allowed(
    graph: PAG,
    previous: Hashable,
    current: Hashable,
    next_node: Hashable,
) -> bool:
    return _is_collider(graph, previous, current, next_node) or graph.is_adjacent(
        previous, next_node
    )


def _is_collider(
    graph: PAG,
    previous: Hashable,
    current: Hashable,
    next_node: Hashable,
) -> bool:
    return graph.has_arrowhead(previous, current) and graph.has_arrowhead(
        next_node, current
    )
