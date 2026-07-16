"""Possible-D-Sep refinement for standard FCI."""

from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Optional

from fci_engine.ci import CITest
from fci_engine.discovery.skeleton import _prepare_data_for_graph
from fci_engine.graph import PAG
from fci_engine.types import SepsetMap, SepsetSourceMap


def possible_dsep(
    graph: PAG,
    x: str,
    y: str,
    max_path_length: Optional[int] = None,
) -> set[str]:
    """Return conservative Possible-D-Sep candidates for ``x`` relative to ``y``.

    A node is reached through a path from ``x`` where each intermediate triple is
    either a collider on that path or is shielded by a triangle. The search is
    implemented over ordered edge states instead of full simple paths, which
    keeps dense finite-sample PAGs from triggering exponential path enumeration.
    """

    if max_path_length is not None and max_path_length < 0:
        raise ValueError("max_path_length must be non-negative.")
    if max_path_length == 0:
        return set()

    candidates: set[str] = set()
    visited_states: set[tuple[str, str]] = set()
    queue: deque[tuple[str, str, int]] = deque()

    for neighbor in graph.neighbors(x):
        state = (x, neighbor)
        queue.append((x, neighbor, 1))
        visited_states.add(state)
        candidates.add(neighbor)

    while queue:
        previous, current, path_length = queue.popleft()
        if max_path_length is not None and path_length >= max_path_length:
            continue

        for next_node in graph.neighbors(current):
            if next_node in {x, previous}:
                continue
            if not _is_pds_step_allowed(graph, previous, current, next_node):
                continue

            next_state = (current, next_node)
            if next_state in visited_states:
                continue
            visited_states.add(next_state)
            candidates.add(next_node)
            queue.append((current, next_node, path_length + 1))

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
    stable: bool = True,
    sepset_selection: str = "max_pvalue",
    allow_nan: bool = False,
) -> tuple[PAG, SepsetMap]:
    """Refine the current skeleton by searching Possible-D-Sep sets."""

    if max_cond_set_size is not None and max_cond_set_size < 0:
        raise ValueError("max_cond_set_size must be non-negative.")
    if sepset_selection not in {"first", "max_pvalue"}:
        raise ValueError("sepset_selection must be 'first' or 'max_pvalue'.")

    normalized_data, node_to_index = _prepare_data_for_graph(
        data,
        graph,
        allow_nan=allow_nan,
    )

    search_graph = graph.copy() if stable else graph
    pending_removals: list[tuple[str, str, set[str]]] = []
    marked_for_removal: set[frozenset[str]] = set()

    for x, y in list(search_graph.edges()):
        edge_key = frozenset((x, y))
        if edge_key in marked_for_removal or not graph.is_adjacent(x, y):
            continue

        candidate_pools = [
            possible_dsep(
                search_graph,
                x,
                y,
                max_path_length=max_path_length,
            ),
            possible_dsep(
                search_graph,
                y,
                x,
                max_path_length=max_path_length,
            ),
        ]
        ordered_pools = [
            [node for node in graph.nodes if node in candidates]
            for candidates in candidate_pools
        ]
        max_size = max((len(pool) for pool in ordered_pools), default=0)
        if max_cond_set_size is not None:
            max_size = min(max_size, max_cond_set_size)

        separating_set: Optional[set[str]] = None
        seen_conditioning_sets: set[frozenset[str]] = set()
        for cond_size in range(max_size + 1):
            best_at_depth: Optional[tuple[float, set[str]]] = None
            for candidate_nodes in ordered_pools:
                if len(candidate_nodes) < cond_size:
                    continue
                for cond_set in combinations(candidate_nodes, cond_size):
                    cond_key = frozenset(cond_set)
                    if cond_key in seen_conditioning_sets:
                        continue
                    seen_conditioning_sets.add(cond_key)
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
                        sepset = set(cond_set)
                        if sepset_selection == "first":
                            separating_set = sepset
                            break
                        if best_at_depth is None or result.p_value > best_at_depth[0]:
                            best_at_depth = (result.p_value, sepset)
                if separating_set is not None:
                    break
            if best_at_depth is not None:
                separating_set = best_at_depth[1]
            if separating_set is not None:
                break

        if separating_set is None:
            continue

        if stable:
            marked_for_removal.add(edge_key)
            pending_removals.append((x, y, separating_set))
        else:
            _remove_edge_with_pdsep(
                graph,
                sepsets,
                x,
                y,
                separating_set,
                sepset_sources=sepset_sources,
            )

    for x, y, sepset in pending_removals:
        _remove_edge_with_pdsep(
            graph,
            sepsets,
            x,
            y,
            sepset,
            sepset_sources=sepset_sources,
        )

    return graph, sepsets


def _remove_edge_with_pdsep(
    graph: PAG,
    sepsets: SepsetMap,
    x: str,
    y: str,
    sepset: set[str],
    sepset_sources: Optional[SepsetSourceMap] = None,
) -> None:
    if graph.is_adjacent(x, y):
        graph.remove_edge(x, y)
    sepsets[(x, y)] = sepset
    sepsets[(y, x)] = set(sepset)
    if sepset_sources is not None:
        sepset_sources[(x, y)] = "pdsep"
        sepset_sources[(y, x)] = "pdsep"


def _is_pds_step_allowed(
    graph: PAG,
    previous: str,
    current: str,
    next_node: str,
) -> bool:
    if _is_collider(graph, previous, current, next_node):
        return True
    if not graph.is_adjacent(previous, next_node):
        return False
    return not _has_definite_noncollider_mark(
        graph,
        previous,
        current,
        next_node,
    )


def _is_collider(
    graph: PAG,
    previous: str,
    current: str,
    next_node: str,
) -> bool:
    return graph.has_arrowhead(previous, current) and graph.has_arrowhead(
        next_node, current
    )


def _has_definite_noncollider_mark(
    graph: PAG,
    previous: str,
    current: str,
    next_node: str,
) -> bool:
    """Return whether a tail proves ``current`` is a noncollider on the path."""

    return graph.has_tail(previous, current) or graph.has_tail(
        next_node,
        current,
    )
