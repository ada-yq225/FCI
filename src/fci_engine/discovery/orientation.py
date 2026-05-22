"""Orientation helpers for FCI discovery."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Mapping
from typing import Optional

from fci_engine.graph import Endpoint, PAG


Triple = tuple[Hashable, Hashable, Hashable]
SepsetMap = Mapping[tuple[Hashable, Hashable], set[Hashable]]


def find_unshielded_triples(graph: PAG) -> list[Triple]:
    """Return unshielded triples ``(x, z, y)`` centered at ``z``."""

    triples: list[Triple] = []
    for z in graph.nodes:
        neighbors = graph.neighbors(z)
        for i, x in enumerate(neighbors):
            for y in neighbors[i + 1 :]:
                if not graph.is_adjacent(x, y):
                    triples.append((x, z, y))
    return triples


def is_unshielded_triple(
    graph: PAG,
    x: Hashable,
    z: Hashable,
    y: Hashable,
) -> bool:
    """Return whether ``x-z-y`` is an unshielded triple."""

    return (
        x != z
        and z != y
        and x != y
        and graph.is_adjacent(x, z)
        and graph.is_adjacent(z, y)
        and not graph.is_adjacent(x, y)
    )


def has_directed_path(
    graph: PAG,
    source: Hashable,
    target: Hashable,
    excluded_edge: Optional[tuple[Hashable, Hashable]] = None,
) -> bool:
    """Return whether a non-empty directed path exists from source to target."""

    if source == target:
        return False

    excluded = frozenset(excluded_edge) if excluded_edge is not None else None
    visited = {source}
    queue: deque[Hashable] = deque([source])

    while queue:
        current = queue.popleft()
        for neighbor in graph.neighbors(current):
            if excluded is not None and frozenset((current, neighbor)) == excluded:
                continue
            if not graph.is_directed_edge(current, neighbor):
                continue
            if neighbor == target:
                return True
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return False


def possible_ancestor(graph: PAG, x: Hashable, y: Hashable) -> bool:
    """Return whether ``x`` is still a possible ancestor of ``y``."""

    return graph.is_possible_ancestor(x, y)


def definite_noncollider(
    graph: PAG,
    x: Hashable,
    z: Hashable,
    y: Hashable,
    sepsets: Optional[SepsetMap] = None,
) -> bool:
    """Return whether ``z`` is a definite noncollider on ``x-z-y``."""

    if x == z or z == y or x == y:
        return False
    if not graph.is_adjacent(x, z) or not graph.is_adjacent(z, y):
        return False
    if graph.has_tail(x, z) or graph.has_tail(y, z):
        return True
    if sepsets is not None and is_unshielded_triple(graph, x, z, y):
        return z in _get_sepset(sepsets, x, y)
    return False


def orient_unshielded_colliders(graph: PAG, sepsets: SepsetMap) -> PAG:
    """Orient unshielded colliders as ``x *-> z <-* y``."""

    for x, z, y in find_unshielded_triples(graph):
        sepset = _get_sepset(sepsets, x, y)
        if z in sepset:
            continue

        _safe_orient_arrowhead(graph, x, z)
        _safe_orient_arrowhead(graph, y, z)
    return graph


def _get_sepset(
    sepsets: SepsetMap,
    x: Hashable,
    y: Hashable,
) -> set[Hashable]:
    return sepsets.get((x, y), sepsets.get((y, x), set()))


def _safe_orient_arrowhead(graph: PAG, x: Hashable, y: Hashable) -> None:
    """Put an arrowhead at ``y`` without erasing stronger endpoint marks."""

    current = graph.get_endpoint(x, y)
    if current is Endpoint.ARROW:
        return
    if current is Endpoint.CIRCLE:
        graph.orient_arrowhead(x, y)
        return
    if current in (Endpoint.NONE, Endpoint.TAIL):
        return
