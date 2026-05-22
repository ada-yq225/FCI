"""Orientation helpers for FCI discovery."""

from __future__ import annotations

from collections.abc import Hashable, Mapping

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
