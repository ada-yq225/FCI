"""Orientation helpers for FCI discovery."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Mapping
from itertools import combinations
from typing import Optional

from fci_engine.ci import CITest
from fci_engine.diagnostics import OrientationEvent
from fci_engine.discovery.skeleton import _prepare_data_for_graph
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


def orient_unshielded_colliders(
    graph: PAG,
    sepsets: SepsetMap,
    trace: Optional[list[OrientationEvent]] = None,
) -> PAG:
    """Orient unshielded colliders as ``x *-> z <-* y``."""

    for x, z, y in find_unshielded_triples(graph):
        sepset = _get_sepset(sepsets, x, y)
        if z in sepset:
            continue

        reason = f"{z!r} not in sepset({x!r}, {y!r})"
        _safe_orient_arrowhead(
            graph,
            x,
            z,
            trace=trace,
            rule="orient_unshielded_colliders",
            reason=reason,
        )
        _safe_orient_arrowhead(
            graph,
            y,
            z,
            trace=trace,
            rule="orient_unshielded_colliders",
            reason=reason,
        )
    return graph


def orient_unshielded_colliders_conservative(
    data: object,
    graph: PAG,
    sepsets: SepsetMap,
    ci_test: CITest,
    max_cond_set_size: Optional[int] = None,
    trace: Optional[list[OrientationEvent]] = None,
) -> tuple[PAG, list[Triple]]:
    """Conservatively orient unshielded colliders.

    For every unshielded triple ``x-z-y``, this searches for additional
    separating sets for ``x`` and ``y`` among their current adjacencies. If all
    discovered separating sets exclude ``z``, the triple is oriented as a
    collider. If all include ``z``, it is treated as a definite noncollider. If
    both kinds are found, the triple is ambiguous and left unoriented.
    """

    if max_cond_set_size is not None and max_cond_set_size < 0:
        raise ValueError("max_cond_set_size must be non-negative.")

    normalized_data, node_to_index = _prepare_data_for_graph(data, graph)
    ambiguous_triples: list[Triple] = []

    for x, z, y in find_unshielded_triples(graph):
        separating_sets = _find_separating_sets_for_pair(
            normalized_data,
            graph,
            x,
            y,
            ci_test,
            node_to_index,
            max_cond_set_size=max_cond_set_size,
            known_sepset=_lookup_known_sepset(sepsets, x, y),
        )
        if not separating_sets:
            continue

        has_center = any(z in sepset for sepset in separating_sets)
        has_without_center = any(z not in sepset for sepset in separating_sets)
        if has_center and has_without_center:
            ambiguous_triples.append((x, z, y))
            continue
        if has_center:
            continue

        reason = (
            f"all separating sets for {x!r}, {y!r} exclude {z!r} "
            "(conservative collider)"
        )
        _safe_orient_arrowhead(
            graph,
            x,
            z,
            trace=trace,
            rule="orient_unshielded_colliders_conservative",
            reason=reason,
        )
        _safe_orient_arrowhead(
            graph,
            y,
            z,
            trace=trace,
            rule="orient_unshielded_colliders_conservative",
            reason=reason,
        )

    return graph, ambiguous_triples


def reset_endpoint_marks(graph: PAG) -> PAG:
    """Reset every remaining edge to ``circle-circle`` while preserving skeleton."""

    for x, y in graph.edges():
        graph.add_circle_edge(x, y)
    return graph


def _get_sepset(
    sepsets: SepsetMap,
    x: Hashable,
    y: Hashable,
) -> set[Hashable]:
    return sepsets.get((x, y), sepsets.get((y, x), set()))


def _lookup_known_sepset(
    sepsets: SepsetMap,
    x: Hashable,
    y: Hashable,
) -> Optional[set[Hashable]]:
    sepset = sepsets.get((x, y), sepsets.get((y, x)))
    if sepset is None:
        return None
    return set(sepset)


def _find_separating_sets_for_pair(
    data: object,
    graph: PAG,
    x: Hashable,
    y: Hashable,
    ci_test: CITest,
    node_to_index: dict[Hashable, int],
    max_cond_set_size: Optional[int],
    known_sepset: Optional[set[Hashable]],
) -> list[set[Hashable]]:
    separating_sets: list[set[Hashable]] = []
    seen: set[frozenset[Hashable]] = set()
    if known_sepset is not None:
        separating_sets.append(set(known_sepset))
        seen.add(frozenset(known_sepset))

    max_size = max_cond_set_size
    pools = [
        [node for node in graph.neighbors(x) if node != y],
        [node for node in graph.neighbors(y) if node != x],
    ]
    if max_size is None:
        max_size = max(len(pool) for pool in pools)

    for candidate_nodes in pools:
        limit = min(len(candidate_nodes), max_size)
        for size in range(limit + 1):
            for cond_set in combinations(candidate_nodes, size):
                frozen = frozenset(cond_set)
                if frozen in seen:
                    continue
                seen.add(frozen)
                result = ci_test.test(
                    data,
                    node_to_index[x],
                    node_to_index[y],
                    tuple(node_to_index[node] for node in cond_set),
                )
                if result.independent:
                    separating_sets.append(set(cond_set))

    return separating_sets


def _safe_orient_arrowhead(
    graph: PAG,
    x: Hashable,
    y: Hashable,
    trace: Optional[list[OrientationEvent]] = None,
    rule: str = "orient_unshielded_colliders",
    reason: str = "",
) -> None:
    """Put an arrowhead at ``y`` without erasing stronger endpoint marks."""

    current = graph.get_endpoint(x, y)
    if current is Endpoint.ARROW:
        return
    if current is Endpoint.CIRCLE:
        before_edge = graph.edge_repr(x, y)
        graph.orient_arrowhead(x, y)
        if trace is not None:
            trace.append(
                OrientationEvent(
                    rule=rule,
                    edge=(x, y),
                    oriented_endpoint=y,
                    before=current,
                    after=Endpoint.ARROW,
                    before_edge=before_edge,
                    after_edge=graph.edge_repr(x, y),
                    reason=reason,
                )
            )
        return
    if current in (Endpoint.NONE, Endpoint.TAIL):
        return
