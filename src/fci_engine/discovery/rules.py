"""Standard FCI PAG orientation rules."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from collections import deque
from itertools import combinations
from typing import Optional

from fci_engine.discovery.orientation import (
    find_unshielded_triples,
    has_directed_path,
)
from fci_engine.diagnostics import OrientationEvent
from fci_engine.graph import Endpoint, PAG


SepsetMap = Mapping[tuple[Hashable, Hashable], set[Hashable]]
Triple = tuple[Hashable, Hashable, Hashable]
AmbiguousTripleKey = tuple[frozenset[Hashable], Hashable]


def apply_orientation_rules(
    graph: PAG,
    sepsets: SepsetMap,
    max_iter: int = 1000,
    max_path_length: Optional[int] = None,
    verbose: bool = False,
    trace: Optional[list[OrientationEvent]] = None,
    ambiguous_triples: Optional[Iterable[Triple]] = None,
) -> PAG:
    """Apply FCI orientation rules until convergence."""

    if max_iter < 0:
        raise ValueError("max_iter must be non-negative.")
    if max_path_length is not None and max_path_length < 0:
        raise ValueError("max_path_length must be non-negative.")

    rules = [
        rule_propagate_arrowheads,
        rule_double_triangle_arrowheads,
        rule_propagate_arrowheads_along_directed_paths,
        rule_avoid_directed_cycles,
        rule_selection_bias_tail_from_undirected,
        rule_selection_bias_tail_from_noncollider,
        rule_orient_tail_along_directed_chain,
    ]

    for iteration in range(max_iter):
        changed = False
        rule_changed = rule_avoid_new_unshielded_colliders(
            graph,
            sepsets,
            trace=trace,
            iteration=iteration,
            ambiguous_triples=ambiguous_triples,
        )
        if verbose and rule_changed:
            print(
                "rule_avoid_new_unshielded_colliders changed graph "
                f"at iteration {iteration}."
            )
        changed = changed or rule_changed
        for rule in rules:
            rule_changed = rule(
                graph,
                sepsets,
                trace=trace,
                iteration=iteration,
            )
            if verbose and rule_changed:
                print(f"{rule.__name__} changed graph at iteration {iteration}.")
            changed = changed or rule_changed
        rule_changed = rule_discriminating_paths(
            graph,
            sepsets,
            max_path_length=max_path_length,
            trace=trace,
            iteration=iteration,
        )
        if verbose and rule_changed:
            print(
                "rule_discriminating_paths changed graph "
                f"at iteration {iteration}."
            )
        changed = changed or rule_changed
        for path_rule in (
            rule_uncovered_circle_path_selection_bias,
            rule_orient_tail_along_uncovered_pd_path,
            rule_orient_tail_with_two_directed_parents,
        ):
            rule_changed = path_rule(
                graph,
                sepsets,
                max_path_length=max_path_length,
                trace=trace,
                iteration=iteration,
            )
            if verbose and rule_changed:
                print(f"{path_rule.__name__} changed graph at iteration {iteration}.")
            changed = changed or rule_changed
        if not changed:
            break
    return graph


def rule_avoid_new_unshielded_colliders(
    graph: PAG,
    sepsets: SepsetMap,
    trace: Optional[list[OrientationEvent]] = None,
    iteration: Optional[int] = None,
    ambiguous_triples: Optional[Iterable[Triple]] = None,
) -> bool:
    """R1: orient ``a *-> b o-* c`` as ``a *-> b --> c`` when unshielded."""

    changed = False
    ambiguous = _normalize_ambiguous_triples(ambiguous_triples)
    for x, z, y in find_unshielded_triples(graph):
        if _is_ambiguous_unshielded_triple(x, z, y, ambiguous):
            continue
        if graph.has_arrowhead(x, z):
            changed = _orient_directed_if_possible(
                graph,
                z,
                y,
                trace=trace,
                rule="R1",
                iteration=iteration,
                reason=f"avoid new unshielded collider {x!r}-{z!r}-{y!r}",
            ) or changed
        if graph.has_arrowhead(y, z):
            changed = (
                _orient_directed_if_possible(
                    graph,
                    z,
                    x,
                    trace=trace,
                    rule="R1",
                    iteration=iteration,
                    reason=f"avoid new unshielded collider {x!r}-{z!r}-{y!r}",
                )
                or changed
            )
    return changed


def _normalize_ambiguous_triples(
    ambiguous_triples: Optional[Iterable[Triple]],
) -> set[AmbiguousTripleKey]:
    if ambiguous_triples is None:
        return set()
    return {
        (frozenset((x, y)), z)
        for x, z, y in ambiguous_triples
        if x != z and z != y and x != y
    }


def _is_ambiguous_unshielded_triple(
    x: Hashable,
    z: Hashable,
    y: Hashable,
    ambiguous: set[AmbiguousTripleKey],
) -> bool:
    return (frozenset((x, y)), z) in ambiguous


def rule_propagate_arrowheads(
    graph: PAG,
    sepsets: SepsetMap,
    trace: Optional[list[OrientationEvent]] = None,
    iteration: Optional[int] = None,
) -> bool:
    """R2: propagate arrowheads through local directed triples."""

    changed = False
    for x, y in graph.edges():
        changed = (
            _apply_local_arrowhead_propagation(
                graph,
                x,
                y,
                trace=trace,
                iteration=iteration,
            )
            or changed
        )
        changed = (
            _apply_local_arrowhead_propagation(
                graph,
                y,
                x,
                trace=trace,
                iteration=iteration,
            )
            or changed
        )
    return changed


def rule_double_triangle_arrowheads(
    graph: PAG,
    sepsets: SepsetMap,
    trace: Optional[list[OrientationEvent]] = None,
    iteration: Optional[int] = None,
) -> bool:
    """R3: orient double-triangle patterns into an existing collider."""

    changed = False
    for a, b, c in _collider_triples(graph):
        if graph.is_adjacent(a, c):
            continue
        for d in graph.neighbors(b):
            if d in (a, b, c):
                continue
            if not graph.has_circle(d, b):
                continue
            if not graph.is_adjacent(a, d) or not graph.is_adjacent(c, d):
                continue
            if not graph.has_circle(a, d) or not graph.has_circle(c, d):
                continue
            changed = (
                _orient_arrowhead_if_circle(
                    graph,
                    d,
                    b,
                    trace=trace,
                    rule="R3",
                    iteration=iteration,
                    reason=(
                        "double-triangle pattern "
                        f"{a!r}->{b!r}<-{c!r} through {d!r}"
                    ),
                )
                or changed
            )
    return changed


def rule_propagate_arrowheads_along_directed_paths(
    graph: PAG,
    sepsets: SepsetMap,
    trace: Optional[list[OrientationEvent]] = None,
    iteration: Optional[int] = None,
) -> bool:
    """Orient arrowheads along existing directed paths."""

    changed = False
    for x, y in graph.edges():
        if has_directed_path(graph, x, y, excluded_edge=(x, y)):
            changed = (
                _orient_arrowhead_if_circle(
                    graph,
                    x,
                    y,
                    trace=trace,
                    rule="R3",
                    iteration=iteration,
                    reason=f"directed path from {x!r} to {y!r}",
                )
                or changed
            )
        if has_directed_path(graph, y, x, excluded_edge=(x, y)):
            changed = (
                _orient_arrowhead_if_circle(
                    graph,
                    y,
                    x,
                    trace=trace,
                    rule="R3",
                    iteration=iteration,
                    reason=f"directed path from {y!r} to {x!r}",
                )
                or changed
            )
    return changed


def rule_avoid_directed_cycles(
    graph: PAG,
    sepsets: SepsetMap,
    trace: Optional[list[OrientationEvent]] = None,
    iteration: Optional[int] = None,
) -> bool:
    """Orient tails on ``a o-> b`` when ``a`` already has a directed path to ``b``."""

    changed = False
    for x, y in graph.edges():
        if graph.has_circle(y, x) and graph.has_arrowhead(x, y):
            if has_directed_path(graph, x, y, excluded_edge=(x, y)):
                changed = (
                    _orient_tail_if_circle(
                        graph,
                        y,
                        x,
                        trace=trace,
                        rule="R2",
                        iteration=iteration,
                        reason=f"avoid directed cycle on {x!r}-{y!r}",
                    )
                    or changed
                )
        if graph.has_circle(x, y) and graph.has_arrowhead(y, x):
            if has_directed_path(graph, y, x, excluded_edge=(x, y)):
                changed = (
                    _orient_tail_if_circle(
                        graph,
                        x,
                        y,
                        trace=trace,
                        rule="R2",
                        iteration=iteration,
                        reason=f"avoid directed cycle on {y!r}-{x!r}",
                    )
                    or changed
                )
    return changed


def rule_discriminating_paths(
    graph: PAG,
    sepsets: SepsetMap,
    max_path_length: Optional[int] = None,
    trace: Optional[list[OrientationEvent]] = None,
    iteration: Optional[int] = None,
) -> bool:
    """R4: orient triples identified by definite discriminating paths."""

    changed = False
    for path in find_discriminating_paths(graph, max_path_length=max_path_length):
        d = path[0]
        a = path[-3]
        b = path[-2]
        c = path[-1]
        sepset = _get_sepset(sepsets, d, c)
        if sepset is None:
            continue

        if b in sepset:
            changed = (
                _orient_tail_if_circle(
                    graph,
                    c,
                    b,
                    trace=trace,
                    rule="R4",
                    iteration=iteration,
                    reason=f"discriminating path {path!r}; {b!r} in sepset",
                )
                or changed
            )
        else:
            changed = (
                _orient_arrowhead_if_circle(
                    graph,
                    a,
                    b,
                    trace=trace,
                    rule="R4",
                    iteration=iteration,
                    reason=f"discriminating path {path!r}; {b!r} not in sepset",
                )
                or changed
            )
            changed = (
                _orient_arrowhead_if_circle(
                    graph,
                    c,
                    b,
                    trace=trace,
                    rule="R4",
                    iteration=iteration,
                    reason=f"discriminating path {path!r}; {b!r} not in sepset",
                )
                or changed
            )
    return changed


def rule_uncovered_circle_path_selection_bias(
    graph: PAG,
    sepsets: SepsetMap,
    max_path_length: Optional[int] = None,
    trace: Optional[list[OrientationEvent]] = None,
    iteration: Optional[int] = None,
) -> bool:
    """R5: orient uncovered circle paths as undirected selection-bias edges."""

    changed = False
    for a, b in list(graph.edges()):
        if not graph.is_circle_edge(a, b):
            continue
        paths = _find_uncovered_circle_paths(
            graph,
            a,
            b,
            max_path_length=max_path_length,
        )
        if not paths:
            continue

        path = paths[0]
        reason = f"uncovered circle path {path!r}"
        changed = _orient_undirected_if_circles(
            graph,
            a,
            b,
            trace=trace,
            rule="R5",
            iteration=iteration,
            reason=reason,
        ) or changed
        for x, y in zip(path, path[1:]):
            changed = _orient_undirected_if_circles(
                graph,
                x,
                y,
                trace=trace,
                rule="R5",
                iteration=iteration,
                reason=reason,
            ) or changed
    return changed


def rule_selection_bias_tail_from_undirected(
    graph: PAG,
    sepsets: SepsetMap,
    trace: Optional[list[OrientationEvent]] = None,
    iteration: Optional[int] = None,
) -> bool:
    """R6: orient ``a --- b o-* c`` as ``a --- b --* c``."""

    changed = False
    for b in graph.nodes:
        for c in graph.neighbors(b):
            if not graph.has_circle(c, b):
                continue
            for a in graph.neighbors(b):
                if a == c:
                    continue
                if not graph.is_undirected_edge(a, b):
                    continue
                changed = (
                    _orient_tail_if_circle(
                        graph,
                        c,
                        b,
                        trace=trace,
                        rule="R6",
                        iteration=iteration,
                        reason=f"{a!r} --- {b!r} o-* {c!r}",
                    )
                    or changed
                )
    return changed


def rule_selection_bias_tail_from_noncollider(
    graph: PAG,
    sepsets: SepsetMap,
    trace: Optional[list[OrientationEvent]] = None,
    iteration: Optional[int] = None,
) -> bool:
    """R7: orient ``a -o b o-* c`` as ``a -o b --* c`` when unshielded."""

    changed = False
    for b in graph.nodes:
        for c in graph.neighbors(b):
            if not graph.has_circle(c, b):
                continue
            for a in graph.neighbors(b):
                if a == c or graph.is_adjacent(a, c):
                    continue
                if graph.get_endpoint(b, a) is not Endpoint.TAIL:
                    continue
                if graph.get_endpoint(a, b) is not Endpoint.CIRCLE:
                    continue
                changed = (
                    _orient_tail_if_circle(
                        graph,
                        c,
                        b,
                        trace=trace,
                        rule="R7",
                        iteration=iteration,
                        reason=f"unshielded {a!r} -o {b!r} o-* {c!r}",
                    )
                    or changed
                )
    return changed


def rule_orient_tail_along_directed_chain(
    graph: PAG,
    sepsets: SepsetMap,
    trace: Optional[list[OrientationEvent]] = None,
    iteration: Optional[int] = None,
) -> bool:
    """R8: orient ``a o-> c`` as ``a --> c`` along directed chains."""

    changed = False
    for a, c in list(graph.edges()):
        changed = (
            _apply_r8_for_order(
                graph,
                a,
                c,
                trace=trace,
                iteration=iteration,
            )
            or changed
        )
        changed = (
            _apply_r8_for_order(
                graph,
                c,
                a,
                trace=trace,
                iteration=iteration,
            )
            or changed
        )
    return changed


def rule_orient_tail_along_uncovered_pd_path(
    graph: PAG,
    sepsets: SepsetMap,
    max_path_length: Optional[int] = None,
    trace: Optional[list[OrientationEvent]] = None,
    iteration: Optional[int] = None,
) -> bool:
    """R9: orient tails using uncovered possibly directed paths."""

    changed = False
    for a, c in list(graph.edges()):
        changed = (
            _apply_r9_for_order(
                graph,
                a,
                c,
                max_path_length=max_path_length,
                trace=trace,
                iteration=iteration,
            )
            or changed
        )
        changed = (
            _apply_r9_for_order(
                graph,
                c,
                a,
                max_path_length=max_path_length,
                trace=trace,
                iteration=iteration,
            )
            or changed
        )
    return changed


def rule_orient_tail_with_two_directed_parents(
    graph: PAG,
    sepsets: SepsetMap,
    max_path_length: Optional[int] = None,
    trace: Optional[list[OrientationEvent]] = None,
    iteration: Optional[int] = None,
) -> bool:
    """R10: orient tails using two directed parents and uncovered paths."""

    changed = False
    for a, c in list(graph.edges()):
        changed = (
            _apply_r10_for_order(
                graph,
                a,
                c,
                max_path_length=max_path_length,
                trace=trace,
                iteration=iteration,
            )
            or changed
        )
        changed = (
            _apply_r10_for_order(
                graph,
                c,
                a,
                max_path_length=max_path_length,
                trace=trace,
                iteration=iteration,
            )
            or changed
        )
    return changed


def find_discriminating_paths(
    graph: PAG,
    max_path_length: Optional[int] = None,
) -> list[tuple[Hashable, ...]]:
    """Return definite discriminating paths ``(d, ..., a, b, c)``.

    A returned path discriminates the final triple ``a-b-c`` for ``b``. Every
    node between ``d`` and ``b`` is a collider on the path and a parent of
    ``c``, and ``d`` is not adjacent to ``c``.
    """

    if max_path_length is not None and max_path_length < 3:
        return []

    paths: list[tuple[Hashable, ...]] = []
    seen: set[tuple[Hashable, ...]] = set()

    for b in graph.nodes:
        for c in graph.neighbors(b):
            if not graph.has_circle(c, b):
                continue
            for a in graph.neighbors(b):
                if a == c:
                    continue
                if not graph.has_arrowhead(b, a):
                    continue
                if not graph.is_directed_edge(a, c):
                    continue

                for path in _search_discriminating_paths_from_suffix(
                    graph,
                    (a, b, c),
                    max_path_length=max_path_length,
                ):
                    if path not in seen:
                        seen.add(path)
                        paths.append(path)
    return paths


def _collider_triples(graph: PAG) -> list[tuple[Hashable, Hashable, Hashable]]:
    triples: list[tuple[Hashable, Hashable, Hashable]] = []
    for b in graph.nodes:
        neighbors = graph.neighbors(b)
        for i, a in enumerate(neighbors):
            if not graph.has_arrowhead(a, b):
                continue
            for c in neighbors[i + 1 :]:
                if graph.has_arrowhead(c, b):
                    triples.append((a, b, c))
    return triples


def _apply_r8_for_order(
    graph: PAG,
    a: Hashable,
    c: Hashable,
    trace: Optional[list[OrientationEvent]] = None,
    iteration: Optional[int] = None,
) -> bool:
    if not _is_circle_arrow_edge(graph, a, c):
        return False

    for b in graph.nodes:
        if b in (a, c):
            continue
        if not graph.is_adjacent(a, b) or not graph.is_directed_edge(b, c):
            continue
        if not graph.has_arrowhead(a, b):
            continue
        endpoint_at_a = graph.get_endpoint(b, a)
        if endpoint_at_a not in (Endpoint.TAIL, Endpoint.CIRCLE):
            continue
        return _orient_tail_if_circle(
            graph,
            c,
            a,
            trace=trace,
            rule="R8",
            iteration=iteration,
            reason=f"{a!r} *-> {b!r} --> {c!r}",
        )
    return False


def _apply_r9_for_order(
    graph: PAG,
    a: Hashable,
    c: Hashable,
    max_path_length: Optional[int] = None,
    trace: Optional[list[OrientationEvent]] = None,
    iteration: Optional[int] = None,
) -> bool:
    if not _is_circle_arrow_edge(graph, a, c):
        return False

    paths = _find_uncovered_possibly_directed_paths(
        graph,
        a,
        c,
        max_path_length=max_path_length,
        excluded_edge=(a, c),
    )
    for path in paths:
        if len(path) < 3:
            continue
        first = path[1]
        if graph.is_adjacent(first, c):
            continue
        return _orient_tail_if_circle(
            graph,
            c,
            a,
            trace=trace,
            rule="R9",
            iteration=iteration,
            reason=f"uncovered possibly directed path {path!r}",
        )
    return False


def _apply_r10_for_order(
    graph: PAG,
    a: Hashable,
    c: Hashable,
    max_path_length: Optional[int] = None,
    trace: Optional[list[OrientationEvent]] = None,
    iteration: Optional[int] = None,
) -> bool:
    if not _is_circle_arrow_edge(graph, a, c):
        return False

    parents = [
        node
        for node in graph.neighbors(c)
        if node != a and graph.is_directed_edge(node, c)
    ]
    for b, d in combinations(parents, 2):
        paths_to_b = _find_uncovered_possibly_directed_paths(
            graph,
            a,
            b,
            max_path_length=max_path_length,
            excluded_edge=(a, c),
        )
        paths_to_d = _find_uncovered_possibly_directed_paths(
            graph,
            a,
            d,
            max_path_length=max_path_length,
            excluded_edge=(a, c),
        )
        for path_b in paths_to_b:
            for path_d in paths_to_d:
                if len(path_b) < 2 or len(path_d) < 2:
                    continue
                first_b = path_b[1]
                first_d = path_d[1]
                if first_b == first_d or graph.is_adjacent(first_b, first_d):
                    continue
                return _orient_tail_if_circle(
                    graph,
                    c,
                    a,
                    trace=trace,
                    rule="R10",
                    iteration=iteration,
                    reason=(
                        f"parents {b!r}, {d!r} of {c!r} with paths "
                        f"{path_b!r} and {path_d!r}"
                    ),
                )
    return False


def _find_uncovered_circle_paths(
    graph: PAG,
    source: Hashable,
    target: Hashable,
    max_path_length: Optional[int] = None,
) -> list[tuple[Hashable, ...]]:
    if max_path_length is not None and max_path_length < 2:
        return []

    paths: list[tuple[Hashable, ...]] = []
    queue: deque[tuple[Hashable, ...]] = deque([(source,)])
    seen: set[tuple[Hashable, ...]] = {(source,)}

    while queue:
        path = queue.popleft()
        current = path[-1]
        path_length = len(path) - 1
        if max_path_length is not None and path_length >= max_path_length:
            continue

        for neighbor in graph.neighbors(current):
            if neighbor in path:
                continue
            if frozenset((current, neighbor)) == frozenset((source, target)):
                continue
            if not graph.is_circle_edge(current, neighbor):
                continue
            next_path = (*path, neighbor)
            if len(next_path) >= 3 and graph.is_adjacent(
                next_path[-3],
                next_path[-1],
            ):
                continue
            if next_path in seen:
                continue
            seen.add(next_path)
            if neighbor == target:
                paths.append(next_path)
            else:
                queue.append(next_path)
    return paths


def _find_uncovered_possibly_directed_paths(
    graph: PAG,
    source: Hashable,
    target: Hashable,
    max_path_length: Optional[int] = None,
    excluded_edge: Optional[tuple[Hashable, Hashable]] = None,
) -> list[tuple[Hashable, ...]]:
    if max_path_length is not None and max_path_length < 1:
        return []

    excluded = frozenset(excluded_edge) if excluded_edge is not None else None
    paths: list[tuple[Hashable, ...]] = []
    queue: deque[tuple[Hashable, ...]] = deque([(source,)])
    seen: set[tuple[Hashable, ...]] = {(source,)}

    while queue:
        path = queue.popleft()
        current = path[-1]
        path_length = len(path) - 1
        if current == target:
            paths.append(path)
            continue
        if max_path_length is not None and path_length >= max_path_length:
            continue

        for neighbor in graph.neighbors(current):
            if neighbor in path:
                continue
            if excluded is not None and frozenset((current, neighbor)) == excluded:
                continue
            if not _is_possibly_directed_step(graph, current, neighbor):
                continue
            next_path = (*path, neighbor)
            if len(next_path) >= 3 and graph.is_adjacent(
                next_path[-3],
                next_path[-1],
            ):
                continue
            if next_path in seen:
                continue
            seen.add(next_path)
            queue.append(next_path)
    return paths


def _is_circle_arrow_edge(graph: PAG, a: Hashable, c: Hashable) -> bool:
    return graph.has_circle(c, a) and graph.has_arrowhead(a, c)


def _is_possibly_directed_step(
    graph: PAG,
    current: Hashable,
    next_node: Hashable,
) -> bool:
    return not graph.has_arrowhead(next_node, current)


def _orient_undirected_if_circles(
    graph: PAG,
    x: Hashable,
    y: Hashable,
    trace: Optional[list[OrientationEvent]] = None,
    rule: str = "",
    iteration: Optional[int] = None,
    reason: str = "",
) -> bool:
    changed = False
    changed = (
        _orient_tail_if_circle(
            graph,
            x,
            y,
            trace=trace,
            rule=rule,
            iteration=iteration,
            reason=reason,
        )
        or changed
    )
    changed = (
        _orient_tail_if_circle(
            graph,
            y,
            x,
            trace=trace,
            rule=rule,
            iteration=iteration,
            reason=reason,
        )
        or changed
    )
    return changed


def _orient_directed_if_possible(
    graph: PAG,
    source: Hashable,
    target: Hashable,
    trace: Optional[list[OrientationEvent]] = None,
    rule: str = "",
    iteration: Optional[int] = None,
    reason: str = "",
) -> bool:
    """Orient ``source`` to ``target`` without overwriting fixed endpoints."""

    if not graph.has_circle(target, source):
        return False

    changed = False
    changed = (
        _orient_tail_if_circle(
            graph,
            target,
            source,
            trace=trace,
            rule=rule,
            iteration=iteration,
            reason=reason,
        )
        or changed
    )
    changed = (
        _orient_arrowhead_if_circle(
            graph,
            source,
            target,
            trace=trace,
            rule=rule,
            iteration=iteration,
            reason=reason,
        )
        or changed
    )
    return changed


def _apply_local_arrowhead_propagation(
    graph: PAG,
    x: Hashable,
    y: Hashable,
    trace: Optional[list[OrientationEvent]] = None,
    iteration: Optional[int] = None,
) -> bool:
    if not graph.has_circle(x, y):
        return False

    for z in graph.nodes:
        if z in (x, y):
            continue
        if not graph.is_adjacent(x, z) or not graph.is_adjacent(z, y):
            continue
        if graph.is_directed_edge(x, z) and graph.has_arrowhead(z, y):
            return _orient_arrowhead_if_circle(
                graph,
                x,
                y,
                trace=trace,
                rule="R2",
                iteration=iteration,
                reason=f"local propagation through {z!r}",
            )
        if graph.has_arrowhead(x, z) and graph.is_directed_edge(z, y):
            return _orient_arrowhead_if_circle(
                graph,
                x,
                y,
                trace=trace,
                rule="R2",
                iteration=iteration,
                reason=f"local propagation through {z!r}",
            )
    return False


def _search_discriminating_paths_from_suffix(
    graph: PAG,
    suffix: tuple[Hashable, Hashable, Hashable],
    max_path_length: Optional[int] = None,
) -> list[tuple[Hashable, ...]]:
    a, b, c = suffix
    paths: list[tuple[Hashable, ...]] = []
    queue: deque[tuple[Hashable, ...]] = deque([(a, b, c)])

    while queue:
        suffix_path = queue.popleft()
        left = suffix_path[0]
        next_node = suffix_path[1]
        path_length = len(suffix_path) - 1
        if max_path_length is not None and path_length >= max_path_length:
            continue

        for candidate in graph.neighbors(left):
            if candidate in suffix_path:
                continue
            if not graph.has_arrowhead(candidate, left):
                continue
            if not graph.has_arrowhead(next_node, left):
                continue

            path = (candidate, *suffix_path)
            if not graph.is_adjacent(candidate, c):
                paths.append(path)
                continue

            if graph.is_directed_edge(candidate, c):
                queue.append(path)

    return paths


def _get_sepset(
    sepsets: SepsetMap,
    x: Hashable,
    y: Hashable,
) -> Optional[set[Hashable]]:
    return sepsets.get((x, y), sepsets.get((y, x)))


def _orient_arrowhead_if_circle(
    graph: PAG,
    x: Hashable,
    y: Hashable,
    trace: Optional[list[OrientationEvent]] = None,
    rule: str = "",
    iteration: Optional[int] = None,
    reason: str = "",
) -> bool:
    """Put an arrowhead at ``y`` only when the endpoint is currently a circle."""

    current = graph.get_endpoint(x, y)
    if current is Endpoint.CIRCLE:
        if has_directed_path(graph, y, x, excluded_edge=(x, y)):
            return False
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
                    iteration=iteration,
                    reason=reason,
                )
            )
        return True
    return False


def _orient_tail_if_circle(
    graph: PAG,
    x: Hashable,
    y: Hashable,
    trace: Optional[list[OrientationEvent]] = None,
    rule: str = "",
    iteration: Optional[int] = None,
    reason: str = "",
) -> bool:
    """Put a tail at ``y`` only when the endpoint is currently a circle."""

    current = graph.get_endpoint(x, y)
    if current is Endpoint.CIRCLE:
        before_edge = graph.edge_repr(x, y)
        graph.orient_tail(x, y)
        if trace is not None:
            trace.append(
                OrientationEvent(
                    rule=rule,
                    edge=(x, y),
                    oriented_endpoint=y,
                    before=current,
                    after=Endpoint.TAIL,
                    before_edge=before_edge,
                    after_edge=graph.edge_repr(x, y),
                    iteration=iteration,
                    reason=reason,
                )
            )
        return True
    return False
