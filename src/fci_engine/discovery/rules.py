"""Standard FCI PAG orientation rules."""

from __future__ import annotations

from collections.abc import Hashable, Mapping

from fci_engine.discovery.orientation import (
    find_unshielded_triples,
    has_directed_path,
)
from fci_engine.graph import Endpoint, PAG


SepsetMap = Mapping[tuple[Hashable, Hashable], set[Hashable]]


def apply_orientation_rules(
    graph: PAG,
    sepsets: SepsetMap,
    max_iter: int = 1000,
    verbose: bool = False,
) -> PAG:
    """Apply FCI orientation rules until convergence."""

    if max_iter < 0:
        raise ValueError("max_iter must be non-negative.")

    rules = [
        rule_avoid_new_unshielded_colliders,
        rule_propagate_arrowheads,
        rule_propagate_arrowheads_along_directed_paths,
        rule_avoid_directed_cycles,
        rule_discriminating_paths,
    ]

    for iteration in range(max_iter):
        changed = False
        for rule in rules:
            rule_changed = rule(graph, sepsets)
            if verbose and rule_changed:
                print(f"{rule.__name__} changed graph at iteration {iteration}.")
            changed = changed or rule_changed
        if not changed:
            break
    return graph


def rule_avoid_new_unshielded_colliders(graph: PAG, sepsets: SepsetMap) -> bool:
    """R1: orient ``a *-> b o-* c`` as ``a *-> b --* c`` when unshielded."""

    changed = False
    for x, z, y in find_unshielded_triples(graph):
        if graph.has_arrowhead(x, z):
            changed = _orient_tail_if_circle(graph, y, z) or changed
        if graph.has_arrowhead(y, z):
            changed = _orient_tail_if_circle(graph, x, z) or changed
    return changed


def rule_propagate_arrowheads(graph: PAG, sepsets: SepsetMap) -> bool:
    """R2: propagate arrowheads through local directed triples."""

    changed = False
    for x, y in graph.edges():
        changed = _apply_local_arrowhead_propagation(graph, x, y) or changed
        changed = _apply_local_arrowhead_propagation(graph, y, x) or changed
    return changed


def rule_propagate_arrowheads_along_directed_paths(
    graph: PAG,
    sepsets: SepsetMap,
) -> bool:
    """Orient arrowheads along existing directed paths."""

    changed = False
    for x, y in graph.edges():
        if has_directed_path(graph, x, y, excluded_edge=(x, y)):
            changed = _orient_arrowhead_if_circle(graph, x, y) or changed
        if has_directed_path(graph, y, x, excluded_edge=(x, y)):
            changed = _orient_arrowhead_if_circle(graph, y, x) or changed
    return changed


def rule_avoid_directed_cycles(graph: PAG, sepsets: SepsetMap) -> bool:
    """Orient tails on ``a o-> b`` when ``a`` already has a directed path to ``b``."""

    changed = False
    for x, y in graph.edges():
        if graph.has_circle(y, x) and graph.has_arrowhead(x, y):
            if has_directed_path(graph, x, y, excluded_edge=(x, y)):
                changed = _orient_tail_if_circle(graph, y, x) or changed
        if graph.has_circle(x, y) and graph.has_arrowhead(y, x):
            if has_directed_path(graph, y, x, excluded_edge=(x, y)):
                changed = _orient_tail_if_circle(graph, x, y) or changed
    return changed


def rule_discriminating_paths(graph: PAG, sepsets: SepsetMap) -> bool:
    """TODO: implement standard FCI discriminating path orientation rule R4."""

    return False


def _apply_local_arrowhead_propagation(
    graph: PAG,
    x: Hashable,
    y: Hashable,
) -> bool:
    if not graph.has_circle(x, y):
        return False

    for z in graph.nodes:
        if z in (x, y):
            continue
        if not graph.is_adjacent(x, z) or not graph.is_adjacent(z, y):
            continue
        if graph.is_directed_edge(x, z) and graph.has_arrowhead(z, y):
            return _orient_arrowhead_if_circle(graph, x, y)
        if graph.has_arrowhead(x, z) and graph.is_directed_edge(z, y):
            return _orient_arrowhead_if_circle(graph, x, y)
    return False


def _orient_arrowhead_if_circle(graph: PAG, x: Hashable, y: Hashable) -> bool:
    """Put an arrowhead at ``y`` only when the endpoint is currently a circle."""

    current = graph.get_endpoint(x, y)
    if current is Endpoint.CIRCLE:
        graph.orient_arrowhead(x, y)
        return True
    return False


def _orient_tail_if_circle(graph: PAG, x: Hashable, y: Hashable) -> bool:
    """Put a tail at ``y`` only when the endpoint is currently a circle."""

    current = graph.get_endpoint(x, y)
    if current is Endpoint.CIRCLE:
        graph.orient_tail(x, y)
        return True
    return False
