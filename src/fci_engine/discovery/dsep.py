"""Hierarchical D-SEP refinement used by FCI+."""

from __future__ import annotations

from collections.abc import Hashable
from itertools import combinations
from typing import Optional

from fci_engine.ci import CITest
from fci_engine.discovery.orientation import _safe_orient_arrowhead
from fci_engine.discovery.skeleton import (
    SepsetMap,
    SepsetSourceMap,
    _prepare_data_for_graph,
)
from fci_engine.graph import Endpoint, PAG


def build_augmented_skeleton(
    graph: PAG,
    sepsets: SepsetMap,
    data: Optional[object] = None,
    ci_test: Optional[CITest] = None,
) -> PAG:
    """Build the FCI+ augmented skeleton approximation.

    The augmented skeleton starts from the current skeleton without orienting
    unshielded colliders. When data and a CI test are supplied, it performs the
    FCI+ single-node augmentation:
    if ``x`` and ``y`` are separated by ``S`` but become dependent after adding
    an adjacent node ``a`` to ``S``, then all existing edges from
    ``{x, y} union S`` into ``a`` receive arrowheads at ``a``.
    """

    augmented = graph.copy()
    if data is not None and ci_test is not None:
        _augment_with_single_node_dependencies(data, augmented, sepsets, ci_test)
    return augmented


def possible_dsep_links(augmented_graph: PAG) -> list[tuple[Hashable, Hashable]]:
    """Return candidate D-SEP links following the FCI+ bidirected pattern.

    A D-SEP link in the augmented skeleton has the characteristic local form
    ``u <-> x <-> y <-> v`` with ``u`` and ``v`` not adjacent. We use that
    pattern when the available endpoint marks support it. For finite-sample
    robustness, a circle endpoint is treated as potentially compatible, but the
    witness nodes must also be connected by not-against-arrowhead paths in the
    cross direction, matching the FCI+ PosDsepLinks criterion.
    """

    candidates: list[tuple[Hashable, Hashable]] = []
    for x, y in augmented_graph.edges():
        if not _edge_can_be_bidirected(augmented_graph, x, y):
            continue
        if _has_dsep_link_witness(augmented_graph, x, y):
            candidates.append((x, y))
    return candidates


def hierarchy(
    seed_nodes: set[Hashable],
    sepsets: SepsetMap,
    exclude_pair: Optional[tuple[Hashable, Hashable]] = None,
) -> set[Hashable]:
    """Return ``HIE(seed_nodes, I)`` from the FCI+ paper."""

    expanded = set(seed_nodes)
    changed = True
    excluded = frozenset(exclude_pair) if exclude_pair is not None else None

    while changed:
        changed = False
        current_nodes = list(expanded)
        for i, x in enumerate(current_nodes):
            for y in current_nodes[i + 1 :]:
                if excluded is not None and frozenset((x, y)) == excluded:
                    continue
                sepset = sepsets.get((x, y), sepsets.get((y, x)))
                if not sepset:
                    continue
                before_size = len(expanded)
                expanded.update(sepset)
                if len(expanded) > before_size:
                    changed = True

    return expanded


def minimal_dsep(
    data: object,
    graph: PAG,
    x: Hashable,
    y: Hashable,
    cond_set: set[Hashable],
    ci_test: CITest,
) -> set[Hashable]:
    """Remove redundant nodes from a D-separating set."""

    normalized_data, node_to_index = _prepare_data_for_graph(data, graph)
    minimized = {node for node in cond_set if node not in {x, y}}

    changed = True
    while changed:
        changed = False
        for node in graph.nodes:
            if node not in minimized:
                continue
            candidate = minimized - {node}
            result = ci_test.test(
                normalized_data,
                node_to_index[x],
                node_to_index[y],
                tuple(
                    node_to_index[item] for item in graph.nodes if item in candidate
                ),
            )
            if result.independent:
                minimized = candidate
                changed = True
                break

    return minimized


def refine_skeleton_with_fci_plus_dsep(
    data: object,
    graph: PAG,
    sepsets: SepsetMap,
    ci_test: CITest,
    max_degree: Optional[int] = None,
    verbose: bool = False,
    sepset_sources: Optional[SepsetSourceMap] = None,
) -> tuple[PAG, SepsetMap]:
    """Refine the skeleton using the FCI+ hierarchical D-SEP search."""

    if max_degree is not None and max_degree < 0:
        raise ValueError("max_degree must be non-negative.")

    normalized_data, node_to_index = _prepare_data_for_graph(data, graph)
    tried_without_update: set[frozenset[Hashable]] = set()
    augmented = build_augmented_skeleton(graph, sepsets, normalized_data, ci_test)
    candidates = possible_dsep_links(augmented)

    while candidates:
        x, y = candidates.pop(0)
        candidate_key = frozenset((x, y))
        if candidate_key in tried_without_update or not graph.is_adjacent(x, y):
            continue

        base_x = [node for node in augmented.neighbors(x) if node != y]
        base_y = [node for node in augmented.neighbors(y) if node != x]
        base_union = [
            node for node in graph.nodes if node in set(base_x) | set(base_y)
        ]
        max_subset_size = len(base_union)
        if max_degree is not None:
            max_subset_size = min(max_subset_size, 2 * max_degree)

        removed = False
        for subset_size in range(1, max_subset_size + 1):
            for subset in combinations(base_union, subset_size):
                if not set(subset) & set(base_x):
                    continue
                if not set(subset) & set(base_y):
                    continue
                if max_degree is not None:
                    if len(set(subset) & set(base_x)) > max_degree:
                        continue
                    if len(set(subset) & set(base_y)) > max_degree:
                        continue
                seed = {x, y, *subset}
                cond_nodes = hierarchy(seed, sepsets, exclude_pair=(x, y)) - {
                    x,
                    y,
                }
                cond_ordered = [node for node in graph.nodes if node in cond_nodes]
                result = ci_test.test(
                    normalized_data,
                    node_to_index[x],
                    node_to_index[y],
                    tuple(node_to_index[node] for node in cond_ordered),
                )
                if verbose:
                    status = "independent" if result.independent else "dependent"
                    print(
                        "FCI+-DSEP("
                        f"{x}, {y} | {set(cond_ordered)}"
                        f") -> {status}"
                    )

                if not result.independent:
                    continue

                minimized = minimal_dsep(
                    data,
                    graph,
                    x,
                    y,
                    set(cond_ordered),
                    ci_test,
                )
                graph.remove_edge(x, y)
                augmented.remove_edge(x, y)
                sepsets[(x, y)] = set(minimized)
                sepsets[(y, x)] = set(minimized)
                if sepset_sources is not None:
                    sepset_sources[(x, y)] = "fci_plus_dsep"
                    sepset_sources[(y, x)] = "fci_plus_dsep"

                augmented = build_augmented_skeleton(
                    graph,
                    sepsets,
                    normalized_data,
                    ci_test,
                )
                candidates = possible_dsep_links(augmented)
                tried_without_update.clear()
                removed = True
                break

        if not removed:
            tried_without_update.add(candidate_key)

    return graph, sepsets


def _edge_can_be_bidirected(graph: PAG, x: Hashable, y: Hashable) -> bool:
    return (
        graph.get_endpoint(x, y) in (Endpoint.ARROW, Endpoint.CIRCLE)
        and graph.get_endpoint(y, x) in (Endpoint.ARROW, Endpoint.CIRCLE)
    )


def _has_dsep_link_witness(graph: PAG, x: Hashable, y: Hashable) -> bool:
    left_witnesses = [
        node
        for node in graph.neighbors(x)
        if node != y and _edge_can_be_bidirected(graph, node, x)
    ]
    right_witnesses = [
        node
        for node in graph.neighbors(y)
        if node != x and _edge_can_be_bidirected(graph, y, node)
    ]
    ancestors_of_x = _not_against_arrowhead_reachable(graph, x)
    ancestors_of_y = _not_against_arrowhead_reachable(graph, y)
    for left in left_witnesses:
        if left not in ancestors_of_y:
            continue
        for right in right_witnesses:
            if right not in ancestors_of_x:
                continue
            if left != right and not graph.is_adjacent(left, right):
                return True
    return False


def _not_against_arrowhead_reachable(graph: PAG, target: Hashable) -> set[Hashable]:
    return {
        node
        for node in graph.nodes
        if node != target and graph.is_possible_ancestor(node, target)
    }


def _augment_with_single_node_dependencies(
    data: object,
    graph: PAG,
    sepsets: SepsetMap,
    ci_test: CITest,
) -> None:
    normalized_data, node_to_index = _prepare_data_for_graph(data, graph)
    seen_pairs: set[frozenset[Hashable]] = set()

    for (x, y), sepset in list(sepsets.items()):
        pair_key = frozenset((x, y))
        if len(pair_key) != 2 or pair_key in seen_pairs:
            continue
        if x not in node_to_index or y not in node_to_index:
            continue
        seen_pairs.add(pair_key)

        sep_nodes = {node for node in sepset if node in node_to_index}
        adjacent = set(graph.neighbors(x)) | set(graph.neighbors(y))
        for sep_node in sep_nodes:
            adjacent.update(graph.neighbors(sep_node))

        protected_nodes = {x, y} | sep_nodes
        candidates = [
            node for node in graph.nodes if node in adjacent - protected_nodes
        ]
        for candidate in candidates:
            cond_nodes = [
                node for node in graph.nodes if node in sep_nodes | {candidate}
            ]
            result = ci_test.test(
                normalized_data,
                node_to_index[x],
                node_to_index[y],
                tuple(node_to_index[node] for node in cond_nodes),
            )
            if result.independent:
                continue

            for source in protected_nodes:
                if source == candidate or not graph.is_adjacent(source, candidate):
                    continue
                _safe_orient_arrowhead(
                    graph,
                    source,
                    candidate,
                    rule="fci_plus_augment_graph",
                    reason=(
                        f"{x!r} and {y!r} become dependent after adding "
                        f"{candidate!r} to their sepset"
                    ),
                )
