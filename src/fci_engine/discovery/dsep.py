"""Hierarchical D-SEP refinement used by FCI+."""

from __future__ import annotations

from collections.abc import Hashable
from itertools import combinations
from typing import Optional

from fci_engine.ci import CITest
from fci_engine.diagnostics import DSEPDiagnostics
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
    allow_nan: bool = False,
) -> PAG:
    """Build the FCI+ augmented skeleton used by Algorithm 2.

    The augmented skeleton starts from the current skeleton without orienting
    unshielded colliders. When data and a CI test are supplied, it performs the
    FCI+ single-node augmentation:
    if ``x`` and ``y`` are separated by ``S`` but become dependent after adding
    an adjacent node ``a`` to ``S``, then all existing edges from
    ``{x, y} union S`` into ``a`` receive arrowheads at ``a``.
    """

    augmented = graph.copy()
    if data is not None and ci_test is not None:
        _augment_with_single_node_dependencies(
            data,
            augmented,
            sepsets,
            ci_test,
            allow_nan=allow_nan,
        )
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
    allow_nan: bool = False,
) -> set[Hashable]:
    """Remove redundant nodes from a D-separating set."""

    normalized_data, node_to_index = _prepare_data_for_graph(
        data,
        graph,
        allow_nan=allow_nan,
    )
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
    sepset_selection: str = "max_pvalue",
    allow_nan: bool = False,
    diagnostics: Optional[DSEPDiagnostics] = None,
) -> tuple[PAG, SepsetMap]:
    """Refine the skeleton using the FCI+ hierarchical D-SEP search.

    This follows the D-SEP stage of Claassen et al.'s Algorithm 2. For each
    candidate edge ``X-Y`` from the augmented skeleton, it enumerates separate
    base subsets ``ZX <= BaseX`` and ``ZY <= BaseY`` of size at most the sparse
    degree bound ``k``. Each pair seeds ``HIE({X,Y} union ZX union ZY, I)``;
    if that hierarchy separates ``X`` and ``Y``, the edge is removed and the
    augmented skeleton/candidate list is rebuilt.
    """

    if max_degree is not None and max_degree < 0:
        raise ValueError("max_degree must be non-negative.")
    if sepset_selection not in {"first", "max_pvalue"}:
        raise ValueError("sepset_selection must be 'first' or 'max_pvalue'.")

    normalized_data, node_to_index = _prepare_data_for_graph(
        data,
        graph,
        allow_nan=allow_nan,
    )
    tried_without_update: set[frozenset[Hashable]] = set()
    attempted_edges: set[frozenset[Hashable]] = set()
    hierarchy_cache: dict[frozenset[Hashable], set[Hashable]] = {}
    augmented = build_augmented_skeleton(
        graph,
        sepsets,
        normalized_data,
        ci_test,
        allow_nan=allow_nan,
    )
    candidates = possible_dsep_links(augmented)

    while candidates:
        x, y = candidates.pop(0)
        candidate_key = frozenset((x, y))
        if diagnostics is not None:
            diagnostics.candidate_edges_seen += 1
            if candidate_key in attempted_edges:
                diagnostics.candidate_revisits += 1
        attempted_edges.add(candidate_key)
        if candidate_key in tried_without_update or not graph.is_adjacent(x, y):
            continue

        base_x = [node for node in augmented.neighbors(x) if node != y]
        base_y = [node for node in augmented.neighbors(y) if node != x]
        removed = False
        tested_conditioning_sets: set[frozenset[Hashable]] = set()
        for depth in _algorithm2_base_depths(base_x, base_y, max_degree=max_degree):
            best_at_depth: Optional[tuple[float, set[Hashable]]] = None
            for zx_candidate, zy_candidate in _base_combinations_at_depth(
                graph,
                base_x,
                base_y,
                depth,
                max_degree=max_degree,
            ):
                seed = {x, y, *zx_candidate, *zy_candidate}
                hierarchy_key = frozenset(seed)
                if diagnostics is not None:
                    diagnostics.hierarchy_queries += 1
                if hierarchy_key in hierarchy_cache:
                    expanded = hierarchy_cache[hierarchy_key]
                    if diagnostics is not None:
                        diagnostics.hierarchy_cache_hits += 1
                else:
                    expanded = hierarchy(seed, sepsets, exclude_pair=(x, y))
                    hierarchy_cache[hierarchy_key] = set(expanded)
                cond_nodes = expanded - {x, y}
                cond_ordered = [node for node in graph.nodes if node in cond_nodes]
                cond_key = frozenset(cond_ordered)
                if cond_key in tested_conditioning_sets:
                    if diagnostics is not None:
                        diagnostics.duplicate_conditioning_skips += 1
                    continue
                tested_conditioning_sets.add(cond_key)
                if diagnostics is not None:
                    diagnostics.max_conditioning_size = max(
                        diagnostics.max_conditioning_size,
                        len(cond_ordered),
                    )
                    diagnostics.ci_tests += 1
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

                cond_set = set(cond_ordered)
                if sepset_selection == "first":
                    best_at_depth = (result.p_value, cond_set)
                    break
                if best_at_depth is None or result.p_value > best_at_depth[0]:
                    best_at_depth = (result.p_value, cond_set)

            if best_at_depth is None:
                continue

            minimized = minimal_dsep(
                data,
                graph,
                x,
                y,
                best_at_depth[1],
                ci_test,
                allow_nan=allow_nan,
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
                allow_nan=allow_nan,
            )
            candidates = possible_dsep_links(augmented)
            tried_without_update.clear()
            hierarchy_cache.clear()
            if diagnostics is not None:
                diagnostics.edges_removed += 1
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


def _algorithm2_base_depths(
    base_x: list[Hashable],
    base_y: list[Hashable],
    max_degree: Optional[int],
) -> range:
    max_x = len(base_x)
    max_y = len(base_y)
    if max_degree is not None:
        max_x = min(max_x, max_degree)
        max_y = min(max_y, max_degree)
    if max_x == 0 or max_y == 0:
        return range(0)
    return range(2, max_x + max_y + 1)


def _base_combinations_at_depth(
    graph: PAG,
    base_x: list[Hashable],
    base_y: list[Hashable],
    depth: int,
    max_degree: Optional[int],
) -> list[tuple[tuple[Hashable, ...], tuple[Hashable, ...]]]:
    """Return Algorithm 2 ``ZX``/``ZY`` pairs with ``|ZX| + |ZY| == depth``."""

    max_x = len(base_x)
    max_y = len(base_y)
    if max_degree is not None:
        max_x = min(max_x, max_degree)
        max_y = min(max_y, max_degree)

    pairs: list[tuple[tuple[Hashable, ...], tuple[Hashable, ...]]] = []
    seen: set[tuple[frozenset[Hashable], frozenset[Hashable]]] = set()
    for size_x in range(1, max_x + 1):
        size_y = depth - size_x
        if size_y < 1 or size_y > max_y:
            continue
        for zx in combinations(base_x, size_x):
            for zy in combinations(base_y, size_y):
                key = (frozenset(zx), frozenset(zy))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(
                    (
                        tuple(node for node in graph.nodes if node in set(zx)),
                        tuple(node for node in graph.nodes if node in set(zy)),
                    )
                )
    return pairs


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
    allow_nan: bool = False,
) -> None:
    normalized_data, node_to_index = _prepare_data_for_graph(
        data,
        graph,
        allow_nan=allow_nan,
    )
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
