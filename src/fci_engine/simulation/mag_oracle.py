"""MAG/PAG oracle helpers for exact m-separation tests."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from fci_engine.ci import CITest, CITestResult
from fci_engine.metrics.accuracy import Shape

Edge = tuple[str, str]


@dataclass(frozen=True)
class MAGSpec:
    """Small maximal ancestral graph specification with exact m-separation.

    This is intended for oracle tests and benchmarks where the target
    independence model matters more than finite-sample data generation.
    """

    nodes: tuple[str, ...]
    directed_edges: tuple[Edge, ...] = ()
    bidirected_edges: tuple[Edge, ...] = ()
    undirected_edges: tuple[Edge, ...] = ()
    pag_shape: Optional[Shape] = None

    def __init__(
        self,
        nodes: Iterable[str],
        directed_edges: Iterable[Edge] = (),
        bidirected_edges: Iterable[Edge] = (),
        undirected_edges: Iterable[Edge] = (),
        pag_shape: Optional[Shape] = None,
    ) -> None:
        object.__setattr__(self, "nodes", tuple(nodes))
        object.__setattr__(self, "directed_edges", tuple(directed_edges))
        object.__setattr__(self, "bidirected_edges", tuple(bidirected_edges))
        object.__setattr__(self, "undirected_edges", tuple(undirected_edges))
        object.__setattr__(self, "pag_shape", pag_shape)
        self._validate()

    def is_m_separated(
        self,
        x: str,
        y: str,
        cond_set: Iterable[str] = (),
    ) -> bool:
        """Return whether ``x`` and ``y`` are m-separated by ``cond_set``."""

        self._validate_node(x)
        self._validate_node(y)
        cond = set(cond_set)
        if x in cond or y in cond:
            raise ValueError("Conditioning set must not contain endpoints.")
        active_collider_targets = self._ancestors_of(cond)
        for path in self._simple_paths(x, y):
            if self._path_m_connects(path, cond, active_collider_targets):
                return False
        return True

    def oracle_shape(self) -> Shape:
        """Return the expected PAG/MAG endpoint shape."""

        if self.pag_shape is not None:
            return dict(self.pag_shape)

        return self.implied_pag_shape()

    def implied_pag_shape(self) -> Shape:
        """Return the endpoint shape implied by explicit MAG adjacencies."""

        order = {node: index for index, node in enumerate(self.nodes)}
        shape: Shape = {}
        for source, target in self.directed_edges:
            edge, endpoints = _ordered_endpoint_pair(
                source,
                target,
                "TAIL",
                "ARROW",
                order,
            )
            shape[edge] = endpoints
        for x, y in self.bidirected_edges:
            edge = _ordered_pair(x, y, order)
            shape[edge] = ("ARROW", "ARROW")
        for x, y in self.undirected_edges:
            edge = _ordered_pair(x, y, order)
            shape[edge] = ("TAIL", "TAIL")
        return shape

    def implied_skeleton_shape(self) -> Shape:
        """Return all adjacencies implied by m-separation.

        Pairs with no separating set are included. Explicit MAG edge marks are
        preserved; otherwise the edge is represented as unresolved ``o-o``.
        """

        order = {node: index for index, node in enumerate(self.nodes)}
        explicit = self.implied_pag_shape()
        shape: Shape = {}
        for i, x in enumerate(self.nodes):
            for y in self.nodes[i + 1 :]:
                if self.minimal_separating_sets(x, y):
                    continue
                edge = _ordered_pair(x, y, order)
                shape[edge] = explicit.get(edge, ("CIRCLE", "CIRCLE"))
        return shape

    def dummy_data(self, n_rows: int = 16) -> pd.DataFrame:
        """Return deterministic placeholder data with matching columns."""

        return pd.DataFrame(
            np.zeros((n_rows, len(self.nodes))),
            columns=list(self.nodes),
        )

    def oracle_ci_test(self, alpha: float = 0.05) -> "MAGOracleCITest":
        """Return a CI test backed by this MAG's exact m-separation model."""

        return MAGOracleCITest(self, alpha=alpha)

    def minimal_separating_sets(self, x: str, y: str) -> list[set[str]]:
        """Enumerate all minimal m-separating sets for one pair."""

        candidates = [node for node in self.nodes if node not in {x, y}]
        minimal: list[set[str]] = []
        for size in range(len(candidates) + 1):
            for cond in combinations(candidates, size):
                cond_set = set(cond)
                if not self.is_m_separated(x, y, cond_set):
                    continue
                if any(existing <= cond_set for existing in minimal):
                    continue
                minimal.append(cond_set)
        return minimal

    def _path_m_connects(
        self,
        path: Sequence[str],
        cond: set[str],
        active_collider_targets: set[str],
    ) -> bool:
        for index in range(1, len(path) - 1):
            prev_node = path[index - 1]
            node = path[index]
            next_node = path[index + 1]
            collider = self._has_arrowhead(prev_node, node) and self._has_arrowhead(
                next_node,
                node,
            )
            if collider:
                if node not in active_collider_targets:
                    return False
            elif node in cond:
                return False
        return True

    def _simple_paths(self, source: str, target: str) -> list[tuple[str, ...]]:
        paths: list[tuple[str, ...]] = []
        stack: list[tuple[str, tuple[str, ...]]] = [(source, (source,))]
        while stack:
            node, path = stack.pop()
            for neighbor in self._neighbors(node):
                if neighbor in path:
                    continue
                next_path = (*path, neighbor)
                if neighbor == target:
                    paths.append(next_path)
                else:
                    stack.append((neighbor, next_path))
        return paths

    def _ancestors_of(self, nodes: set[str]) -> set[str]:
        ancestors = set(nodes)
        parents = {node: set[str]() for node in self.nodes}
        for source, target in self.directed_edges:
            parents[target].add(source)
        queue: deque[str] = deque(nodes)
        while queue:
            node = queue.popleft()
            for parent in parents[node]:
                if parent in ancestors:
                    continue
                ancestors.add(parent)
                queue.append(parent)
        return ancestors

    def _neighbors(self, node: str) -> list[str]:
        adjacent = set()
        for x, y in self._all_edges():
            if x == node:
                adjacent.add(y)
            elif y == node:
                adjacent.add(x)
        return [candidate for candidate in self.nodes if candidate in adjacent]

    def _has_arrowhead(self, source: str, target: str) -> bool:
        if (source, target) in self.directed_edges:
            return True
        if (target, source) in self.directed_edges:
            return False
        if _unordered_edge(source, target) in {
            _unordered_edge(x, y) for x, y in self.bidirected_edges
        }:
            return True
        return False

    def _all_edges(self) -> tuple[Edge, ...]:
        return self.directed_edges + self.bidirected_edges + self.undirected_edges

    def _validate(self) -> None:
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("MAGSpec nodes must be unique.")
        for edge in self._all_edges():
            if len(edge) != 2 or edge[0] == edge[1]:
                raise ValueError(f"Invalid MAG edge: {edge!r}.")
            self._validate_node(edge[0])
            self._validate_node(edge[1])
        seen: set[frozenset[str]] = set()
        for edge in self._all_edges():
            key = frozenset(edge)
            if key in seen:
                raise ValueError(f"Duplicate MAG adjacency: {edge!r}.")
            seen.add(key)

    def _validate_node(self, node: str) -> None:
        if node not in self.nodes:
            raise ValueError(f"Unknown MAG node: {node!r}.")


class MAGOracleCITest(CITest):
    """Conditional independence oracle backed by a ``MAGSpec``."""

    def __init__(self, mag: MAGSpec, alpha: float = 0.05) -> None:
        super().__init__(alpha=alpha)
        self.mag = mag
        self.names = list(mag.nodes)

    def test(
        self,
        data: np.ndarray,
        x: int,
        y: int,
        cond_set: Sequence[int],
    ) -> CITestResult:
        x_name = self.names[x]
        y_name = self.names[y]
        cond_names = [self.names[index] for index in cond_set]
        independent = self.mag.is_m_separated(x_name, y_name, cond_names)
        return CITestResult(
            independent=independent,
            p_value=0.99 if independent else 0.001,
            statistic=None,
            method="mag_oracle",
            n_samples=None,
        )


def canonical_dsep_mag() -> MAGSpec:
    """Return a small MAG with the canonical FCI D-sep pattern.

    The path ``X <-> U <-> Z -> V <-> Y`` is blocked by noncollider ``Z`` only
    when conditioning includes ``Z`` together with the adjacent ancestors
    ``U`` and ``V``.
    """

    return MAGSpec(
        nodes=["X", "U", "Z", "V", "Y"],
        directed_edges=[
            ("Z", "V"),
        ],
        bidirected_edges=[
            ("X", "U"),
            ("U", "Z"),
            ("V", "Y"),
        ],
        pag_shape={
            ("X", "U"): ("ARROW", "ARROW"),
            ("U", "Z"): ("ARROW", "ARROW"),
            ("Z", "V"): ("TAIL", "ARROW"),
            ("V", "Y"): ("ARROW", "ARROW"),
        },
    )


def _unordered_edge(x: str, y: str) -> frozenset[str]:
    return frozenset((x, y))


def _ordered_pair(x: str, y: str, order: dict[str, int]) -> Edge:
    return (x, y) if order[x] <= order[y] else (y, x)


def _ordered_endpoint_pair(
    source: str,
    target: str,
    endpoint_source: str,
    endpoint_target: str,
    order: dict[str, int],
) -> tuple[Edge, tuple[str, str]]:
    if order[source] <= order[target]:
        return (source, target), (endpoint_source, endpoint_target)
    return (target, source), (endpoint_target, endpoint_source)
