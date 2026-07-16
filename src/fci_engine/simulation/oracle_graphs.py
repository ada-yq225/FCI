"""Helpers for generating reference PAG shapes from causal graph specs."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional

from fci_engine.metrics.accuracy import NormalizedShape, Shape

DirectedEdge = tuple[str, str]


@dataclass(frozen=True)
class CausalGraphSpec:
    """A small DAG-with-latents specification for benchmark oracle generation.

    The generated shape is a conservative structural reference:

    - observed directed effects are represented as ``o->`` by default
    - observed effects marked in ``definite_directed_edges`` become ``-->``
    - latent common causes induce ``<->`` between their observed descendants

    This is intended for controlled benchmarks and regression tests. It is not
    a replacement for a full MAG/PAG oracle enumerator.
    """

    observed_nodes: tuple[str, ...]
    latent_nodes: tuple[str, ...] = ()
    directed_edges: tuple[DirectedEdge, ...] = ()
    definite_directed_edges: frozenset[DirectedEdge] = field(default_factory=frozenset)

    def __init__(
        self,
        observed_nodes: Iterable[str],
        latent_nodes: Iterable[str] = (),
        directed_edges: Iterable[DirectedEdge] = (),
        definite_directed_edges: Iterable[DirectedEdge] = (),
    ) -> None:
        observed = tuple(observed_nodes)
        latent = tuple(latent_nodes)
        edges = tuple(directed_edges)
        object.__setattr__(self, "observed_nodes", observed)
        object.__setattr__(self, "latent_nodes", latent)
        object.__setattr__(self, "directed_edges", edges)
        object.__setattr__(
            self,
            "definite_directed_edges",
            frozenset(definite_directed_edges),
        )
        self._validate()

    def to_pag_shape(self) -> Shape:
        """Return a conservative reference PAG shape over observed nodes."""

        shape: NormalizedShape = {}
        observed = set(self.observed_nodes)
        order = {node: index for index, node in enumerate(self.observed_nodes)}

        for source, target in self._observed_directed_relations():
            edge = _ordered_pair(source, target, order)
            if edge == (source, target):
                endpoints = (
                    ("TAIL", "ARROW")
                    if (source, target) in self.definite_directed_edges
                    else ("CIRCLE", "ARROW")
                )
            else:
                endpoints = (
                    ("ARROW", "TAIL")
                    if (source, target) in self.definite_directed_edges
                    else ("ARROW", "CIRCLE")
                )
            shape[edge] = _merge_endpoints(shape.get(edge), endpoints)

        for latent in self.latent_nodes:
            descendants = self._observed_children_through_latents(latent)
            for i, x in enumerate(descendants):
                for y in descendants[i + 1 :]:
                    if x not in observed or y not in observed:
                        continue
                    edge = _ordered_pair(x, y, order)
                    shape[edge] = _merge_endpoints(shape.get(edge), ("ARROW", "ARROW"))

        return shape

    def _observed_directed_relations(self) -> set[DirectedEdge]:
        relations: set[DirectedEdge] = set()
        observed = set(self.observed_nodes)
        for source in self.observed_nodes:
            queue: deque[tuple[str, bool]] = deque([(source, False)])
            visited: set[tuple[str, bool]] = {(source, False)}
            while queue:
                node, crossed_latent = queue.popleft()
                for child in self._children(node):
                    child_is_latent = child in self.latent_nodes
                    next_crossed_latent = crossed_latent or child_is_latent
                    state = (child, next_crossed_latent)
                    if state in visited:
                        continue
                    visited.add(state)
                    if child in observed and child != source:
                        if crossed_latent or not child_is_latent:
                            relations.add((source, child))
                        continue
                    queue.append((child, next_crossed_latent))
        return relations

    def _has_directed_path(self, source: str, target: str) -> bool:
        if source == target:
            return True
        visited = {source}
        queue: deque[str] = deque([source])
        while queue:
            node = queue.popleft()
            for child in self._children(node):
                if child == target:
                    return True
                if child in visited:
                    continue
                visited.add(child)
                queue.append(child)
        return False

    def _observed_children_through_latents(self, latent: str) -> list[str]:
        observed = set(self.observed_nodes)
        latent_nodes = set(self.latent_nodes)
        reached: list[str] = []
        queue: deque[str] = deque([latent])
        visited = {latent}
        while queue:
            node = queue.popleft()
            for child in self._children(node):
                if child in observed:
                    reached.append(child)
                    continue
                if child not in latent_nodes or child in visited:
                    continue
                visited.add(child)
                queue.append(child)
        return [node for node in self.observed_nodes if node in set(reached)]

    def _children(self, node: str) -> list[str]:
        return [target for source, target in self.directed_edges if source == node]

    def _validate(self) -> None:
        all_nodes = set(self.observed_nodes) | set(self.latent_nodes)
        if len(all_nodes) != len(self.observed_nodes) + len(self.latent_nodes):
            raise ValueError("Observed and latent node names must be unique.")
        for source, target in self.directed_edges:
            if source == target:
                raise ValueError("Directed edges require distinct nodes.")
            if source not in all_nodes or target not in all_nodes:
                raise ValueError(
                    f"Unknown directed edge endpoint: {(source, target)!r}."
                )
        for edge in self.definite_directed_edges:
            if edge not in self.directed_edges:
                raise ValueError(
                    "definite_directed_edges must be a subset of directed_edges."
                )


def _ordered_pair(x: str, y: str, order: dict[str, int]) -> tuple[str, str]:
    return (x, y) if order[x] <= order[y] else (y, x)


def _merge_endpoints(
    current: Optional[tuple[str, str]],
    incoming: tuple[str, str],
) -> tuple[str, str]:
    if current is None:
        return incoming
    return (
        _merge_endpoint(current[0], incoming[0]),
        _merge_endpoint(current[1], incoming[1]),
    )


def _merge_endpoint(current: str, incoming: str) -> str:
    if current == incoming:
        return current
    if "ARROW" in {current, incoming}:
        return "ARROW"
    if "TAIL" in {current, incoming}:
        return "TAIL"
    return "CIRCLE"
