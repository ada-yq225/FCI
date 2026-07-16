"""Partial Ancestral Graph representation."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from fci_engine.graph.endpoint import Endpoint


class PAG:
    """A Partial Ancestral Graph over observed node names.

    Endpoints are stored for ordered node pairs. For an edge ``x-y``,
    ``get_endpoint(x, y)`` returns the endpoint located at ``y``.
    """

    def __init__(self, nodes: Sequence[str]) -> None:
        self._nodes = tuple(nodes)
        if len(set(self._nodes)) != len(self._nodes):
            raise ValueError("PAG nodes must be unique.")

        self._node_set = set(self._nodes)
        self._endpoints: dict[str, dict[str, Endpoint]] = {
            x: {y: Endpoint.NONE for y in self._nodes if y != x} for x in self._nodes
        }

    @property
    def nodes(self) -> tuple[str, ...]:
        """Return nodes in insertion order."""

        return self._nodes

    def add_edge(
        self,
        x: str,
        y: str,
        endpoint_x: Endpoint,
        endpoint_y: Endpoint,
    ) -> None:
        """Add or replace an edge between ``x`` and ``y``.

        ``endpoint_x`` is the mark at ``x`` and ``endpoint_y`` is the mark at
        ``y``.
        """

        self._validate_pair(x, y)
        if Endpoint.NONE in (endpoint_x, endpoint_y):
            raise ValueError("Edges must use non-NONE endpoints.")

        self._endpoints[x][y] = endpoint_y
        self._endpoints[y][x] = endpoint_x

    def add_circle_edge(self, x: str, y: str) -> None:
        """Add an unoriented ``x o-o y`` edge."""

        self.add_edge(x, y, Endpoint.CIRCLE, Endpoint.CIRCLE)

    def remove_edge(self, x: str, y: str) -> None:
        """Remove the edge between ``x`` and ``y`` if present."""

        self._validate_pair(x, y)
        self._endpoints[x][y] = Endpoint.NONE
        self._endpoints[y][x] = Endpoint.NONE

    def is_adjacent(self, x: str, y: str) -> bool:
        """Return whether ``x`` and ``y`` are adjacent."""

        self._validate_pair(x, y)
        return (
            self._endpoints[x][y] is not Endpoint.NONE
            and self._endpoints[y][x] is not Endpoint.NONE
        )

    def neighbors(self, x: str) -> list[str]:
        """Return adjacent nodes to ``x`` in graph node order."""

        self._validate_node(x)
        return [y for y in self._nodes if y != x and self.is_adjacent(x, y)]

    def get_endpoint(self, x: str, y: str) -> Endpoint:
        """Return the endpoint at ``y`` on edge ``x-y``."""

        self._validate_pair(x, y)
        return self._endpoints[x][y]

    def set_endpoint(self, x: str, y: str, endpoint: Endpoint) -> None:
        """Set the endpoint at ``y`` on edge ``x-y``."""

        self._validate_pair(x, y)
        if endpoint is Endpoint.NONE:
            raise ValueError("Use remove_edge() to remove PAG edges.")
        if not self.is_adjacent(x, y):
            raise ValueError(f"Cannot orient non-adjacent nodes: {x!r}, {y!r}.")

        self._endpoints[x][y] = endpoint

    def orient_arrowhead(self, x: str, y: str) -> None:
        """Put an arrowhead at ``y`` on edge ``x-y``."""

        self.set_endpoint(x, y, Endpoint.ARROW)

    def orient_tail(self, x: str, y: str) -> None:
        """Put a tail at ``y`` on edge ``x-y``."""

        self.set_endpoint(x, y, Endpoint.TAIL)

    def has_arrowhead(self, x: str, y: str) -> bool:
        """Return whether edge ``x-y`` has an arrowhead at ``y``."""

        return self.get_endpoint(x, y) is Endpoint.ARROW

    def has_tail(self, x: str, y: str) -> bool:
        """Return whether edge ``x-y`` has a tail at ``y``."""

        return self.get_endpoint(x, y) is Endpoint.TAIL

    def has_circle(self, x: str, y: str) -> bool:
        """Return whether edge ``x-y`` has a circle at ``y``."""

        return self.get_endpoint(x, y) is Endpoint.CIRCLE

    def is_directed_edge(self, x: str, y: str) -> bool:
        """Return whether the edge is oriented ``x --> y``."""

        return (
            self.is_adjacent(x, y)
            and self.get_endpoint(y, x) is Endpoint.TAIL
            and self.get_endpoint(x, y) is Endpoint.ARROW
        )

    def is_bidirected_edge(self, x: str, y: str) -> bool:
        """Return whether the edge is oriented ``x <-> y``."""

        return (
            self.is_adjacent(x, y)
            and self.get_endpoint(y, x) is Endpoint.ARROW
            and self.get_endpoint(x, y) is Endpoint.ARROW
        )

    def is_undirected_edge(self, x: str, y: str) -> bool:
        """Return whether the edge is oriented ``x --- y``."""

        return (
            self.is_adjacent(x, y)
            and self.get_endpoint(y, x) is Endpoint.TAIL
            and self.get_endpoint(x, y) is Endpoint.TAIL
        )

    def is_circle_edge(self, x: str, y: str) -> bool:
        """Return whether the edge is unoriented ``x o-o y``."""

        return (
            self.is_adjacent(x, y)
            and self.get_endpoint(y, x) is Endpoint.CIRCLE
            and self.get_endpoint(x, y) is Endpoint.CIRCLE
        )

    def is_possible_ancestor(self, x: str, y: str) -> bool:
        """Return whether ``x`` can still be an ancestor of ``y``.

        This follows possibly directed paths: when traversing an edge from
        ``current`` to ``next``, an arrowhead at ``current`` blocks that
        traversal.
        """

        self._validate_node(x)
        self._validate_node(y)
        if x == y:
            return True

        visited = {x}
        queue: deque[str] = deque([x])
        while queue:
            current = queue.popleft()
            for next_node in self.neighbors(current):
                if next_node in visited:
                    continue
                if self.has_arrowhead(next_node, current):
                    continue
                if next_node == y:
                    return True
                visited.add(next_node)
                queue.append(next_node)
        return False

    def is_definite_ancestor(self, x: str, y: str) -> bool:
        """Return whether ``x`` reaches ``y`` through directed edges only."""

        self._validate_node(x)
        self._validate_node(y)
        if x == y:
            return True

        visited = {x}
        queue: deque[str] = deque([x])
        while queue:
            current = queue.popleft()
            for next_node in self.neighbors(current):
                if next_node in visited:
                    continue
                if not self.is_directed_edge(current, next_node):
                    continue
                if next_node == y:
                    return True
                visited.add(next_node)
                queue.append(next_node)
        return False

    def possible_causes(self, y: str) -> list[str]:
        """Return nodes that may be ancestors of ``y``."""

        self._validate_node(y)
        return [x for x in self._nodes if x != y and self.is_possible_ancestor(x, y)]

    def definite_causes(self, y: str) -> list[str]:
        """Return nodes that are connected to ``y`` by directed paths."""

        self._validate_node(y)
        return [x for x in self._nodes if x != y and self.is_definite_ancestor(x, y)]

    def edges(self) -> list[tuple[str, str]]:
        """Return each unordered adjacent pair once in node order."""

        edge_pairs: list[tuple[str, str]] = []
        for i, x in enumerate(self._nodes):
            for y in self._nodes[i + 1 :]:
                if self.is_adjacent(x, y):
                    edge_pairs.append((x, y))
        return edge_pairs

    def copy(self) -> "PAG":
        """Return an independent copy of the graph."""

        copied = PAG(self._nodes)
        copied._endpoints = {
            x: endpoints.copy() for x, endpoints in self._endpoints.items()
        }
        return copied

    def edge_repr(self, x: str, y: str) -> str:
        """Return a compact PAG representation for edge ``x-y``."""

        if not self.is_adjacent(x, y):
            raise ValueError(f"Nodes are not adjacent: {x!r}, {y!r}.")

        endpoint_x = self.get_endpoint(y, x)
        endpoint_y = self.get_endpoint(x, y)
        return f"{x} {self._left_mark(endpoint_x)}-{self._right_mark(endpoint_y)} {y}"

    def to_edge_list(self) -> list[tuple[str, str, Endpoint, Endpoint]]:
        """Return edge tuples ``(x, y, endpoint_x, endpoint_y)``."""

        return [
            (x, y, self.get_endpoint(y, x), self.get_endpoint(x, y))
            for x, y in self.edges()
        ]

    def _validate_node(self, node: str) -> None:
        if node not in self._node_set:
            raise KeyError(f"Unknown PAG node: {node!r}.")

    def _validate_pair(self, x: str, y: str) -> None:
        self._validate_node(x)
        self._validate_node(y)
        if x == y:
            raise ValueError("PAG edges require two distinct nodes.")

    @staticmethod
    def _left_mark(endpoint: Endpoint) -> str:
        marks = {
            Endpoint.TAIL: "-",
            Endpoint.ARROW: "<",
            Endpoint.CIRCLE: "o",
        }
        return marks[endpoint]

    @staticmethod
    def _right_mark(endpoint: Endpoint) -> str:
        marks = {
            Endpoint.TAIL: "-",
            Endpoint.ARROW: ">",
            Endpoint.CIRCLE: "o",
        }
        return marks[endpoint]
