"""Background knowledge constraints for FCI orientation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional

from fci_engine.diagnostics import OrientationEvent
from fci_engine.graph import Endpoint, PAG


DirectedEdge = tuple[str, str]


@dataclass
class BackgroundKnowledge:
    """Required and forbidden directed-edge constraints.

    A required constraint ``X -> Y`` orients an existing ``X-Y`` edge as
    ``X --> Y``. A forbidden constraint ``X -> Y`` orients an existing ``X-Y``
    edge as ``Y --> X``, matching the common FCI background-knowledge behavior.
    """

    required_edges: set[DirectedEdge] = field(default_factory=set)
    forbidden_edges: set[DirectedEdge] = field(default_factory=set)

    def __init__(
        self,
        required_edges: Optional[Iterable[DirectedEdge]] = None,
        forbidden_edges: Optional[Iterable[DirectedEdge]] = None,
    ) -> None:
        self.required_edges = set(required_edges or ())
        self.forbidden_edges = set(forbidden_edges or ())
        self._validate()

    def require(self, source: str, target: str) -> "BackgroundKnowledge":
        """Require ``source -> target`` and return ``self`` for chaining."""

        self.required_edges.add((source, target))
        self._validate()
        return self

    def forbid(self, source: str, target: str) -> "BackgroundKnowledge":
        """Forbid ``source -> target`` and return ``self`` for chaining."""

        self.forbidden_edges.add((source, target))
        self._validate()
        return self

    def is_required(self, source: str, target: str) -> bool:
        """Return whether ``source -> target`` is required."""

        return (source, target) in self.required_edges

    def is_forbidden(self, source: str, target: str) -> bool:
        """Return whether ``source -> target`` is forbidden."""

        return (source, target) in self.forbidden_edges

    def _validate(self) -> None:
        for source, target in self.required_edges | self.forbidden_edges:
            if source == target:
                raise ValueError("Background knowledge edges require distinct nodes.")
        conflicts = self.required_edges & self.forbidden_edges
        if conflicts:
            raise ValueError(
                "Background knowledge cannot both require and forbid the same "
                f"directed edge: {sorted(conflicts)!r}."
            )
        reverse_required = {
            (target, source) for source, target in self.required_edges
        }
        opposite_required = self.required_edges & reverse_required
        if opposite_required:
            raise ValueError(
                "Background knowledge cannot require both directions for an "
                f"edge: {sorted(opposite_required)!r}."
            )


def apply_background_knowledge(
    graph: PAG,
    knowledge: Optional[BackgroundKnowledge],
    trace: Optional[list[OrientationEvent]] = None,
) -> PAG:
    """Apply orientation constraints to all currently adjacent pairs."""

    if knowledge is None:
        return graph

    for x, y in graph.edges():
        if knowledge.is_required(str(x), str(y)):
            _orient_directed_by_knowledge(graph, x, y, trace, "required")
        elif knowledge.is_required(str(y), str(x)):
            _orient_directed_by_knowledge(graph, y, x, trace, "required")
        elif knowledge.is_forbidden(str(x), str(y)):
            _orient_directed_by_knowledge(graph, y, x, trace, "forbidden")
        elif knowledge.is_forbidden(str(y), str(x)):
            _orient_directed_by_knowledge(graph, x, y, trace, "forbidden")
    return graph


def _orient_directed_by_knowledge(
    graph: PAG,
    source: str,
    target: str,
    trace: Optional[list[OrientationEvent]],
    constraint_type: str,
) -> None:
    reason = f"background knowledge {constraint_type}: {source!r} -> {target!r}"
    _set_endpoint_by_knowledge(
        graph,
        target,
        source,
        Endpoint.TAIL,
        trace,
        reason,
    )
    _set_endpoint_by_knowledge(
        graph,
        source,
        target,
        Endpoint.ARROW,
        trace,
        reason,
    )


def _set_endpoint_by_knowledge(
    graph: PAG,
    x: str,
    y: str,
    endpoint: Endpoint,
    trace: Optional[list[OrientationEvent]],
    reason: str,
) -> None:
    current = graph.get_endpoint(x, y)
    if current is endpoint:
        return
    if current is Endpoint.NONE:
        return

    before_edge = graph.edge_repr(x, y)
    graph.set_endpoint(x, y, endpoint)
    if trace is not None:
        trace.append(
            OrientationEvent(
                rule="background_knowledge",
                edge=(x, y),
                oriented_endpoint=y,
                before=current,
                after=endpoint,
                before_edge=before_edge,
                after_edge=graph.edge_repr(x, y),
                reason=reason,
            )
        )
