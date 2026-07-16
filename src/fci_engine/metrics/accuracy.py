"""Accuracy metrics for comparing PAGs with oracle or reference shapes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from fci_engine.diagnostics import OrientationEvent
from fci_engine.graph import Endpoint, PAG


EndpointLike = Union[Endpoint, str]
Shape = Mapping[tuple[str, str], tuple[EndpointLike, EndpointLike]]
MutableShape = dict[tuple[str, str], tuple[EndpointLike, EndpointLike]]
NormalizedShape = dict[tuple[str, str], tuple[str, str]]


@dataclass(frozen=True)
class PAGComparison:
    """Comparison between an expected PAG shape and an actual PAG shape."""

    expected_edges: int
    actual_edges: int
    true_positive_edges: int
    false_positive_edges: int
    false_negative_edges: int
    exact_edge_matches: int
    endpoint_matches: int
    endpoint_total: int

    @property
    def skeleton_precision(self) -> float:
        return _safe_divide(self.true_positive_edges, self.actual_edges)

    @property
    def skeleton_recall(self) -> float:
        return _safe_divide(self.true_positive_edges, self.expected_edges)

    @property
    def skeleton_f1(self) -> float:
        return _f1(self.skeleton_precision, self.skeleton_recall)

    @property
    def exact_edge_precision(self) -> float:
        return _safe_divide(self.exact_edge_matches, self.actual_edges)

    @property
    def exact_edge_recall(self) -> float:
        return _safe_divide(self.exact_edge_matches, self.expected_edges)

    @property
    def exact_edge_f1(self) -> float:
        return _f1(self.exact_edge_precision, self.exact_edge_recall)

    @property
    def endpoint_accuracy(self) -> float:
        return _safe_divide(self.endpoint_matches, self.endpoint_total)

    def summary(self) -> str:
        """Return a compact one-line metric summary."""

        return (
            f"skeleton_f1={self.skeleton_f1:.3f}, "
            f"endpoint_accuracy={self.endpoint_accuracy:.3f}, "
            f"exact_edge_f1={self.exact_edge_f1:.3f}, "
            f"fp={self.false_positive_edges}, fn={self.false_negative_edges}"
        )


@dataclass(frozen=True)
class PAGESemanticComparison:
    """Compatibility-aware comparison for PAG endpoint marks.

    Exact scoring treats ``o->`` and ``-->`` as different. This semantic score
    still reports that difference, but classifies it as a compatible
    over-orientation rather than a definite contradiction.
    """

    expected_edges: int
    actual_edges: int
    true_positive_edges: int
    false_positive_edges: int
    false_negative_edges: int
    exact_edges: int
    compatible_edges: int
    over_oriented_edges: int
    under_oriented_edges: int
    mixed_oriented_edges: int
    contradicted_edges: int
    endpoint_exact: int
    endpoint_compatible: int
    endpoint_over_oriented: int
    endpoint_under_oriented: int
    endpoint_contradicted: int
    endpoint_total: int

    @property
    def semantic_edge_precision(self) -> float:
        return _safe_divide(self.compatible_edges, self.actual_edges)

    @property
    def semantic_edge_recall(self) -> float:
        return _safe_divide(self.compatible_edges, self.expected_edges)

    @property
    def semantic_edge_f1(self) -> float:
        return _f1(self.semantic_edge_precision, self.semantic_edge_recall)

    @property
    def compatible_endpoint_accuracy(self) -> float:
        return _safe_divide(self.endpoint_compatible, self.endpoint_total)

    @property
    def exact_endpoint_accuracy(self) -> float:
        return _safe_divide(self.endpoint_exact, self.endpoint_total)

    def summary(self) -> str:
        """Return a compact one-line semantic metric summary."""

        return (
            f"semantic_f1={self.semantic_edge_f1:.3f}, "
            f"compatible_endpoint_acc={self.compatible_endpoint_accuracy:.3f}, "
            f"exact_endpoint_acc={self.exact_endpoint_accuracy:.3f}, "
            f"over={self.over_oriented_edges}, under={self.under_oriented_edges}, "
            f"conflict={self.contradicted_edges}, "
            f"fp={self.false_positive_edges}, fn={self.false_negative_edges}"
        )


@dataclass(frozen=True)
class PAGDifference:
    """One edge-level difference between expected and actual PAG shapes."""

    edge: tuple[str, str]
    kind: str
    expected: Optional[tuple[str, str]]
    actual: Optional[tuple[str, str]]
    endpoint_status: tuple[str, str] = ("missing", "missing")
    orientation_events: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """Return a compact human-readable difference summary."""

        return (
            f"{self.edge[0]}-{self.edge[1]} {self.kind}: "
            f"expected={self.expected}, actual={self.actual}, "
            f"events={len(self.orientation_events)}"
        )


def shape_from_pag(graph: PAG) -> NormalizedShape:
    """Convert a PAG into a normalized edge-shape dictionary."""

    return {
        (str(x), str(y)): (endpoint_x.name, endpoint_y.name)
        for x, y, endpoint_x, endpoint_y in graph.to_edge_list()
    }


def compare_pag_shapes(expected: Shape, actual: Shape) -> PAGComparison:
    """Compare actual PAG endpoints against an expected oracle/reference shape."""

    expected_shape = _normalize_shape(expected)
    actual_shape = _normalize_shape(actual)
    expected_edges = set(expected_shape)
    actual_edges = set(actual_shape)
    common_edges = expected_edges & actual_edges

    exact_edge_matches = sum(
        expected_shape[edge] == actual_shape[edge] for edge in common_edges
    )
    endpoint_matches = 0
    for edge in common_edges:
        endpoint_matches += int(expected_shape[edge][0] == actual_shape[edge][0])
        endpoint_matches += int(expected_shape[edge][1] == actual_shape[edge][1])

    endpoint_total = 2 * len(expected_edges | actual_edges)
    return PAGComparison(
        expected_edges=len(expected_edges),
        actual_edges=len(actual_edges),
        true_positive_edges=len(common_edges),
        false_positive_edges=len(actual_edges - expected_edges),
        false_negative_edges=len(expected_edges - actual_edges),
        exact_edge_matches=exact_edge_matches,
        endpoint_matches=endpoint_matches,
        endpoint_total=endpoint_total,
    )


def compare_pag_to_shape(graph: PAG, expected: Shape) -> PAGComparison:
    """Compare a learned PAG directly with an expected edge-shape dictionary."""

    return compare_pag_shapes(expected, shape_from_pag(graph))


def compare_pag_shapes_semantic(
    expected: Shape, actual: Shape
) -> PAGESemanticComparison:
    """Compare PAGs while distinguishing over/under orientation from conflicts."""

    expected_shape = _normalize_shape(expected)
    actual_shape = _normalize_shape(actual)
    expected_edges = set(expected_shape)
    actual_edges = set(actual_shape)
    common_edges = expected_edges & actual_edges

    exact_edges = 0
    compatible_edges = 0
    over_edges = 0
    under_edges = 0
    mixed_edges = 0
    contradicted_edges = 0
    endpoint_exact = 0
    endpoint_compatible = 0
    endpoint_over = 0
    endpoint_under = 0
    endpoint_contradicted = 0

    for edge in common_edges:
        expected_endpoints = expected_shape[edge]
        actual_endpoints = actual_shape[edge]
        statuses = (
            _endpoint_semantic_status(
                expected_endpoints[0],
                actual_endpoints[0],
            ),
            _endpoint_semantic_status(
                expected_endpoints[1],
                actual_endpoints[1],
            ),
        )
        endpoint_exact += sum(status == "exact" for status in statuses)
        endpoint_compatible += sum(status != "contradicted" for status in statuses)
        endpoint_over += sum(status == "over_oriented" for status in statuses)
        endpoint_under += sum(status == "under_oriented" for status in statuses)
        endpoint_contradicted += sum(status == "contradicted" for status in statuses)

        if statuses == ("exact", "exact"):
            exact_edges += 1
        if "contradicted" in statuses:
            contradicted_edges += 1
            continue

        compatible_edges += 1
        has_over = "over_oriented" in statuses
        has_under = "under_oriented" in statuses
        if has_over and has_under:
            mixed_edges += 1
        elif has_over:
            over_edges += 1
        elif has_under:
            under_edges += 1

    endpoint_total = 2 * len(expected_edges | actual_edges)
    return PAGESemanticComparison(
        expected_edges=len(expected_edges),
        actual_edges=len(actual_edges),
        true_positive_edges=len(common_edges),
        false_positive_edges=len(actual_edges - expected_edges),
        false_negative_edges=len(expected_edges - actual_edges),
        exact_edges=exact_edges,
        compatible_edges=compatible_edges,
        over_oriented_edges=over_edges,
        under_oriented_edges=under_edges,
        mixed_oriented_edges=mixed_edges,
        contradicted_edges=contradicted_edges,
        endpoint_exact=endpoint_exact,
        endpoint_compatible=endpoint_compatible,
        endpoint_over_oriented=endpoint_over,
        endpoint_under_oriented=endpoint_under,
        endpoint_contradicted=endpoint_contradicted,
        endpoint_total=endpoint_total,
    )


def compare_pag_to_shape_semantic(
    graph: PAG, expected: Shape
) -> PAGESemanticComparison:
    """Compare a learned PAG with semantic endpoint compatibility classes."""

    return compare_pag_shapes_semantic(expected, shape_from_pag(graph))


def explain_pag_differences(
    expected: Shape,
    actual: Shape,
    orientation_trace: Optional[Sequence[OrientationEvent]] = None,
) -> list[PAGDifference]:
    """Return edge-level PAG differences with optional orientation-rule trace."""

    expected_shape = _normalize_shape(expected)
    actual_shape = _normalize_shape(actual)
    differences: list[PAGDifference] = []
    for edge in sorted(set(expected_shape) | set(actual_shape)):
        expected_endpoints = expected_shape.get(edge)
        actual_endpoints = actual_shape.get(edge)
        if expected_endpoints is None:
            differences.append(
                PAGDifference(
                    edge=edge,
                    kind="extra_edge",
                    expected=None,
                    actual=actual_endpoints,
                    orientation_events=_events_for_edge(orientation_trace, edge),
                )
            )
            continue
        if actual_endpoints is None:
            differences.append(
                PAGDifference(
                    edge=edge,
                    kind="missing_edge",
                    expected=expected_endpoints,
                    actual=None,
                    orientation_events=_events_for_edge(orientation_trace, edge),
                )
            )
            continue
        statuses = (
            _endpoint_semantic_status(
                expected_endpoints[0],
                actual_endpoints[0],
            ),
            _endpoint_semantic_status(
                expected_endpoints[1],
                actual_endpoints[1],
            ),
        )
        if statuses == ("exact", "exact"):
            continue
        if "contradicted" in statuses:
            kind = "endpoint_conflict"
        elif "over_oriented" in statuses and "under_oriented" in statuses:
            kind = "mixed_endpoint_difference"
        elif "over_oriented" in statuses:
            kind = "over_oriented"
        else:
            kind = "under_oriented"
        differences.append(
            PAGDifference(
                edge=edge,
                kind=kind,
                expected=expected_endpoints,
                actual=actual_endpoints,
                endpoint_status=statuses,
                orientation_events=_events_for_edge(orientation_trace, edge),
            )
        )
    return differences


def _normalize_shape(shape: Shape) -> NormalizedShape:
    normalized = {}
    for edge, endpoints in shape.items():
        x, y = edge
        endpoint_x, endpoint_y = endpoints
        normalized[(str(x), str(y))] = (
            _normalize_endpoint(endpoint_x),
            _normalize_endpoint(endpoint_y),
        )
    return normalized


def _normalize_endpoint(endpoint: EndpointLike) -> str:
    if isinstance(endpoint, Endpoint):
        return endpoint.name
    return str(endpoint).upper()


def _endpoint_semantic_status(expected: str, actual: str) -> str:
    if expected == actual:
        return "exact"
    if expected == "CIRCLE" and actual in {"ARROW", "TAIL"}:
        return "over_oriented"
    if actual == "CIRCLE" and expected in {"ARROW", "TAIL"}:
        return "under_oriented"
    return "contradicted"


def _events_for_edge(
    orientation_trace: Optional[Sequence[OrientationEvent]],
    edge: tuple[str, str],
) -> list[dict[str, Any]]:
    if orientation_trace is None:
        return []
    edge_key = frozenset(edge)
    events = []
    for event in orientation_trace:
        if frozenset(str(node) for node in event.edge) != edge_key:
            continue
        events.append(
            {
                "rule": event.rule,
                "edge": [str(event.edge[0]), str(event.edge[1])],
                "oriented_endpoint": str(event.oriented_endpoint),
                "before": event.before.name,
                "after": event.after.name,
                "before_edge": event.before_edge,
                "after_edge": event.after_edge,
                "iteration": event.iteration,
                "reason": event.reason,
            }
        )
    return events


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
