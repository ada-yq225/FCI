"""Accuracy metrics for comparing PAGs with oracle or reference shapes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from fci_engine.graph import Endpoint, PAG


EndpointLike = Union[Endpoint, str]
Shape = dict[tuple[str, str], tuple[EndpointLike, EndpointLike]]
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


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
