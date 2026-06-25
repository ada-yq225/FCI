"""Diagnostic event records emitted by FCI runs."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Optional

from fci_engine.graph import Endpoint


@dataclass(frozen=True)
class CITraceEvent:
    """A single conditional-independence query observed by the cache."""

    query_index: int
    x: Hashable
    y: Hashable
    cond_set: tuple[Hashable, ...]
    independent: bool
    p_value: float
    statistic: Optional[float]
    method: str
    n_samples: Optional[int]
    cache_hit: bool


@dataclass(frozen=True)
class OrientationEvent:
    """A single endpoint orientation made by collider or rule propagation."""

    rule: str
    edge: tuple[Hashable, Hashable]
    oriented_endpoint: Hashable
    before: Endpoint
    after: Endpoint
    before_edge: str
    after_edge: str
    iteration: Optional[int] = None
    reason: str = ""


@dataclass
class DSEPDiagnostics:
    """Summary counters for the FCI+ hierarchical D-SEP stage."""

    candidate_edges_seen: int = 0
    candidate_revisits: int = 0
    hierarchy_queries: int = 0
    hierarchy_cache_hits: int = 0
    duplicate_conditioning_skips: int = 0
    ci_tests: int = 0
    edges_removed: int = 0
    max_conditioning_size: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return JSON-friendly diagnostic counters."""

        return {
            "candidate_edges_seen": self.candidate_edges_seen,
            "candidate_revisits": self.candidate_revisits,
            "hierarchy_queries": self.hierarchy_queries,
            "hierarchy_cache_hits": self.hierarchy_cache_hits,
            "duplicate_conditioning_skips": self.duplicate_conditioning_skips,
            "ci_tests": self.ci_tests,
            "edges_removed": self.edges_removed,
            "max_conditioning_size": self.max_conditioning_size,
        }
