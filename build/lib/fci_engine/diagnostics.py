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
