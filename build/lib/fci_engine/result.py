"""Result object returned by the public FCI API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from fci_engine.config import FCIConfig
from fci_engine.diagnostics import CITraceEvent, OrientationEvent
from fci_engine.graph import PAG


@dataclass(frozen=True)
class FCIResult:
    """Container for a completed FCI run."""

    graph: PAG
    sepsets: dict[tuple[str, str], set[str]]
    ci_test_count: int
    cache_hits: int
    elapsed_time: float
    config: FCIConfig
    orientation_trace: list[OrientationEvent] = field(default_factory=list)
    ci_test_trace: list[CITraceEvent] = field(default_factory=list)
    sepset_sources: dict[tuple[str, str], str] = field(default_factory=dict)
    bootstrap_edge_frequencies: Optional[dict[str, float]] = None

    def summary(self) -> str:
        """Return a compact human-readable summary."""

        return "\n".join(
            [
                "FCIResult",
                f"- nodes: {len(self.graph.nodes)}",
                f"- edges: {len(self.graph.edges())}",
                f"- separating sets: {len(self.sepsets)}",
                f"- CI tests: {self.ci_test_count}",
                f"- cache hits: {self.cache_hits}",
                f"- orientation events: {len(self.orientation_trace)}",
                f"- CI trace events: {len(self.ci_test_trace)}",
                f"- elapsed time: {self.elapsed_time:.4f}s",
                f"- alpha: {self.config.alpha}",
                f"- Possible-D-Sep: {self.config.do_pdsep}",
            ]
        )
