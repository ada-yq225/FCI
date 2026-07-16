"""Result object returned by the public FCI API."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from fci_engine.config import FCIConfig
from fci_engine.diagnostics import CITraceEvent, OrientationEvent
from fci_engine.graph import PAG

if TYPE_CHECKING:
    import networkx as nx
    import pandas as pd


@dataclass(frozen=True)
class EdgeExplanation:
    """Audit record explaining the evidence attached to one PAG edge."""

    x: str
    y: str
    edge_exists: bool
    edge_repr: Optional[str]
    endpoint_x: Optional[str]
    endpoint_y: Optional[str]
    sepset: Optional[list[str]]
    sepset_source: Optional[str]
    orientation_events: list[dict[str, Any]]
    ci_tests: list[dict[str, Any]]
    bootstrap_frequency: Optional[float] = None

    def summary(self) -> str:
        """Return a compact human-readable explanation summary."""

        edge_text = self.edge_repr if self.edge_exists else f"{self.x} ... {self.y}"
        sepset_text = self.sepset if self.sepset is not None else "not separated"
        return (
            f"{edge_text}\n"
            f"- edge exists: {self.edge_exists}\n"
            f"- sepset: {sepset_text}\n"
            f"- sepset source: {self.sepset_source or 'none'}\n"
            f"- direct CI tests: {len(self.ci_tests)}\n"
            f"- orientation events: {len(self.orientation_events)}"
        )


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
    ambiguous_triples: list[tuple[str, str, str]] = field(default_factory=list)
    bootstrap_edge_frequencies: Optional[dict[str, float]] = None
    dsep_diagnostics: Optional[dict[str, int]] = None
    algorithm: str = "fci"
    n_samples: Optional[int] = None
    alpha_was_auto: bool = False

    @property
    def nodes(self) -> tuple[str, ...]:
        """Return learned variable names in input-column order."""

        return self.graph.nodes

    @property
    def edges(self) -> list[tuple[str, str]]:
        """Return learned PAG adjacencies."""

        return self.graph.edges()

    def summary(self) -> str:
        """Return a compact human-readable summary."""

        lines = [
            "FCIResult",
            f"- algorithm: {self.algorithm}",
            f"- samples: {self.n_samples if self.n_samples is not None else 'unknown'}",
            f"- nodes: {len(self.graph.nodes)}",
            f"- edges: {len(self.graph.edges())}",
            f"- separating sets: {len(self.sepsets)}",
            f"- CI tests: {self.ci_test_count}",
            f"- cache hits: {self.cache_hits}",
            f"- orientation events: {len(self.orientation_trace)}",
            f"- ambiguous triples: {len(self.ambiguous_triples)}",
        ]
        if self.algorithm == "fci_plus":
            lines.extend(
                [
                    f"- FCI+ sparsity bound: {self.config.sparsity_bound}",
                    f"- D-SEP diagnostics: {_format_dsep_summary(self.dsep_diagnostics)}",
                ]
            )
        else:
            lines.extend(
                [
                    f"- Possible-D-Sep: {self.config.do_pdsep}",
                    f"- stable Possible-D-Sep: {self.config.pdsep_stable}",
                ]
            )
        lines.extend(
            [
                f"- CI trace events: {len(self.ci_test_trace)}",
                f"- elapsed time: {self.elapsed_time:.4f}s",
                f"- alpha: {self.config.alpha}",
                f"- alpha selected by heuristic: {self.alpha_was_auto}",
                f"- stable skeleton: {self.config.skeleton_stable}",
                f"- sepset selection: {self.config.sepset_selection}",
                f"- conservative orientation: {self.config.conservative_orientation}",
                f"- orientation strategy: {self.config.orientation_strategy}",
            ]
        )
        return "\n".join(lines)

    def assumption_notes(self) -> list[str]:
        """Return interpretation and CI-test assumptions for this run."""

        notes = [
            (
                "The output is a PAG equivalence class, not a unique DAG; "
                "retained adjacencies are not automatically direct causal effects."
            )
        ]
        methods = {event.method for event in self.ci_test_trace}
        if "fisher_z" in methods:
            notes.append(
                "Fisher-Z relies on continuous approximately linear-Gaussian "
                "relationships and valid partial-correlation tests."
            )
        if "mv_fisher_z" in methods:
            notes.append(
                "Missing-value Fisher-Z uses query-wise complete cases, so "
                "effective sample size can differ between CI tests."
            )
        if methods & {"chi_square", "g_square"}:
            notes.append(
                "Discrete chi-square/G-square tests require adequate expected "
                "cell counts for every tested contingency table."
            )
        if "kernel_ci" in methods:
            notes.append(
                "Kernel CI uses finite permutation or Gamma approximations and "
                "can be sensitive to kernel and regularization settings."
            )
        if self.algorithm == "fci_plus" and self.config.sparsity_bound is not None:
            notes.append(
                "FCI+ paper guarantees require faithfulness and a true MAG "
                f"maximum degree no larger than k={self.config.sparsity_bound}."
            )
        if self.alpha_was_auto:
            notes.append(
                "alpha='auto' used a sample-size heuristic; it is not a "
                "dataset-specific error-rate guarantee."
            )
        if self.bootstrap_edge_frequencies is not None:
            notes.append(
                "Bootstrap frequencies measure resampling stability and do not "
                "remove systematic conditional-independence-test bias."
            )
        return notes

    def to_edge_records(self) -> list[dict[str, Any]]:
        """Return final PAG edges as JSON/pandas-friendly records."""

        records = []
        for x, y, endpoint_x, endpoint_y in self.graph.to_edge_list():
            edge_text = self.graph.edge_repr(x, y)
            records.append(
                {
                    "x": str(x),
                    "y": str(y),
                    "endpoint_x": endpoint_x.name,
                    "endpoint_y": endpoint_y.name,
                    "edge": edge_text,
                    "bootstrap_frequency": _lookup_frequency(
                        self.bootstrap_edge_frequencies,
                        edge_text,
                    ),
                }
            )
        return records

    def to_pandas_edges(self) -> "pd.DataFrame":
        """Return final PAG edges as a ``pandas.DataFrame``."""

        import pandas as pd

        return pd.DataFrame(self.to_edge_records())

    def to_networkx(self) -> "nx.Graph[str]":
        """Return a ``networkx.Graph`` with PAG endpoint marks as edge attributes."""

        import networkx as nx

        graph: nx.Graph[str] = nx.Graph()
        graph.add_nodes_from(self.graph.nodes)
        for record in self.to_edge_records():
            graph.add_edge(
                record["x"],
                record["y"],
                endpoint_x=record["endpoint_x"],
                endpoint_y=record["endpoint_y"],
                edge=record["edge"],
                bootstrap_frequency=record["bootstrap_frequency"],
            )
        return graph

    def explain_edge(self, x: str, y: str) -> EdgeExplanation:
        """Return CI and orientation evidence associated with one node pair."""

        if x not in self.graph.nodes or y not in self.graph.nodes:
            raise KeyError(f"Unknown node pair: {x!r}, {y!r}.")
        if x == y:
            raise ValueError("Edge explanations require two distinct nodes.")

        edge_exists = self.graph.is_adjacent(x, y)
        if edge_exists:
            edge_repr = self.graph.edge_repr(x, y)
            endpoint_x = self.graph.get_endpoint(y, x).name
            endpoint_y = self.graph.get_endpoint(x, y).name
        else:
            edge_repr = None
            endpoint_x = None
            endpoint_y = None

        sepset = _lookup_sepset(self.sepsets, x, y)
        sepset_source = self.sepset_sources.get((x, y), self.sepset_sources.get((y, x)))
        orientation_events = [
            _orientation_event_to_dict(event)
            for event in self.orientation_trace
            if _same_unordered_pair(event.edge[0], event.edge[1], x, y)
        ]
        ci_tests = [
            _ci_event_to_dict(event)
            for event in self.ci_test_trace
            if _same_unordered_pair(event.x, event.y, x, y)
        ]

        return EdgeExplanation(
            x=str(x),
            y=str(y),
            edge_exists=edge_exists,
            edge_repr=edge_repr,
            endpoint_x=endpoint_x,
            endpoint_y=endpoint_y,
            sepset=sorted(str(node) for node in sepset) if sepset is not None else None,
            sepset_source=sepset_source,
            orientation_events=orientation_events,
            ci_tests=ci_tests,
            bootstrap_frequency=_lookup_frequency(
                self.bootstrap_edge_frequencies,
                edge_repr,
            ),
        )

    def to_dict(self, include_traces: bool = True) -> dict[str, Any]:
        """Return a JSON-serializable result dictionary."""

        payload: dict[str, Any] = {
            "algorithm": self.algorithm,
            "n_samples": self.n_samples,
            "nodes": list(self.graph.nodes),
            "edges": self.to_edge_records(),
            "sepsets": _sepset_records(self.sepsets, self.sepset_sources),
            "ambiguous_triples": [
                [str(x), str(z), str(y)] for x, z, y in self.ambiguous_triples
            ],
            "ci_test_count": self.ci_test_count,
            "cache_hits": self.cache_hits,
            "elapsed_time": self.elapsed_time,
            "config": _config_to_dict(self.config),
            "alpha_was_auto": self.alpha_was_auto,
            "assumption_notes": self.assumption_notes(),
            "bootstrap_edge_frequencies": self.bootstrap_edge_frequencies,
            "dsep_diagnostics": self.dsep_diagnostics,
        }
        if include_traces:
            payload["orientation_trace"] = [
                _orientation_event_to_dict(event) for event in self.orientation_trace
            ]
            payload["ci_test_trace"] = [
                _ci_event_to_dict(event) for event in self.ci_test_trace
            ]
        return payload

    def to_json(self, include_traces: bool = True, indent: int = 2) -> str:
        """Return a JSON string for this result."""

        return json.dumps(
            self.to_dict(include_traces=include_traces),
            indent=indent,
            sort_keys=True,
        )

    def save_json(
        self,
        path: Union[str, Path],
        include_traces: bool = True,
        indent: int = 2,
    ) -> None:
        """Write this result as JSON."""

        Path(path).write_text(
            self.to_json(include_traces=include_traces, indent=indent),
            encoding="utf-8",
        )

    def to_interactive_report(
        self,
        title: str = "FCI Interactive PAG Report",
    ) -> str:
        """Return a standalone interactive HTML report for this result."""

        from fci_engine.reports import render_interactive_report

        return render_interactive_report(self, title=title)

    def save_interactive_report(
        self,
        path: Union[str, Path],
        title: str = "FCI Interactive PAG Report",
    ) -> None:
        """Write a standalone interactive HTML report for this result."""

        Path(path).write_text(
            self.to_interactive_report(title=title),
            encoding="utf-8",
        )

    def save_artifacts(
        self,
        directory: Union[str, Path],
        *,
        stem: str = "fci_result",
        include_traces: bool = False,
        report_title: str = "FCI Interactive PAG Report",
    ) -> dict[str, Path]:
        """Save the common result artifacts for an applied analysis.

        The output bundle contains a JSON audit record, a CSV edge table, and
        a standalone interactive HTML report. Returned paths are absolute.
        """

        output_directory = Path(directory).expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "json": output_directory / f"{stem}.json",
            "edges_csv": output_directory / f"{stem}_edges.csv",
            "report_html": output_directory / f"{stem}_report.html",
        }
        self.save_json(paths["json"], include_traces=include_traces)
        self.to_pandas_edges().to_csv(paths["edges_csv"], index=False)
        self.save_interactive_report(
            paths["report_html"],
            title=report_title,
        )
        return paths


def _lookup_sepset(
    sepsets: dict[tuple[str, str], set[str]],
    x: str,
    y: str,
) -> Optional[set[str]]:
    return sepsets.get((x, y), sepsets.get((y, x)))


def _same_unordered_pair(a: object, b: object, x: object, y: object) -> bool:
    return frozenset((a, b)) == frozenset((x, y))


def _lookup_frequency(
    frequencies: Optional[dict[str, float]],
    edge_repr: Optional[str],
) -> Optional[float]:
    if frequencies is None or edge_repr is None:
        return None
    return frequencies.get(edge_repr)


def _format_dsep_summary(diagnostics: Optional[dict[str, int]]) -> str:
    if diagnostics is None:
        return "none"
    return (
        f"candidates={diagnostics.get('candidate_edges_seen', 0)}, "
        f"removed={diagnostics.get('edges_removed', 0)}, "
        f"ci={diagnostics.get('ci_tests', 0)}"
    )


def _orientation_event_to_dict(event: OrientationEvent) -> dict[str, Any]:
    return {
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


def _ci_event_to_dict(event: CITraceEvent) -> dict[str, Any]:
    return {
        "query_index": event.query_index,
        "x": str(event.x),
        "y": str(event.y),
        "cond_set": [str(node) for node in event.cond_set],
        "independent": event.independent,
        "p_value": event.p_value,
        "statistic": event.statistic,
        "method": event.method,
        "n_samples": event.n_samples,
        "cache_hit": event.cache_hit,
    }


def _config_to_dict(config: FCIConfig) -> dict[str, Any]:
    return {
        "alpha": config.alpha,
        "ci_test": (
            type(config.ci_test).__name__ if config.ci_test is not None else None
        ),
        "max_cond_set_size": config.max_cond_set_size,
        "sparsity_bound": config.sparsity_bound,
        "max_path_length": config.max_path_length,
        "do_pdsep": config.do_pdsep,
        "skeleton_stable": config.skeleton_stable,
        "pdsep_stable": config.pdsep_stable,
        "sepset_selection": config.sepset_selection,
        "conservative_colliders": config.conservative_colliders,
        "conservative_orientation": config.conservative_orientation,
        "orientation_strategy": config.orientation_strategy,
        "background_knowledge": (
            repr(config.background_knowledge)
            if config.background_knowledge is not None
            else None
        ),
        "verbose": config.verbose,
    }


def _sepset_records(
    sepsets: dict[tuple[str, str], set[str]],
    sources: dict[tuple[str, str], str],
) -> list[dict[str, Any]]:
    records = []
    seen: set[frozenset[str]] = set()
    for (x, y), sepset in sepsets.items():
        key = frozenset((x, y))
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "x": str(x),
                "y": str(y),
                "conditioning_set": sorted(str(node) for node in sepset),
                "source": sources.get((x, y), sources.get((y, x))),
            }
        )
    return records
