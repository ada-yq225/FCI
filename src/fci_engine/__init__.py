"""Fast Causal Inference tools for PAG learning under latent confounding."""

from fci_engine.api import fci, fci_plus
from fci_engine.discovery.fci import FCI
from fci_engine.discovery.fci_plus import FCIPlus
from fci_engine.graph import Endpoint, PAG
from fci_engine.knowledge import BackgroundKnowledge
from fci_engine.metrics import (
    BenchmarkAggregate,
    BenchmarkResult,
    PAGComparison,
    aggregate_benchmark_results,
    bootstrap_edge_frequencies,
    bootstrap_adjacency_frequencies,
    compare_pag_shapes,
    compare_pag_to_shape,
    format_benchmark_leaderboard,
    format_benchmark_results,
    run_oracle_benchmark,
    shape_from_pag,
    stable_fci,
)
from fci_engine.result import EdgeExplanation, FCIResult
from fci_engine.simulation import OracleCase, default_oracle_cases

__version__ = "0.1.0"

__all__ = [
    "Endpoint",
    "EdgeExplanation",
    "FCI",
    "FCIPlus",
    "FCIResult",
    "BackgroundKnowledge",
    "BenchmarkAggregate",
    "BenchmarkResult",
    "OracleCase",
    "PAGComparison",
    "PAG",
    "aggregate_benchmark_results",
    "bootstrap_adjacency_frequencies",
    "bootstrap_edge_frequencies",
    "compare_pag_shapes",
    "compare_pag_to_shape",
    "default_oracle_cases",
    "fci",
    "fci_plus",
    "format_benchmark_results",
    "format_benchmark_leaderboard",
    "run_oracle_benchmark",
    "shape_from_pag",
    "stable_fci",
]
