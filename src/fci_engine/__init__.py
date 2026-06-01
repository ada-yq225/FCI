"""Fast Causal Inference tools for PAG learning under latent confounding."""

from fci_engine.api import fci, fci_plus
from fci_engine.ci import (
    ChiSquareTest,
    FisherZTest,
    GSquareTest,
    KernelCITest,
    MissingValueFisherZTest,
)
from fci_engine.config import FCIConfig
from fci_engine.discovery.fci import FCI
from fci_engine.discovery.fci_plus import FCIPlus
from fci_engine.graph import Endpoint, PAG
from fci_engine.knowledge import BackgroundKnowledge
from fci_engine.metrics import (
    BenchmarkAggregate,
    BenchmarkResult,
    PAGDifference,
    PAGComparison,
    PAGESemanticComparison,
    aggregate_benchmark_results,
    bootstrap_edge_frequencies,
    bootstrap_adjacency_frequencies,
    compare_pag_shapes_semantic,
    compare_pag_shapes,
    compare_pag_to_shape_semantic,
    compare_pag_to_shape,
    explain_pag_differences,
    format_benchmark_leaderboard,
    format_benchmark_results,
    run_oracle_benchmark,
    shape_from_pag,
    stable_fci,
    stable_fci_plus,
)
from fci_engine.reports import render_interactive_report
from fci_engine.result import EdgeExplanation, FCIResult
from fci_engine.simulation import (
    CausalGraphSpec,
    OracleCase,
    default_oracle_cases,
    realistic_oracle_cases,
)

__version__ = "0.1.0"

__all__ = [
    "Endpoint",
    "EdgeExplanation",
    "FCI",
    "FCIConfig",
    "FCIPlus",
    "FCIResult",
    "BackgroundKnowledge",
    "BenchmarkAggregate",
    "BenchmarkResult",
    "CausalGraphSpec",
    "ChiSquareTest",
    "OracleCase",
    "FisherZTest",
    "GSquareTest",
    "KernelCITest",
    "MissingValueFisherZTest",
    "PAGDifference",
    "PAGComparison",
    "PAGESemanticComparison",
    "PAG",
    "aggregate_benchmark_results",
    "bootstrap_adjacency_frequencies",
    "bootstrap_edge_frequencies",
    "compare_pag_shapes_semantic",
    "compare_pag_shapes",
    "compare_pag_to_shape_semantic",
    "compare_pag_to_shape",
    "default_oracle_cases",
    "realistic_oracle_cases",
    "fci",
    "fci_plus",
    "format_benchmark_results",
    "format_benchmark_leaderboard",
    "explain_pag_differences",
    "run_oracle_benchmark",
    "render_interactive_report",
    "shape_from_pag",
    "stable_fci",
    "stable_fci_plus",
]
