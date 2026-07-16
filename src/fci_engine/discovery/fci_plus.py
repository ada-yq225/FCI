"""Integrated FCI+ estimator and sparse hierarchical D-SEP pipeline."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any, Optional

from typing_extensions import Unpack

from fci_engine.config import (
    FCIConfig,
    FCIOptions,
    FCIPlusConfig,
    FCIPlusPaperOptions,
)
from fci_engine.diagnostics import DSEPDiagnostics
from fci_engine.discovery.base import BaseFCIEstimator
from fci_engine.discovery.dsep import refine_skeleton_with_fci_plus_dsep
from fci_engine.result import FCIResult


class FCIPlus(BaseFCIEstimator):
    """Estimator for FCI+ with reusable practical and paper profiles."""

    algorithm = "fci_plus"
    config_class = FCIPlusConfig

    def __init__(
        self,
        config: Optional[FCIPlusConfig] = None,
        **kwargs: Unpack[FCIOptions],
    ) -> None:
        super().__init__(config=config, **kwargs)

    @classmethod
    def practical(cls, **overrides: Unpack[FCIOptions]) -> "FCIPlus":
        """Create a bounded conservative finite-sample estimator."""

        return cls(FCIPlusConfig.practical(**overrides))

    @classmethod
    def paper(cls, **overrides: Unpack[FCIPlusPaperOptions]) -> "FCIPlus":
        """Create an estimator using literal Algorithm 2 search settings."""

        return cls(FCIPlusConfig.paper(**overrides))

    @classmethod
    def from_profile(cls, profile: str, **overrides: Any) -> "FCIPlus":
        """Create an estimator from the ``practical`` or ``paper`` profile."""

        return cls(FCIPlusConfig.from_profile(profile, **overrides))

    def _normalize_algorithm_config(self, config: FCIConfig) -> FCIConfig:
        return replace(config, do_pdsep=False)

    def fit(self, data: object) -> FCIResult:
        """Run FCI+ and return an :class:`FCIResult`."""

        start_time = perf_counter()
        run = self._prepare_run(data)
        graph, sepsets = self._learn_initial_skeleton(run)

        sparsity_bound = run.config.sparsity_bound
        if sparsity_bound is None:
            sparsity_bound = run.config.max_cond_set_size

        diagnostics = DSEPDiagnostics()
        graph, sepsets = refine_skeleton_with_fci_plus_dsep(
            run.data,
            graph,
            sepsets,
            run.ci_test,
            max_degree=sparsity_bound,
            verbose=run.config.verbose,
            sepset_sources=run.sepset_sources,
            sepset_selection=run.config.sepset_selection,
            allow_nan=run.allow_nan,
            diagnostics=diagnostics,
        )

        self._orient_colliders(run, graph, sepsets, reset=True)
        self._apply_orientation_rules(run, graph, sepsets)
        return self._build_result(
            run,
            graph,
            sepsets,
            start_time=start_time,
            dsep_diagnostics=diagnostics.to_dict(),
        )
