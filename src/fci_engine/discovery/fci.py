"""End-to-end standard FCI discovery pipeline."""

from __future__ import annotations

from time import perf_counter
from typing import Optional

from typing_extensions import Unpack

from fci_engine.config import FCIConfig, FCIOptions
from fci_engine.discovery.base import BaseFCIEstimator
from fci_engine.discovery.pdsep import refine_skeleton_with_pdsep
from fci_engine.result import FCIResult


class FCI(BaseFCIEstimator):
    """User-facing estimator for standard Fast Causal Inference."""

    algorithm = "fci"

    def __init__(
        self,
        config: Optional[FCIConfig] = None,
        **kwargs: Unpack[FCIOptions],
    ) -> None:
        super().__init__(config=config, **kwargs)

    def fit(self, data: object) -> FCIResult:
        """Run standard FCI and return an :class:`FCIResult`."""

        start_time = perf_counter()
        run = self._prepare_run(data)
        graph, sepsets = self._learn_initial_skeleton(run)

        self._orient_colliders(run, graph, sepsets)
        if run.config.do_pdsep:
            graph, sepsets = refine_skeleton_with_pdsep(
                run.data,
                graph,
                sepsets,
                run.ci_test,
                max_cond_set_size=run.config.max_cond_set_size,
                max_path_length=run.config.max_path_length,
                verbose=run.config.verbose,
                sepset_sources=run.sepset_sources,
                stable=run.config.pdsep_stable,
                sepset_selection=run.config.sepset_selection,
                allow_nan=run.allow_nan,
            )
            self._orient_colliders(
                run,
                graph,
                sepsets,
                reset=True,
                clear_trace=True,
            )

        self._apply_orientation_rules(run, graph, sepsets)
        return self._build_result(
            run,
            graph,
            sepsets,
            start_time=start_time,
        )
