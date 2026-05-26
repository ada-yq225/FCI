"""FCI+ discovery pipeline."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Optional

from fci_engine.ci import CITestCache, FisherZTest
from fci_engine.config import FCIConfig
from fci_engine.diagnostics import OrientationEvent
from fci_engine.discovery.dsep import refine_skeleton_with_fci_plus_dsep
from fci_engine.discovery.fci import _name_ci_trace
from fci_engine.discovery.orientation import (
    orient_unshielded_colliders,
    reset_endpoint_marks,
)
from fci_engine.discovery.rules import apply_orientation_rules
from fci_engine.discovery.skeleton import create_complete_pag, learn_initial_skeleton
from fci_engine.knowledge import apply_background_knowledge
from fci_engine.result import FCIResult
from fci_engine.utils.validation import validate_numeric_data


class FCIPlus:
    """Estimator for the FCI+ sparse hierarchical D-SEP variant."""

    def __init__(self, config: Optional[FCIConfig] = None, **kwargs: object) -> None:
        if config is None:
            self.config = FCIConfig(**kwargs)
        elif kwargs:
            self.config = replace(config, **kwargs)
        else:
            self.config = config

        self.variable_names: list[str] = []
        self.result_: Optional[FCIResult] = None
        self.ci_test_cache_: Optional[CITestCache] = None

    def fit(self, data: object) -> FCIResult:
        """Run FCI+ and return an ``FCIResult``."""

        start_time = perf_counter()
        normalized_data, variable_names = validate_numeric_data(data)
        self.variable_names = variable_names
        n_samples = normalized_data.shape[0]

        resolved_alpha = self.config.alpha
        if resolved_alpha == "auto":
            if n_samples < 1000:
                resolved_alpha = 0.05
            elif n_samples < 5000:
                resolved_alpha = 0.01
            else:
                resolved_alpha = 0.001
            if self.config.verbose:
                print(f"Auto-selected alpha={resolved_alpha} for n_samples={n_samples}")

        resolved_config = replace(self.config, alpha=resolved_alpha, do_pdsep=False)
        base_ci_test = resolved_config.ci_test
        if base_ci_test is None:
            base_ci_test = FisherZTest(alpha=resolved_alpha)
        ci_test = CITestCache(base_ci_test)
        self.ci_test_cache_ = ci_test

        orientation_trace: list[OrientationEvent] = []
        sepset_sources: dict[tuple[str, str], str] = {}

        graph = create_complete_pag(variable_names)
        graph, sepsets = learn_initial_skeleton(
            normalized_data,
            graph,
            ci_test,
            max_cond_set_size=resolved_config.max_cond_set_size,
            verbose=resolved_config.verbose,
            sepset_sources=sepset_sources,
        )

        graph, sepsets = refine_skeleton_with_fci_plus_dsep(
            normalized_data,
            graph,
            sepsets,
            ci_test,
            max_degree=resolved_config.max_cond_set_size,
            verbose=resolved_config.verbose,
            sepset_sources=sepset_sources,
        )

        reset_endpoint_marks(graph)
        apply_background_knowledge(
            graph,
            resolved_config.background_knowledge,
            trace=orientation_trace,
        )
        orient_unshielded_colliders(graph, sepsets, trace=orientation_trace)
        apply_orientation_rules(
            graph,
            sepsets,
            max_path_length=resolved_config.max_path_length,
            verbose=resolved_config.verbose,
            trace=orientation_trace,
        )

        result = FCIResult(
            graph=graph,
            sepsets=sepsets,
            ci_test_count=ci_test.n_tests_total,
            cache_hits=ci_test.n_cache_hits,
            elapsed_time=perf_counter() - start_time,
            config=resolved_config,
            orientation_trace=orientation_trace,
            ci_test_trace=_name_ci_trace(ci_test.trace, variable_names),
            sepset_sources=sepset_sources,
            algorithm="fci_plus",
        )
        self.result_ = result
        return result
