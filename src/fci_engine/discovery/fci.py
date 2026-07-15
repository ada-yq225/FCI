"""End-to-end standard FCI discovery pipeline."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Optional

from fci_engine.ci import CITestCache, FisherZTest
from fci_engine.config import FCIConfig
from fci_engine.diagnostics import CITraceEvent, OrientationEvent
from fci_engine.discovery.orientation import (
    orient_unshielded_colliders_conservative,
    orient_unshielded_colliders,
    reset_endpoint_marks,
)
from fci_engine.discovery.pdsep import refine_skeleton_with_pdsep
from fci_engine.discovery.rules import apply_orientation_rules
from fci_engine.discovery.skeleton import create_complete_pag, learn_initial_skeleton
from fci_engine.knowledge import apply_background_knowledge
from fci_engine.result import FCIResult
from fci_engine.utils.validation import validate_numeric_data


class FCI:
    """User-facing estimator for standard Fast Causal Inference."""

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
        """Run standard FCI and return an ``FCIResult``."""

        start_time = perf_counter()
        allow_nan = getattr(self.config.ci_test, "allow_nan", False)
        normalized_data, variable_names = validate_numeric_data(
            data,
            allow_nan=allow_nan,
        )
        self.variable_names = variable_names
        n_samples = normalized_data.shape[0]

        # Resolve auto alpha
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
                
        resolved_config = replace(self.config, alpha=resolved_alpha)
        if resolved_config.orientation_strategy == "robust":
            resolved_config = replace(resolved_config, conservative_colliders=True)

        base_ci_test = resolved_config.ci_test
        if base_ci_test is None:
            base_ci_test = FisherZTest(alpha=resolved_alpha)
        else:
            resolved_config = replace(resolved_config, alpha=base_ci_test.alpha)
        allow_nan = getattr(base_ci_test, "allow_nan", False)
        ci_test = CITestCache(base_ci_test)
        self.ci_test_cache_ = ci_test

        orientation_trace: list[OrientationEvent] = []
        ambiguous_triples: list[tuple[str, str, str]] = []
        sepset_sources: dict[tuple[str, str], str] = {}

        graph = create_complete_pag(variable_names)
        graph, sepsets = learn_initial_skeleton(
            normalized_data,
            graph,
            ci_test,
            max_cond_set_size=resolved_config.max_cond_set_size,
            verbose=resolved_config.verbose,
            sepset_sources=sepset_sources,
            stable=resolved_config.skeleton_stable,
            sepset_selection=resolved_config.sepset_selection,
            allow_nan=allow_nan,
        )

        apply_background_knowledge(
            graph,
            resolved_config.background_knowledge,
            trace=orientation_trace,
        )
        if resolved_config.conservative_colliders:
            graph, ambiguous_triples = orient_unshielded_colliders_conservative(
                normalized_data,
                graph,
                sepsets,
                ci_test,
                max_cond_set_size=resolved_config.max_cond_set_size,
                trace=orientation_trace,
                allow_nan=allow_nan,
            )
        else:
            orient_unshielded_colliders(graph, sepsets, trace=orientation_trace)

        if resolved_config.do_pdsep:
            graph, sepsets = refine_skeleton_with_pdsep(
                normalized_data,
                graph,
                sepsets,
                ci_test,
                max_cond_set_size=resolved_config.max_cond_set_size,
                max_path_length=resolved_config.max_path_length,
                verbose=resolved_config.verbose,
                sepset_sources=sepset_sources,
                stable=resolved_config.pdsep_stable,
                sepset_selection=resolved_config.sepset_selection,
                allow_nan=allow_nan,
            )
            reset_endpoint_marks(graph)
            orientation_trace.clear()
            ambiguous_triples = []
            apply_background_knowledge(
                graph,
                resolved_config.background_knowledge,
                trace=orientation_trace,
            )
            if resolved_config.conservative_colliders:
                graph, ambiguous_triples = orient_unshielded_colliders_conservative(
                    normalized_data,
                    graph,
                    sepsets,
                    ci_test,
                    max_cond_set_size=resolved_config.max_cond_set_size,
                    trace=orientation_trace,
                    allow_nan=allow_nan,
                )
            else:
                orient_unshielded_colliders(graph, sepsets, trace=orientation_trace)

        apply_orientation_rules(
            graph,
            sepsets,
            max_path_length=resolved_config.max_path_length,
            verbose=resolved_config.verbose,
            trace=orientation_trace,
            ambiguous_triples=ambiguous_triples,
            conservative_orientation=resolved_config.conservative_orientation,
            orientation_strategy=resolved_config.orientation_strategy,
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
            ambiguous_triples=ambiguous_triples,
            algorithm="fci",
        )
        self.result_ = result
        return result


def _name_ci_trace(
    trace: list[CITraceEvent],
    variable_names: list[str],
) -> list[CITraceEvent]:
    named_trace: list[CITraceEvent] = []
    for event in trace:
        named_trace.append(
            replace(
                event,
                x=_name_index(event.x, variable_names),
                y=_name_index(event.y, variable_names),
                cond_set=tuple(
                    _name_index(node, variable_names) for node in event.cond_set
                ),
            )
        )
    return named_trace


def _name_index(node: object, variable_names: list[str]) -> object:
    if isinstance(node, int) and 0 <= node < len(variable_names):
        return variable_names[node]
    return node
