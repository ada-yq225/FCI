"""Shared estimator lifecycle for FCI-family discovery algorithms."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Optional, Type

from fci_engine.ci import CITestCache, FisherZTest
from fci_engine.config import FCIConfig
from fci_engine.diagnostics import CITraceEvent, OrientationEvent
from fci_engine.discovery.orientation import (
    orient_unshielded_colliders,
    orient_unshielded_colliders_conservative,
    reset_endpoint_marks,
)
from fci_engine.discovery.rules import apply_orientation_rules
from fci_engine.discovery.skeleton import create_complete_pag, learn_initial_skeleton
from fci_engine.graph import PAG
from fci_engine.knowledge import apply_background_knowledge
from fci_engine.result import FCIResult
from fci_engine.types import Array
from fci_engine.utils.validation import validate_numeric_data


@dataclass
class DiscoveryRun:
    """Mutable state shared by one FCI-family estimator run."""

    data: Array
    variable_names: list[str]
    config: FCIConfig
    alpha_was_auto: bool
    ci_test: CITestCache
    allow_nan: bool
    orientation_trace: list[OrientationEvent]
    ambiguous_triples: list[tuple[str, str, str]]
    sepset_sources: dict[tuple[str, str], str]


class BaseFCIEstimator:
    """Common data preparation, orientation, and result assembly lifecycle."""

    algorithm = "fci"
    config_class: Type[FCIConfig] = FCIConfig

    def __init__(self, config: Optional[FCIConfig] = None, **kwargs: Any) -> None:
        if config is None:
            self.config = self.config_class(**kwargs)
        elif kwargs:
            self.config = replace(config, **kwargs)
        else:
            self.config = config

        self.variable_names: list[str] = []
        self.result_: Optional[FCIResult] = None
        self.ci_test_cache_: Optional[CITestCache] = None

    @property
    def graph_(self) -> PAG:
        """Return the learned PAG after :meth:`fit` has completed."""

        return self.get_result().graph

    def get_result(self) -> FCIResult:
        """Return the fitted result or raise a clear pre-fit error."""

        if self.result_ is None:
            raise RuntimeError("Estimator is not fitted. Call fit(data) first.")
        return self.result_

    def fit_predict(self, data: object) -> PAG:
        """Fit the estimator and return the learned PAG directly."""

        return self.fit(data).graph

    def fit(self, data: object) -> FCIResult:
        """Run discovery and return an :class:`FCIResult`."""

        raise NotImplementedError

    def _prepare_run(self, data: object) -> DiscoveryRun:
        initial_allow_nan = getattr(self.config.ci_test, "allow_nan", False)
        normalized_data, variable_names = validate_numeric_data(
            data,
            allow_nan=initial_allow_nan,
        )
        self.variable_names = variable_names

        resolved_config = self._resolve_config(
            self.config,
            n_samples=normalized_data.shape[0],
        )
        base_ci_test = resolved_config.ci_test
        if base_ci_test is None:
            base_ci_test = FisherZTest(alpha=float(resolved_config.alpha))
        else:
            resolved_config = replace(resolved_config, alpha=base_ci_test.alpha)

        ci_test = CITestCache(base_ci_test)
        self.ci_test_cache_ = ci_test
        return DiscoveryRun(
            data=normalized_data,
            variable_names=variable_names,
            config=resolved_config,
            alpha_was_auto=self.config.alpha == "auto",
            ci_test=ci_test,
            allow_nan=getattr(base_ci_test, "allow_nan", False),
            orientation_trace=[],
            ambiguous_triples=[],
            sepset_sources={},
        )

    def _resolve_config(self, config: FCIConfig, *, n_samples: int) -> FCIConfig:
        resolved_alpha = config.alpha
        if resolved_alpha == "auto":
            resolved_alpha = _auto_alpha(n_samples)
            if config.verbose:
                print(f"Auto-selected alpha={resolved_alpha} for n_samples={n_samples}")

        resolved = replace(config, alpha=resolved_alpha)
        if resolved.orientation_strategy == "robust":
            resolved = replace(resolved, conservative_colliders=True)
        return self._normalize_algorithm_config(resolved)

    def _normalize_algorithm_config(self, config: FCIConfig) -> FCIConfig:
        return config

    def _learn_initial_skeleton(
        self,
        run: DiscoveryRun,
    ) -> tuple[PAG, dict[tuple[str, str], set[str]]]:
        graph = create_complete_pag(run.variable_names)
        return learn_initial_skeleton(
            run.data,
            graph,
            run.ci_test,
            max_cond_set_size=run.config.max_cond_set_size,
            verbose=run.config.verbose,
            sepset_sources=run.sepset_sources,
            stable=run.config.skeleton_stable,
            sepset_selection=run.config.sepset_selection,
            allow_nan=run.allow_nan,
        )

    def _orient_colliders(
        self,
        run: DiscoveryRun,
        graph: PAG,
        sepsets: dict[tuple[str, str], set[str]],
        *,
        reset: bool = False,
        clear_trace: bool = False,
    ) -> None:
        if reset:
            reset_endpoint_marks(graph)
        if clear_trace:
            run.orientation_trace.clear()

        apply_background_knowledge(
            graph,
            run.config.background_knowledge,
            trace=run.orientation_trace,
        )
        if run.config.conservative_colliders:
            _, run.ambiguous_triples = orient_unshielded_colliders_conservative(
                run.data,
                graph,
                sepsets,
                run.ci_test,
                max_cond_set_size=run.config.max_cond_set_size,
                trace=run.orientation_trace,
                allow_nan=run.allow_nan,
            )
        else:
            orient_unshielded_colliders(
                graph,
                sepsets,
                trace=run.orientation_trace,
            )
            run.ambiguous_triples = []

    def _apply_orientation_rules(
        self,
        run: DiscoveryRun,
        graph: PAG,
        sepsets: dict[tuple[str, str], set[str]],
    ) -> None:
        apply_orientation_rules(
            graph,
            sepsets,
            max_path_length=run.config.max_path_length,
            verbose=run.config.verbose,
            trace=run.orientation_trace,
            ambiguous_triples=run.ambiguous_triples,
            conservative_orientation=run.config.conservative_orientation,
            orientation_strategy=run.config.orientation_strategy,
        )

    def _build_result(
        self,
        run: DiscoveryRun,
        graph: PAG,
        sepsets: dict[tuple[str, str], set[str]],
        *,
        start_time: float,
        dsep_diagnostics: Optional[dict[str, int]] = None,
    ) -> FCIResult:
        result = FCIResult(
            graph=graph,
            sepsets=sepsets,
            ci_test_count=run.ci_test.n_tests_total,
            cache_hits=run.ci_test.n_cache_hits,
            elapsed_time=perf_counter() - start_time,
            config=run.config,
            orientation_trace=run.orientation_trace,
            ci_test_trace=_name_ci_trace(
                run.ci_test.trace,
                run.variable_names,
            ),
            sepset_sources=run.sepset_sources,
            ambiguous_triples=run.ambiguous_triples,
            dsep_diagnostics=dsep_diagnostics,
            algorithm=self.algorithm,
            n_samples=run.data.shape[0],
            alpha_was_auto=run.alpha_was_auto,
        )
        self.result_ = result
        return result


def _auto_alpha(n_samples: int) -> float:
    if n_samples < 1000:
        return 0.05
    if n_samples < 5000:
        return 0.01
    return 0.001


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
