"""Configuration objects for the public FCI and FCI+ APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

from typing_extensions import TypedDict, Unpack

from fci_engine.ci import CITest
from fci_engine.knowledge import BackgroundKnowledge


Alpha = Union[float, Literal["auto"]]
SepsetSelection = Literal["first", "max_pvalue"]
OrientationStrategy = Literal["standard", "conservative", "leaf", "robust"]


class FCIOptions(TypedDict, total=False):
    """Keyword options accepted by the public FCI-family estimators."""

    alpha: Alpha
    ci_test: Optional[CITest]
    max_cond_set_size: Optional[int]
    sparsity_bound: Optional[int]
    max_path_length: Optional[int]
    do_pdsep: bool
    skeleton_stable: bool
    pdsep_stable: bool
    sepset_selection: SepsetSelection
    conservative_colliders: bool
    conservative_orientation: bool
    orientation_strategy: OrientationStrategy
    background_knowledge: Optional[BackgroundKnowledge]
    verbose: bool


class FCIPaperOptions(TypedDict, total=False):
    """Overrides that preserve the Spirtes et al. FCI search schedule."""

    alpha: float
    ci_test: Optional[CITest]
    background_knowledge: Optional[BackgroundKnowledge]
    verbose: bool


class FCIPlusPaperOptions(FCIPaperOptions, total=False):
    """Inputs to Claassen et al.'s Algorithm 2 paper profile."""

    k: int


@dataclass(frozen=True)
class FCIConfig:
    """Configuration for standard FCI discovery.

    Attributes:
        alpha: The significance level for the conditional independence tests.
            If "auto", it dynamically scales based on sample size (N):
            N < 1000 uses 0.05
            N < 5000 uses 0.01
            N >= 5000 uses 0.001
        ci_test: Pre-configured CITest instance.
        max_cond_set_size: Maximum size of conditioning set.
        sparsity_bound: Maximum node degree bound ``k`` used by FCI+'s sparse
            hierarchical D-SEP base search. When omitted, FCI+ uses
            ``max_cond_set_size`` for backward compatibility.
        max_path_length: Maximum path length for orientation rules.
        do_pdsep: Whether to refine skeleton using Possible-D-SEP.
        skeleton_stable: Snapshot adjacency sets within each conditioning depth
            to avoid order-dependent skeleton deletion.
        pdsep_stable: Snapshot the PAG at the start of Possible-D-SEP refinement
            to avoid order-dependent candidate search.
        sepset_selection: How to choose among multiple separating sets at the
            same search depth. "max_pvalue" spends more CI tests to keep the
            strongest independence evidence; "first" keeps traditional early
            stopping behavior.
        conservative_colliders: Use conservative unshielded-collider
            orientation by checking multiple separating sets and leaving
            ambiguous triples unoriented.
        conservative_orientation: Skip tail-producing orientation propagation
            rules after collider orientation. This favors keeping circle
            endpoints over aggressive finite-sample orientation.
        orientation_strategy: Controls collider and tail-producing orientation
            rules: "standard" applies all implemented PAG rules,
            "conservative" keeps arrowhead rules only, "leaf" applies
            arrowhead rules plus R1 when the newly directed endpoint is a leaf
            in the current PAG, and "robust" combines conservative collider
            orientation with the leaf-tail rule profile.
        background_knowledge: Required and forbidden orientation constraints.
        verbose: Enable detailed logging output.
    """

    alpha: Alpha = 0.05
    ci_test: Optional[CITest] = None
    max_cond_set_size: Optional[int] = None
    sparsity_bound: Optional[int] = None
    max_path_length: Optional[int] = None
    do_pdsep: bool = True
    skeleton_stable: bool = True
    pdsep_stable: bool = True
    sepset_selection: SepsetSelection = "max_pvalue"
    conservative_colliders: bool = False
    conservative_orientation: bool = False
    orientation_strategy: OrientationStrategy = "standard"
    background_knowledge: Optional[BackgroundKnowledge] = None
    verbose: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.alpha, str):
            if self.alpha != "auto":
                raise ValueError("If string, alpha must be 'auto'.")
        elif not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1.")

        if self.max_cond_set_size is not None and self.max_cond_set_size < 0:
            raise ValueError("max_cond_set_size must be non-negative.")
        if self.sparsity_bound is not None and self.sparsity_bound < 0:
            raise ValueError("sparsity_bound must be non-negative.")
        if self.max_path_length is not None and self.max_path_length < 0:
            raise ValueError("max_path_length must be non-negative.")
        if self.sepset_selection not in {"first", "max_pvalue"}:
            raise ValueError("sepset_selection must be 'first' or 'max_pvalue'.")
        if self.orientation_strategy not in {
            "standard",
            "conservative",
            "leaf",
            "robust",
        }:
            raise ValueError(
                "orientation_strategy must be 'standard', 'conservative', "
                "'leaf', or 'robust'."
            )

    @classmethod
    def practical(cls, **overrides: Unpack[FCIOptions]) -> "FCIConfig":
        """Return the stable finite-sample configuration used by default."""

        values: dict[str, Any] = {
            "alpha": 0.05,
            "skeleton_stable": True,
            "pdsep_stable": True,
            "sepset_selection": "max_pvalue",
            "do_pdsep": True,
            "orientation_strategy": "standard",
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def paper(cls, **overrides: Unpack[FCIPaperOptions]) -> "FCIConfig":
        """Return the unbounded FCI search schedule from Spirtes et al. (2000).

        The skeleton and Possible-D-SEP stages use first-found minimal
        separating sets. The PC stage updates immediately; Possible-D-SEP
        candidates are computed from the initially oriented graph for that
        stage, matching the book's ``F``/``F'`` construction. The final
        orientation phase uses the complete PAG rule schedule implemented by
        this package.
        """

        values: dict[str, Any] = {
            "alpha": 0.05,
            "max_cond_set_size": None,
            "max_path_length": None,
            "do_pdsep": True,
            "skeleton_stable": False,
            "pdsep_stable": True,
            "sepset_selection": "first",
            "conservative_colliders": False,
            "conservative_orientation": False,
            "orientation_strategy": "standard",
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_profile(
        cls,
        profile: str,
        **overrides: Any,
    ) -> "FCIConfig":
        """Build the ``"practical"`` or ``"paper"`` FCI profile."""

        normalized = profile.strip().lower().replace("-", "_")
        if normalized == "practical":
            return cls.practical(**overrides)
        if normalized in {"paper", "spirtes_2000"}:
            return cls.paper(**overrides)
        raise ValueError("Unknown FCI profile. Expected 'practical' or 'paper'.")


@dataclass(frozen=True)
class FCIPlusConfig(FCIConfig):
    """FCI+-specific configuration with practical and paper-aligned profiles.

    FCI+ does not use standard FCI's Possible-D-SEP stage, so ``do_pdsep`` is
    disabled by default. Use :meth:`practical` for a bounded, finite-sample
    oriented configuration or :meth:`paper` for the Algorithm 2 profile.
    """

    do_pdsep: bool = False

    @classmethod
    def practical(
        cls,
        **overrides: Unpack[FCIOptions],
    ) -> "FCIPlusConfig":
        """Return a bounded conservative finite-sample profile."""

        values_override: dict[str, Any] = dict(overrides)
        max_cond_set_size = values_override.pop("max_cond_set_size", 3)
        sparsity_bound = values_override.pop("sparsity_bound", None)
        if sparsity_bound is None:
            sparsity_bound = max_cond_set_size
        values: dict[str, Any] = {
            "alpha": "auto",
            "max_cond_set_size": max_cond_set_size,
            "sparsity_bound": sparsity_bound,
            "sepset_selection": "max_pvalue",
            "skeleton_stable": True,
            "orientation_strategy": "robust",
            "conservative_colliders": True,
            "do_pdsep": False,
        }
        values.update(values_override)
        values["do_pdsep"] = False
        return cls(**values)

    @classmethod
    def paper(
        cls,
        **overrides: Unpack[FCIPlusPaperOptions],
    ) -> "FCIPlusConfig":
        """Return Claassen et al.'s literal Algorithm 2 search profile."""

        values_override: dict[str, Any] = dict(overrides)
        k = values_override.pop("k", 3)
        alpha = values_override.pop("alpha", 0.05)
        if k < 0:
            raise ValueError("k must be non-negative.")
        values: dict[str, Any] = {
            "alpha": alpha,
            "max_cond_set_size": k,
            "sparsity_bound": k,
            "max_path_length": None,
            "skeleton_stable": False,
            "pdsep_stable": False,
            "sepset_selection": "first",
            "orientation_strategy": "standard",
            "conservative_colliders": False,
            "conservative_orientation": False,
            "do_pdsep": False,
        }
        values.update(values_override)
        values["max_cond_set_size"] = k
        values["sparsity_bound"] = k
        values["max_path_length"] = None
        values["skeleton_stable"] = False
        values["pdsep_stable"] = False
        values["sepset_selection"] = "first"
        values["orientation_strategy"] = "standard"
        values["conservative_colliders"] = False
        values["conservative_orientation"] = False
        values["do_pdsep"] = False
        return cls(**values)

    @classmethod
    def from_profile(
        cls,
        profile: str,
        **overrides: Any,
    ) -> "FCIPlusConfig":
        """Build a named FCI+ profile.

        Supported profiles are ``"practical"`` and ``"paper"``.
        """

        normalized = profile.strip().lower().replace("-", "_")
        if normalized == "practical":
            return cls.practical(**overrides)
        if normalized in {"paper", "paper_aligned"}:
            return cls.paper(**overrides)
        raise ValueError("Unknown FCI+ profile. Expected 'practical' or 'paper'.")
