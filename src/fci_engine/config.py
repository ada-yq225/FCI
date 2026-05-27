"""Configuration for the public FCI API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from fci_engine.ci import CITest
from fci_engine.knowledge import BackgroundKnowledge


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
        orientation_strategy: Controls tail-producing orientation rules:
            "standard" applies all implemented PAG rules, "conservative" keeps
            arrowhead rules only, and "leaf" applies arrowhead rules plus R1
            when the newly directed endpoint is a leaf in the current PAG.
        background_knowledge: Required and forbidden orientation constraints.
        verbose: Enable detailed logging output.
    """

    alpha: Union[float, str] = "auto"
    ci_test: Optional[CITest] = None
    max_cond_set_size: Optional[int] = None
    max_path_length: Optional[int] = None
    do_pdsep: bool = True
    skeleton_stable: bool = True
    pdsep_stable: bool = True
    sepset_selection: str = "max_pvalue"
    conservative_colliders: bool = False
    conservative_orientation: bool = False
    orientation_strategy: str = "standard"
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
        if self.max_path_length is not None and self.max_path_length < 0:
            raise ValueError("max_path_length must be non-negative.")
        if self.sepset_selection not in {"first", "max_pvalue"}:
            raise ValueError("sepset_selection must be 'first' or 'max_pvalue'.")
        if self.orientation_strategy not in {"standard", "conservative", "leaf"}:
            raise ValueError(
                "orientation_strategy must be 'standard', 'conservative', or 'leaf'."
            )
