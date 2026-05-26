"""Configuration for the public FCI API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fci_engine.ci import CITest


@dataclass(frozen=True)
class FCIConfig:
    """Configuration for standard FCI discovery."""

    alpha: float = 0.05
    ci_test: Optional[CITest] = None
    max_cond_set_size: Optional[int] = None
    max_path_length: Optional[int] = None
    do_pdsep: bool = True
    verbose: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1.")
        if self.max_cond_set_size is not None and self.max_cond_set_size < 0:
            raise ValueError("max_cond_set_size must be non-negative.")
        if self.max_path_length is not None and self.max_path_length < 0:
            raise ValueError("max_path_length must be non-negative.")
