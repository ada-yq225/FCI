"""Conditional independence test interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

from fci_engine.types import Array


@dataclass(frozen=True)
class CITestResult:
    """Result of a conditional independence test."""

    independent: bool
    p_value: float
    statistic: Optional[float]
    method: str
    n_samples: Optional[int]


class CITest(ABC):
    """Abstract base class for conditional independence tests."""

    allow_nan = False

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1.")
        self.alpha = alpha

    @abstractmethod
    def test(
        self,
        data: Array,
        x: int,
        y: int,
        cond_set: Sequence[int],
    ) -> CITestResult:
        """Test whether ``x`` and ``y`` are independent given ``cond_set``."""
