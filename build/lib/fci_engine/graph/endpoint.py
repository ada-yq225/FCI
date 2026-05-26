"""Endpoint marks used by Partial Ancestral Graph edges."""

from __future__ import annotations

from enum import Enum


class Endpoint(Enum):
    """Endpoint type for one side of a PAG edge."""

    NONE = "none"
    TAIL = "tail"
    ARROW = "arrow"
    CIRCLE = "circle"

    def __str__(self) -> str:
        return self.value
