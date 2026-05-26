"""Public function API for FCI discovery."""

from __future__ import annotations

from fci_engine.discovery.fci import FCI
from fci_engine.discovery.fci_plus import FCIPlus
from fci_engine.result import FCIResult


def fci(data: object, **kwargs: object) -> FCIResult:
    """Run standard FCI on ``data`` using the provided configuration options."""

    return FCI(**kwargs).fit(data)


def fci_plus(data: object, **kwargs: object) -> FCIResult:
    """Run FCI+ on ``data`` using the provided configuration options."""

    return FCIPlus(**kwargs).fit(data)
