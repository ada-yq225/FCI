"""Public function API for FCI discovery."""

from __future__ import annotations

from typing import Any, Literal, Optional, overload

from typing_extensions import Unpack

from fci_engine.config import (
    FCIConfig,
    FCIOptions,
    FCIPaperOptions,
    FCIPlusConfig,
    FCIPlusPaperOptions,
)
from fci_engine.discovery.fci import FCI
from fci_engine.discovery.fci_plus import FCIPlus
from fci_engine.result import FCIResult


@overload
def fci(
    data: object,
    config: Optional[FCIConfig] = None,
    *,
    profile: None = None,
    **kwargs: Unpack[FCIOptions],
) -> FCIResult: ...


@overload
def fci(
    data: object,
    config: None = None,
    *,
    profile: Literal["paper", "spirtes_2000"],
    **kwargs: Unpack[FCIPaperOptions],
) -> FCIResult: ...


@overload
def fci(
    data: object,
    config: None = None,
    *,
    profile: Literal["practical"],
    **kwargs: Unpack[FCIOptions],
) -> FCIResult: ...


def fci(
    data: object,
    config: Optional[FCIConfig] = None,
    *,
    profile: Optional[str] = None,
    **kwargs: Any,
) -> FCIResult:
    """Run standard FCI with explicit options or a named profile."""

    if profile is not None:
        if config is not None:
            raise ValueError("Pass either config or profile, not both.")
        config = FCIConfig.from_profile(profile, **kwargs)
        kwargs = {}

    return FCI(config=config, **kwargs).fit(data)


@overload
def fci_plus(
    data: object,
    config: Optional[FCIPlusConfig] = None,
    *,
    profile: None = None,
    **kwargs: Unpack[FCIOptions],
) -> FCIResult: ...


@overload
def fci_plus(
    data: object,
    config: None = None,
    *,
    profile: Literal["paper", "paper_aligned"],
    **kwargs: Unpack[FCIPlusPaperOptions],
) -> FCIResult: ...


@overload
def fci_plus(
    data: object,
    config: None = None,
    *,
    profile: Literal["practical"],
    **kwargs: Unpack[FCIOptions],
) -> FCIResult: ...


def fci_plus(
    data: object,
    config: Optional[FCIPlusConfig] = None,
    *,
    profile: Optional[str] = None,
    **kwargs: Any,
) -> FCIResult:
    """Run FCI+ with explicit options or a named integrated profile.

    Args:
        data: A numeric pandas DataFrame or two-dimensional NumPy array.
        config: An explicit :class:`FCIPlusConfig`.
        profile: Optional ``"practical"`` or ``"paper"`` profile.
        **kwargs: Configuration overrides.
    """

    if profile is not None:
        if config is not None:
            raise ValueError("Pass either config or profile, not both.")
        config = FCIPlusConfig.from_profile(profile, **kwargs)
        kwargs = {}

    return FCIPlus(config=config, **kwargs).fit(data)
