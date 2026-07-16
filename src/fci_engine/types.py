"""Shared type aliases for graph discovery and public result objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Union

from numpy.typing import NDArray


Array = NDArray[Any]
Node = str
NodePair = tuple[Node, Node]
Triple = tuple[Node, Node, Node]
Sepset = set[Node]
SepsetMap = dict[NodePair, Sepset]
SepsetMapping = Mapping[NodePair, Sepset]
SepsetSourceMap = dict[NodePair, str]
CIIndex = int
TraceNode = Union[CIIndex, Node]
