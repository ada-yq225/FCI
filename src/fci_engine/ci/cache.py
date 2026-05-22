"""Caching wrapper for conditional independence tests."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np

from fci_engine.ci.base import CITest, CITestResult


CacheKey = tuple[frozenset[Hashable], frozenset[Hashable]]


class CITestCache(CITest):
    """Cache conditional independence test results by query."""

    def __init__(self, ci_test: CITest) -> None:
        self.ci_test = ci_test
        self.alpha = ci_test.alpha
        self._cache: dict[CacheKey, CITestResult] = {}
        self.n_tests_total = 0
        self.n_cache_hits = 0

    def test(
        self,
        data: np.ndarray,
        x: Hashable,
        y: Hashable,
        cond_set: Sequence[Hashable] = (),
    ) -> CITestResult:
        """Run or retrieve the cached result for a CI query."""

        self.n_tests_total += 1
        key = self._make_key(x, y, cond_set)
        if key in self._cache:
            self.n_cache_hits += 1
            return self._cache[key]

        result = self.ci_test.test(data, x, y, tuple(cond_set))
        self._cache[key] = result
        return result

    def clear(self) -> None:
        """Clear cached results and reset cache counters."""

        self._cache.clear()
        self.n_tests_total = 0
        self.n_cache_hits = 0

    @staticmethod
    def _make_key(
        x: Hashable,
        y: Hashable,
        cond_set: Sequence[Hashable],
    ) -> CacheKey:
        try:
            pair = frozenset((x, y))
            conditioning = frozenset(cond_set)
        except TypeError as exc:
            raise TypeError("CI cache keys must be hashable.") from exc
        return pair, conditioning
