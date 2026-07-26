import numpy as np

from fci_engine.ci import CITest, CITestCache, CITestResult


class CountingCITest(CITest):
    def __init__(self) -> None:
        super().__init__(alpha=0.05)
        self.n_underlying_calls = 0

    def test(
        self,
        data: np.ndarray,
        x: int,
        y: int,
        cond_set: tuple[int, ...],
    ) -> CITestResult:
        self.n_underlying_calls += 1
        return CITestResult(
            independent=True,
            p_value=0.8,
            statistic=0.1,
            method="counting",
            n_samples=data.shape[0],
        )


def test_repeated_ci_queries_hit_cache() -> None:
    data = np.ones((10, 3))
    data.setflags(write=False)
    underlying = CountingCITest()
    cached = CITestCache(underlying)

    first = cached.test(data, 0, 1, [2])
    second = cached.test(data, 0, 1, [2])

    assert first is second
    assert underlying.n_underlying_calls == 1
    assert cached.n_tests_total == 2
    assert cached.n_cache_hits == 1


def test_reversed_x_y_queries_hit_same_cache_entry() -> None:
    data = np.ones((10, 3))
    data.setflags(write=False)
    underlying = CountingCITest()
    cached = CITestCache(underlying)

    first = cached.test(data, 0, 1, [2])
    second = cached.test(data, 1, 0, [2])

    assert first is second
    assert underlying.n_underlying_calls == 1
    assert cached.n_tests_total == 2
    assert cached.n_cache_hits == 1


def test_conditioning_set_order_does_not_affect_cache_key() -> None:
    data = np.ones((10, 4))
    data.setflags(write=False)
    underlying = CountingCITest()
    cached = CITestCache(underlying)

    first = cached.test(data, 0, 1, [2, 3])
    second = cached.test(data, 0, 1, [3, 2])

    assert first is second
    assert underlying.n_underlying_calls == 1
    assert cached.n_cache_hits == 1


def test_cache_clear_resets_entries_and_counters() -> None:
    data = np.ones((10, 3))
    data.setflags(write=False)
    underlying = CountingCITest()
    cached = CITestCache(underlying)
    cached.test(data, 0, 1, [2])
    cached.test(data, 0, 1, [2])

    cached.clear()
    cached.test(data, 0, 1, [2])

    assert underlying.n_underlying_calls == 2
    assert cached.n_tests_total == 1
    assert cached.n_cache_hits == 0


def test_writable_data_is_not_cached_because_it_can_change_in_place() -> None:
    data = np.ones((10, 3))
    underlying = CountingCITest()
    cached = CITestCache(underlying)

    cached.test(data, 0, 1, [2])
    data[:, 0] = 2.0
    cached.test(data, 0, 1, [2])

    assert underlying.n_underlying_calls == 2
    assert cached.n_cache_hits == 0


def test_switching_read_only_datasets_clears_query_entries() -> None:
    first_data = np.ones((10, 3))
    second_data = np.zeros((10, 3))
    first_data.setflags(write=False)
    second_data.setflags(write=False)
    underlying = CountingCITest()
    cached = CITestCache(underlying)

    cached.test(first_data, 0, 1, [2])
    cached.test(second_data, 0, 1, [2])

    assert underlying.n_underlying_calls == 2
    assert cached.n_cache_hits == 0


def test_making_cached_data_writable_invalidates_old_entries() -> None:
    data = np.ones((10, 3))
    data.setflags(write=False)
    underlying = CountingCITest()
    cached = CITestCache(underlying)

    cached.test(data, 0, 1, [2])
    data.setflags(write=True)
    data[:, 0] = 2.0
    cached.test(data, 0, 1, [2])
    data.setflags(write=False)
    cached.test(data, 0, 1, [2])

    assert underlying.n_underlying_calls == 3
    assert cached.n_cache_hits == 0
