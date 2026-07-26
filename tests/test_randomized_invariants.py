import numpy as np
import pytest

from fci_engine import fci, fci_plus
from fci_engine.graph import Endpoint


@pytest.mark.parametrize("learner", [fci, fci_plus])
def test_randomized_runs_preserve_pag_and_diagnostic_invariants(learner) -> None:
    rng = np.random.default_rng(20260726)

    for n_variables in range(1, 7):
        data = rng.normal(size=(45, n_variables))
        if n_variables >= 2:
            data[:, 1] += 0.7 * data[:, 0]

        result = learner(
            data,
            max_cond_set_size=min(2, max(0, n_variables - 2)),
            max_path_length=4,
        )

        assert result.nodes == tuple(f"X{i}" for i in range(n_variables))
        assert result.ci_test_count == len(result.ci_test_trace)
        assert 0 <= result.cache_hits <= result.ci_test_count
        for x, y in result.edges:
            assert x != y
            assert result.graph.get_endpoint(x, y) is not Endpoint.NONE
            assert result.graph.get_endpoint(y, x) is not Endpoint.NONE
        for (x, y), separating_set in result.sepsets.items():
            assert result.sepsets[(y, x)] == separating_set
            assert x not in separating_set
            assert y not in separating_set
