"""Missing-value Fisher-Z conditional independence test."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fci_engine.ci.base import CITestResult
from fci_engine.ci.fisher_z import FisherZTest


class MissingValueFisherZTest(FisherZTest):
    """Fisher-Z test with query-wise complete-case deletion.

    For each CI query, only rows finite in ``x``, ``y``, and ``cond_set`` are
    retained. This keeps missingness local to the variables involved in the
    query instead of dropping rows globally.
    """

    method = "mv_fisher_z"
    allow_nan = True

    def test(
        self,
        data: object,
        x: int,
        y: int,
        cond_set: Sequence[int] = (),
    ) -> CITestResult:
        if not isinstance(data, np.ndarray):
            raise TypeError("MissingValueFisherZTest expects a numpy.ndarray.")
        if data.ndim != 2:
            raise ValueError(
                "MissingValueFisherZTest expects a two-dimensional data array."
            )

        cond_tuple = tuple(cond_set)
        self._validate_indices(data.shape[1], x, y, cond_tuple)
        variables = (x, y, *cond_tuple)
        query_data = np.asarray(data[:, variables], dtype=float)
        finite_mask = np.all(np.isfinite(query_data), axis=1)
        complete_data = query_data[finite_mask]

        if complete_data.shape[0] < 4:
            raise ValueError(
                "MissingValueFisherZTest requires at least four complete rows "
                "for the queried variables."
            )

        remapped_cond_set = tuple(range(2, 2 + len(cond_tuple)))
        result = super().test(complete_data, 0, 1, remapped_cond_set)
        return CITestResult(
            independent=result.independent,
            p_value=result.p_value,
            statistic=result.statistic,
            method=self.method,
            n_samples=complete_data.shape[0],
        )
