import os
import tempfile
import warnings
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pandas as pd
import pytest

from fci_engine import fci, shape_from_pag
from fci_engine.ci import FisherZTest, KernelCITest
from fci_engine.metrics import Shape, compare_pag_shapes


os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))
causal_learn = pytest.importorskip("causallearn.search.ConstraintBased.FCI")


def make_nonlinear_common_cause_data(
    n_samples: int = 260,
    seed: int = 23,
) -> pd.DataFrame:
    """Known graph: Z -> X and Z -> Y with nonlinear mechanisms."""

    rng = np.random.default_rng(seed)
    z = rng.uniform(-2.0, 2.0, size=n_samples)
    shared_nonlinear_signal = z**2
    x = 0.4 * z + shared_nonlinear_signal + rng.normal(scale=0.25, size=n_samples)
    y = -0.4 * z + shared_nonlinear_signal + rng.normal(scale=0.25, size=n_samples)
    return pd.DataFrame({"Z": z, "X": x, "Y": y})


def nonlinear_common_cause_oracle() -> Shape:
    return {
        ("Z", "X"): ("CIRCLE", "CIRCLE"),
        ("Z", "Y"): ("CIRCLE", "CIRCLE"),
    }


def _fci_engine_shape(
    data: pd.DataFrame,
    ci_test,
) -> dict[tuple[str, str], tuple[str, str]]:
    result = fci(
        data,
        alpha=ci_test.alpha,
        ci_test=ci_test,
        max_cond_set_size=1,
        max_path_length=2,
        do_pdsep=False,
    )
    return shape_from_pag(result.graph)


def _causal_learn_kci_shape(
    data: pd.DataFrame,
) -> dict[tuple[str, str], tuple[str, str]]:
    labels = list(data.columns)
    label_order = {label: index for index, label in enumerate(labels)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with redirect_stdout(StringIO()):
            graph, _ = causal_learn.fci(
                data.to_numpy(),
                independence_test_method="kci",
                alpha=0.05,
                depth=1,
                max_path_length=2,
                verbose=False,
                show_progress=False,
                node_names=labels,
            )

    shape = {}
    for edge in graph.get_graph_edges():
        node1 = edge.get_node1().get_name()
        node2 = edge.get_node2().get_name()
        endpoint1 = str(edge.get_endpoint1())
        endpoint2 = str(edge.get_endpoint2())
        if label_order[node1] <= label_order[node2]:
            shape[(node1, node2)] = (endpoint1, endpoint2)
        else:
            shape[(node2, node1)] = (endpoint2, endpoint1)
    return shape


def test_kernel_ci_fci_beats_fisherz_on_nonlinear_common_cause() -> None:
    data = make_nonlinear_common_cause_data()
    oracle = nonlinear_common_cause_oracle()

    fisher_shape = _fci_engine_shape(data, FisherZTest(alpha=0.05))
    kernel_shape = _fci_engine_shape(
        data,
        KernelCITest(alpha=0.05, n_permutations=99, random_state=0),
    )

    fisher_score = compare_pag_shapes(oracle, fisher_shape)
    kernel_score = compare_pag_shapes(oracle, kernel_shape)

    assert fisher_score.false_positive_edges == 1
    assert kernel_score.false_positive_edges == 0
    assert kernel_score.exact_edge_f1 > fisher_score.exact_edge_f1


def test_kernel_ci_fci_matches_causal_learn_kci_on_nonlinear_oracle() -> None:
    data = make_nonlinear_common_cause_data()
    oracle = nonlinear_common_cause_oracle()

    engine_shape = _fci_engine_shape(
        data,
        KernelCITest(alpha=0.05, n_permutations=99, random_state=0),
    )
    causal_shape = _causal_learn_kci_shape(data)

    engine_score = compare_pag_shapes(oracle, engine_shape)
    causal_score = compare_pag_shapes(oracle, causal_shape)

    assert engine_score.exact_edge_f1 == 1.0
    assert causal_score.exact_edge_f1 == 1.0
    assert engine_score.exact_edge_f1 >= causal_score.exact_edge_f1
