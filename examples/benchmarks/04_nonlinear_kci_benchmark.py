"""Nonlinear oracle benchmark for fci_engine KernelCITest.

Known graph:
    Z -> X
    Z -> Y

The shared mechanism is nonlinear, so Fisher-Z tends to keep a false X-Y edge.
Kernel CI should remove X-Y after conditioning on Z.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from contextlib import redirect_stdout
from io import StringIO
from time import perf_counter
from typing import Optional

import numpy as np
import pandas as pd

from fci_engine import fci, shape_from_pag
from fci_engine.ci import FisherZTest, KernelCITest
from fci_engine.metrics import Shape, compare_pag_shapes


os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))


def make_nonlinear_common_cause_data(
    n_samples: int = 260,
    seed: int = 23,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.uniform(-2.0, 2.0, size=n_samples)
    shared_nonlinear_signal = z**2
    x = 0.4 * z + shared_nonlinear_signal + rng.normal(scale=0.25, size=n_samples)
    y = -0.4 * z + shared_nonlinear_signal + rng.normal(scale=0.25, size=n_samples)
    return pd.DataFrame({"Z": z, "X": x, "Y": y})


def oracle_shape() -> Shape:
    return {
        ("Z", "X"): ("CIRCLE", "CIRCLE"),
        ("Z", "Y"): ("CIRCLE", "CIRCLE"),
    }


def run_fci_engine(data: pd.DataFrame, ci_test) -> tuple[dict, float]:
    start = perf_counter()
    result = fci(
        data,
        alpha=ci_test.alpha,
        ci_test=ci_test,
        max_cond_set_size=1,
        max_path_length=2,
        do_pdsep=False,
    )
    elapsed = perf_counter() - start
    return shape_from_pag(result.graph), elapsed


def run_causal_learn_kci(data: pd.DataFrame) -> Optional[tuple[dict, float]]:
    try:
        from causallearn.search.ConstraintBased.FCI import fci as causal_learn_fci
    except ImportError:
        return None

    labels = list(data.columns)
    label_order = {label: index for index, label in enumerate(labels)}
    start = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with redirect_stdout(StringIO()):
            graph, _ = causal_learn_fci(
                data.to_numpy(),
                independence_test_method="kci",
                alpha=0.05,
                depth=1,
                max_path_length=2,
                verbose=False,
                show_progress=False,
                node_names=labels,
            )
    elapsed = perf_counter() - start

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
    return shape, elapsed


def print_score(name: str, shape: dict, elapsed: float, oracle: Shape) -> None:
    score = compare_pag_shapes(oracle, shape)
    print(f"{name:22s} {score.summary()} time={elapsed:.3f}s")
    print(f"  edges: {shape}")


def main() -> None:
    data = make_nonlinear_common_cause_data()
    oracle = oracle_shape()
    print("Nonlinear oracle benchmark")
    print("True PAG shape:", oracle)

    fisher_shape, fisher_time = run_fci_engine(data, FisherZTest(alpha=0.05))
    print_score("fci_engine Fisher-Z", fisher_shape, fisher_time, oracle)

    kernel_shape, kernel_time = run_fci_engine(
        data,
        KernelCITest(alpha=0.05, n_permutations=99, random_state=0),
    )
    print_score("fci_engine KernelCI", kernel_shape, kernel_time, oracle)

    causal_result = run_causal_learn_kci(data)
    if causal_result is None:
        print("causal-learn KCI      not installed")
    else:
        causal_shape, causal_time = causal_result
        print_score("causal-learn KCI", causal_shape, causal_time, oracle)


if __name__ == "__main__":
    main()
