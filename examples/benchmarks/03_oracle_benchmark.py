"""Oracle benchmark comparing fci_engine and causal-learn.

The oracle shapes below are written from the known synthetic data-generating
graphs. causal-learn is used as a comparator, not as the source of truth.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from time import perf_counter

import numpy as np
import pandas as pd

from fci_engine import fci, shape_from_pag
from fci_engine.metrics import Shape, compare_pag_shapes

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))


def make_medical_demo_data(n_samples: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hidden = rng.normal(size=n_samples)
    x1 = rng.normal(size=n_samples)
    x2 = rng.normal(size=n_samples)
    a = 0.8 * x1 + 0.8 * hidden + rng.normal(scale=0.4, size=n_samples)
    b = 0.8 * x2 + 0.8 * hidden + rng.normal(scale=0.4, size=n_samples)
    d = 0.8 * a + rng.normal(scale=0.4, size=n_samples)
    return pd.DataFrame({"X1": x1, "X2": x2, "A": a, "B": b, "D": d})


def make_independent_noise_data(n_samples: int = 8000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(n_samples, 6)),
        columns=["X0", "X1", "X2", "X3", "X4", "X5"],
    )


def medical_oracle_shape() -> Shape:
    return {
        ("X1", "A"): ("CIRCLE", "ARROW"),
        ("X2", "B"): ("CIRCLE", "ARROW"),
        ("A", "B"): ("ARROW", "ARROW"),
        ("A", "D"): ("TAIL", "ARROW"),
    }


def empty_oracle_shape() -> Shape:
    return {}


def run_fci_engine(
    data: pd.DataFrame,
    alpha: object,
) -> tuple[dict[tuple[str, str], tuple[str, str]], float]:
    start = perf_counter()
    result = fci(data, alpha=alpha, max_cond_set_size=2, max_path_length=3)
    elapsed = perf_counter() - start
    return shape_from_pag(result.graph), elapsed


def run_causal_learn(
    data: pd.DataFrame,
    alpha: float,
) -> tuple[dict[tuple[str, str], tuple[str, str]], float]:
    from causallearn.search.ConstraintBased.FCI import fci as causal_learn_fci

    labels = list(data.columns)
    label_order = {label: index for index, label in enumerate(labels)}
    start = perf_counter()
    with redirect_stdout(StringIO()):
        graph, _ = causal_learn_fci(
            data.to_numpy(),
            independence_test_method="fisherz",
            alpha=alpha,
            depth=2,
            max_path_length=3,
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


def print_case(
    name: str,
    data: pd.DataFrame,
    oracle: Shape,
    engine_alpha: object,
    causal_learn_alpha: float,
) -> None:
    engine_shape, engine_time = run_fci_engine(data, alpha=engine_alpha)
    causal_shape, causal_time = run_causal_learn(data, alpha=causal_learn_alpha)
    engine_score = compare_pag_shapes(oracle, engine_shape)
    causal_score = compare_pag_shapes(oracle, causal_shape)

    print(f"\n{name}")
    print(f"  fci_engine  {engine_score.summary()} time={engine_time:.4f}s")
    print(f"  causal-learn {causal_score.summary()} time={causal_time:.4f}s")
    print(f"  fci_engine edges:  {engine_shape}")
    print(f"  causal-learn edges: {causal_shape}")


def main() -> None:
    print("Oracle benchmark: fci_engine vs causal-learn")
    print_case(
        "Known latent medical graph, same alpha",
        make_medical_demo_data(),
        medical_oracle_shape(),
        engine_alpha=0.001,
        causal_learn_alpha=0.001,
    )
    print_case(
        "Independent high-N noise, package defaults",
        make_independent_noise_data(),
        empty_oracle_shape(),
        engine_alpha="auto",
        causal_learn_alpha=0.05,
    )


if __name__ == "__main__":
    main()
