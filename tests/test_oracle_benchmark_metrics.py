import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from fci_engine import fci, shape_from_pag
from fci_engine.metrics import compare_pag_shapes

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))
causal_learn = pytest.importorskip("causallearn.search.ConstraintBased.FCI")


def causal_learn_shape(
    data: pd.DataFrame,
    alpha: float,
    depth: int = 2,
    max_path_length: int = 3,
) -> dict[tuple[str, str], tuple[str, str]]:
    labels = list(data.columns)
    label_order = {label: index for index, label in enumerate(labels)}
    graph, _ = causal_learn.fci(
        data.to_numpy(),
        independence_test_method="fisherz",
        alpha=alpha,
        depth=depth,
        max_path_length=max_path_length,
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


def test_pag_comparison_scores_false_positive_edges() -> None:
    comparison = compare_pag_shapes(
        {},
        {("X0", "X1"): ("CIRCLE", "CIRCLE")},
    )

    assert comparison.expected_edges == 0
    assert comparison.actual_edges == 1
    assert comparison.false_positive_edges == 1
    assert comparison.exact_edge_f1 == 0.0


def test_fci_engine_default_beats_causal_learn_default_on_noise_oracle() -> None:
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        rng.normal(size=(8000, 6)),
        columns=["X0", "X1", "X2", "X3", "X4", "X5"],
    )
    oracle_shape = {}

    engine_result = fci(data, max_cond_set_size=2, max_path_length=3)
    engine_shape = shape_from_pag(engine_result.graph)
    causal_shape = causal_learn_shape(data, alpha=0.05)

    engine_score = compare_pag_shapes(oracle_shape, engine_shape)
    causal_score = compare_pag_shapes(oracle_shape, causal_shape)

    assert engine_result.config.alpha == 0.001
    assert engine_score.false_positive_edges == 0
    assert causal_score.false_positive_edges > engine_score.false_positive_edges
    assert engine_score.exact_edge_f1 > causal_score.exact_edge_f1


def test_same_alpha_medical_oracle_matches_for_both_implementations() -> None:
    rng = np.random.default_rng(42)
    hidden = rng.normal(size=8000)
    x1 = rng.normal(size=8000)
    x2 = rng.normal(size=8000)
    a = 0.8 * x1 + 0.8 * hidden + rng.normal(scale=0.4, size=8000)
    b = 0.8 * x2 + 0.8 * hidden + rng.normal(scale=0.4, size=8000)
    d = 0.8 * a + rng.normal(scale=0.4, size=8000)
    data = pd.DataFrame({"X1": x1, "X2": x2, "A": a, "B": b, "D": d})
    oracle_shape = {
        ("X1", "A"): ("CIRCLE", "ARROW"),
        ("X2", "B"): ("CIRCLE", "ARROW"),
        ("A", "B"): ("ARROW", "ARROW"),
        ("A", "D"): ("TAIL", "ARROW"),
    }

    engine_shape = shape_from_pag(
        fci(data, alpha=0.001, max_cond_set_size=2, max_path_length=3).graph
    )
    causal_shape = causal_learn_shape(data, alpha=0.001)

    assert compare_pag_shapes(oracle_shape, engine_shape).exact_edge_f1 == 1.0
    assert compare_pag_shapes(oracle_shape, causal_shape).exact_edge_f1 == 1.0
