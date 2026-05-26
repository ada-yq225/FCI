import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from fci_engine import fci

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))
causal_learn = pytest.importorskip("causallearn.search.ConstraintBased.FCI")


def make_medical_demo_data(n_samples: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hidden = rng.normal(size=n_samples)
    x1 = rng.normal(size=n_samples)
    x2 = rng.normal(size=n_samples)
    a = 0.8 * x1 + 0.8 * hidden + rng.normal(scale=0.4, size=n_samples)
    b = 0.8 * x2 + 0.8 * hidden + rng.normal(scale=0.4, size=n_samples)
    d = 0.8 * a + rng.normal(scale=0.4, size=n_samples)
    return pd.DataFrame({"X1": x1, "X2": x2, "A": a, "B": b, "D": d})


def make_complex_ten_variable_data(
    n_samples: int = 8000,
    seed: int = 11,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    u1 = rng.normal(size=n_samples)
    u2 = rng.normal(size=n_samples)
    u3 = rng.normal(size=n_samples)

    a = rng.normal(size=n_samples)
    b = 0.75 * a + rng.normal(scale=0.4, size=n_samples)
    c = rng.normal(size=n_samples)
    d = 0.7 * b + 0.55 * u1 + rng.normal(scale=0.4, size=n_samples)
    e = 0.75 * c + 0.55 * u1 + rng.normal(scale=0.4, size=n_samples)
    f = 0.65 * d + 0.55 * u2 + rng.normal(scale=0.4, size=n_samples)
    g = 0.65 * e + 0.55 * u2 + rng.normal(scale=0.4, size=n_samples)
    h = 0.6 * f + 0.45 * g + rng.normal(scale=0.4, size=n_samples)
    i = 0.65 * h + 0.55 * u3 + rng.normal(scale=0.4, size=n_samples)
    j = 0.65 * c + 0.55 * u3 + rng.normal(scale=0.4, size=n_samples)

    return pd.DataFrame(
        {
            "A": a,
            "B": b,
            "C": c,
            "D": d,
            "E": e,
            "F": f,
            "G": g,
            "H": h,
            "I": i,
            "J": j,
        }
    )


def make_instrumented_two_latent_data(
    n_samples: int = 12000,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    u1 = rng.normal(size=n_samples)
    u2 = rng.normal(size=n_samples)
    i1 = rng.normal(size=n_samples)
    i2 = rng.normal(size=n_samples)
    i3 = rng.normal(size=n_samples)

    a = 0.8 * i1 + 0.7 * u1 + rng.normal(scale=0.35, size=n_samples)
    b = 0.7 * i2 + 0.7 * u1 + 0.6 * u2 + rng.normal(scale=0.35, size=n_samples)
    c = 0.8 * i3 + 0.7 * u2 + rng.normal(scale=0.35, size=n_samples)
    d = 0.7 * a + 0.5 * b + rng.normal(scale=0.4, size=n_samples)
    e = 0.7 * c + 0.5 * d + rng.normal(scale=0.4, size=n_samples)

    return pd.DataFrame(
        {"I1": i1, "I2": i2, "I3": i3, "A": a, "B": b, "C": c, "D": d, "E": e}
    )


def fci_engine_shape(
    data: pd.DataFrame,
) -> dict[tuple[str, str], tuple[str, str]]:
    result = fci(data, alpha=0.001, max_cond_set_size=3, max_path_length=4)
    return {
        (x, y): (endpoint_x.name, endpoint_y.name)
        for x, y, endpoint_x, endpoint_y in result.graph.to_edge_list()
    }


def causal_learn_shape(
    data: pd.DataFrame,
) -> dict[tuple[str, str], tuple[str, str]]:
    labels = list(data.columns)
    label_order = {label: index for index, label in enumerate(labels)}
    graph, _ = causal_learn.fci(
        data.to_numpy(),
        independence_test_method="fisherz",
        alpha=0.001,
        depth=3,
        max_path_length=4,
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


def agreement_scores(
    expected: dict[tuple[str, str], tuple[str, str]],
    actual: dict[tuple[str, str], tuple[str, str]],
) -> tuple[float, float, float]:
    expected_edges = set(expected)
    actual_edges = set(actual)
    union = expected_edges | actual_edges
    common = expected_edges & actual_edges
    if not union:
        return 1.0, 1.0, 1.0

    skeleton_score = len(common) / len(union)
    exact_score = sum(expected[edge] == actual[edge] for edge in common) / len(union)

    endpoint_matches = 0
    endpoint_total = 2 * len(union)
    for edge in common:
        endpoint_matches += int(expected[edge][0] == actual[edge][0])
        endpoint_matches += int(expected[edge][1] == actual[edge][1])
    endpoint_score = endpoint_matches / endpoint_total
    return skeleton_score, endpoint_score, exact_score


@pytest.mark.parametrize(
    "data_factory",
    [
        make_medical_demo_data,
        make_complex_ten_variable_data,
    ],
)
def test_complex_reference_shapes_match_causal_learn_exactly(data_factory) -> None:
    data = data_factory()

    assert fci_engine_shape(data) == causal_learn_shape(data)


def test_instrumented_two_latent_case_has_high_but_not_exact_reference_agreement() -> None:
    data = make_instrumented_two_latent_data()
    engine_shape = fci_engine_shape(data)
    reference_shape = causal_learn_shape(data)

    skeleton_score, endpoint_score, exact_score = agreement_scores(
        reference_shape,
        engine_shape,
    )

    assert skeleton_score >= 0.80
    assert endpoint_score >= 0.70
    assert exact_score >= 0.60


def test_instrumented_two_latent_case_exact_match_causal_learn() -> None:
    data = make_instrumented_two_latent_data()

    assert fci_engine_shape(data) == causal_learn_shape(data)
