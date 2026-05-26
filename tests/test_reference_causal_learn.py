import os
import tempfile

import numpy as np
import pytest

from fci_engine import fci

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))
causal_learn = pytest.importorskip("causallearn.search.ConstraintBased.FCI")


def make_chain_data(n_samples: int = 2000, seed: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_samples)
    y = 0.9 * x + rng.normal(scale=0.4, size=n_samples)
    z = 0.9 * y + rng.normal(scale=0.4, size=n_samples)
    return np.column_stack([x, y, z])


def make_fork_data(n_samples: int = 2000, seed: int = 41) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n_samples)
    x = 0.9 * y + rng.normal(scale=0.4, size=n_samples)
    z = 0.9 * y + rng.normal(scale=0.4, size=n_samples)
    return np.column_stack([x, y, z])


def make_collider_data(n_samples: int = 2000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_samples)
    z = rng.normal(size=n_samples)
    y = 0.9 * x + 0.9 * z + rng.normal(scale=0.4, size=n_samples)
    return np.column_stack([x, y, z])


def make_latent_confounder_data(
    n_samples: int = 2000,
    seed: int = 43,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hidden = rng.normal(size=n_samples)
    x = 0.9 * hidden + rng.normal(scale=0.35, size=n_samples)
    y = 0.8 * hidden + rng.normal(scale=0.35, size=n_samples)
    w = rng.normal(size=n_samples)
    return np.column_stack([x, y, w])


def fci_engine_shape(
    data: np.ndarray,
    labels: list[str],
) -> dict[tuple[str, str], tuple[str, str]]:
    result = fci(data, alpha=0.001, max_cond_set_size=2, max_path_length=3)
    return {
        (labels[int(x[1:])], labels[int(y[1:])]): (endpoint_x.name, endpoint_y.name)
        for x, y, endpoint_x, endpoint_y in result.graph.to_edge_list()
    }


def causal_learn_shape(
    data: np.ndarray,
    labels: list[str],
) -> dict[tuple[str, str], tuple[str, str]]:
    graph, _ = causal_learn.fci(
        data,
        independence_test_method="fisherz",
        alpha=0.001,
        depth=2,
        max_path_length=3,
        verbose=False,
        show_progress=False,
    )
    label_order = {label: index for index, label in enumerate(labels)}
    name_to_label = {f"X{index + 1}": label for index, label in enumerate(labels)}
    shape = {}

    for edge in graph.get_graph_edges():
        node1 = name_to_label[edge.get_node1().get_name()]
        node2 = name_to_label[edge.get_node2().get_name()]
        endpoint1 = str(edge.get_endpoint1())
        endpoint2 = str(edge.get_endpoint2())
        if label_order[node1] <= label_order[node2]:
            shape[(node1, node2)] = (endpoint1, endpoint2)
        else:
            shape[(node2, node1)] = (endpoint2, endpoint1)
    return shape


@pytest.mark.parametrize(
    ("data_factory", "labels"),
    [
        (make_chain_data, ["X", "Y", "Z"]),
        (make_fork_data, ["X", "Y", "Z"]),
        (make_collider_data, ["X", "Y", "Z"]),
        (make_latent_confounder_data, ["X", "Y", "W"]),
    ],
)
def test_reference_shapes_match_causal_learn(data_factory, labels) -> None:
    data = data_factory()

    assert fci_engine_shape(data, labels) == causal_learn_shape(data, labels)
