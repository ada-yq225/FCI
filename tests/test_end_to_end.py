import numpy as np
import pandas as pd

from fci_engine import FCI, fci


def make_chain_data(n_samples: int = 4000, seed: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_samples)
    y = 0.9 * x + rng.normal(scale=0.4, size=n_samples)
    z = 0.9 * y + rng.normal(scale=0.4, size=n_samples)
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def make_fork_data(n_samples: int = 4000, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n_samples)
    x = 0.9 * y + rng.normal(scale=0.4, size=n_samples)
    z = 0.9 * y + rng.normal(scale=0.4, size=n_samples)
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def make_collider_data(n_samples: int = 4000, seed: int = 22) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_samples)
    z = rng.normal(size=n_samples)
    y = 0.9 * x + 0.9 * z + rng.normal(scale=0.4, size=n_samples)
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def make_latent_confounder_data(
    n_samples: int = 4000,
    seed: int = 23,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hidden = rng.normal(size=n_samples)
    x = 0.9 * hidden + rng.normal(scale=0.35, size=n_samples)
    y = 0.8 * hidden + rng.normal(scale=0.35, size=n_samples)
    w = rng.normal(size=n_samples)
    return pd.DataFrame({"X": x, "Y": y, "W": w})


def edge_reprs(result) -> set[str]:
    return {result.graph.edge_repr(x, y) for x, y in result.graph.edges()}


def test_chain_structure_end_to_end() -> None:
    result = fci(make_chain_data(), alpha=0.001, max_cond_set_size=2)

    assert edge_reprs(result) == {"X o-o Y", "Y o-o Z"}
    assert result.sepsets[("X", "Z")] == {"Y"}


def test_fork_structure_end_to_end() -> None:
    result = fci(make_fork_data(), alpha=0.001, max_cond_set_size=2)

    assert edge_reprs(result) == {"X o-o Y", "Y o-o Z"}
    assert result.sepsets[("X", "Z")] == {"Y"}


def test_collider_structure_end_to_end() -> None:
    result = fci(make_collider_data(), alpha=0.001, max_cond_set_size=2)

    assert edge_reprs(result) == {"X o-> Y", "Y <-o Z"}
    assert result.sepsets[("X", "Z")] == set()


def test_latent_confounder_structure_end_to_end() -> None:
    result = fci(make_latent_confounder_data(), alpha=0.001, max_cond_set_size=2)

    assert edge_reprs(result) == {"X o-o Y"}
    assert not result.graph.is_adjacent("X", "W")
    assert not result.graph.is_adjacent("Y", "W")


def test_end_to_end_with_pandas_dataframe_names() -> None:
    frame = make_collider_data().rename(
        columns={"X": "load", "Y": "latency", "Z": "network"}
    )

    estimator = FCI(alpha=0.001, max_cond_set_size=2)
    result = estimator.fit(frame)

    assert estimator.variable_names == ["load", "latency", "network"]
    assert result.graph.nodes == ("load", "latency", "network")
    assert edge_reprs(result) == {"load o-> latency", "latency <-o network"}
