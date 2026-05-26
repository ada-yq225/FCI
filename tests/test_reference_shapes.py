import numpy as np
import pandas as pd

from fci_engine import fci


def edge_reprs(result) -> set[str]:
    return {result.graph.edge_repr(x, y) for x, y in result.graph.edges()}


def make_chain_data(n_samples: int = 4000, seed: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_samples)
    y = 0.9 * x + rng.normal(scale=0.4, size=n_samples)
    z = 0.9 * y + rng.normal(scale=0.4, size=n_samples)
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def make_fork_data(n_samples: int = 4000, seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n_samples)
    x = 0.9 * y + rng.normal(scale=0.4, size=n_samples)
    z = 0.9 * y + rng.normal(scale=0.4, size=n_samples)
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def make_collider_data(n_samples: int = 4000, seed: int = 32) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_samples)
    z = rng.normal(size=n_samples)
    y = 0.9 * x + 0.9 * z + rng.normal(scale=0.4, size=n_samples)
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def make_latent_confounder_data(
    n_samples: int = 4000,
    seed: int = 33,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hidden = rng.normal(size=n_samples)
    x = 0.9 * hidden + rng.normal(scale=0.35, size=n_samples)
    y = 0.8 * hidden + rng.normal(scale=0.35, size=n_samples)
    w = rng.normal(size=n_samples)
    return pd.DataFrame({"X": x, "Y": y, "W": w})


def test_reference_chain_shape() -> None:
    result = fci(make_chain_data(), alpha=0.001, max_cond_set_size=2)

    assert result.graph.edges() == [("X", "Y"), ("Y", "Z")]
    assert edge_reprs(result) == {"X o-o Y", "Y o-o Z"}


def test_reference_fork_shape() -> None:
    result = fci(make_fork_data(), alpha=0.001, max_cond_set_size=2)

    assert result.graph.edges() == [("X", "Y"), ("Y", "Z")]
    assert edge_reprs(result) == {"X o-o Y", "Y o-o Z"}


def test_reference_collider_shape() -> None:
    result = fci(make_collider_data(), alpha=0.001, max_cond_set_size=2)

    assert result.graph.edges() == [("X", "Y"), ("Y", "Z")]
    assert edge_reprs(result) == {"X o-> Y", "Y <-o Z"}


def test_reference_latent_confounder_shape() -> None:
    result = fci(
        make_latent_confounder_data(),
        alpha=0.001,
        max_cond_set_size=2,
    )

    assert result.graph.edges() == [("X", "Y")]
    assert edge_reprs(result) == {"X o-o Y"}
