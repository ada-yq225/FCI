"""Basic FCI examples with continuous Gaussian data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fci_engine import fci


def print_result(title: str, data: pd.DataFrame) -> None:
    result = fci(data, alpha="auto", max_cond_set_size=2)

    print(f"\n{title}")
    print(result.summary())
    print("Edges:")
    for x, y in result.graph.edges():
        print(f"- {result.graph.edge_repr(x, y)}")


def make_chain_data(n_samples: int = 3000, seed: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_samples)
    y = 0.9 * x + rng.normal(scale=0.4, size=n_samples)
    z = 0.9 * y + rng.normal(scale=0.4, size=n_samples)
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def make_collider_data(n_samples: int = 3000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_samples)
    z = rng.normal(size=n_samples)
    y = 0.9 * x + 0.9 * z + rng.normal(scale=0.4, size=n_samples)
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


if __name__ == "__main__":
    print_result("Chain example: X -> Y -> Z", make_chain_data())
    print_result("Collider example: X -> Y <- Z", make_collider_data())
