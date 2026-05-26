"""Latent confounder demonstration for FCI."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fci_engine import fci


def make_latent_confounder_data(
    n_samples: int = 3000,
    seed: int = 12,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hidden_incident = rng.normal(size=n_samples)
    cpu_pressure = 0.9 * hidden_incident + rng.normal(scale=0.35, size=n_samples)
    error_rate = 0.8 * hidden_incident + rng.normal(scale=0.35, size=n_samples)
    independent_metric = rng.normal(size=n_samples)
    return pd.DataFrame(
        {
            "cpu_pressure": cpu_pressure,
            "error_rate": error_rate,
            "independent_metric": independent_metric,
        }
    )


if __name__ == "__main__":
    result = fci(
        make_latent_confounder_data(),
        alpha="auto",
        max_cond_set_size=2,
    )

    print("Latent confounder example")
    print(result.summary())
    print("Edges:")
    for x, y in result.graph.edges():
        print(f"- {result.graph.edge_repr(x, y)}")
