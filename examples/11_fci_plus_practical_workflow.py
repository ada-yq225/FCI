"""End-to-end FCI+ workflow for an applied continuous dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fci_engine import fci_plus


def make_example_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate observed data with an unobserved common cause."""

    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n_samples)
    exposure = 0.8 * latent + rng.normal(scale=0.6, size=n_samples)
    biomarker = 0.7 * exposure + rng.normal(scale=0.6, size=n_samples)
    outcome = (
        0.9 * latent
        + 0.6 * biomarker
        + rng.normal(
            scale=0.6,
            size=n_samples,
        )
    )
    return pd.DataFrame(
        {
            "exposure": exposure,
            "biomarker": biomarker,
            "outcome": outcome,
        }
    )


def main() -> None:
    data = make_example_data()
    result = fci_plus(
        data,
        profile="practical",
        max_cond_set_size=2,
    )

    print(result.summary())
    print(result.to_pandas_edges().to_string(index=False))

    output_directory = Path(__file__).resolve().parent / "fci_plus_output"
    paths = result.save_artifacts(
        output_directory,
        stem="practical_example",
    )
    for artifact, path in paths.items():
        print(f"{artifact}: {path}")


if __name__ == "__main__":
    main()
