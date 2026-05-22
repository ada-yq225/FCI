"""Show a small data-to-PAG demonstration.

This example uses synthetic data and a hand-specified expected PAG. It is meant
to make the package output intuitive before the FCI discovery algorithm is
implemented.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from fci_engine.graph import Endpoint, PAG


def make_toy_data(n_samples: int = 200, seed: int = 7) -> pd.DataFrame:
    """Create toy observables with one hidden confounder."""

    rng = np.random.default_rng(seed)

    service_load = rng.normal(size=n_samples)
    latent_incident = rng.normal(size=n_samples)

    latency = 0.9 * service_load + rng.normal(scale=0.35, size=n_samples)
    timeouts = 0.8 * latency + rng.normal(scale=0.35, size=n_samples)
    cpu_pressure = 0.9 * latent_incident + rng.normal(scale=0.3, size=n_samples)
    error_rate = 0.8 * latent_incident + rng.normal(scale=0.3, size=n_samples)

    return pd.DataFrame(
        {
            "service_load": service_load,
            "latency": latency,
            "timeouts": timeouts,
            "cpu_pressure": cpu_pressure,
            "error_rate": error_rate,
        }
    )


def expected_pag_for_toy_data() -> PAG:
    """Return the expected PAG for the toy data-generating process."""

    pag = PAG(["service_load", "latency", "timeouts", "cpu_pressure", "error_rate"])
    pag.add_edge("service_load", "latency", Endpoint.TAIL, Endpoint.ARROW)
    pag.add_edge("latency", "timeouts", Endpoint.TAIL, Endpoint.ARROW)
    pag.add_edge("cpu_pressure", "error_rate", Endpoint.ARROW, Endpoint.ARROW)
    return pag


def format_demo() -> str:
    """Build a readable demo output."""

    data = make_toy_data()
    pag = expected_pag_for_toy_data()

    lines = [
        "Input data sample:",
        data.head(6).round(3).to_string(index=False),
        "",
        "Observed correlation matrix:",
        data.corr().round(2).to_string(),
        "",
        "Expected PAG result:",
    ]
    lines.extend(f"- {pag.edge_repr(x, y)}" for x, y in pag.edges())
    lines.extend(
        [
            "",
            "Causal queries on the PAG:",
            f"- definite causes of timeouts: {pag.definite_causes('timeouts')}",
            f"- possible causes of timeouts: {pag.possible_causes('timeouts')}",
            f"- definite causes of error_rate: {pag.definite_causes('error_rate')}",
            f"- possible causes of error_rate: {pag.possible_causes('error_rate')}",
            "",
            "Interpretation:",
            "- service_load --> latency --> timeouts is a directed causal chain.",
            "- cpu_pressure <-> error_rate indicates a latent common cause signal.",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_demo())
