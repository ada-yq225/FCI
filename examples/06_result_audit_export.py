"""Audit and export a learned PAG result."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from fci_engine import fci


def main() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=1000)
    y = 0.8 * x + rng.normal(scale=0.4, size=1000)
    z = 0.8 * y + rng.normal(scale=0.4, size=1000)
    data = pd.DataFrame({"X": x, "Y": y, "Z": z})

    result = fci(data, alpha=0.001, max_cond_set_size=2)

    print(result.summary())
    print("\nEdges")
    print(result.to_pandas_edges())

    print("\nExplanation for X-Z")
    print(result.explain_edge("X", "Z").summary())

    graph = result.to_networkx()
    print(f"\nNetworkX graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    output_path = Path(tempfile.gettempdir()) / "fci_engine_result.json"
    result.save_json(output_path, include_traces=True)
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
