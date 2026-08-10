"""Generate a bounded exact-oracle scaling comparison for FCI and FCI+.

The benchmark deliberately includes two sparse graph families. A directed
chain has no difficult D-SEP link, while the second family embeds the
Claassen et al. Figure 4(b) D-SEP structure alongside isolated observed
variables. Both are exact-oracle experiments with maximum MAG degree at most
three. They measure implementation work; they do not establish a universal
runtime ranking or estimate finite-sample statistical accuracy.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median
from typing import Any, Callable

from fci_engine import FCIResult, fci, fci_plus
from fci_engine.simulation import MAGSpec


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "data" / "scaling_benchmark.csv"
DEFAULT_NODE_COUNTS = (5, 10, 20, 40, 80)
FIELDNAMES = (
    "graph_family",
    "nodes",
    "true_max_degree",
    "algorithm",
    "profile",
    "sparsity_bound",
    "repeats",
    "median_elapsed_seconds",
    "ci_test_count",
    "learned_edges",
    "target_edges",
    "exact_skeleton_recovered",
    "ci_test_ratio_to_fci",
    "runtime_ratio_to_fci",
)


def generate_scaling_benchmark(
    output_path: Path = DEFAULT_OUTPUT,
    *,
    node_counts: tuple[int, ...] = DEFAULT_NODE_COUNTS,
    repeats: int = 3,
) -> Path:
    """Run both paper profiles over two sparse exact-oracle graph families."""

    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    if not node_counts or any(count < 5 for count in node_counts):
        raise ValueError("node_counts must contain values of at least five.")

    rows: list[dict[str, Any]] = []
    families: tuple[tuple[str, Callable[[int], MAGSpec], int], ...] = (
        ("directed_chain", _directed_chain, 2),
        ("figure4b_plus_isolates", _figure4b_plus_isolates, 3),
    )
    for family_name, factory, true_max_degree in families:
        for node_count in node_counts:
            mag = factory(node_count)
            target_edges = len(mag.directed_edges) + len(mag.bidirected_edges)
            target_skeleton = {
                frozenset(edge) for edge in (*mag.directed_edges, *mag.bidirected_edges)
            }
            group_rows = []
            for algorithm in ("fci", "fci_plus"):
                results = [
                    _fit_exact_oracle(mag, algorithm=algorithm) for _ in range(repeats)
                ]
                first = results[0]
                elapsed = [result.elapsed_time for result in results]
                learned_edges = len(first.edges)
                learned_skeleton = {frozenset((x, y)) for x, y in first.edges}
                group_rows.append(
                    {
                        "graph_family": family_name,
                        "nodes": node_count,
                        "true_max_degree": true_max_degree,
                        "algorithm": algorithm,
                        "profile": (
                            "spirtes_2000_paper"
                            if algorithm == "fci"
                            else "claassen_2013_paper"
                        ),
                        "sparsity_bound": "" if algorithm == "fci" else 3,
                        "repeats": repeats,
                        "median_elapsed_seconds": float(median(elapsed)),
                        "ci_test_count": first.ci_test_count,
                        "learned_edges": learned_edges,
                        "target_edges": target_edges,
                        "exact_skeleton_recovered": (
                            learned_skeleton == target_skeleton
                        ),
                    }
                )
            baseline = next(row for row in group_rows if row["algorithm"] == "fci")
            for row in group_rows:
                row["ci_test_ratio_to_fci"] = (
                    row["ci_test_count"] / baseline["ci_test_count"]
                )
                row["runtime_ratio_to_fci"] = (
                    row["median_elapsed_seconds"] / baseline["median_elapsed_seconds"]
                )
                rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _fit_exact_oracle(mag: MAGSpec, *, algorithm: str) -> FCIResult:
    if algorithm == "fci":
        return fci(
            mag.dummy_data(),
            profile="paper",
            ci_test=mag.oracle_ci_test(),
        )
    if algorithm == "fci_plus":
        return fci_plus(
            mag.dummy_data(),
            profile="paper",
            k=3,
            ci_test=mag.oracle_ci_test(),
        )
    raise ValueError(f"Unknown algorithm: {algorithm!r}.")


def _directed_chain(node_count: int) -> MAGSpec:
    nodes = tuple(f"X{index:03d}" for index in range(node_count))
    return MAGSpec(
        nodes=nodes,
        directed_edges=tuple(zip(nodes, nodes[1:])),
    )


def _figure4b_plus_isolates(node_count: int) -> MAGSpec:
    core = ("Z", "U", "V", "X", "Y")
    isolates = tuple(f"N{index:03d}" for index in range(node_count - len(core)))
    return MAGSpec(
        nodes=(*core, *isolates),
        directed_edges=(("Z", "U"), ("Z", "V"), ("U", "Y"), ("V", "X")),
        bidirected_edges=(("X", "U"), ("V", "Y")),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--node-counts", type=int, nargs="+", default=DEFAULT_NODE_COUNTS
    )
    args = parser.parse_args()
    output = generate_scaling_benchmark(
        args.output,
        node_counts=tuple(args.node_counts),
        repeats=args.repeats,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
