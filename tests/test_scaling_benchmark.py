from __future__ import annotations

import csv

from reports.generate_scaling_benchmark import generate_scaling_benchmark


def test_scaling_benchmark_records_both_families_and_algorithms(tmp_path) -> None:
    output = generate_scaling_benchmark(
        tmp_path / "scaling.csv",
        node_counts=(10,),
        repeats=1,
    )

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4
    assert {row["graph_family"] for row in rows} == {
        "directed_chain",
        "figure4b_plus_isolates",
    }
    assert {row["algorithm"] for row in rows} == {"fci", "fci_plus"}
    assert all(row["exact_skeleton_recovered"] == "True" for row in rows)
    assert all(float(row["ci_test_ratio_to_fci"]) > 0 for row in rows)
