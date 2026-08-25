"""Large realistic oracle benchmark for fci_engine.

The cases are synthetic but scenario-shaped: each one has a known data
generating graph and a hand-written expected PAG shape. Optional external
comparators are causal-learn FCI and R pcalg::fciPlus when available.
"""

from __future__ import annotations

from fci_engine.metrics import (
    aggregate_benchmark_results,
    format_benchmark_leaderboard,
    format_benchmark_results,
    run_oracle_benchmark,
)
from fci_engine.simulation import default_oracle_cases, realistic_oracle_cases


def main() -> None:
    cases = [
        *default_oracle_cases(),
        *realistic_oracle_cases(n_repeats=2, n_samples=6000),
    ]
    results = run_oracle_benchmark(
        cases,
        include_causal_learn=True,
        include_pcalg=True,
    )

    print("Large realistic oracle benchmark")
    print(f"cases={len(cases)}")
    print()
    print(format_benchmark_results(results))
    print("\nLeaderboard")
    print(format_benchmark_leaderboard(results))
    print("\nPer-case edge details")
    for case in cases:
        print(f"\n{case.name}")
        if case.notes:
            print(f"  notes: {case.notes}")
        print(f"  oracle: {_format_shape(case.oracle_shape)}")
        for result in results:
            if result.case_name != case.name:
                continue
            if result.skipped:
                print(f"  {result.algorithm}: skipped ({result.skipped_reason})")
            else:
                assert result.comparison is not None
                print(
                    f"  {result.algorithm}: "
                    f"{result.comparison.summary()} "
                    f"time={result.elapsed_time:.4f}s ci={result.ci_test_count}"
                )
                print(f"    edges: {_format_shape(result.edges)}")

    print("\nCompleted algorithms")
    for aggregate in aggregate_benchmark_results(results):
        print(f"  {aggregate.summary()}")


def _format_shape(shape: dict[tuple[str, str], tuple[object, object]]) -> str:
    if not shape:
        return "{}"
    parts = []
    for (x, y), (endpoint_x, endpoint_y) in sorted(shape.items()):
        parts.append(_edge_text(x, y, endpoint_x, endpoint_y))
    return "{" + ", ".join(parts) + "}"


def _edge_text(x: str, y: str, endpoint_x: object, endpoint_y: object) -> str:
    left = _mark(endpoint_x)
    right = _mark(endpoint_y)
    if (left, right) == ("o", "o"):
        return f"{x} o-o {y}"
    if (left, right) == ("o", ">"):
        return f"{x} o-> {y}"
    if (left, right) == ("-", ">"):
        return f"{x} --> {y}"
    if (left, right) == (">", ">"):
        return f"{x} <-> {y}"
    if (left, right) == ("-", "-"):
        return f"{x} --- {y}"
    if (left, right) == (">", "o"):
        return f"{x} <-o {y}"
    if (left, right) == (">", "-"):
        return f"{x} <-- {y}"
    return f"{x} {left}-{right} {y}"


def _mark(endpoint: object) -> str:
    text = getattr(endpoint, "name", str(endpoint)).upper()
    return {
        "TAIL": "-",
        "ARROW": ">",
        "CIRCLE": "o",
        "NONE": ".",
    }.get(text, text)


if __name__ == "__main__":
    main()
