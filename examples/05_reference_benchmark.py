"""Run preset oracle benchmarks for fci_engine, causal-learn, and pcalg.

R pcalg comparison is optional. It runs only when both Rscript and pcalg are
available in the local environment.
"""

from __future__ import annotations

from fci_engine.metrics import (
    format_benchmark_leaderboard,
    format_benchmark_results,
    run_oracle_benchmark,
)
from fci_engine.simulation import default_oracle_cases


def main() -> None:
    results = run_oracle_benchmark(
        default_oracle_cases(),
        include_causal_learn=True,
        include_pcalg=True,
    )
    print(format_benchmark_results(results))
    print("\nLeaderboard")
    print(format_benchmark_leaderboard(results))


if __name__ == "__main__":
    main()
