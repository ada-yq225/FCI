import pytest

from fci_engine.metrics.benchmark import (
    _parse_pcalg_edges,
    aggregate_benchmark_results,
    format_benchmark_leaderboard,
    format_benchmark_results,
    run_oracle_benchmark,
    run_pcalg_fci_plus,
)
from fci_engine.simulation import (
    default_oracle_cases,
    make_independent_noise_case,
    make_latent_medical_case,
    realistic_oracle_cases,
)


def test_oracle_benchmark_runs_engine_algorithms() -> None:
    case = make_independent_noise_case(n_samples=400, n_variables=4, seed=11)

    results = run_oracle_benchmark(
        [case],
        include_causal_learn=False,
        include_pcalg=False,
    )

    assert [result.algorithm for result in results] == [
        "fci_engine.fci",
        "fci_engine.fci_plus",
    ]
    assert all(not result.skipped for result in results)
    assert all(result.comparison is not None for result in results)
    assert all(result.comparison.false_positive_edges == 0 for result in results)


def test_benchmark_format_includes_scores() -> None:
    case = make_latent_medical_case(n_samples=1200, seed=12)
    results = run_oracle_benchmark(
        [case],
        include_causal_learn=False,
        include_pcalg=False,
    )

    formatted = format_benchmark_results(results)

    assert "latent_medical" in formatted
    assert "fci_engine.fci_plus" in formatted
    assert "exact_edge_f1" in formatted


def test_benchmark_leaderboard_aggregates_by_algorithm() -> None:
    case = make_independent_noise_case(n_samples=400, n_variables=4, seed=14)
    results = run_oracle_benchmark(
        [case],
        include_causal_learn=False,
        include_pcalg=False,
    )

    aggregates = aggregate_benchmark_results(results)
    formatted = format_benchmark_leaderboard(results)

    assert {aggregate.algorithm for aggregate in aggregates} == {
        "fci_engine.fci",
        "fci_engine.fci_plus",
    }
    assert all(aggregate.mean_exact_edge_f1 == 1.0 for aggregate in aggregates)
    assert "endpoint_acc" in formatted


def test_realistic_oracle_cases_run_engine_algorithms() -> None:
    cases = realistic_oracle_cases(n_repeats=1, n_samples=600)

    results = run_oracle_benchmark(
        cases[:2],
        include_causal_learn=False,
        include_pcalg=False,
    )

    assert len(cases) == 5
    assert {case.name for case in cases} >= {
        "hospital_triage",
        "microservice_incident",
    }
    assert all(result.comparison is not None for result in results)
    assert all(not result.skipped for result in results)


def test_pcalg_runner_skips_cleanly_without_rscript() -> None:
    case = make_independent_noise_case(n_samples=200, n_variables=3, seed=13)

    result = run_pcalg_fci_plus(case)

    if result.skipped:
        assert "Rscript" in result.skipped_reason or "pcalg" in result.skipped_reason
    else:
        assert result.comparison is not None


def test_fci_plus_oracle_accuracy_matches_or_exceeds_pcalg_when_available() -> None:
    results = run_oracle_benchmark(
        default_oracle_cases(),
        include_causal_learn=False,
        include_pcalg=True,
        include_kernel_ci=False,
    )
    pcalg_results = [
        result for result in results if result.algorithm == "pcalg.fciPlus"
    ]
    if any(result.skipped for result in pcalg_results):
        pytest.skip("pcalg is not available in this environment.")

    aggregates = {
        aggregate.algorithm: aggregate
        for aggregate in aggregate_benchmark_results(results)
    }
    engine = aggregates["fci_engine.fci_plus"]
    pcalg = aggregates["pcalg.fciPlus"]

    assert engine.mean_exact_edge_f1 >= pcalg.mean_exact_edge_f1
    assert engine.mean_skeleton_f1 >= pcalg.mean_skeleton_f1
    assert engine.mean_endpoint_accuracy >= pcalg.mean_endpoint_accuracy


def test_parse_pcalg_pag_edges() -> None:
    output = """
noise
PCALG_EDGE_BEGIN
A,B,3,2
A,C,2,2
B,C,1,1
PCALG_EDGE_END
"""

    assert _parse_pcalg_edges(output) == {
        ("A", "B"): ("TAIL", "ARROW"),
        ("A", "C"): ("ARROW", "ARROW"),
        ("B", "C"): ("CIRCLE", "CIRCLE"),
    }
