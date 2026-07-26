"""Known-truth regression tests for the finite-sample application profile."""

from __future__ import annotations

from statistics import mean

from fci_engine import (
    compare_pag_shapes,
    compare_pag_shapes_semantic,
    fci_plus,
    realistic_oracle_cases,
    shape_from_pag,
)


def test_practical_profile_improves_seeded_finite_sample_endpoint_recovery() -> None:
    """Conservative application settings must improve the seeded suite."""

    paper_scores: list[tuple[float, float, float]] = []
    robust_scores: list[tuple[float, float, float]] = []
    for case in realistic_oracle_cases(n_repeats=3, n_samples=2_500):
        bound = case.sparsity_bound or case.max_cond_set_size
        assert bound is not None
        paper = fci_plus(
            case.data,
            profile="paper",
            k=bound,
            alpha=case.alpha,
        )
        robust = fci_plus(
            case.data,
            profile="practical",
            max_cond_set_size=case.max_cond_set_size,
            sparsity_bound=case.sparsity_bound,
            alpha=case.alpha,
            max_path_length=case.max_path_length,
        )
        for result, scores in (
            (paper, paper_scores),
            (robust, robust_scores),
        ):
            learned = shape_from_pag(result.graph)
            exact = compare_pag_shapes(case.oracle_shape, learned)
            semantic = compare_pag_shapes_semantic(case.oracle_shape, learned)
            scores.append(
                (
                    exact.skeleton_f1,
                    exact.exact_edge_f1,
                    semantic.semantic_edge_f1,
                )
            )

    paper_means = tuple(
        mean(score[index] for score in paper_scores) for index in range(3)
    )
    robust_means = tuple(
        mean(score[index] for score in robust_scores) for index in range(3)
    )

    assert robust_means[0] >= paper_means[0]
    assert robust_means[1] > paper_means[1]
    assert robust_means[2] > paper_means[2]
