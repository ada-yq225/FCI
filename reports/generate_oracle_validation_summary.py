"""Freeze the Figure 4(b) exact-oracle and seeded SEM validation results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from fci_engine import (
    compare_pag_shapes,
    compare_pag_shapes_semantic,
    fci,
    fci_plus,
    sample_canonical_dsep_data,
    shape_from_pag,
)
from fci_engine.result import FCIResult
from fci_engine.simulation import canonical_dsep_mag


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "data" / "oracle_validation_summary.csv"

FIELDNAMES = (
    "fixture_id",
    "regime",
    "algorithm",
    "profile",
    "ci_test",
    "seed",
    "n_samples",
    "target_sha256",
    "target_edges",
    "alpha",
    "max_cond_set_size",
    "sparsity_bound",
    "max_path_length",
    "sepset_selection",
    "orientation_strategy",
    "skeleton_f1",
    "exact_edge_f1",
    "semantic_edge_f1",
    "endpoint_accuracy",
    "ci_test_count",
    "xy_present",
    "xy_separator",
    "xy_separator_source",
    "exact_target_recovered",
    "status",
)

EXACT_SETTINGS: dict[str, Any] = {
    "max_cond_set_size": None,
    "sparsity_bound": None,
    "max_path_length": None,
    "sepset_selection": "first",
    "orientation_strategy": "standard",
}

FINITE_SETTINGS: dict[str, Any] = {
    "alpha": 0.001,
    "max_cond_set_size": 3,
    "sparsity_bound": 3,
    "max_path_length": None,
    "sepset_selection": "first",
    "orientation_strategy": "standard",
}


def generate_oracle_validation_summary(
    output_path: Path = DEFAULT_OUTPUT,
) -> Path:
    """Execute and serialize the frozen validation rows deterministically."""

    mag = canonical_dsep_mag()
    target_shape = mag.oracle_shape()
    target_sha256 = _target_sha256(target_shape)
    rows: list[dict[str, str]] = []

    for algorithm_name, algorithm in (("fci", fci), ("fci_plus", fci_plus)):
        result = algorithm(
            mag.dummy_data(),
            ci_test=mag.oracle_ci_test(),
            **EXACT_SETTINGS,
        )
        rows.append(
            _result_row(
                result=result,
                target_shape=target_shape,
                target_sha256=target_sha256,
                regime="exact_oracle",
                algorithm=algorithm_name,
                profile="unbounded_exact_oracle_standard_orientation",
                ci_test="mag_m_separation_oracle",
                seed=None,
                n_samples=None,
                settings=EXACT_SETTINGS,
            )
        )

    for n_samples in (5_000, 50_000):
        data = sample_canonical_dsep_data(n_samples=n_samples, seed=1)
        for algorithm_name, algorithm in (("fci", fci), ("fci_plus", fci_plus)):
            result = algorithm(data, **FINITE_SETTINGS)
            rows.append(
                _result_row(
                    result=result,
                    target_shape=target_shape,
                    target_sha256=target_sha256,
                    regime="finite_sample",
                    algorithm=algorithm_name,
                    profile="advisor_showcase_fixed_fisher_z",
                    ci_test="fisher_z",
                    seed=1,
                    n_samples=n_samples,
                    settings=FINITE_SETTINGS,
                )
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _result_row(
    *,
    result: FCIResult,
    target_shape: dict[tuple[str, str], tuple[str, str]],
    target_sha256: str,
    regime: str,
    algorithm: str,
    profile: str,
    ci_test: str,
    seed: int | None,
    n_samples: int | None,
    settings: dict[str, Any],
) -> dict[str, str]:
    learned_shape = shape_from_pag(result.graph)
    exact = compare_pag_shapes(target_shape, learned_shape)
    semantic = compare_pag_shapes_semantic(target_shape, learned_shape)
    xy_key = ("X", "Y")
    separator = sorted(result.sepsets.get(xy_key, set()))
    exact_recovered = exact.exact_edge_f1 == 1.0
    return {
        "fixture_id": "claassen_2013_figure4b_mag",
        "regime": regime,
        "algorithm": algorithm,
        "profile": profile,
        "ci_test": ci_test,
        "seed": "" if seed is None else str(seed),
        "n_samples": "" if n_samples is None else str(n_samples),
        "target_sha256": target_sha256,
        "target_edges": str(len(target_shape)),
        "alpha": _setting(settings, "alpha"),
        "max_cond_set_size": _setting(settings, "max_cond_set_size"),
        "sparsity_bound": _setting(settings, "sparsity_bound"),
        "max_path_length": _setting(settings, "max_path_length"),
        "sepset_selection": str(settings["sepset_selection"]),
        "orientation_strategy": str(settings["orientation_strategy"]),
        "skeleton_f1": f"{exact.skeleton_f1:.6f}",
        "exact_edge_f1": f"{exact.exact_edge_f1:.6f}",
        "semantic_edge_f1": f"{semantic.semantic_edge_f1:.6f}",
        "endpoint_accuracy": f"{exact.endpoint_accuracy:.6f}",
        "ci_test_count": str(result.ci_test_count),
        "xy_present": str(xy_key in learned_shape).lower(),
        "xy_separator": "|".join(separator),
        "xy_separator_source": result.sepset_sources.get(xy_key, ""),
        "exact_target_recovered": str(exact_recovered).lower(),
        "status": "completed",
    }


def _setting(settings: dict[str, Any], key: str) -> str:
    if key not in settings:
        return "not_applicable"
    value = settings.get(key)
    return "unbounded" if value is None else str(value)


def _target_sha256(
    shape: dict[tuple[str, str], tuple[str, str]],
) -> str:
    records = [
        {
            "x": x,
            "y": y,
            "endpoint_x": endpoint_x,
            "endpoint_y": endpoint_y,
        }
        for (x, y), (endpoint_x, endpoint_y) in sorted(shape.items())
    ]
    payload = json.dumps(
        records,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = generate_oracle_validation_summary(args.output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
