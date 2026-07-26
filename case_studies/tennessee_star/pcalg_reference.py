"""Load and optionally refresh the independent CRAN pcalg FCI+ benchmark."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from case_studies.tennessee_star.study import (
    OUTPUT_DIR,
    PROCESSED_DIR,
    TEMPORAL_TIERS,
    PreparedStudy,
)

HERE = Path(__file__).resolve().parent
R_SCRIPT = HERE / "pcalg_reference.R"
PCALG_RUNS_PATH = OUTPUT_DIR / "star_pcalg_runs.csv"
PCALG_EDGES_PATH = OUTPUT_DIR / "star_pcalg_edges.csv"
PCALG_ORDER_AUDIT_PATH = OUTPUT_DIR / "star_pcalg_order_audit.csv"
PCALG_ALGORITHM = "pcalg_fci_plus"


def find_rscript(explicit: Path | None = None) -> Path | None:
    """Return an available Rscript executable."""

    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Rscript does not exist: {candidate}")
        return candidate

    discovered = shutil.which("Rscript")
    if discovered is not None:
        return Path(discovered)
    for candidate in (
        Path("/opt/homebrew/bin/Rscript"),
        Path("/usr/local/bin/Rscript"),
        Path("/Library/Frameworks/R.framework/Resources/bin/Rscript"),
    ):
        if candidate.exists():
            return candidate
    return None


def refresh_pcalg_reference(
    *,
    output_directory: Path = OUTPUT_DIR,
    rscript: Path | None = None,
    alpha: float = 0.05,
    benchmark_repeats: int = 3,
    order_audit: bool = True,
    timeout: int = 900,
) -> None:
    """Run the committed R benchmark and replace its CSV artifacts."""

    executable = find_rscript(rscript)
    if executable is None:
        raise RuntimeError(
            "Rscript is not available. Install R and CRAN pcalg, or pass "
            "--rscript explicitly."
        )
    command = [
        str(executable),
        str(R_SCRIPT),
        "--input-dir",
        str(PROCESSED_DIR),
        "--output-dir",
        str(output_directory),
        "--alpha",
        str(alpha),
        "--benchmark-repeats",
        str(benchmark_repeats),
    ]
    if not order_audit:
        command.append("--no-order-audit")
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"pcalg FCI+ benchmark failed: {details[-2000:]}")


def load_pcalg_reference(
    study: PreparedStudy,
    *,
    output_directory: Path = OUTPUT_DIR,
    expected_alpha: float = 0.05,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load validated pcalg run records for inclusion in the report payload."""

    runs_path = output_directory / PCALG_RUNS_PATH.name
    edges_path = output_directory / PCALG_EDGES_PATH.name
    audit_path = output_directory / PCALG_ORDER_AUDIT_PATH.name
    missing = [
        path for path in (runs_path, edges_path, audit_path) if not path.exists()
    ]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"Missing pcalg benchmark artifacts: {names}. Run "
            "`Rscript case_studies/tennessee_star/pcalg_reference.R`."
        )

    run_frame = pd.read_csv(runs_path)
    edge_frame = pd.read_csv(edges_path)
    audit_frame = pd.read_csv(audit_path)
    records: list[dict[str, Any]] = []
    for row in run_frame.to_dict(orient="records"):
        panel_name = str(row["panel"])
        if panel_name not in study.panels:
            raise ValueError(f"Unknown pcalg panel: {panel_name}")
        panel = study.panels[panel_name]
        if int(row["samples"]) != len(panel.data):
            raise ValueError(f"Stale pcalg sample count for {panel_name}.")
        if int(row["nodes"]) != len(panel.data.columns):
            raise ValueError(f"Stale pcalg node count for {panel_name}.")
        if abs(float(row["alpha"]) - expected_alpha) > 1e-12:
            raise ValueError(f"Stale pcalg alpha for {panel_name}.")

        panel_edges = edge_frame.loc[edge_frame["panel"] == panel_name, :]
        edge_records = [
            {
                "x": str(edge["x"]),
                "y": str(edge["y"]),
                "endpoint_x": str(edge["endpoint_x"]),
                "endpoint_y": str(edge["endpoint_y"]),
                "edge": str(edge["edge"]),
                "bootstrap_frequency": None,
            }
            for edge in panel_edges.to_dict(orient="records")
        ]
        temporal_flags = _temporal_flags(edge_records)
        panel_audit = audit_frame.loc[audit_frame["panel"] == panel_name, :]
        elapsed_runs = [
            float(value)
            for value in str(row["elapsed_runs_seconds"]).split(";")
            if value
        ]
        warnings = _optional_text(row.get("warnings"))
        records.append(
            {
                "panel": panel_name,
                "algorithm": PCALG_ALGORITHM,
                "samples": int(row["samples"]),
                "nodes": int(row["nodes"]),
                "node_names": list(panel.data.columns),
                "edges": int(row["edges"]),
                "ci_tests": int(row["ci_tests"]),
                "median_elapsed_seconds": float(row["median_elapsed_seconds"]),
                "elapsed_runs_seconds": elapsed_runs,
                "temporal_flags": temporal_flags,
                "pag_edges": edge_records,
                "bootstrap_adjacencies": [],
                "assumption_notes": [
                    "Independent reference implementation from CRAN pcalg; "
                    "external package output is not treated as ground truth.",
                    "The R runner supplies the same compact-table G-square "
                    "decision rule as the Python implementation.",
                    "pcalg::fciPlus does not expose the local implementation's "
                    "explicit k sparsity-bound parameter.",
                ],
                "implementation": {
                    "language": "R",
                    "r_version": str(row["r_version"]),
                    "package": "pcalg",
                    "package_version": str(row["pcalg_version"]),
                    "ci_test": str(row["ci_test"]),
                    "selection_bias": bool(row["selection_bias"]),
                    "warning_count": int(row["warning_count"]),
                    "warnings": warnings,
                },
                "order_audit": {
                    "orderings_checked": int(row["orderings_checked"]),
                    "exact_pag_match_rate": float(row["exact_pag_match_rate"]),
                    "mean_skeleton_jaccard": float(row["mean_skeleton_jaccard"]),
                    "rows": [
                        {
                            "ordering": str(item["ordering"]),
                            "first_variable": str(item["first_variable"]),
                            "exact_pag_match": bool(item["exact_pag_match"]),
                            "skeleton_jaccard": float(item["skeleton_jaccard"]),
                            "edges": int(item["edges"]),
                            "ci_tests": int(item["ci_tests"]),
                            "elapsed_seconds": float(item["elapsed_seconds"]),
                        }
                        for item in panel_audit.to_dict(orient="records")
                    ],
                },
            }
        )

    metadata = {
        "algorithm": PCALG_ALGORITHM,
        "r_script": _portable_path(R_SCRIPT),
        "runs_csv": _portable_path(runs_path),
        "edges_csv": _portable_path(edges_path),
        "order_audit_csv": _portable_path(audit_path),
        "r_version": str(run_frame.iloc[0]["r_version"]),
        "pcalg_version": str(run_frame.iloc[0]["pcalg_version"]),
        "ci_test": str(run_frame.iloc[0]["ci_test"]),
        "alpha": float(run_frame.iloc[0]["alpha"]),
    }
    return records, metadata


def _temporal_flags(edges: list[dict[str, Any]]) -> list[str]:
    flags = []
    for edge in edges:
        x = str(edge["x"])
        y = str(edge["y"])
        endpoint_x = str(edge["endpoint_x"])
        endpoint_y = str(edge["endpoint_y"])
        if endpoint_x == "TAIL" and endpoint_y == "ARROW":
            source, target = x, y
        elif endpoint_x == "ARROW" and endpoint_y == "TAIL":
            source, target = y, x
        else:
            continue
        if TEMPORAL_TIERS.get(source, 0) > TEMPORAL_TIERS.get(target, 0):
            flags.append(f"{source} --> {target}")
    return sorted(flags)


def _optional_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _portable_path(path: Path) -> str:
    repository_root = HERE.parents[1]
    try:
        return str(path.resolve().relative_to(repository_root))
    except ValueError:
        return str(path.resolve())
