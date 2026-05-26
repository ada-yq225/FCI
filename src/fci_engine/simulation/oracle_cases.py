"""Preset oracle cases for FCI/FCI+ benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from fci_engine.metrics.accuracy import Shape


@dataclass(frozen=True)
class OracleCase:
    """Synthetic data with a hand-written expected PAG shape."""

    name: str
    data: pd.DataFrame
    oracle_shape: Shape
    alpha: float = 0.001
    max_cond_set_size: Optional[int] = 2
    max_path_length: Optional[int] = 3
    use_kernel_ci: bool = False
    notes: str = ""


def make_independent_noise_case(
    n_samples: int = 8000,
    n_variables: int = 6,
    seed: int = 0,
) -> OracleCase:
    """Independent Gaussian variables; the expected PAG is empty."""

    rng = np.random.default_rng(seed)
    columns = [f"X{index}" for index in range(n_variables)]
    data = pd.DataFrame(rng.normal(size=(n_samples, n_variables)), columns=columns)
    return OracleCase(
        name="independent_noise",
        data=data,
        oracle_shape={},
        alpha=0.001,
        max_cond_set_size=2,
        max_path_length=3,
        notes="All observed variables are mutually independent.",
    )


def make_latent_medical_case(
    n_samples: int = 8000,
    seed: int = 42,
) -> OracleCase:
    """Small latent-confounder graph with instruments and one directed effect."""

    rng = np.random.default_rng(seed)
    hidden = rng.normal(size=n_samples)
    x1 = rng.normal(size=n_samples)
    x2 = rng.normal(size=n_samples)
    a = 0.8 * x1 + 0.8 * hidden + rng.normal(scale=0.4, size=n_samples)
    b = 0.8 * x2 + 0.8 * hidden + rng.normal(scale=0.4, size=n_samples)
    d = 0.8 * a + rng.normal(scale=0.4, size=n_samples)
    data = pd.DataFrame({"X1": x1, "X2": x2, "A": a, "B": b, "D": d})
    return OracleCase(
        name="latent_medical",
        data=data,
        oracle_shape={
            ("X1", "A"): ("CIRCLE", "ARROW"),
            ("X2", "B"): ("CIRCLE", "ARROW"),
            ("A", "B"): ("ARROW", "ARROW"),
            ("A", "D"): ("TAIL", "ARROW"),
        },
        alpha=0.001,
        max_cond_set_size=2,
        max_path_length=3,
        notes="X1 -> A, X2 -> B, latent H confounds A/B, and A -> D.",
    )


def make_nonlinear_common_cause_case(
    n_samples: int = 260,
    seed: int = 23,
) -> OracleCase:
    """Nonlinear common cause where Fisher-Z keeps a false X-Y edge."""

    rng = np.random.default_rng(seed)
    z = rng.uniform(-2.0, 2.0, size=n_samples)
    shared_nonlinear_signal = z**2
    x = 0.4 * z + shared_nonlinear_signal + rng.normal(scale=0.25, size=n_samples)
    y = -0.4 * z + shared_nonlinear_signal + rng.normal(scale=0.25, size=n_samples)
    data = pd.DataFrame({"Z": z, "X": x, "Y": y})
    return OracleCase(
        name="nonlinear_common_cause",
        data=data,
        oracle_shape={
            ("Z", "X"): ("CIRCLE", "CIRCLE"),
            ("Z", "Y"): ("CIRCLE", "CIRCLE"),
        },
        alpha=0.05,
        max_cond_set_size=1,
        max_path_length=2,
        use_kernel_ci=True,
        notes="Known graph Z -> X and Z -> Y with nonlinear mechanisms.",
    )


def make_sparse_latent_case(
    n_samples: int = 12000,
    seed: int = 7,
) -> OracleCase:
    """Sparse graph with two latent confounders and three instruments."""

    rng = np.random.default_rng(seed)
    u1 = rng.normal(size=n_samples)
    u2 = rng.normal(size=n_samples)
    i1 = rng.normal(size=n_samples)
    i2 = rng.normal(size=n_samples)
    i3 = rng.normal(size=n_samples)

    a = 0.8 * i1 + 0.7 * u1 + rng.normal(scale=0.35, size=n_samples)
    b = 0.7 * i2 + 0.7 * u1 + 0.6 * u2 + rng.normal(scale=0.35, size=n_samples)
    c = 0.8 * i3 + 0.7 * u2 + rng.normal(scale=0.35, size=n_samples)
    d = 0.7 * a + 0.5 * b + rng.normal(scale=0.4, size=n_samples)
    e = 0.7 * c + 0.5 * d + rng.normal(scale=0.4, size=n_samples)
    data = pd.DataFrame(
        {"I1": i1, "I2": i2, "I3": i3, "A": a, "B": b, "C": c, "D": d, "E": e}
    )
    return OracleCase(
        name="sparse_two_latent",
        data=data,
        oracle_shape={
            ("I1", "A"): ("CIRCLE", "ARROW"),
            ("I2", "B"): ("CIRCLE", "ARROW"),
            ("I3", "C"): ("CIRCLE", "ARROW"),
            ("A", "B"): ("ARROW", "ARROW"),
            ("B", "C"): ("ARROW", "ARROW"),
            ("A", "D"): ("CIRCLE", "ARROW"),
            ("B", "D"): ("CIRCLE", "ARROW"),
            ("C", "E"): ("CIRCLE", "ARROW"),
            ("D", "E"): ("TAIL", "ARROW"),
        },
        alpha=0.001,
        max_cond_set_size=3,
        max_path_length=4,
        notes="Sparse graph with instruments, latent confounders, and a directed chain.",
    )


def default_oracle_cases() -> list[OracleCase]:
    """Return the default benchmark suite."""

    return [
        make_independent_noise_case(),
        make_latent_medical_case(),
        make_nonlinear_common_cause_case(),
        make_sparse_latent_case(),
    ]
