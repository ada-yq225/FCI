"""Simulation utilities for synthetic causal data."""

from fci_engine.simulation.oracle_cases import (
    OracleCase,
    default_oracle_cases,
    make_independent_noise_case,
    make_latent_medical_case,
    make_nonlinear_common_cause_case,
    make_sparse_latent_case,
)

__all__ = [
    "OracleCase",
    "default_oracle_cases",
    "make_independent_noise_case",
    "make_latent_medical_case",
    "make_nonlinear_common_cause_case",
    "make_sparse_latent_case",
]
