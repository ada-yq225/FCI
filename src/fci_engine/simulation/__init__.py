"""Simulation utilities for synthetic causal data."""

from fci_engine.simulation.oracle_graphs import CausalGraphSpec
from fci_engine.simulation.mag_oracle import (
    MAGOracleCITest,
    MAGSpec,
    canonical_dsep_mag,
    sample_canonical_dsep_data,
    spirtes_latent_reference_mag,
    zhang_orientation_reference_mag,
)
from fci_engine.simulation.oracle_cases import (
    OracleCase,
    default_oracle_cases,
    make_independent_noise_case,
    make_latent_medical_case,
    make_hospital_triage_case,
    make_microservice_incident_case,
    make_finance_risk_case,
    make_manufacturing_quality_case,
    make_enterprise_monitoring_case,
    make_nonlinear_common_cause_case,
    make_sparse_latent_case,
    realistic_oracle_cases,
)

__all__ = [
    "OracleCase",
    "CausalGraphSpec",
    "MAGOracleCITest",
    "MAGSpec",
    "canonical_dsep_mag",
    "sample_canonical_dsep_data",
    "spirtes_latent_reference_mag",
    "zhang_orientation_reference_mag",
    "default_oracle_cases",
    "realistic_oracle_cases",
    "make_enterprise_monitoring_case",
    "make_finance_risk_case",
    "make_hospital_triage_case",
    "make_independent_noise_case",
    "make_latent_medical_case",
    "make_manufacturing_quality_case",
    "make_microservice_incident_case",
    "make_nonlinear_common_cause_case",
    "make_sparse_latent_case",
]
