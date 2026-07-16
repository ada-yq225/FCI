"""Preset oracle cases for FCI/FCI+ benchmarking."""

from __future__ import annotations

from collections.abc import Callable
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
    sparsity_bound: Optional[int] = None
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
        sparsity_bound=2,
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
        sparsity_bound=2,
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
        sparsity_bound=1,
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
        sparsity_bound=3,
        max_path_length=4,
        notes="Sparse graph with instruments, latent confounders, and a directed chain.",
    )


def make_hospital_triage_case(
    n_samples: int = 9000,
    seed: int = 101,
) -> OracleCase:
    """Clinical triage-style graph with latent illness severity."""

    rng = np.random.default_rng(seed)
    latent_severity = rng.normal(size=n_samples)
    genetic_risk = rng.normal(size=n_samples)
    protocol = rng.normal(size=n_samples)
    comorbidity = rng.normal(size=n_samples)

    inflammation = (
        0.75 * genetic_risk
        + 0.70 * latent_severity
        + 0.35 * comorbidity
        + rng.normal(scale=0.35, size=n_samples)
    )
    oxygen = 0.75 * latent_severity + rng.normal(scale=0.35, size=n_samples)
    treatment = (
        0.75 * protocol + 0.55 * inflammation + rng.normal(scale=0.40, size=n_samples)
    )
    recovery = 0.70 * treatment - 0.55 * oxygen + rng.normal(scale=0.40, size=n_samples)

    data = pd.DataFrame(
        {
            "GeneticRisk": genetic_risk,
            "Protocol": protocol,
            "Comorbidity": comorbidity,
            "Inflammation": inflammation,
            "Oxygen": oxygen,
            "Treatment": treatment,
            "Recovery": recovery,
        }
    )
    return OracleCase(
        name="hospital_triage",
        data=data,
        oracle_shape={
            ("GeneticRisk", "Inflammation"): ("CIRCLE", "ARROW"),
            ("Comorbidity", "Inflammation"): ("CIRCLE", "ARROW"),
            ("Protocol", "Treatment"): ("CIRCLE", "ARROW"),
            ("Inflammation", "Oxygen"): ("ARROW", "ARROW"),
            ("Inflammation", "Treatment"): ("CIRCLE", "ARROW"),
            ("Oxygen", "Recovery"): ("CIRCLE", "ARROW"),
            ("Treatment", "Recovery"): ("TAIL", "ARROW"),
        },
        alpha=0.001,
        max_cond_set_size=3,
        sparsity_bound=3,
        max_path_length=4,
        notes=(
            "Observed risk/protocol variables, latent illness severity "
            "confounding inflammation and oxygen, and treatment -> recovery."
        ),
    )


def make_microservice_incident_case(
    n_samples: int = 10000,
    seed: int = 202,
) -> OracleCase:
    """Microservice monitoring graph with a hidden network incident."""

    rng = np.random.default_rng(seed)
    hidden_network = rng.normal(size=n_samples)
    traffic = rng.normal(size=n_samples)
    deploy = rng.normal(size=n_samples)
    cache = rng.normal(size=n_samples)

    queue = 0.80 * traffic + rng.normal(scale=0.35, size=n_samples)
    auth_latency = (
        0.75 * queue + 0.65 * hidden_network + rng.normal(scale=0.35, size=n_samples)
    )
    payment_latency = (
        0.70 * cache + 0.65 * hidden_network + rng.normal(scale=0.35, size=n_samples)
    )
    error_rate = (
        0.65 * auth_latency
        + 0.55 * payment_latency
        + 0.45 * deploy
        + rng.normal(scale=0.40, size=n_samples)
    )

    data = pd.DataFrame(
        {
            "Traffic": traffic,
            "Deploy": deploy,
            "CacheHit": cache,
            "QueueDepth": queue,
            "AuthLatency": auth_latency,
            "PaymentLatency": payment_latency,
            "ErrorRate": error_rate,
        }
    )
    return OracleCase(
        name="microservice_incident",
        data=data,
        oracle_shape={
            ("Traffic", "QueueDepth"): ("CIRCLE", "ARROW"),
            ("CacheHit", "PaymentLatency"): ("CIRCLE", "ARROW"),
            ("Deploy", "ErrorRate"): ("CIRCLE", "ARROW"),
            ("QueueDepth", "AuthLatency"): ("CIRCLE", "ARROW"),
            ("AuthLatency", "PaymentLatency"): ("ARROW", "ARROW"),
            ("AuthLatency", "ErrorRate"): ("CIRCLE", "ARROW"),
            ("PaymentLatency", "ErrorRate"): ("CIRCLE", "ARROW"),
        },
        alpha=0.001,
        max_cond_set_size=3,
        sparsity_bound=3,
        max_path_length=4,
        notes=(
            "Traffic and cache affect service metrics while an unobserved "
            "network incident confounds auth/payment latency."
        ),
    )


def make_finance_risk_case(
    n_samples: int = 10000,
    seed: int = 303,
) -> OracleCase:
    """Financial risk graph with a hidden market factor."""

    rng = np.random.default_rng(seed)
    market_factor = rng.normal(size=n_samples)
    rate_shock = rng.normal(size=n_samples)
    liquidity = rng.normal(size=n_samples)
    leverage = rng.normal(size=n_samples)

    credit_spread = (
        0.75 * rate_shock
        + 0.65 * market_factor
        + rng.normal(scale=0.35, size=n_samples)
    )
    equity_vol = (
        0.70 * liquidity + 0.65 * market_factor + rng.normal(scale=0.35, size=n_samples)
    )
    default_prob = (
        0.70 * credit_spread + 0.55 * leverage + rng.normal(scale=0.40, size=n_samples)
    )
    loss = (
        0.75 * default_prob
        + 0.45 * equity_vol
        + rng.normal(
            scale=0.40,
            size=n_samples,
        )
    )

    data = pd.DataFrame(
        {
            "RateShock": rate_shock,
            "Liquidity": liquidity,
            "Leverage": leverage,
            "CreditSpread": credit_spread,
            "EquityVol": equity_vol,
            "DefaultProb": default_prob,
            "Loss": loss,
        }
    )
    return OracleCase(
        name="finance_risk",
        data=data,
        oracle_shape={
            ("RateShock", "CreditSpread"): ("CIRCLE", "ARROW"),
            ("Liquidity", "EquityVol"): ("CIRCLE", "ARROW"),
            ("Leverage", "DefaultProb"): ("CIRCLE", "ARROW"),
            ("CreditSpread", "EquityVol"): ("ARROW", "ARROW"),
            ("CreditSpread", "DefaultProb"): ("CIRCLE", "ARROW"),
            ("DefaultProb", "Loss"): ("CIRCLE", "ARROW"),
            ("EquityVol", "Loss"): ("CIRCLE", "ARROW"),
        },
        alpha=0.001,
        max_cond_set_size=3,
        sparsity_bound=3,
        max_path_length=4,
        notes="Market factor is hidden; credit/equity risk share latent exposure.",
    )


def make_manufacturing_quality_case(
    n_samples: int = 9000,
    seed: int = 404,
) -> OracleCase:
    """Manufacturing quality graph with hidden ambient conditions."""

    rng = np.random.default_rng(seed)
    ambient = rng.normal(size=n_samples)
    setpoint = rng.normal(size=n_samples)
    maintenance = rng.normal(size=n_samples)
    operator = rng.normal(size=n_samples)

    chamber_temp = (
        0.75 * setpoint + 0.65 * ambient + rng.normal(scale=0.35, size=n_samples)
    )
    vibration = (
        0.70 * maintenance + 0.65 * ambient + rng.normal(scale=0.35, size=n_samples)
    )
    pressure = 0.70 * operator + rng.normal(scale=0.35, size=n_samples)
    defect_rate = (
        0.65 * chamber_temp
        + 0.55 * vibration
        + 0.45 * pressure
        + rng.normal(scale=0.40, size=n_samples)
    )

    data = pd.DataFrame(
        {
            "Setpoint": setpoint,
            "Maintenance": maintenance,
            "Operator": operator,
            "ChamberTemp": chamber_temp,
            "Vibration": vibration,
            "Pressure": pressure,
            "DefectRate": defect_rate,
        }
    )
    return OracleCase(
        name="manufacturing_quality",
        data=data,
        oracle_shape={
            ("Setpoint", "ChamberTemp"): ("CIRCLE", "ARROW"),
            ("Maintenance", "Vibration"): ("CIRCLE", "ARROW"),
            ("Operator", "Pressure"): ("CIRCLE", "ARROW"),
            ("ChamberTemp", "Vibration"): ("ARROW", "ARROW"),
            ("ChamberTemp", "DefectRate"): ("CIRCLE", "ARROW"),
            ("Vibration", "DefectRate"): ("CIRCLE", "ARROW"),
            ("Pressure", "DefectRate"): ("CIRCLE", "ARROW"),
        },
        alpha=0.001,
        max_cond_set_size=3,
        sparsity_bound=3,
        max_path_length=4,
        notes="Unobserved ambient conditions confound temperature and vibration.",
    )


def make_enterprise_monitoring_case(
    n_samples: int = 12000,
    seed: int = 505,
) -> OracleCase:
    """Larger monitoring graph with two hidden operational factors."""

    rng = np.random.default_rng(seed)
    platform_load = rng.normal(size=n_samples)
    network_jitter = rng.normal(size=n_samples)
    deploy_a = rng.normal(size=n_samples)
    deploy_b = rng.normal(size=n_samples)
    marketing = rng.normal(size=n_samples)
    cache_policy = rng.normal(size=n_samples)

    api_cpu = (
        0.75 * deploy_a + 0.65 * platform_load + rng.normal(scale=0.35, size=n_samples)
    )
    db_cpu = (
        0.70 * deploy_b + 0.65 * platform_load + rng.normal(scale=0.35, size=n_samples)
    )
    search_qps = 0.75 * marketing + rng.normal(scale=0.35, size=n_samples)
    cache_miss = 0.75 * cache_policy + rng.normal(scale=0.35, size=n_samples)
    api_latency = (
        0.65 * api_cpu
        + 0.45 * search_qps
        + 0.65 * network_jitter
        + rng.normal(scale=0.40, size=n_samples)
    )
    db_latency = (
        0.65 * db_cpu
        + 0.45 * cache_miss
        + 0.65 * network_jitter
        + rng.normal(scale=0.40, size=n_samples)
    )
    checkout_latency = (
        0.60 * api_latency + 0.60 * db_latency + rng.normal(scale=0.40, size=n_samples)
    )
    error_budget = 0.75 * checkout_latency + rng.normal(
        scale=0.40,
        size=n_samples,
    )

    data = pd.DataFrame(
        {
            "DeployA": deploy_a,
            "DeployB": deploy_b,
            "Marketing": marketing,
            "CachePolicy": cache_policy,
            "ApiCPU": api_cpu,
            "DbCPU": db_cpu,
            "SearchQPS": search_qps,
            "CacheMiss": cache_miss,
            "ApiLatency": api_latency,
            "DbLatency": db_latency,
            "CheckoutLatency": checkout_latency,
            "ErrorBudget": error_budget,
        }
    )
    return OracleCase(
        name="enterprise_monitoring",
        data=data,
        oracle_shape={
            ("DeployA", "ApiCPU"): ("CIRCLE", "ARROW"),
            ("DeployB", "DbCPU"): ("CIRCLE", "ARROW"),
            ("Marketing", "SearchQPS"): ("CIRCLE", "ARROW"),
            ("CachePolicy", "CacheMiss"): ("CIRCLE", "ARROW"),
            ("ApiCPU", "DbCPU"): ("ARROW", "ARROW"),
            ("ApiCPU", "ApiLatency"): ("CIRCLE", "ARROW"),
            ("DbCPU", "DbLatency"): ("CIRCLE", "ARROW"),
            ("SearchQPS", "ApiLatency"): ("CIRCLE", "ARROW"),
            ("CacheMiss", "DbLatency"): ("CIRCLE", "ARROW"),
            ("ApiLatency", "DbLatency"): ("ARROW", "ARROW"),
            ("ApiLatency", "CheckoutLatency"): ("CIRCLE", "ARROW"),
            ("DbLatency", "CheckoutLatency"): ("CIRCLE", "ARROW"),
            ("CheckoutLatency", "ErrorBudget"): ("TAIL", "ARROW"),
        },
        alpha=0.001,
        max_cond_set_size=3,
        sparsity_bound=3,
        max_path_length=4,
        notes=(
            "Larger operational graph with hidden platform load and network "
            "jitter creating two latent-confounded metric pairs."
        ),
    )


def realistic_oracle_cases(
    n_repeats: int = 1,
    n_samples: Optional[int] = None,
) -> list[OracleCase]:
    """Return realistic synthetic benchmark cases with hand-written PAG shapes."""

    factories: list[Callable[..., OracleCase]] = [
        make_hospital_triage_case,
        make_microservice_incident_case,
        make_finance_risk_case,
        make_manufacturing_quality_case,
        make_enterprise_monitoring_case,
    ]
    cases: list[OracleCase] = []
    for repeat in range(n_repeats):
        seed_offset = 1000 * repeat
        for factory in factories:
            kwargs: dict[str, int] = {
                "seed": seed_offset + _base_seed_for_factory(factory)
            }
            if n_samples is not None:
                kwargs["n_samples"] = n_samples
            case = factory(**kwargs)
            if n_repeats > 1:
                case = OracleCase(
                    name=f"{case.name}_r{repeat + 1}",
                    data=case.data,
                    oracle_shape=case.oracle_shape,
                    alpha=case.alpha,
                    max_cond_set_size=case.max_cond_set_size,
                    sparsity_bound=case.sparsity_bound,
                    max_path_length=case.max_path_length,
                    use_kernel_ci=case.use_kernel_ci,
                    notes=case.notes,
                )
            cases.append(case)
    return cases


def default_oracle_cases() -> list[OracleCase]:
    """Return the default benchmark suite."""

    return [
        make_independent_noise_case(),
        make_latent_medical_case(),
        make_nonlinear_common_cause_case(),
        make_sparse_latent_case(),
    ]


def _base_seed_for_factory(factory: Callable[..., OracleCase]) -> int:
    seeds: dict[Callable[..., OracleCase], int] = {
        make_hospital_triage_case: 101,
        make_microservice_incident_case: 202,
        make_finance_risk_case: 303,
        make_manufacturing_quality_case: 404,
        make_enterprise_monitoring_case: 505,
    }
    return seeds[factory]
