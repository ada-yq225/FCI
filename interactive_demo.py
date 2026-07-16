import numpy as np
import pandas as pd
from fci_engine import fci


EXPECTED_ADJACENCIES = {
    frozenset(("X1", "A")),
    frozenset(("X2", "B")),
    frozenset(("A", "B")),
    frozenset(("A", "D")),
}


def generate_medical_data(n: int, seed: int = 42) -> pd.DataFrame:
    # Scenario: medical data analysis.
    # U1 (latent): hidden genetic predisposition, not observed.

    # Observed data:
    # X1 (topical treatment): affects only A.
    # X2 (internal therapy): affects only B.
    # A  (surface symptom): <- X1, U1.
    # B  (internal inflammation): <- X2, U1.
    # D  (fever response): <- A.

    np.random.seed(seed)

    # Latent variable.
    U1 = np.random.normal(0, 1, n)

    # Independent intervention drivers.
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)

    A = 0.8 * X1 + 0.8 * U1 + np.random.normal(0, 0.4, n)
    B = 0.8 * X2 + 0.8 * U1 + np.random.normal(0, 0.4, n)

    D = 0.8 * A + np.random.normal(0, 0.4, n)

    return pd.DataFrame({"X1": X1, "X2": X2, "A": A, "B": B, "D": D})


def validate_result(result):
    observed_adjacencies = {frozenset(edge) for edge in result.graph.edges()}
    missing = EXPECTED_ADJACENCIES - observed_adjacencies
    extra = observed_adjacencies - EXPECTED_ADJACENCIES

    print("\nValidation results:")
    if not missing and not extra:
        print("  PASS Skeleton exactly matches: X1-A, X2-B, A-B, A-D")
    else:
        print(f"  FAIL Skeleton mismatch: missing={missing}, extra={extra}")

    checks = [
        (
            "X1 has an arrowhead into A",
            result.graph.is_adjacent("X1", "A")
            and result.graph.has_arrowhead("X1", "A"),
        ),
        (
            "X2 has an arrowhead into B",
            result.graph.is_adjacent("X2", "B")
            and result.graph.has_arrowhead("X2", "B"),
        ),
        (
            "A-B is identified as latent confounding A <-> B",
            result.graph.is_adjacent("A", "B")
            and result.graph.is_bidirected_edge("A", "B"),
        ),
        (
            "The A endpoint on A-D is a tail",
            result.graph.is_adjacent("A", "D") and result.graph.has_tail("D", "A"),
        ),
    ]

    for description, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  {status} {description}")

    if result.graph.is_adjacent("A", "D"):
        edge = result.graph.edge_repr("A", "D")
        if edge == "A --> D":
            print("  PASS A-D is fully oriented as A --> D")
        else:
            print(
                "  WARN A-D is only partially oriented as "
                f"{edge}; the causal-learn reference often gives A --> D on this data"
            )


def run_experiment(n: int, description: str):
    print(f"\n{'=' * 55}")
    print(f"Experiment: {description} (sample size N={n})")
    print(f"{'=' * 55}")

    df = generate_medical_data(n)

    print("Running the FCI engine...")
    result = fci(df, alpha="auto", verbose=False)

    print(f"Auto-selected alpha = {result.config.alpha} for N={n}")
    print("\nGraph discovery complete. Edge interpretation:\n")

    edges_found = []
    for x, y in result.graph.edges():
        edges_found.append(result.graph.edge_repr(x, y))

    for edge_str in sorted(edges_found):
        if "<->" in edge_str:
            print(f"  {edge_str:12}  <-- latent confounding signal")
        elif "->" in edge_str:
            print(f"  {edge_str:12}  <-- directed effect")
        else:
            print(f"  {edge_str:12}  <-- partially oriented or ambiguous")

    validate_result(result)


def main():
    print("Medical data topology [oracle view]:")
    print("   1. Hidden confounder U1 -> (A, B)")
    print("   2. Instrument X1 -> A")
    print("   3. Instrument X2 -> B")
    print("   4. Direct causal effect A -> D")
    print("\nExpected observed skeleton: X1-A, X2-B, A-B, A-D")
    print("Key expectation: A-B should appear as latent confounding A <-> B;")
    print("A-D is usually A --> D in the reference output.\n")

    # Use different sample-size regimes to exercise the auto alpha schedule.
    run_experiment(500, "small emergency-care sample")
    run_experiment(2500, "medium screening-database sample")
    run_experiment(8000, "large national-population sample")


if __name__ == "__main__":
    main()
