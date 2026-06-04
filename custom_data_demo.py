import pandas as pd
import numpy as np
from pathlib import Path

# Note: adjusting the PYTHONPATH or making sure fci_engine is importable is important.
# I'll rely on the terminal running from the project root.
from fci_engine import fci
from fci_engine.reports import render_interactive_report

def create_custom_data(n_samples=3000):
    # Setting seed for reproducibility
    np.random.seed(42)
    
    # Simulated ground-truth causal structure:
    # Latent variable Z affects both X and Y, introducing confounding.
    # Other variables:
    # V -> X (independent instrumental variable)
    # X -> Y (confounded by the latent variable Z)
    # Y -> W (observed outcome Y causes W)
    # P -> W (independent factor causing W)
    
    Z_latent = np.random.normal(0, 1, n_samples)
    
    V = np.random.normal(0, 1, n_samples)
    P = np.random.normal(0, 1, n_samples)
    
    X = 0.8 * V + 0.8 * Z_latent + np.random.normal(0, 0.3, n_samples)
    Y = 0.7 * X + 0.6 * Z_latent + np.random.normal(0, 0.3, n_samples)
    W = 0.9 * Y + 0.8 * P + np.random.normal(0, 0.4, n_samples)
    
    # Build the DataFrame with only observed variables exposed to the algorithm.
    df = pd.DataFrame({"V": V, "X": X, "Y": Y, "W": W, "P": P})
    return df

def main():
    print("1. Generating custom test data without a predefined template...")
    df = create_custom_data()
    print(f"   Data shape: {df.shape}")
    print(f"   Observed variables: {list(df.columns)}")
    
    print("\n2. Running causal inference with the FCI engine to find the PAG structure...")
    # alpha="auto" selects the significance level automatically.
    result = fci(df, alpha="auto", max_cond_set_size=3)
    
    print("\n3. Analysis complete. Summary:")
    print(result.summary())
    
    print("\nEdge endpoint representations:")
    for x, y in result.graph.edges():
        print(f"  {result.graph.edge_repr(x, y)}")

    print("\n4. Generating the interactive report...")
    report_html = render_interactive_report(result, title="Custom Data CI/PAG Analysis")
    
    output_path = Path("custom_data_report.html")
    output_path.write_text(report_html, encoding="utf-8")
    
    print(f"\nReport generated: {output_path.absolute()}")

if __name__ == "__main__":
    main()
