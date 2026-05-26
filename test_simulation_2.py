import numpy as np
import pandas as pd
from src.fci_engine import fci

def generate_complex_structure(n=5000):
    np.random.seed(42)
    # X1 -> X2 <- X3
    # X2 -> X4
    X1 = np.random.normal(0, 1, n)
    X3 = np.random.normal(0, 1, n)
    X2 = 0.8 * X1 + 0.8 * X3 + np.random.normal(0, 0.5, n)
    X4 = 0.8 * X2 + np.random.normal(0, 0.5, n)
    
    df = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3, "X4": X4})
    return df

def generate_mbias(n=5000):
    np.random.seed(42)
    # U1 -> X1, U1 -> X2
    # U2 -> X2, U2 -> X3
    U1 = np.random.normal(0, 1, n)
    U2 = np.random.normal(0, 1, n)
    X1 = 0.8 * U1 + np.random.normal(0, 0.5, n)
    X3 = 0.8 * U2 + np.random.normal(0, 0.5, n)
    X2 = 0.8 * U1 + 0.8 * U2 + np.random.normal(0, 0.5, n)
    df = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})
    return df

if __name__ == "__main__":
    df = generate_complex_structure()
    result = fci(df, alpha=0.01)
    print("Complex Structure: X1 -> X2 <- X3, X2 -> X4")
    print("Actual Edges:")
    for x, y in result.graph.edges():
        print(f"- {result.graph.edge_repr(x, y)}")
        
    df2 = generate_mbias()
    result2 = fci(df2, alpha=0.01)
    print("\nMBias Structure: U1->X1, U1->X2, U2->X2, U2->X3 (observed X1,X2,X3)")
    print("Actual Edges:")
    for x, y in result2.graph.edges():
        print(f"- {result2.graph.edge_repr(x, y)}")
