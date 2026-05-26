import numpy as np
import pandas as pd
from src.fci_engine import fci

def generate_y_structure(n=5000):
    # D -> A -> B -> C
    # L -> A
    # L -> C
    # Ground truth for observed (D, A, B, C):
    # D -> A <-> C
    # A -> B -> C
    # So A is a collider for D and C? No, D -> A, L->A, L->C (which means A <-> C)
    # Wait, simple Y structure to find <->
    
    # Let's do: X1 -> X2, X3 -> X2, X2 -> X4
    # With a latent confounder between X1 and X3.
    # U -> X1
    # U -> X3
    # X1 -> X2
    # X3 -> X2
    # X2 -> X4
    # Observed: X1, X3, X2, X4
    # Unshielded colliders: X1 -> X2 <- X3. 
    # But X1 and X3 are correlated due to U. So X1 and X3 are connected!
    # Wait, if U -> X1, U -> X3, then X1 and X3 are adjacent in the PAG (X1 <-> X3).
    # Then X1, X2, X3 is a shielded triple. Not an unshielded collider.
    # To get a <-> edge between say A and B, we need:
    # U -> A, U -> B.
    # Plus, A needs an instrumental variable I1 -> A. B needs I2 -> B.
    # Then I1 -> A <-> B <- I2.
    # Let's simulate:
    # I1 -> A
    # I2 -> B
    # U -> A
    # U -> B
    
    np.random.seed(42)
    U = np.random.normal(0, 1, n)
    I1 = np.random.normal(0, 1, n)
    I2 = np.random.normal(0, 1, n)
    
    A = 0.8 * I1 + 0.9 * U + np.random.normal(0, 0.5, n)
    B = 0.8 * I2 + 0.9 * U + np.random.normal(0, 0.5, n)
    
    df = pd.DataFrame({"I1": I1, "I2": I2, "A": A, "B": B})
    return df

if __name__ == "__main__":
    df = generate_y_structure()
    result = fci(df, alpha=0.01)
    print("Latent confounder with Instruments")
    print("Expected PAG:")
    print("I1 o-> A")
    print("I2 o-> B")
    print("A <-> B")
    print("\nActual Edges:")
    for x, y in result.graph.edges():
        print(f"- {result.graph.edge_repr(x, y)}")
