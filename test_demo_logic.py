import numpy as np
import pandas as pd
from fci_engine import fci

n = 2500
np.random.seed(42)

I1 = np.random.normal(0, 1, n)
I2 = np.random.normal(0, 1, n)
U1 = np.random.normal(0, 1, n)

A = 0.8 * I1 + 0.8 * U1 + np.random.normal(0, 0.4, n)
B = 0.8 * I2 + 0.8 * U1 + np.random.normal(0, 0.4, n)
D = 0.8 * A + np.random.normal(0, 0.4, n)

df = pd.DataFrame({"I1": I1, "I2": I2, "A": A, "B": B, "D": D})

result = fci(df, alpha="auto")
for x, y in result.graph.edges():
    print(f"{x} {y}: {result.graph.edge_repr(x, y)}")

