import numpy as np
import pandas as pd
from fci_engine import fci

np.random.seed(42)
n = 5000
U1 = np.random.normal(0, 1, n)
U2 = np.random.normal(0, 1, n)
A = 0.8 * U1 + np.random.normal(0, 0.4, n)
B = 0.7 * U1 + 0.7 * U2 + np.random.normal(0, 0.4, n)
C = 0.8 * U2 + np.random.normal(0, 0.4, n)
D = 0.8 * A + np.random.normal(0, 0.4, n)
E = 0.7 * D + 0.8 * C + np.random.normal(0, 0.4, n)
df = pd.DataFrame({"A": A, "B": B, "C": C, "D": D, "E": E})

print("Alpha = 0.001")
res = fci(df, alpha=0.001)
for x, y in res.graph.edges():
    print(res.graph.edge_repr(x, y))
