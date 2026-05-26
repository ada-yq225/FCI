import numpy as np
import pandas as pd
import fci_engine
from fci_engine.ci.fisher_z import FisherZTest

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

test = FisherZTest(alpha=0.01)
data, _ = fci_engine.utils.validation.validate_numeric_data(df)
print("B ⊥ D | A:", test.test(data, 1, 3, (0,)))
print("A ⊥ C | empty:", test.test(data, 0, 2, ()))
print("D ⊥ C | empty:", test.test(data, 3, 2, ()))
