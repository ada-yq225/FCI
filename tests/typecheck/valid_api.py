"""MyPy fixture for supported public configuration calls."""

import numpy as np

from fci_engine import FCI, FCIPlus, fci, fci_plus


data = np.zeros((20, 3))

fci(data, alpha=0.01, max_cond_set_size=2, orientation_strategy="standard")
fci(data, profile="paper", alpha=0.01)
fci_plus(data, alpha=0.01, sparsity_bound=2)
fci_plus(data, profile="practical", max_cond_set_size=2)
fci_plus(data, profile="paper", k=2, alpha=0.01)

FCI(alpha=0.01, max_cond_set_size=2)
FCIPlus.practical(max_cond_set_size=2)
FCIPlus.paper(k=2, alpha=0.01)
