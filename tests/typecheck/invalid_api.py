"""MyPy fixture that must fail on misspelled or profile-invalid options."""

import numpy as np

from fci_engine import fci, fci_plus


data = np.zeros((20, 3))

fci(data, aplha=0.01)
fci_plus(data, profile="paper", max_cond_set_size=2)
