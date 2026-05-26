# Quickstart

This package exposes two public entry points:

- `fci_engine.fci(data, **kwargs)`
- `fci_engine.FCI(...).fit(data)`
- `fci_engine.fci_plus(data, **kwargs)`
- `fci_engine.FCIPlus(...).fit(data)`

Both return an `FCIResult` containing the learned PAG, separating sets, CI test
counts, cache hits, elapsed time, and configuration.

## Install

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Run FCI On A DataFrame

```python
import numpy as np
import pandas as pd
from fci_engine import fci

rng = np.random.default_rng(7)
n = 1000

x = rng.normal(size=n)
y = 0.8 * x + rng.normal(scale=0.5, size=n)
z = 0.8 * y + rng.normal(scale=0.5, size=n)

data = pd.DataFrame({"X": x, "Y": y, "Z": z})
result = fci(data, alpha=0.01, max_cond_set_size=2)

print(result.summary())
for a, b in result.graph.edges():
    print(result.graph.edge_repr(a, b))
```

DataFrame column names become graph node names. NumPy arrays are also accepted;
their variables are named `X0`, `X1`, `X2`, and so on.

## Configuration

```python
from fci_engine import FCI

estimator = FCI(
    alpha=0.01,
    max_cond_set_size=3,
    max_path_length=4,
    do_pdsep=True,
    skeleton_stable=True,
    verbose=False,
)
result = estimator.fit(data)
```

The default conditional independence test is Fisher-Z, intended for continuous
Gaussian-style data. Stable skeleton search is enabled by default so edge
removals within one conditioning depth do not change later candidate sets at
the same depth.

## Run FCI+

```python
from fci_engine import FCIPlus, fci_plus

result = fci_plus(data, alpha=0.01, max_cond_set_size=3)

estimator = FCIPlus(alpha=0.01, max_cond_set_size=3)
result = estimator.fit(data)
```

FCI+ uses the same `FCIResult` and PAG representation as standard FCI. The
difference is the refinement stage: standard FCI uses Possible-D-Sep search,
while FCI+ uses a sparse hierarchical D-SEP search driven by separating sets
already discovered in earlier stages.
