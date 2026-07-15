# Quickstart

This package exposes two public entry points:

- `fci_engine.fci(data, **kwargs)`
- `fci_engine.FCI(...).fit(data)`
- `fci_engine.fci_plus(data, **kwargs)`
- `fci_engine.FCIPlus(...).fit(data)`
- `fci_engine.stable_fci(data, **kwargs)`
- `fci_engine.stable_fci_plus(data, **kwargs)`

Both return an `FCIResult` containing the learned PAG, separating sets, CI test
counts, cache hits, elapsed time, and configuration.

## Install

Supported Python versions are 3.9, 3.10, 3.11, 3.12, and 3.13. Source builds
need `setuptools>=61`; normal modern `pip` build isolation installs that build
backend automatically.

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

result.save_interactive_report("fci_report.html")
```

DataFrame column names become graph node names. NumPy arrays are also accepted;
their variables are named `X0`, `X1`, `X2`, and so on.

The HTML report is self-contained. Open it in a browser and click any retained
PAG edge to see deterministic explanations of the endpoint marks, the
skeleton/CI evidence that kept the edge, and the orientation-rule evidence that
changed any endpoints.

## Configuration

```python
from fci_engine import FCI

estimator = FCI(
    alpha=0.01,
    max_cond_set_size=3,
    max_path_length=4,
    do_pdsep=True,
    skeleton_stable=True,
    pdsep_stable=True,
    sepset_selection="max_pvalue",
    conservative_colliders=False,
    conservative_orientation=False,
    orientation_strategy="robust",
    verbose=False,
)
result = estimator.fit(data)
```

The default conditional independence test is Fisher-Z, intended for continuous
Gaussian-style data. Stable skeleton search is enabled by default so edge
removals within one conditioning depth do not change later candidate sets at
the same depth. Stable Possible-D-Sep refinement is also enabled by default so
later PDS candidate paths use a start-of-stage PAG snapshot.
The default `sepset_selection="max_pvalue"` spends extra CI tests to keep the
strongest separating set found at the first successful conditioning depth. Use
`sepset_selection="first"` if you need traditional early-stopping behavior.
Set `conservative_colliders=True` to use Conservative-FCI-style collider
orientation and report ambiguous unshielded triples instead of forcing a
direction.
Set `conservative_orientation=True` when you want to keep arrowhead evidence
but skip tail-producing propagation rules, producing a more cautious PAG.
Set `orientation_strategy="leaf"` to use a middle ground: avoid most
tail-producing propagation in dense regions while still allowing R1 to orient
clear leaf effects.
Set `orientation_strategy="robust"` to also enable conservative collider
checks, which reduces endpoint conflicts in finite-sample settings.

## Missing Values

Default Fisher-Z rejects missing values. Use `MissingValueFisherZTest` when you
want query-wise complete-case deletion inside the public FCI or FCI+ pipelines:

```python
from fci_engine import MissingValueFisherZTest, fci

result = fci(data_with_nan, ci_test=MissingValueFisherZTest(alpha=0.01))
```

## Run FCI+

```python
from fci_engine import FCIPlus, fci_plus

result = fci_plus(
    data,
    alpha=0.01,
    max_cond_set_size=3,
    sparsity_bound=3,
    max_path_length=None,
    sepset_selection="first",
    orientation_strategy="standard",
)

estimator = FCIPlus(
    alpha=0.01,
    max_cond_set_size=3,
    sparsity_bound=3,
    max_path_length=None,
    sepset_selection="first",
    orientation_strategy="standard",
)
result = estimator.fit(data)
```

FCI+ uses the same `FCIResult` and PAG representation as standard FCI. The
difference is the refinement stage: standard FCI uses Possible-D-Sep search,
while FCI+ uses a sparse hierarchical D-SEP search driven by separating sets
already discovered in earlier stages.

The explicit settings above are the paper-aligned profile. `alpha="auto"`,
max-p-value sepset selection, finite path caps, and robust/leaf orientation are
optional finite-sample engineering choices.

## Stability-Selected FCI+

For research data where finite-sample false positives matter, run FCI+ with
bootstrap filtering:

```python
from fci_engine import stable_fci_plus

result = stable_fci_plus(
    data,
    n_bootstraps=50,
    edge_threshold=0.6,
    alpha="auto",
    max_cond_set_size=3,
    orientation_strategy="robust",
)
```

The returned `FCIResult` contains the filtered PAG and bootstrap support for
the final retained edges.

## Visual Oracle Report

For a larger sanity check on known synthetic structures:

```bash
PYTHONPATH=src python examples/08_visual_benchmark_report.py
```

Open `examples/realistic_benchmark_report.html` to inspect aggregate scores,
semantic PAG compatibility metrics, and side-by-side true versus learned PAGs.
