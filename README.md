# fci-engine: Fast Causal Inference for Auditable PAG Learning

`fci-engine` learns **Partial Ancestral Graphs (PAGs)** from observational data
with standard **Fast Causal Inference (FCI)** and an FCI+ style sparse D-SEP
refinement.

It is designed for cases where a simple correlation graph or PC-style algorithm
is too optimistic: latent confounding, selection effects, missing values,
finite-sample instability, and production audit requirements.

```mermaid
flowchart LR
    A[pandas or NumPy data] --> B[Input validation]
    B --> C[CI tests + cache]
    C --> D[Stable skeleton search]
    D --> E[Possible-D-Sep or FCI+ D-Sep]
    E --> F[Collider + PAG rules]
    F --> G[PAG, traces, exports]
```

## What You Get

| Capability | Why it matters |
| --- | --- |
| Standard FCI API | Learn a PAG under latent confounding instead of forcing a single DAG. |
| FCI+ API | Use sparse hierarchical D-SEP refinement for faster candidate search on larger sparse graphs. |
| Stable skeleton search | Reduces order dependence by deferring edge removals within each conditioning depth. |
| Accuracy-first sepsets | Keeps the strongest independence evidence at the same depth instead of the first passing set. |
| Multiple CI tests | Fisher-Z, missing-value Fisher-Z, chi-square, G-square, and kernel CI are available. |
| PAG diagnostics | Records CI traces, sepset sources, orientation events, and edge explanations. |
| Oracle benchmarks | Compare outputs against known PAG shapes, causal-learn, and R `pcalg` when installed. |
| Audit exports | Save edge tables, JSON, and NetworkX graphs for downstream review. |

## PAG Output At A Glance

FCI does **not** claim to recover one unique true DAG. It returns a PAG: a graph
that represents the causal facts identifiable from the observed conditional
independence structure.

```text
X --> Y    X is an ancestor/cause candidate of Y
X <-> Y    latent confounding is supported
X o-> Y    Y is not an ancestor of X, but X's endpoint is still uncertain
X o-o Y    direction is not identifiable from the current evidence
X --- Y    selection-bias style undirected dependence
```

Example latent-confounder output:

```text
I1 o-> A
I2 o-> B
A  <-> B
```

## When To Use Which Entry Point

| Entry point | Best use case |
| --- | --- |
| `fci(data)` | Standard FCI pipeline for general continuous data. |
| `fci_plus(data)` | Sparse graphs where broad Possible-D-Sep search is too expensive. |
| `FCI(config).fit(data)` | Estimator-style usage with explicit configuration and reusable objects. |
| `stable_fci(data)` | Bootstrap stability selection when finite-sample reliability matters. |
| `run_oracle_benchmark(...)` | Regression testing against preset known graph structures. |

## Benchmark Snapshot

The visual benchmark report compares hand-written oracle PAGs, `fci_engine`
outputs, and optional R `pcalg::fciPlus` outputs:

```bash
PYTHONPATH=src python examples/08_visual_benchmark_report.py
open examples/realistic_benchmark_report.html
```

The report includes:

- side-by-side oracle / learned / R-package graphs;
- exact-edge F1 and compatibility-aware semantic F1;
- per-edge missing, extra, under-oriented, and over-oriented differences;
- orientation-rule traces for explainability.

## Installation

Supported Python versions: **3.9, 3.10, 3.11, 3.12, and 3.13**.
Source builds require a PEP 621 capable build backend; standard modern `pip`
build isolation handles this automatically, or use `setuptools>=61`.

To use `fci-engine` locally in your projects, clone the repository and install it using pip:

```bash
git clone <your-repo-url>
cd FCI
pip install .
```

For development (includes testing and linting tools):
```bash
pip install -e ".[dev]"
```

The CI matrix runs the test suite on every supported Python minor version.

## Quick Start

Once installed, you can simply `import` and run the graph extraction algorithm on a `pandas.DataFrame` or `numpy.ndarray`.

### 1. Basic Causal Graph (Chain)

```python
import numpy as np
import pandas as pd
from fci_engine import fci

# 1. Generate Synthetic Data: X -> Y -> Z
np.random.seed(42)
X = np.random.normal(size=2000)
Y = 0.8 * X + np.random.normal(size=2000)
Z = 0.8 * Y + np.random.normal(size=2000)
df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

# 2. Run the FCI Engine
result = fci(df, alpha="auto")

# 3. View the Resulting PAG (Partial Ancestral Graph)
print(result.summary())
print("\nIdentified Edges:")
for x, y in result.graph.edges():
    print(f"- {result.graph.edge_repr(x, y)}")
```

### 2. Identifying Latent Confounders

One of FCI's greatest strengths is recognizing unobserved hidden variables via bidirectional arrows (`<->`).

```python
import numpy as np
import pandas as pd
from fci_engine import fci

# Generate data: U (latent) -> X1 & X3. X1 -> X2, X3 -> X2.
np.random.seed(42)
U = np.random.normal(0, 1, 2000)  # Unobserved Confounder
I1 = np.random.normal(0, 1, 2000)
I2 = np.random.normal(0, 1, 2000)

A = 0.8 * I1 + 0.9 * U + np.random.normal(0, 0.5, 2000)
B = 0.8 * I2 + 0.9 * U + np.random.normal(0, 0.5, 2000)

# We only observe I1, I2, A, B. We don't observe U!
df = pd.DataFrame({"I1": I1, "I2": I2, "A": A, "B": B})

result = fci(df, alpha="auto")
for x, y in result.graph.edges():
    if "<->" in result.graph.edge_repr(x, y):
        print(f"Latent Confounder Detected!: {result.graph.edge_repr(x, y)}")
        # Output: A <-> B
```

### 3. Object-Oriented Estimator 

If you prefer `scikit-learn`-style usage:

```python
from fci_engine import FCI, FCIConfig

# Configure the solver (Try alpha="auto" for dynamic thresholding!)
config = FCIConfig(
    alpha="auto",
    max_cond_set_size=3,
    do_pdsep=True,
    skeleton_stable=True,
    pdsep_stable=True,
    sepset_selection="max_pvalue",
    conservative_colliders=True,
    conservative_orientation=False,
    orientation_strategy="robust",
)
estimator = FCI(config)

# Run solver
result = estimator.fit(df)
```

### 4. FCI+

FCI+ keeps the same user-facing result type but replaces standard FCI's broad
Possible-D-Sep search with a sparse hierarchical D-SEP refinement:

```python
from fci_engine import FCIPlus, fci_plus

result = fci_plus(df, alpha="auto", max_cond_set_size=3)

estimator = FCIPlus(alpha=0.01, max_cond_set_size=3)
result = estimator.fit(df)
```

### 5. Missing Values

The public FCI and FCI+ pipelines support query-wise complete-case Fisher-Z when
you explicitly choose the missing-value CI test:

```python
from fci_engine import MissingValueFisherZTest, fci

result = fci(
    df_with_nan,
    ci_test=MissingValueFisherZTest(alpha=0.01),
    max_cond_set_size=3,
)
```

Rows are filtered per CI query, using only the variables in `x`, `y`, and the
current conditioning set. Default Fisher-Z still rejects missing values so that
silent data loss does not happen accidentally.

---

## Theory and Algorithms

### The Need for FCI

Standard Causal Discovery algorithms like PC assume **Causal Sufficiency**: the assumption that there are *no unmeasured common causes (latent confounders)*. In real-world data, this is almost never true. If you run a PC algorithm on data that has latent variables, it frequently draws incorrect causal pathways.

**FCI (Fast Causal Inference)** drops this assumption. Instead of learning a single Directed Acyclic Graph (DAG) or a CPDAG, it learns a **PAG (Partial Ancestral Graph)**.

### Reading a PAG (Endpoint Meaning)

A PAG contains several types of endpoints denoting sets of DAGs (Markov equivalence classes) consistent with the data constraints:
*   `X --> Y` (Tail to Arrow): $X$ is a cause of $Y$.
*   `X <-> Y` (Arrow to Arrow): Spurious correlation; there is an unobserved latent confounder causing both $X$ and $Y$ ($X \leftarrow U \rightarrow Y$).
*   `X o-> Y` (Circle to Arrow): It is either $X \rightarrow Y$ or $X \leftrightarrow Y$. $X$ is purely not an effect of $Y$.
*   `X o-o Y` (Circle to Circle): No information is known about the direction. (Could be $\rightarrow, \leftarrow, \text{or} \leftrightarrow$).
*   `X --- Y` (Tail to Tail): Known as a selection-bias edge. Very rare unless dealing with uniquely conditioned datasets.

### FCI Stages

The structural learning happens in four phases directly implemented in `fci-engine/discovery`:
1.  **FAS (Fast Adjacency Search)**: Iteratively searches for Conditional Independence (CI) up to constraint length $N$ from both endpoints' current adjacency sets to remove structurally non-essential edges. Yields the un-oriented Skeleton and `Sepsets`. When multiple separating sets succeed at the same depth, the default `sepset_selection="max_pvalue"` keeps the strongest independence evidence for later orientation rules.
2.  **Unshielded Colliders Discovery**: Orients V-structures ($X \circ\!\!\to Z \gets\!\!\circ Y$).
3.  **Possible-D-SEP**: Generates larger conditioning candidates that earlier
    rounds may miss. The implementation uses finite edge-state BFS instead of
    enumerating every simple path, avoiding path explosion on dense PAGs.
4.  **Zhang's Orientation Rules (R1 - R10)**: Iteratively closes reasoning gaps over the generated graph by tracing causal endpoints to monotonic conclusions without cycles.

---

## Why `fci-engine`? (Diagnostics System)

Unlike black-box implementations of FCI, `fci-engine` makes debugging causal topologies explicit. You can view exactly why a graph resolved the way it did:

```python
result = fci(df, alpha="auto")

# Review the history of how endpoints were changed
for event in result.orientation_trace:
    print(f"Rule {event.rule} triggered by: {event.reason}")
    print(f"  Change: {event.before_edge} => {event.after_edge}")
```

For production systems, results can be exported and audited:

```python
edge_table = result.to_pandas_edges()
networkx_graph = result.to_networkx()
result.save_json("fci_result.json")

print(result.explain_edge("X", "Y").summary())
```

For finite-sample robustness checks, use the stability-selection wrapper:

```python
from fci_engine import stable_fci

stable_result = stable_fci(df, n_bootstraps=50, edge_threshold=0.6, alpha="auto")
```

Background knowledge can force or forbid directions on edges that remain in the
learned skeleton:

```python
from fci_engine import BackgroundKnowledge, fci

knowledge = BackgroundKnowledge(
    required_edges={("treatment", "outcome")},
    forbidden_edges={("outcome", "treatment")},
)
result = fci(df, background_knowledge=knowledge)
```

Run the preset oracle benchmark suite to compare standard FCI, FCI+, optional
causal-learn, and optional R `pcalg::fciPlus`:

```python
from fci_engine import (
    default_oracle_cases,
    format_benchmark_leaderboard,
    format_benchmark_results,
    run_oracle_benchmark,
)

results = run_oracle_benchmark(default_oracle_cases())
print(format_benchmark_results(results))
print(format_benchmark_leaderboard(results))
```

The benchmark output includes both strict exact-edge F1 and compatibility-aware
semantic F1. Semantic scoring is useful for PAGs because `o->` versus `-->`
can be a compatible certainty difference rather than a contradiction.

You can generate the visual benchmark report with highlighted true/learned
differences and per-edge orientation-rule traces:

```bash
PYTHONPATH=src python examples/08_visual_benchmark_report.py
```

## Current Scope And Limitations

- **Continuous Data**: Uses Fisher-Z as default CI test via Numpy arrays and
  supports covariance/correlation sufficient-statistics input at the CI-test
  layer.
- **Missing Values**: `MissingValueFisherZTest` supports query-wise
  complete-case Fisher-Z in both `fci(...)` and `fci_plus(...)`.
- **Discrete Data**: Provided Chi-square and G-square implementations.
- **Nonlinear CI**: `KernelCITest` provides RBF-HSIC for unconditional tests and
  kernel-ridge residualized KCI-style conditional tests for nonlinear data.
- **Soundness & Convergence**: Supports deterministic Zhang-style rule closures (R1-R10) eliminating false cyclic patterns.
- **Stability Selection**: `stable_fci` can filter edges with weak bootstrap support.
- **Background Knowledge**: Required and forbidden edge directions are supported.
- **FCI+**: Available as `fci_plus(...)` / `FCIPlus`; this first release uses
  hierarchical D-SEP refinement and reuses the standard FCI orientation rules.
- **Reference Benchmarks**: Preset oracle cases can compare `fci_engine`,
  causal-learn, and R `pcalg::fciPlus` when those optional tools are installed.

---

See `docs/quickstart.md`, `docs/theory.md`, and `docs/api.md` for more advanced workflows and architectural layout.
