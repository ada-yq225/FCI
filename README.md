# fci-engine: Industrial-Grade Fast Causal Inference

`fci-engine` is a highly modular, transparent, and robust Python package for learning **Partial Ancestral Graphs (PAGs)** from observational data using the **Fast Causal Inference (FCI)** algorithm.

Unlike standard PC algorithms that assume all variables are observed (*Causal Sufficiency*), FCI is designed to recover causal structures under the presence of **Latent Confounders** and **Selection Bias**. 

This package provides a modern Pythonic API, heavily optimizing CI-test caching, and features an exclusive **Trace & Diagnostics System** to make causal discovery explainable step-by-step.

---

## 🛠 Features

* **Causal Discovery with Latent Variables**: Specifically handles unobserved confounding and identifies `X <-> Y` structures.
* **Modern Pythonic API**: Clean `dataclass`-driven models and endpoint abstractions. Easy to import with standard Python mechanics: `from fci_engine import fci, fci_plus`.
* **Standard Zhang's Rules**: Fully and strictly implements J. Zhang's orientation rules (R1-R10) and Possible-D-SEP (PD-SEP) for rigorous soundness and completeness.
* **FCI+ Variant**: Provides `fci_plus(...)` / `FCIPlus` with a sparse hierarchical D-SEP refinement inspired by Claassen, Mooij, and Heskes (2013).
* **Order-Stable Skeleton Search**: The initial PC-style skeleton stage snapshots adjacency sets per conditioning depth and applies removals after the depth completes, reducing order dependence.
* **Exceptional Explainability**: Built-in tracking of `OrientationEvent` and `CITraceEvent`. Allows you to easily debug *why* a specific algorithmic decision (e.g., directing an arrow) was made.
* **Performance Optimizations**: Out-of-the-box `CITestCache` radically cuts down redundant Conditional Independence tests.

## 📦 Installation

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

## 🚀 Quick Start

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
config = FCIConfig(alpha="auto", max_cond_set_size=3, do_pdsep=True)
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

---

## 🧠 Theory and Algorithms

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
1.  **FAS (Fast Adjacency Search)**: Iteratively searches for Conditional Independence (CI) up to constraint length $N$ from both endpoints' current adjacency sets to remove structurally non-essential edges. Yields the un-oriented Skeleton and `Sepsets`.
2.  **Unshielded Colliders Discovery**: Orients V-structures ($X \circ\!\!\to Z \gets\!\!\circ Y$).
3.  **Possible-D-SEP**: Generates larger condition tests specifically seeking structural confounds that earlier rounds might have missed out due to arbitrary graph topologies.
4.  **Zhang's Orientation Rules (R1 - R10)**: Iteratively closes reasoning gaps over the generated graph by tracing causal endpoints to monotonic conclusions without cycles.

---

## 🔍 Why `fci-engine`? (Diagnostics System)

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

## Current Scope And Limitations

- **Continuous Data**: Uses Fisher-Z as default CI test via Numpy arrays and
  supports covariance/correlation sufficient-statistics input at the CI-test
  layer.
- **Missing Values**: `MissingValueFisherZTest` supports query-wise
  complete-case Fisher-Z.
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
