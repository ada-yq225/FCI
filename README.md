# fci-engine: Fast Causal Inference for Auditable PAG Learning

`fci-engine` learns **Partial Ancestral Graphs (PAGs)** from observational data
with standard **Fast Causal Inference (FCI)** and the sparse hierarchical D-SEP
stage from FCI+ Algorithm 2.

It is designed for cases where a simple correlation graph or PC-style algorithm
is too optimistic: latent confounding, selection effects, missing values,
finite-sample instability, and production audit requirements.

## 60-Second Summary For Reviewers

- **Problem**: learn causal structure from observational data when hidden
  confounders may exist.
- **Output**: a PAG, not an overconfident single DAG.
- **Main API**: `fci(data)`, `fci_plus(data)`, `stable_fci(data)`,
  and `stable_fci_plus(data)`.
- **Research focus**: stable skeleton discovery, accuracy-first separating-set
  selection, conservative orientation options, bootstrap stability filtering,
  and explicit audit traces.
- **Validation**: published oracle PAG cases, variable-order checks, seeded
  latent-SEM integration tests, discrete/nonlinear/missing-data paths, optional
  reference-package comparisons, and a Python 3.9-3.13 CI matrix.
- **Advisor view**: [open the committed evidence-first showcase](examples/advisor_showcase.html)
  or regenerate it from the current checkout.

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
| Audit exports | Save a JSON audit record, CSV edge table, and interactive HTML report in one call. |
| D-SEP diagnostics | Inspect FCI+ candidate edges, revisits, hierarchy-cache hits, skipped duplicate conditioning sets, and D-SEP CI tests. |

## PAG Output At A Glance

FCI does **not** claim to recover one unique true DAG. It returns a PAG: a graph
that represents the causal facts identifiable from the observed conditional
independence structure.

```text
X --> Y    X is an invariant ancestor of Y (not necessarily a direct cause)
X <-> Y    invariant arrowheads; compatible with latent confounding
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
| `fci(data)` | Stable finite-sample FCI defaults with explicit alpha `0.05`. |
| `fci(data, profile="paper")` | Spirtes et al. adjacency and Possible-D-SEP search schedule. |
| `fci_plus(data, profile="practical")` | Bounded conservative FCI+ profile; validate it for the dataset at hand. |
| `FCI(config).fit(data)` | Estimator-style usage with explicit configuration and reusable objects. |
| `FCIPlus.practical(...).fit(data)` | Reusable bounded FCI+ estimator with conservative finite-sample options. |
| `FCIPlus.paper(...).fit(data)` | Literal Algorithm 2 search settings for validation and research comparison. |
| `stable_fci(data)` | Bootstrap stability selection when finite-sample reliability matters. |
| `stable_fci_plus(data)` | Bootstrap stability selection with the FCI+ sparse D-SEP pipeline. |
| `run_oracle_benchmark(...)` | Regression testing against preset known graph structures. |

## FCI+ Usage Guide

### 1. Prepare the data

Pass a numeric `pandas.DataFrame` when possible. Its column names become PAG
node names, which makes every result and export easier to interpret.

```python
import pandas as pd

data = pd.read_csv("observational_data.csv")
feature_columns = ["exposure", "biomarker", "outcome", "site"]
data = data[feature_columns]
```

The default Fisher-Z test expects continuous numeric data without missing or
infinite values. See the missing-value and custom-CI examples below when those
assumptions do not match the dataset.

### 2. Run the bounded practical FCI+ profile

For an initial bounded analysis, the `practical` profile enables
stable skeleton discovery, strongest-at-depth separating-set selection, a
bounded sparse D-SEP search, automatic alpha selection, and robust finite-sample
orientation. It is a convenience profile, not a universal statistical optimum;
compare important results with standard FCI and sensitivity settings.

```python
from fci_engine import fci_plus

result = fci_plus(
    data,
    profile="practical",
    max_cond_set_size=3,
)

print(result.summary())
print(result.to_pandas_edges())
```

`max_cond_set_size=3` is an example bounded starting point. Increasing it can
discover separators involving more variables but
can substantially increase runtime and finite-sample error. The practical
profile uses the same value as the FCI+ sparsity bound unless
`sparsity_bound` is specified separately.

### 3. Use a reusable estimator

Use `FCIPlus` when the configuration should be named, reused, inspected, or
passed through an application service.

```python
from fci_engine import FCIPlus

estimator = FCIPlus.practical(
    max_cond_set_size=3,
    alpha="auto",
)
result = estimator.fit(data)

# Available after fit:
pag = estimator.graph_
same_result = estimator.get_result()

# Or return the PAG directly:
pag = estimator.fit_predict(data)
```

The estimator stores only run metadata and the latest fitted result. Call
`fit` again to analyze another dataset with the same configuration.

### 4. Run the paper-aligned profile

Use the `paper` profile for Algorithm 2 validation or comparison with another
FCI+ implementation. It uses the same `k` for ordinary conditioning and the
sparse hierarchical D-SEP bound, keeps first-found separating sets, and applies
the standard orientation profile.

```python
from fci_engine import fci_plus

paper_result = fci_plus(
    data,
    profile="paper",
    k=3,
    alpha=0.01,
)
```

This profile fixes both PC adjacency depth and the hierarchical base size to
the same `k`, uses immediate graph updates and first-found minimal separating
sets, and disables standard FCI's Possible-D-SEP stage. It is intended for
algorithm fidelity checks, not as an automatic finite-sample recommendation.

### 5. Read and export the result

`FCIResult` keeps the learned PAG together with the configuration, separating
sets, CI-test trace, orientation trace, runtime, and FCI+ D-SEP diagnostics.

```python
# Compact edge table for analysis code
edges = result.to_pandas_edges()

# PAG edge notation
for x, y in result.edges:
    print(result.graph.edge_repr(x, y))

# Inspect the evidence for one node pair
print(result.explain_edge("exposure", "outcome").summary())

# Write a complete applied-analysis bundle
paths = result.save_artifacts(
    "outputs",
    stem="study_fci_plus",
)
print(paths["json"])
print(paths["edges_csv"])
print(paths["report_html"])
print(result.assumption_notes())
```

The JSON file is the machine-readable audit record, the CSV is convenient for
downstream analysis, and the standalone HTML report supports interactive edge
inspection. Use `include_traces=True` in `save_artifacts` when the JSON export
must contain every CI and orientation event.

Do not interpret every retained adjacency as a direct causal effect. FCI+
returns a PAG representing an equivalence class under its assumptions.
Arrowheads, tails, and circles describe identifiable endpoint information.

### 6. Choose the important parameters

| Parameter | Practical meaning |
| --- | --- |
| `profile="practical"` | Bounded conservative convenience profile for finite-sample exploration. |
| `profile="paper"` | Literal Algorithm 2 search settings for research validation. |
| `alpha` | CI-test threshold; default is `0.05`. `"auto"` is an opt-in sample-size heuristic recorded in `result.alpha_was_auto`. |
| `max_cond_set_size` | Maximum ordinary conditioning depth; larger values cost more CI tests. |
| `sparsity_bound` | FCI+ hierarchical D-SEP degree bound; defaults to `max_cond_set_size` in the practical profile. |
| `orientation_strategy` | `"robust"` is cautious for applied work; `"standard"` follows the full implemented rule schedule. |
| `sepset_selection` | `"max_pvalue"` favors stronger finite-sample evidence; `"first"` matches traditional early stopping. |
| `ci_test` | Replace Fisher-Z for missing, discrete, nonlinear, or domain-specific data. |

### 7. Missing values and alternative data types

Missing continuous data requires an explicit missing-value CI test:

```python
from fci_engine import MissingValueFisherZTest, fci_plus

result = fci_plus(
    data_with_missing_values,
    profile="practical",
    ci_test=MissingValueFisherZTest(alpha=0.01),
    max_cond_set_size=3,
)
```

For discrete data use `ChiSquareTest` or `GSquareTest`. For nonlinear
continuous relationships use `KernelCITest`. A custom test can implement the
public `CITest` interface and be passed through the same `ci_test` argument.

### 8. Add bootstrap stability analysis when needed

Bootstrap filtering is a sensitivity analysis around FCI+, not part of the
paper's oracle algorithm:

```python
from fci_engine import stable_fci_plus

stable = stable_fci_plus(
    data,
    profile="practical",
    n_bootstraps=50,
    n_jobs=4,
    edge_threshold=0.6,
    max_cond_set_size=3,
)
```

## Accuracy And Reliability Strategy

`fci-engine` does not rely on one single trick. The implementation combines
several safeguards that matter in finite-sample research data:

| Source of error | Mitigation in this package |
| --- | --- |
| Order-dependent skeleton deletion | Stable adjacency snapshots per conditioning depth. |
| Near-threshold separating sets | `sepset_selection="max_pvalue"` records the strongest independence evidence at the first successful depth. |
| Overconfident collider orientation | Conservative collider mode can leave ambiguous triples unoriented. |
| Dense Possible-D-Sep path growth | Ordered edge-state BFS avoids full simple-path enumeration. |
| Finite-sample false positives | `stable_fci(...)` and `stable_fci_plus(...)` filter weak bootstrap edges. |
| Hard-to-audit decisions | CI traces, sepset sources, orientation traces, and edge explanations are exported. |

The correct way to claim accuracy is to run a transparent benchmark on known
oracle graphs. This repository includes that workflow instead of treating any
external implementation as ground truth.

## Paper-Aligned Validation Snapshot

The strongest regression fixture is the D-SEP MAG in FCI+ Figure 4(b). Exact
m-separation supplies the CI answers, so this test isolates algorithm fidelity
from sampling error. Both algorithms recover the complete published PAG; FCI+
removes `X-Y` in its hierarchical D-SEP stage with the unique separator
`{U, V, Z}`.

| Reproducible run | FCI+ | Standard FCI | What it shows |
| --- | --- | --- | --- |
| Figure 4(b), exact oracle | Exact PAG; 63 CI queries | Exact PAG; 102 CI queries | 39 fewer queries (about 38%) on this fixture only. |
| Seeded latent SEM, `N=5,000` | Extra `X <-> Y`; exact F1 0.923 | Exact PAG; exact F1 1.000 | Finite-sample CI error can prevent FCI+ candidate recognition. |
| Same SEM, `N=50,000` | Exact PAG; exact F1 1.000 | Exact PAG; exact F1 1.000 | A reproducible integration result, not a universal sample-size threshold. |

The committed tests also cover the complete PAG for three published reference
graphs and four representative variable orders for Figure 4(b). The oracle
result establishes correctness under exact CI answers; it does not promise
finite-sample recovery on arbitrary data.

```bash
python -m pytest -q
python -m ruff check src tests examples
python -m mypy
```

Current validation is checked by Pytest, Ruff, strict MyPy, a wheel build, and
an installed-wheel smoke test. GitHub Actions runs Pytest independently on
Python 3.9, 3.10, 3.11, 3.12, and 3.13.
Local validation for this revision: **256 passed** on Python 3.14; **243 passed,
4 optional-reference skips** on Python 3.9.6; strict MyPy passed under both
interpreters.
The full graphs, diagnostics, same-data FCI baseline, configuration, and
limitations are collected in the
[advisor showcase](examples/advisor_showcase.html). Its metrics are computed by
[`examples/10_fci_plus_advisor_showcase.py`](examples/10_fci_plus_advisor_showcase.py),
not entered by hand.

## Benchmark Snapshot

The visual benchmark report compares hand-written oracle PAGs, `fci_engine`
outputs, and optional R `pcalg::fciPlus` outputs:

```bash
PYTHONPATH=src python examples/08_visual_benchmark_report.py
open examples/realistic_benchmark_report.html
```

The report includes:

- side-by-side oracle / learned / R-package graphs;
- click-to-explain graph edges with expected endpoints, actual endpoints,
  endpoint status, and recorded orientation-rule evidence;
- exact-edge F1 and compatibility-aware semantic F1;
- per-edge missing, extra, under-oriented, and over-oriented differences;
- orientation-rule traces for explainability.

The committed HTML is a presentation snapshot, not a universal dominance
claim. Regenerate it after any algorithm or configuration change and review the
per-case edge explanations before citing its aggregate numbers. This visual
report uses a different cohort from the paper-aligned Figure 4(b) experiment
above.

## Tennessee STAR Applied Case Study

The repository now includes a separate real-data application to the Tennessee
Student/Teacher Achievement Ratio randomized class-size experiment:

- [case-study methodology and reproduction guide](case_studies/tennessee_star/README.md)
- [standalone visual report](case_studies/tennessee_star/output/star_case_study_report.html)
- [machine-readable result summary](case_studies/tennessee_star/output/star_case_study_summary.json)

The application is intentionally outside `src/fci_engine`: STAR-specific data
coding, cohort selection, school-cluster bootstrap, visualizations, and
researcher conclusions are not mixed into the reusable algorithm package.

Rebuild every STAR artifact with:

```bash
PYTHONPATH=src python -m case_studies.tennessee_star.run_case_study
```

## Installation

Supported Python versions: **3.9, 3.10, 3.11, 3.12, and 3.13**.
Source builds require a PEP 621 capable build backend; standard modern `pip`
build isolation handles this automatically, or use `setuptools>=77`.

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

CI runs Pytest on every supported Python minor version. Separate jobs run Ruff,
strict MyPy, build the wheel, install it into a clean virtual environment, and
execute an import/FCI+ smoke test.

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

# 2. Run FCI with an explicit statistical threshold
result = fci(df, alpha=0.01)

# 3. View the Resulting PAG (Partial Ancestral Graph)
print(result.summary())
print("\nIdentified Edges:")
for x, y in result.graph.edges():
    print(f"- {result.graph.edge_repr(x, y)}")
```

### 2. Finding Structures Compatible With Latent Confounding

FCI can return invariant bidirected endpoints (`<->`). This is compatible with
latent confounding, but it does not identify or prove one particular hidden
variable.

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

result = fci(df, alpha=0.01)
for x, y in result.graph.edges():
    if "<->" in result.graph.edge_repr(x, y):
        print(f"Bidirected PAG edge: {result.graph.edge_repr(x, y)}")
        # Possible example output: A <-> B (finite samples can differ)
```

### 3. Object-Oriented Estimator

If you prefer `scikit-learn`-style usage:

```python
from fci_engine import FCI, FCIConfig

# Configure a stable finite-sample run.
config = FCIConfig(
    alpha=0.01,
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
from fci_engine import FCIPlus, FCIPlusConfig, fci_plus

# Bounded conservative convenience profile
result = fci_plus(df, profile="practical", max_cond_set_size=3)

# Equivalent reusable estimator
estimator = FCIPlus.practical(max_cond_set_size=3)
result = estimator.fit(df)

# Explicit configuration object for application settings
config = FCIPlusConfig.practical(
    max_cond_set_size=3,
    sparsity_bound=3,
    alpha="auto",
)
result = FCIPlus(config).fit(df)
```

See the full [FCI+ Usage Guide](#fci-usage-guide) above for paper-aligned
settings, result interpretation, exports, missing values, and bootstrap
stability analysis.

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

Standard causal-discovery algorithms such as PC assume **causal sufficiency**:
there are no unmeasured common causes among the modeled variables. This
assumption is often questionable in observational studies. When it is violated,
PC's causal guarantees need not hold and the learned graph can be misleading.

**FCI (Fast Causal Inference)** drops this assumption. Instead of learning a single Directed Acyclic Graph (DAG) or a CPDAG, it learns a **PAG (Partial Ancestral Graph)**.

### Reading a PAG (Endpoint Meaning)

A PAG represents a Markov-equivalence class of MAGs; those MAGs summarize the
observed-margin constraints induced by latent-variable causal DAGs. Its endpoint
marks have the following meanings:
*   `X --> Y` (Tail to Arrow): $X$ is an invariant ancestor of $Y$ in the represented equivalence class; the edge need not be a direct effect.
*   `X <-> Y` (Arrow to Arrow): neither endpoint is an ancestor of the other in the represented MAG/PAG. This is compatible with latent confounding, but does not identify a specific latent variable.
*   `X o-> Y` (Circle to Arrow): the arrowhead at $Y$ rules out $Y$ as an ancestor of $X$; the endpoint at $X$ remains unresolved.
*   `X o-o Y` (Circle to Circle): No information is known about the direction. (Could be $\rightarrow, \leftarrow, \text{or} \leftrightarrow$).
*   `X --- Y` (Tail to Tail): compatible with selection effects in the represented ancestral graph, but does not prove one specific selection mechanism.

### Standard FCI stages

The `profile="paper"` search follows the FCI construction in Chapter 6 of
Spirtes, Glymour, and Scheines:

1. **PC-style adjacency search** over ordered endpoint adjacency sets, recording
   the first separating set found at increasing conditioning depth.
2. **Initial unshielded-collider orientation** from the recorded separating
   sets.
3. **Possible-D-SEP refinement**. `Possible-D-SEP(A,B)` and
   `Possible-D-SEP(B,A)` are searched as two separate candidate pools; the
   implementation never creates a conditioning set by mixing the pools.
   Candidate reachability is computed from the initially oriented graph for
   this stage, as in the book's `F`/`F'` construction.
4. **Reset and final orientation** after all remaining false adjacencies are
   removed.

The 2000 book proves that this returns a partially oriented inducing-path graph
and explicitly notes that completeness was then unknown. To return the complete
PAG expected by the 2013 FCI+ paper, this package uses the later complete
R1-R10 orientation schedule: close R1-R4, apply R5, close R6-R7, then close
R8-R10.

### FCI+ Algorithm 2 mapping

`fci_plus(..., profile="paper", k=...)` maps directly to Claassen, Mooij, and
Heskes (2013), Algorithm 2:

- line 1: PC adjacency search with bound `k`;
- lines 2-3: augmented skeleton and Lemma 4 candidate D-SEP links;
- lines 4-22: separate `BaseX` / `BaseY` subset loops, recursive `HIE`,
  minimal D-SEP reduction, augmented-graph rebuild, and candidate revisiting;
- line 23: complete FCI PAG orientation.

The augmented skeleton uses only invariant arrowheads produced when adding one
node to a known minimal separating set destroys independence. Hierarchy caches
are keyed by both seed nodes and the excluded candidate edge, so one candidate
cannot reuse another candidate's `HIE` result.

### Paper-aligned oracle validation

The regression suite includes FCI+ Figure 4(b), Zhang's orientation example,
and the Spirtes (1997) latent-variable example. The Figure 4(b) test verifies
that PC leaves `X-Y`, FCI+ removes it only in the hierarchical D-SEP stage with
the unique separator `{U, V, Z}`, and the complete PAG is unchanged across four
representative variable orders. A seeded 50,000-row latent linear SEM supplies
a separate finite-sample integration test.

```bash
python -m pytest tests/test_published_reference_graphs.py -q
```

The oracle guarantees require the causal Markov and faithfulness assumptions,
an underlying acyclic causal DAG (latent and selection variables are allowed),
sound/complete CI answers, and—for FCI+'s polynomial bound—a sparse observed
MAG with maximum degree `k`. Finite samples do not inherit the oracle guarantee.

Primary specifications:
[Spirtes, Glymour, and Scheines (2000), *Causation, Prediction, and Search*](https://www.cs.cmu.edu/afs/cs.cmu.edu/project/learn-43/lib/photoz/.g/web/.g/group/group2/g/opera/g/scottd/fullbook.pdf),
[Claassen, Mooij, and Heskes (2013), Algorithm 2](https://www.auai.org/uai2013/prints/papers/121.pdf),
and [Zhang's complete PAG rules](https://doi.org/10.1016/j.artint.2008.08.001).

---

## Why `fci-engine`? (Diagnostics System)

Unlike black-box implementations of FCI, `fci-engine` makes debugging causal topologies explicit. You can view exactly why a graph resolved the way it did:

```python
result = fci(df, alpha=0.01)

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
result.save_interactive_report("fci_report.html")

print(result.explain_edge("X", "Y").summary())
```

The interactive report is designed for user data, not only benchmarks. It is a
standalone HTML file: click any retained PAG edge to inspect deterministic
endpoint meanings, skeleton/CI evidence, orientation-rule evidence, and a plain
English reasoning paragraph. The explanatory text is template-based and
auditable; it is not generated by a language model.

For finite-sample robustness checks, use the stability-selection wrapper:

```python
from fci_engine import stable_fci

stable_result = stable_fci(
    df,
    n_bootstraps=50,
    n_jobs=4,
    edge_threshold=0.6,
    alpha=0.01,
)
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
    run_pcalg_comparison_benchmark,
)

results = run_oracle_benchmark(default_oracle_cases())
print(format_benchmark_results(results))
print(format_benchmark_leaderboard(results))

pcalg_results = run_pcalg_comparison_benchmark(default_oracle_cases())
print(format_benchmark_leaderboard(pcalg_results))
```

To write the focused FCI+ versus `pcalg::fciPlus` comparison as HTML and CSV:

```bash
PYTHONPATH=src python examples/09_pcalg_fci_plus_comparison.py
```

For an advisor-facing showcase that explains the Algorithm 2 alignment,
FCI+ D-SEP diagnostics, reference comparisons, and the user API in one HTML
page:

```bash
PYTHONPATH=src python examples/10_fci_plus_advisor_showcase.py
open examples/advisor_showcase.html
```

The benchmark output includes both strict exact-edge F1 and compatibility-aware
semantic F1. Semantic scoring is useful for PAGs because `o->` versus `-->`
can be a compatible certainty difference rather than a contradiction.

You can generate the visual benchmark report with highlighted true/learned
differences and per-edge orientation-rule traces:

```bash
PYTHONPATH=src python examples/08_visual_benchmark_report.py
```

## Assumptions, interpretation, and non-goals

- **Continuous Data**: Uses Fisher-Z as default CI test via Numpy arrays and
  supports covariance/correlation sufficient-statistics input at the CI-test
  layer.
- **Missing Values**: `MissingValueFisherZTest` supports query-wise
  complete-case Fisher-Z in both `fci(...)` and `fci_plus(...)`.
- **Discrete Data**: Provided Chi-square and G-square implementations.
- **Nonlinear CI**: `KernelCITest` provides RBF-HSIC for unconditional tests and
  kernel-ridge residualized KCI-style conditional tests for nonlinear data.
- **Orientation Rules**: Applies the standard Zhang R1-R10 schedule. Oracle
  guarantees require Markov, faithfulness, acyclicity, and exact-CI assumptions;
  finite-sample errors remain possible.
- **Stability Selection**: `stable_fci` and `stable_fci_plus` can parallelize
  bootstrap replicates with `n_jobs`. Stability does not correct systematic CI
  bias.
- **Background Knowledge**: Required and forbidden edge directions are supported.
- **FCI+**: Available as `fci_plus(...)` / `FCIPlus`, with hierarchical D-SEP
  refinement and the standard FCI orientation rules. The API can separate the
  sparse degree bound from the conditioning-set cap as an engineering extension;
  `profile="practical"` is a bounded convenience profile, while
  `profile="paper"` enforces the Algorithm 2 search schedule and uses one `k`
  for both stages.
- **Reference Benchmarks**: Preset oracle cases can compare `fci_engine`,
  causal-learn, and R `pcalg::fciPlus` when those optional tools are installed.
- **Interpretation**: The result is a PAG, not a unique DAG, causal-effect
  estimator, adjustment-set finder, or intervention model. Use
  `result.assumption_notes()` and the audit exports before downstream
  interpretation.
- **Statistical guarantees**: Markov, faithfulness, valid CI decisions, and the
  FCI+ degree bound are theorem assumptions. No implementation can turn
  finite-sample CI errors into an oracle guarantee.
- **Typing boundary**: strict MyPy checks all package code. pandas and NetworkX
  use installed stubs, SciPy uses a narrow repository stub for the APIs called
  here, and the optional untyped causal-learn import is isolated behind a typed
  protocol rather than covered by a global `ignore_missing_imports`.

---

See `docs/quickstart.md`, `docs/theory.md`, and `docs/api.md` for more advanced workflows and architectural layout.
