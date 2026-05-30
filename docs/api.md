# API Reference

The package supports Python 3.9 through 3.13.

## `fci_engine.fci`

```python
from fci_engine import fci

result = fci(data, alpha=0.05, do_pdsep=True)
```

Runs standard FCI and returns an `FCIResult`.

Accepted input:

- `pandas.DataFrame`: column names become variable names
- `numpy.ndarray`: columns are named `X0`, `X1`, `X2`, ...

## `fci_engine.fci_plus`

```python
from fci_engine import fci_plus

result = fci_plus(data, alpha=0.05, max_cond_set_size=3)
```

Runs FCI+ and returns an `FCIResult`. FCI+ keeps the same graph, sepset, trace,
and CI-cache interfaces as standard FCI, but replaces the broad Possible-D-Sep
refinement with a sparse hierarchical D-SEP search.

## `fci_engine.FCI`

```python
from fci_engine import FCI

estimator = FCI(alpha=0.01, max_cond_set_size=3)
result = estimator.fit(data)
```

Configuration options:

- `alpha`: significance level for the default Fisher-Z CI test
- `ci_test`: custom conditional independence test
- `max_cond_set_size`: maximum conditioning set size
- `max_path_length`: maximum Possible-D-Sep path length
- `do_pdsep`: whether to run Possible-D-Sep refinement
- `skeleton_stable`: snapshot adjacency sets within each conditioning depth and
  defer removals until the depth is complete, reducing order dependence
- `pdsep_stable`: snapshot the PAG at the start of Possible-D-Sep refinement
  so candidate paths are not affected by earlier removals in the same stage
- `sepset_selection`: choose how separating sets are recorded when several
  candidate sets at the same search depth imply independence. The default
  `"max_pvalue"` scans the full depth and keeps the strongest independence
  evidence; `"first"` preserves early-stopping behavior.
- `conservative_colliders`: use Conservative-FCI-style unshielded collider
  orientation; triples with conflicting separating sets are reported as
  ambiguous instead of being forced into a collider
- `conservative_orientation`: keep arrowhead-producing orientation rules but
  skip tail-producing propagation rules; useful when audits prefer a less
  committed PAG over possible over-orientation
- `orientation_strategy`: `"standard"` applies all implemented PAG orientation
  rules, `"conservative"` keeps arrowhead rules only, and `"leaf"` keeps the
  conservative behavior in dense graph regions while still allowing R1 to
  direct leaf endpoints. `"robust"` combines Conservative-FCI-style collider
  checks with the leaf-tail rule profile.
- `background_knowledge`: required and forbidden orientation constraints
- `verbose`: print CI and orientation progress

## `fci_engine.FCIPlus`

```python
from fci_engine import FCIPlus

estimator = FCIPlus(alpha=0.01, max_cond_set_size=3)
result = estimator.fit(data)
```

`FCIPlus` uses the same configuration dataclass as `FCI`. In FCI+, the
`max_cond_set_size` value is also used as the sparsity bound for the base
adjacency subsets in the hierarchical D-SEP search.

## `FCIResult`

Fields:

- `graph`: learned `PAG`
- `sepsets`: separating sets
- `ci_test_count`: total cached CI test queries
- `cache_hits`: number of cache hits
- `elapsed_time`: runtime in seconds
- `config`: `FCIConfig`
- `orientation_trace`: endpoint before/after events with the rule that changed them
- `ci_test_trace`: CI query records including p-values and cache-hit status
- `sepset_sources`: whether each sepset came from initial skeleton search, PDS,
  or FCI+ hierarchical D-SEP
- `ambiguous_triples`: unshielded triples skipped by conservative collider
  orientation because separating sets disagree
- `bootstrap_edge_frequencies`: optional user-populated stability summary

Use `result.summary()` for a compact text summary.

Industrial export and audit helpers:

```python
edge_table = result.to_pandas_edges()
networkx_graph = result.to_networkx()
payload = result.to_dict()
result.save_json("fci_result.json")

print(result.explain_edge("X", "Y").summary())
```

- `to_pandas_edges()`: edge table with endpoint marks and edge strings
- `to_networkx()`: `networkx.Graph` with PAG endpoint marks as edge attributes
- `to_dict()` / `to_json()` / `save_json()`: JSON-safe result export
- `explain_edge(x, y)`: direct CI tests, sepset, sepset source, and orientation
  events related to a node pair

## CI Tests

Built-in CI tests:

- `FisherZTest`: default continuous Gaussian-style test
- `FisherZTest` also accepts sufficient-statistics mappings with
  `{"correlation": corr, "n_samples": n}` or
  `{"covariance": cov, "n_samples": n}`
- `MissingValueFisherZTest`: query-wise complete-case Fisher-Z for arrays or
  DataFrames with missing values in the public `fci(...)` and `fci_plus(...)`
  pipelines
- `ChiSquareTest`: Pearson chi-square test for discrete variables
- `GSquareTest`: likelihood-ratio G-square test for discrete variables
- `KernelCITest`: RBF kernel CI test for nonlinear dependence. Empty
  conditioning sets use an unconditional HSIC permutation test; non-empty
  conditioning sets use a kernel-ridge residualized KCI-style statistic with a
  Gamma null approximation.

Example:

```python
from fci_engine import KernelCITest, MissingValueFisherZTest

result = fci(data, ci_test=MissingValueFisherZTest(alpha=0.01))
nonlinear = fci(data, ci_test=KernelCITest(alpha=0.05, n_permutations=200))
```

## Stability Diagnostics

```python
from fci_engine import bootstrap_edge_frequencies

frequencies = bootstrap_edge_frequencies(data, n_bootstraps=20, alpha=0.01)
```

The keys are exact PAG edge strings such as `X o-> Y`; values are bootstrap
frequencies in `[0, 1]`.

For stability selection, use `stable_fci`:

```python
from fci_engine import stable_fci

result = stable_fci(data, n_bootstraps=50, edge_threshold=0.6, alpha="auto")
```

This runs standard FCI on the full data, then removes final edges that appear in
less than `edge_threshold` of bootstrap runs. It is an engineering robustness
wrapper, not a replacement for the standard FCI algorithm.

## Oracle Benchmarks

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

The benchmark runner compares `fci_engine.fci`, `fci_engine.fci_plus`, optional
causal-learn FCI, and optional R `pcalg::fciPlus`. If `Rscript` or `pcalg` is
not installed, the pcalg row is returned with a skip reason instead of failing.

Each completed row contains strict exact-edge metrics and semantic PAG metrics.
The semantic score counts `o->` versus `-->` as a compatible certainty
difference, while still reporting whether an endpoint was over-oriented,
under-oriented, or contradicted.

For hand-written oracle cases, use `CausalGraphSpec`:

```python
from fci_engine import CausalGraphSpec

spec = CausalGraphSpec(
    observed_nodes=["X", "Y"],
    latent_nodes=["U"],
    directed_edges=[("U", "X"), ("U", "Y")],
)
expected_pag_shape = spec.to_pag_shape()
```

To generate the HTML report with side-by-side true and learned PAGs plus
edge-level difference explanations:

```bash
PYTHONPATH=src python examples/08_visual_benchmark_report.py
```

## Background Knowledge

```python
from fci_engine import BackgroundKnowledge, fci

knowledge = BackgroundKnowledge(
    required_edges={("X", "Y")},
    forbidden_edges={("Z", "Y")},
)
result = fci(data, background_knowledge=knowledge)
```

Required constraints orient existing edges as `X --> Y`. Forbidden constraints
orient existing edges in the reverse direction. Background-knowledge changes are
recorded in `result.orientation_trace` with rule `background_knowledge`.

## `PAG`

Common methods:

- `edges()`
- `edge_repr(x, y)`
- `to_edge_list()`
- `neighbors(x)`
- `is_adjacent(x, y)`
- `get_endpoint(x, y)`
- `possible_causes(y)`
- `definite_causes(y)`

Remember that `get_endpoint(x, y)` returns the endpoint at `y` on edge `x-y`.

## Reference Comparison Tests

Optional reference tests compare small benchmark shapes with causal-learn:

```bash
pip install -e ".[reference]"
PYTHONPATH=src pytest tests/test_reference_causal_learn.py
```
