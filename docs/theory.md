# Theory Notes

## Causal Sufficiency

Causal sufficiency means that every common cause of the measured variables is
also measured. The PC algorithm usually relies on this assumption. Real
observational systems often violate it: hidden load, unlogged incidents,
genetic factors, socioeconomic variables, or measurement pipelines may act as
unobserved common causes.

## Latent Confounding

Latent confounding occurs when an unobserved variable causally affects two or
more observed variables. FCI is designed to remain conservative in this setting.
Instead of forcing a single DAG, it learns a PAG over the observed variables.

An edge like `X <-> Y` says that neither endpoint can be an ancestor of the
other in the represented equivalence class. It is compatible with latent
confounding, but it does not identify a particular latent variable.

## Selection Bias

Selection bias can arise when the data are conditioned on an event that is
itself affected by variables in the system. FCI's PAG representation is designed
for maximal ancestral graphs, which can represent latent and selection effects
more flexibly than a DAG over only observed variables.

## PAG Endpoint Meanings

Endpoint marks describe constraints:

- `TAIL`: a tail endpoint, displayed as `-`
- `ARROW`: an arrowhead endpoint, displayed as `<` or `>`
- `CIRCLE`: unresolved endpoint, displayed as `o`
- `NONE`: no edge

Examples:

- `X --> Y`: tail at `X`, arrowhead at `Y`
- `X o-> Y`: uncertain endpoint at `X`, arrowhead at `Y`
- `X <-> Y`: arrowheads at both endpoints
- `X o-o Y`: unresolved at both endpoints

A PAG is an equivalence-class object. It is not a unique true DAG. In
particular, `X --> Y` is an invariant ancestral statement and need not be a
direct causal effect.

## Skeleton Discovery

The initial skeleton stage starts with a complete circle-edge PAG. It removes an
edge `X-Y` when a conditional independence test finds a separating set `S` such
that `X` is independent of `Y` given `S`. Separating sets are stored for later
orientation. Conditioning candidates are searched from the current adjacency
sets of both endpoints, which better matches PC/FAS behavior and avoids pushing
ordinary adjacency-search deletions into the later Possible-D-Sep stage.

## Possible-D-Sep

FCI extends PC with Possible-D-Sep refinement. In the presence of hidden
variables, the separating set for two observed variables may not be contained
only in their current adjacency sets. Possible-D-Sep searches a conservative set
of graph-reachable candidates and runs additional CI tests.

This package includes a readable conservative Possible-D-Sep implementation with
an optional `max_path_length` limit. Candidate reachability is computed with a
finite ordered-edge-state BFS rather than full simple-path enumeration, which
keeps dense or cyclic PAGs from triggering avoidable path explosion.

## Orientation Rules

After skeleton discovery, FCI orients unshielded colliders using separating
sets. It then follows the complete-rule schedule: close R1-R4, apply R5, close
R6-R7, and finally close R8-R10.

The standard discriminating path rule is included. The orientation rule suite is
kept readable and tested against small reference shapes, including an optional
causal-learn comparison test.

The standard strategy implements Zhang's R1-R10 rules:

- R1: avoid introducing new unshielded colliders
- R2: propagate an arrowhead through either local directed-chain pattern
- R3: orient double-triangle arrowheads
- R4: use discriminating paths for collider/noncollider orientation
- R5: orient uncovered circle paths as undirected selection-bias edges
- R6: propagate tails out of undirected selection-bias edges
- R7: propagate tails from definite noncollider patterns
- R8: orient tails along directed chains
- R9: orient tails along uncovered possibly directed paths
- R10: orient tails from two directed parents with suitable uncovered paths

Each endpoint change can be inspected through `FCIResult.orientation_trace`.

## FCI+ Hierarchical D-SEP

The FCI+ pipeline follows Claassen, Mooij, and Heskes' sparse hierarchical
D-SEP search in Algorithm 2: it builds an augmented skeleton, identifies
candidates only through the literal bidirected witness pattern
`U <-> X <-> Y <-> V` plus the two not-against-arrowhead cross paths, and
enumerates separate endpoint-base subsets in the paper's nested `n=1..k`,
`m=1..k` order.

The implementation deliberately separates two limits:

- `max_cond_set_size` limits ordinary PC/FCI conditioning-set depth.
- `sparsity_bound` is the FCI+ sparse degree bound used for the D-SEP base
  subsets.

Finite-sample engineering choices are explicit rather than hidden.
`sepset_selection="first"`, equal PC/D-SEP bounds, unlimited orientation paths,
and `orientation_strategy="standard"` are the paper-aligned settings.
`sepset_selection="max_pvalue"`, `alpha="auto"`, conservative/leaf orientation,
and path limits are optional engineering profiles, not claims from the paper.
The result includes
`dsep_diagnostics` counters for candidate edges, revisits, hierarchy-cache
hits, duplicate conditioning-set skips, D-SEP CI tests, removed edges, and the
maximum D-SEP conditioning size.

## Finite-Sample Sepset Selection

In exact oracle CI, any valid separating set at the first successful
conditioning depth is enough to remove an edge. With finite samples, however,
the first successful set can be a near-threshold accident, and that recorded
sepset later controls unshielded-collider orientation. For accuracy-first
behavior, the default `sepset_selection="max_pvalue"` scans all candidate
sets at the same depth and records the independent set with the largest
p-value. This costs more CI tests but reduces orientation sensitivity to
candidate ordering. Set `sepset_selection="first"` for traditional
early-stopping behavior.

## Assumptions, Guarantee, And Complexity

The soundness/completeness statements are oracle results. They assume causal
Markov and faithfulness, an underlying acyclic causal DAG, and correct
conditional-independence answers; latent and selection variables may be
present. FCI+ additionally assumes that the observed MAG is sparse with maximum
degree `k`. Under those assumptions FCI+ replaces only the D-SEP stage and must
return the same complete PAG as standard FCI.

The proved worst-case number of CI tests is `O(N^(2(k+2)))`. The main paper and
supplement theorem agree on this exponent; the `N^(2(k+1))` text on the
supplement's arXiv abstract page is inconsistent with its own theorem. Larger
finite-sample conditioning sets also have lower statistical power, so
polynomial oracle complexity does not imply better finite-sample accuracy.

Primary sources:

- [FCI+ paper and Algorithm 2](https://auai.org/uai2013/prints/papers/121.pdf)
- [FCI+ proof supplement](https://staff.fnwi.uva.nl/j.m.mooij/articles/UAI2013_121_sup.pdf)
- [Zhang (2008), complete orientation rules](https://doi.org/10.1016/j.artint.2008.08.001)
- [Readable R1-R10 rule table and schedule](https://www.cs.ru.nl/~tomc/docs/UAI2011_LCCausD.pdf)

## Published Oracle Regression Graphs

`tests/test_published_reference_graphs.py` runs both FCI and FCI+ against exact
m-separation on three published structures. The flagship FCI+ Figure 4(b) case
has the unique separator `{U, V, Z}` for `X-Y`, where `Z` is adjacent to neither
endpoint. The test requires the edge source to be `fci_plus_dsep`, compares all
PAG endpoints, and repeats the run under multiple variable orders. A separate
seeded latent linear SEM checks the same graph with 50,000 sampled rows.

## Reference Implementation Caveat

External packages are differential references, not truth. In particular,
`pcalg` 2.7-12's `PosDsepLinks` creates `NAAA` entries only for indices `2:p`
but can later read `NAAA[[1]]`. On Figure 4(b), this can retain the false
`X-Y` link when an endpoint occupies index 1. The regression suite therefore
tests variable permutations directly instead of copying pcalg output. See the
[CRAN FCI+ source](https://rdrr.io/cran/pcalg/src/R/fciPlus.R).

## Background Knowledge

Domain knowledge can constrain orientations after skeleton discovery. A required
constraint `X -> Y` orients an existing edge as `X --> Y`; a forbidden
constraint `X -> Y` orients an existing edge as `Y --> X`. These constraints do
not force edges to exist when CI tests remove them from the skeleton.

## Limitations

- Fisher-Z is the default CI test and is intended for continuous numeric data.
- Chi-square and G-square CI tests are available for discrete variables.
- Fisher-Z accepts covariance/correlation sufficient statistics at the CI-test
  layer.
- Missing-value Fisher-Z is available through query-wise complete-case deletion
  in both standard FCI and FCI+.
- Kernel CI is available through RBF-HSIC for unconditional tests and a
  kernel-ridge residualized KCI-style conditional statistic for nonlinear data.
- Finite-sample CI decisions may be unstable near the significance threshold.
- FCI+ candidate recognition needs invariant arrowheads in the augmented
  skeleton. At moderate sample sizes these may be missed, so standard FCI can
  be more reliable even when FCI+ uses fewer oracle CI tests.
- Output is a PAG, not a DAG.
- FCI+ is available as a sparse hierarchical D-SEP implementation with explicit
  finite-sample diagnostics. It shares the same PAG orientation stage as
  standard FCI and is primarily intended to reduce the broad Possible-D-Sep
  search cost in sparse graphs.
