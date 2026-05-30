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

An edge like `X <-> Y` often signals that neither endpoint can be an ancestor of
the other in the represented equivalence class, which is commonly interpreted as
evidence compatible with latent confounding.

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

A PAG is an equivalence-class object. It is not a unique true DAG.

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
sets. It then applies PAG orientation rules iteratively to propagate reliable
endpoint information while avoiding new unshielded colliders and directed
cycles.

The standard discriminating path rule is included. The orientation rule suite is
kept readable and tested against small reference shapes, including an optional
causal-learn comparison test.

The implementation also includes conservative Zhang-style R5-R10 rules:

- uncovered circle paths for selection-bias undirected edges
- tail propagation from undirected and definite noncollider patterns
- tail orientation along directed chains
- tail orientation from uncovered possibly directed paths
- tail orientation from two directed parents with suitable uncovered paths

Each endpoint change can be inspected through `FCIResult.orientation_trace`.

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
- Output is a PAG, not a DAG.
- FCI+ is available as a sparse hierarchical D-SEP variant. It shares the same
  PAG orientation stage as standard FCI and is primarily intended to reduce the
  broad Possible-D-Sep search cost in sparse graphs.
