# FCI and FCI+ source dossier

Evidence freeze: **2026-07-26**

This dossier separates four kinds of statements that the report must not
collapse:

1. claims made by the original FCI and FCI+ sources;
2. later results needed to use modern PAG terminology;
3. repository implementation facts;
4. finite-sample or locally executed evidence.

The core project contribution is an independently implemented, auditable
Python package with distinct paper-aligned and engineering profiles. It is not
a new proof of FCI or FCI+ correctness, and the Tennessee STAR case study is
an application of the package rather than part of either algorithm.

## Spirtes, Glymour, and Scheines (2000)

### Primary source and scope

The standard-FCI source is Peter Spirtes, Clark Glymour, and Richard
Scheines, *Causation, Prediction, and Search*, second edition, MIT Press,
copyright 2000, especially Chapter 6, §§6.3–6.7, printed pp. 129–147.
The MIT Press catalog dates the volume to 2001, while its copyright and
Library of Congress data say 2000; this project follows the conventional
“Spirtes et al. (2000)” citation.

Primary links:

- [MIT Press catalog](https://mitpress.mit.edu/9780262194402/causation-prediction-and-search/)
- [Full-text course mirror](https://ics.uci.edu/~dechter/courses/ics-295cr/2021-22_Q2_Winter/reading/Causation_Prediction_and_Search.pdf)

The source assumes an acyclic causal graph and an observed distribution that
supplies correct conditional-independence decisions under causal Markov and
Faithfulness assumptions. FCI permits omitted variables in the full graph; it
does not assume that the measured set is causally sufficient (Chapter 3,
§§3.4.1 and 3.4.3, printed pp. 29–31; Chapter 6, §§6.3 and 6.7,
printed pp. 129–130 and 139).

### Algorithm schedule

The book's “Fast Causal Inference Algorithm” (Chapter 6, §6.7, printed
pp. 144–145) has the following evidence-bearing stages:

1. begin with a complete observed-variable graph and run a PC-style adjacency
   search, storing separating sets;
2. orient unshielded colliders from separator membership;
3. search for additional separators in Possible-D-SEP from both ordered
   endpoint directions;
4. remove any newly separated adjacency and record its separator;
5. reset the surviving endpoint marks, reorient colliders from the final
   separating sets, and apply the original orientation phase to closure.

Possible-D-SEP is defined immediately before that algorithm on printed
p. 144. A node is reachable from an endpoint along a path whose every
intermediate triple is either a collider or a triangle whose center is not a
definite noncollider. It is deliberately a computable superset of D-SEP, not
an estimate of a unique causal neighborhood. Searching both endpoint orders
is part of the algorithmic contract.

For an unshielded triple \(A-B-C\), the collider decision is separator based:
orient arrowheads into \(B\) when \(B\notin\operatorname{Sepset}(A,C)\)
(“Causal Inference Algorithm,” step C, printed p. 139; reused by FCI on
p. 145). The final phase includes discriminating-path logic and repeats until
no further mark can be oriented (definition and Figure 6.14, pp. 138–140).

### What the 2000 theorem does—and does not—establish

Theorem 6.4 (printed p. 145) states that, given Faithfulness and correct
population-level decisions, FCI returns a partially oriented inducing-path
graph for the observed margin. The paragraph immediately after the theorem
explicitly says that the authors did not know whether this output is always
maximally informative. Therefore:

- the 2000 result supports correctness/soundness under its assumptions;
- it does **not** by itself support a claim of orientation completeness for a
  modern completed PAG;
- the repository's `orientation_strategy="spirtes_2000"` must not be
  described as Zhang-complete.

Modern MAG/PAG and selection-variable language is supported by Richardson
and Spirtes (2002), “Ancestral Graph Markov Models,” *Annals of Statistics*
30(4), 962–1030, DOI
[10.1214/aos/1031689015](https://doi.org/10.1214/aos/1031689015).
Orientation completeness is supported by Jiji Zhang (2008), “On the
completeness of orientation rules for causal discovery in the presence of
latent confounders and selection bias,” *Artificial Intelligence*
172(16–17), 1873–1896, DOI
[10.1016/j.artint.2008.08.001](https://doi.org/10.1016/j.artint.2008.08.001).
Those later results must be cited when the report uses the phrase “completed
PAG.”

## Claassen, Mooij, and Heskes (2013)

### Primary source and theorem scope

The FCI+ source is Tom Claassen, Joris M. Mooij, and Tom Heskes, “Learning
Sparse Causal Models is not NP-hard,” *Proceedings of UAI 2013*, pp. 172–181.

Primary links:

- [UAI proceedings paper](https://www.auai.org/uai2013/prints/papers/121.pdf)
- [arXiv record](https://arxiv.org/abs/1309.6824v1)
- [proof supplement](https://arxiv.org/abs/1411.1557v1)

Definition 1 and the `C-LEARN` problem (p. 173), Algorithm 2 (p. 178), the
complexity analysis (p. 179), and supplement Theorem 1 (PDF pp. 10–11)
establish the restricted result. Given:

- a constant-time, exact conditional-independence oracle;
- Faithfulness to an underlying causal DAG;
- latent variables and selection bias being allowed;
- a constant bound \(k\) on the degree of the true observed-variable MAG;

FCI+ recovers a sound and complete PAG using at most worst-case order
\(O(N^{2(k+2)})\) independence queries. This is fixed-\(k\) oracle/query
complexity. It is neither a finite-sample accuracy guarantee nor a claim that
general unrestricted causal discovery is not NP-hard.

The supplement's arXiv abstract says \(N^{2(k+1)}\), but its Theorem 1,
proof, and final sentence agree with the main paper on
\(N^{2(k+2)}\). The report uses the theorem/body value and records the
abstract discrepancy rather than silently choosing between them.

### Algorithm 2 decomposition

Algorithm 2 (p. 178) replaces standard FCI's potentially large
Possible-D-SEP subset search with a sparse, hierarchy-guided D-SEP stage:

1. `PCAdjSearch(V,O,k)` builds the initial skeleton and minimal independence
   set.
2. The algorithm constructs an augmented skeleton \(G^+\) (Definition 3 and
   Lemma 2(1), p. 176). If adding a single adjacent node to a known minimal
   separator destroys an independence, the resulting invariant
   non-ancestry information supplies arrowheads in this temporary graph.
3. Lemma 4 (p. 176) recognizes candidate D-SEP links through the literal
   bidirected pattern \(U\leftrightarrow X\leftrightarrow
   Y\leftrightarrow V\), nonadjacent witnesses \(U,V\), and cross-direction
   paths that do not run against arrowheads.
4. Definitions 4–5 and Lemmas 5–7 (pp. 177–178) define the recursive
   hierarchy \(HIE(\cdot,I)\), explain why separators learned for relevant
   ancestral pairs reveal required D-SEP nodes, and justify revisiting
   candidates after each successful removal.
5. Lines 6–12 enumerate at most \(k\)-node adjacent-ancestor bases at both
   endpoints; lines 13–18 test/minimize the hierarchical separator, update
   the independence set, rebuild the augmentation, and reconsider candidates.
6. Line 23 invokes complete FCI orientation. That completeness is delegated
   to the established Zhang (2008) PAG rule set rather than proved anew in
   the UAI paper.

The main paper's complexity analysis (p. 179) counts
\(O(N^{k+2})\) initial PC queries, \(O(N^3)\) augmentation queries, up to
\(O(N^2)\) candidate links, \(O(N^{2k})\) paired base choices per
candidate, and an additional \(O(N^2)\) candidate-rechecking factor. The
dominant query term is therefore \(O(N^{2(k+2)})\). The authors explicitly
say that this can remain infeasible for large \(N\), that their
implementation was not optimized, and that large hierarchy-derived
conditioning sets may reduce finite-sample power (§6, p. 179).

### Figure 4(b) boundary

Figure 4(b) (pp. 175–176) is a five-node **MAG** example, not a printed
completed PAG. It exhibits a false \(X-Y\) link surviving the adjacency
stage and the separator \(\{U,V,Z\}\), where \(Z\) is not adjacent to
either endpoint. The repository's `canonical_dsep_mag()` encodes that MAG and
uses a hand-authored completed-PAG shape for regression testing. Consequently:

- the six MAG edges and the exhibited separator are directly source based;
- removal provenance can be validated with the exact repository oracle;
- the completed endpoint target is a repository derivation, not a PAG copied
  from Figure 4(b);
- the paper exhibits the separator but does not call it uniquely minimal.

## Source-to-implementation mapping

The table distinguishes paper-aligned components from engineering extensions.
Line numbers are intentionally omitted because symbols are more stable report
locators than a transient checkout position.

| Source concept | Repository symbol | Evidence / validation | Status |
|---|---|---|---|
| FCI stage ordering | `src/fci_engine/discovery/fci.py::FCI.fit` | `tests/test_published_reference_graphs.py` | Paper-aligned when used with `FCIConfig.paper()` |
| Complete start and PC adjacency search | `src/fci_engine/discovery/skeleton.py::create_complete_pag`; `learn_initial_skeleton` | skeleton and published-reference tests | Core paper mechanism |
| Possible-D-SEP path criterion | `src/fci_engine/discovery/pdsep.py::possible_dsep`; `_is_pds_step_allowed` | `tests/test_pdsep.py` | Core FCI mechanism |
| Ordered Possible-D-SEP separator search | `src/fci_engine/discovery/pdsep.py::refine_skeleton_with_pdsep` | `tests/test_pdsep.py` | Core FCI mechanism |
| FCI+ temporary augmented skeleton | `src/fci_engine/discovery/dsep.py::build_augmented_skeleton`; `_augment_with_single_node_dependencies` | `tests/test_fci_plus.py` | Algorithm 2 mechanism |
| Lemma 4 candidate recognition | `src/fci_engine/discovery/dsep.py::possible_dsep_links`; `_has_dsep_link_witness` | strict bidirected/cross-path tests in `tests/test_fci_plus.py` | Algorithm 2 mechanism |
| Recursive hierarchy | `src/fci_engine/discovery/dsep.py::hierarchy`; `_hierarchy_cache_key` | hierarchy fixed-point tests | Algorithm 2 mechanism; cache is an engineering optimization |
| Paired endpoint bases bounded by \(k\) | `src/fci_engine/discovery/dsep.py::_algorithm2_base_sizes`; `_base_combinations_for_sizes` | Algorithm 2 loop tests | Algorithm 2 mechanism |
| Hierarchical removal, minimization, and revisiting | `src/fci_engine/discovery/dsep.py::refine_skeleton_with_fci_plus_dsep`; `minimal_dsep` | Figure 4(b) provenance test | Algorithm 2 mechanism |
| FCI+ estimator integration | `src/fci_engine/discovery/fci_plus.py::FCIPlus.fit`; `FCIPlus.paper` | `tests/test_fci_plus.py`; `tests/test_published_reference_graphs.py` | Paper-aligned only under the explicit paper profile |
| Collider orientation and reset | `src/fci_engine/discovery/orientation.py::orient_unshielded_colliders`; `reset_endpoint_marks` | orientation tests | Source-aligned; conservative variant is an extension |
| Original and complete rule schedules | `src/fci_engine/discovery/rules.py::apply_orientation_rules` | `tests/test_orientation_rules.py`; `tests/test_oracle_pag_rules.py` | `spirtes_2000` is historical; `standard` uses later Zhang completion |
| PAG endpoint storage and queries | `src/fci_engine/graph/pag.py::PAG` | `tests/test_pag.py`; `tests/test_pag_queries.py` | Modern representation supported by later MAG/PAG literature |
| Exact m-separation validation | `src/fci_engine/simulation/mag_oracle.py::MAGOracleCITest`; `MAGSpec.is_m_separated` | `tests/test_oracle_graphs.py`; `tests/test_published_reference_graphs.py` | Validation instrument, not a real-data oracle |
| Cross-library finite-sample comparison | public `fci()` against causal-learn FCI | `tests/test_reference_causal_learn.py`; `tests/test_reference_complex_causal_learn.py` | Executed reference evidence, not a proof |
| Stable skeletons and max-p separators | `FCIConfig.practical`; `skeleton_stable`; `sepset_selection="max_pvalue"` | config, order, and robustness tests | Engineering extension |
| Conservative/leaf/robust orientation | `orientation_strategy` values `conservative`, `leaf`, and `robust` | `tests/test_robust_application_profile.py` | Engineering extension; no complete-PAG theorem claimed |
| Automatic alpha and finite search caps | `alpha="auto"`; `max_cond_set_size`; `max_path_length` | config and auto-alpha tests | Engineering extension; caps can weaken paper guarantees |
| Bootstrap and order audits | stable wrapper and STAR application audit helpers | stability and STAR tests | Application workflow, not FCI+ Algorithm 2 |
| Trace and artifact exports | `src/fci_engine/result.py::FCIResult` | `tests/test_result_exports.py` | Engineering extension for auditability |

The paper profiles preserve the conceptual boundary:

- `FCIConfig.paper()` uses unbounded conditioning/path search, immediate
  skeleton deletion, first-found separators, Possible-D-SEP, and the original
  2000 orientation schedule.
- `FCIPlusConfig.paper(k=...)` requires an explicit \(k\), uses the same bound
  for the adjacency and sparse-base searches, disables standard FCI's
  Possible-D-SEP stage, and selects the Zhang-complete standard rule schedule
  required by Claassen et al.'s complete-PAG claim.
- `practical` and `robust` profiles are finite-sample engineering workflows.
  They are not redefinitions of either paper's algorithm.

## Claims that require finite-sample qualification

1. **Oracle recovery is not sample recovery.** The theorems assume correct CI
   decisions. Fisher-Z additionally assumes continuous approximately Gaussian
   data; discrete, missing-value, and nonlinear tests have their own
   conditions and power limitations.
2. **Faithfulness is untestable from one finite observational sample.**
   Near-unfaithfulness can make separator and collider decisions unstable.
3. **A user-supplied \(k\) is not evidence of true sparsity.** If the true
   observed-MAG degree exceeds the supplied `sparsity_bound`, FCI+'s theorem
   does not apply and needed base sets can be omitted.
4. **Search caps trade completeness for feasibility.** A finite conditioning
   or path bound can retain false adjacencies or miss orientations.
5. **The robust profile deliberately sacrifices orientation information.**
   Conservative colliders and tail restrictions can improve order stability
   without producing a maximally informative PAG.
6. **Polynomial query complexity is conditional.** The
   \(O(N^{2(k+2)})\) result treats \(k\) and oracle-query time as constants;
   it is not an elapsed-time guarantee for statistical CI tests.
7. **PAGs are equivalence classes.** A retained adjacency does not by itself
   prove a direct effect or identify a particular latent variable. Circles are
   unresolved endpoints; output is not a unique DAG, adjustment set, or
   treatment-effect estimate.
8. **Selection-bias semantics need explicit design information.** An
   undirected PAG edge can be compatible with selection, but the algorithm
   does not automatically identify the selection mechanism.
9. **Software smoke tests establish callability only.** Local execution of
   `pcalg`, causal-learn, or this package does not establish common accuracy
   without a matched known-truth benchmark and normalized endpoint semantics.
10. **The STAR application remains observational structure discovery.** Its
    randomized assignment can support a separate experimental comparison,
    but a learned PAG alone does not estimate the class-size treatment effect.
11. **Fixed-\(k\) polynomial complexity is not per-instance dominance.** The
    committed 5--80 node exact-oracle audit recovers every target skeleton but
    shows that augmented-skeleton overhead can make FCI+ use more queries on
    easy chains or after isolated variables are added. The defensible empirical
    claim is workload-dependent query reduction, not universal superiority.

## Software-comparison evidence policy

The companion `software_landscape.json` uses `evidence_kind="executed"` only
when a named local version completed a small smoke call. Tetrad is
documentation-only because it was not executed in this environment. Capability
absence statements are narrow: they describe the inspected official API, not
what a user might construct externally.

Standard FCI, FCI+, RFCI, GFCI, and other hybrid FCI-family algorithms are not
interchangeable. In particular, causal-learn exposes standard FCI but no
public FCI+ entry point was identified; Tetrad exposes several FCI-family
algorithms but no drop-in Claassen et al. `FciPlus` API was established.
Empirical rankings must include only matched executed algorithms, while
documentation-only rows may be used for feature comparisons.
