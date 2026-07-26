## Research Brief

The presentation must explain the original FCI and FCI+ algorithms, trace
their implementation into this repository, compare the package with
established software, and report the Tennessee STAR application without
mixing design-based treatment evidence with PAG-based structural discovery.

The supplied repository already contains the implementation, exact-oracle
tests, STAR outputs, and a source dossier. External research was limited to
the original algorithm sources, later PAG-orientation work, the official STAR
dataset record, and official software documentation needed for a dated
capability comparison.

## Original FCI

Spirtes, Glymour, and Scheines give the standard FCI schedule in Chapter 6 of
the second edition of *Causation, Prediction, and Search*. The algorithm uses
an initial adjacency search, collider information, a non-local
Possible-D-SEP refinement, endpoint reset, and a final orientation phase. Its
population-level guarantee assumes Faithfulness and correct conditional-
independence decisions. The book explicitly leaves maximal informativeness
of the original orientation schedule unresolved; the modern completeness
claim therefore requires later work, especially Zhang (2008).

## FCI+

Claassen, Mooij, and Heskes replace the potentially large standard
Possible-D-SEP subset search with augmented-skeleton, D-SEP-link, and
hierarchy logic. Under a fixed maximum observed-MAG degree bound, Faithfulness,
and a constant-time exact conditional-independence oracle, their analysis gives
worst-case query order \(O(N^{2(k+2)})\). This is a conditional query-complexity
result, not a finite-sample accuracy or wall-clock guarantee.

The paper's Figure 4(b) is a MAG example that exhibits the separator
\(\{U,V,Z\}\). It is not a printed completed PAG, so the repository's completed
endpoint target must be described as a derived regression fixture.

## Established software

The checked landscape contains two executed external references and one
documentation-only system. CRAN `pcalg` 2.7-12 exposes standard FCI and
`fciPlus`. The locally installed causal-learn 0.1.4.7 exposes standard FCI and
several named conditional-independence methods, but no public FCI+ entry point
was identified in the inspected official API. Tetrad 7.6.10 documents standard
FCI and several distinct FCI-family algorithms; it was not executed here and
no drop-in Claassen FCI+ API was established.

The defensible local distinction is integrated auditability: explicit paper
profiles, structured CI and orientation traces, separating-set provenance,
D-SEP diagnostics, per-edge explanations, and bundled export/stability
workflows. This is a feature and reproducibility claim, not a universal
accuracy or speed ranking.

## Tennessee STAR

The official dataset contains 11,601 student records. The repository constructs
a kindergarten assignment cohort of 6,325 students across 79 schools and then
three separate analysis panels. The randomized-arm contrast is the primary
basis for the early treatment conclusion; the learned PAGs provide a separate
structural and sensitivity analysis. Later grade-3 complete-case contrasts and
endpoint marks require stronger selection and robustness qualifications.

## Sources

- https://mitpress.mit.edu/9780262194402/causation-prediction-and-search/
- https://www.auai.org/uai2013/prints/papers/121.pdf
- https://arxiv.org/abs/1411.1557v1
- https://doi.org/10.1016/j.artint.2008.08.001
- https://doi.org/10.1214/aos/1031689015
- https://doi.org/10.7910/DVN/SIWH9F
- https://cran.r-project.org/package=pcalg
- https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/FCI.html
- https://github.com/cmu-phil/tetrad/releases/latest
