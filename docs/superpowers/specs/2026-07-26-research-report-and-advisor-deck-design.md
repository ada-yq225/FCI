# FCI+ Research Report and Advisor Deck Design

## Objective

Produce two coordinated, entirely English research artifacts:

1. a rewritten thesis-style LaTeX/PDF report that presents the scientific
   foundations, independent implementation, validation, software comparison,
   Tennessee STAR application, causal interpretation, and limitations as one
   coherent research argument; and
2. a visually concise, editable PowerPoint deck for a 15–20 minute formal
   advisor presentation covering the complete workflow and principal results.

Both artifacts must distinguish algorithm validation from empirical
application. The literal paper profiles for standard FCI and FCI+ remain
unchanged; the practical robustness profile is presented as a separate
finite-sample application strategy.

## Audience and Tone

The primary audience is an academic advisor evaluating:

- understanding of the original FCI and FCI+ literature;
- depth and independence of the Python implementation;
- correctness and validation evidence;
- differences from established software;
- methodological care in the Tennessee STAR application; and
- whether the causal claims are appropriately bounded by the evidence.

The report uses formal academic prose. The presentation uses shorter
defense-style statements, visual evidence, and explicit takeaways. Neither
artifact may market exploratory PAG endpoints as identified treatment effects.

## Research Narrative

The central narrative is:

> latent confounding and selection complicate causal discovery; standard FCI
> addresses this setting but has expensive Possible-D-SEP search; FCI+ uses a
> logical characterization to obtain polynomial behavior under bounded
> sparsity; this project independently implements and validates both methods,
> adds a separately labeled robust application workflow, and demonstrates the
> distinction between algorithmic output and defensible causal interpretation
> using Tennessee STAR.

## Literature Analysis

The rewritten report will substantially expand its treatment of the two
foundational sources:

- Spirtes, Glymour, and Scheines (2000), *Causation, Prediction, and Search*;
- Claassen, Mooij, and Heskes (2013), “Learning Sparse Causal Models is not
  NP-hard.”

For each source, the report will cover:

- problem setting and graphical objects;
- assumptions and oracle versus finite-sample scope;
- algorithm stages and invariants;
- separating sets and collider logic;
- Possible-D-SEP or hierarchical D-SEP machinery;
- orientation rules and PAG semantics;
- stated complexity result and the conditions under which it applies;
- what the source establishes, and what it does not establish; and
- a source-to-code mapping showing the corresponding repository modules,
  functions, tests, and diagnostics.

Claims about the papers must be checked against the original text. Short
quotations, if any, stay within copyright limits; the default is precise
paraphrase with page, theorem, definition, or algorithm references when
available.

## Implementation Analysis

The implementation chapter will explain:

- package boundaries and public API;
- PAG endpoint representation and invariants;
- conditional-independence test interfaces;
- stable and paper-aligned skeleton discovery;
- separating-set selection and provenance;
- discriminating paths and orientation closure;
- FCI Possible-D-SEP search;
- FCI+ augmented skeleton and hierarchical D-SEP stages;
- paper, practical, and reusable estimator profiles;
- traces, exported artifacts, validation utilities, and error handling;
- computational trade-offs introduced by robust application settings; and
- installation, reproducibility, typing, testing, and CI.

The chapter will use a traceable matrix:

| Published concept | Repository implementation | Validation evidence |
| --- | --- | --- |
| Standard FCI stage/rule | Exact module and symbol | Exact-oracle or regression test |
| FCI+ characterization/stage | Exact module and symbol | Figure 4(b), oracle, or unit test |
| Engineering extension | Exact module and option | Finite-sample or usability test |

Extensions are explicitly labeled as engineering or finite-sample choices,
never as content from the original papers.

## Established-Software Comparison

The report will contain two comparison layers.

### Executed comparison

Where compatible interfaces and dependencies permit, run matched inputs
through:

- the self-implemented standard FCI paper profile;
- the self-implemented FCI+ paper profile;
- the self-implemented robust FCI+ application profile;
- R `pcalg::fciPlus`; and
- Python `causal-learn` FCI.

The executed comparison will report:

- input and CI-test compatibility;
- skeleton and endpoint agreement;
- exact and semantic PAG metrics on known-truth cases;
- CI-test counts and runtime;
- order, bootstrap, and sensitivity behavior where supported; and
- reproducibility details, including software versions.

Different default CI tests or endpoint conventions must not be presented as a
fair exact comparison without normalization.

### Feature-level comparison

Primary documentation and publications will be used to compare the project
with established systems such as `pcalg`, `causal-learn`, and Tetrad on:

- availability of standard FCI and FCI+;
- language and installation model;
- latent-confounder and selection-bias support;
- CI-test extensibility;
- paper-profile reproducibility;
- traceability of separating sets and orientations;
- finite-sample robustness workflow;
- order/bootstrap/sensitivity audit support;
- artifact export and explanation API; and
- scope limitations.

Only locally executed results are called benchmarks. Documentation-derived
capabilities are labeled as such and timestamped to July 2026.

## Paper Structure

1. Abstract
2. Executive Summary
3. Introduction and Research Questions
4. Graphical and Statistical Foundations
5. Standard FCI in Spirtes et al. (2000)
6. FCI+ in Claassen et al. (2013)
7. Independent Python Implementation
8. Validation Design and Results
9. Comparison with Established Software
10. Tennessee STAR Study Design
11. STAR Analysis Methods
12. STAR Results
13. Causal Interpretation and Evidence Hierarchy
14. Critical Self-Assessment
15. Conclusions and Future Work
16. Reproducibility and Detailed Appendices

The implementation and application chapters remain separate even though the
overall paper tells one connected story.

## Evidence Hierarchy

Empirical conclusions will be ordered by evidential strength:

1. randomized-design contrasts and design facts;
2. conclusions stable across implementations, variable order, bootstrap
   samples, and sensitivity settings;
3. stable skeleton adjacencies with unresolved endpoints;
4. implementation-specific or order-sensitive PAG endpoints; and
5. exploratory patterns that require external validation.

The principal STAR interpretation remains:

- small classes improve observed early achievement relative to regular
  classes;
- a regular class with an aide does not show a comparable advantage;
- later achievement evidence is weaker because observation at grade 3 is
  selective;
- robust discovery supports several stable adjacencies but does not uniquely
  identify a DAG or estimate a treatment effect; and
- unresolved circle endpoints are a defensible expression of uncertainty, not
  a failed result.

## Advisor Presentation

The PowerPoint will contain approximately 15 slides:

1. title and one-sentence contribution;
2. research problem and motivation;
3. causal setting: latent variables, selection, MAGs, and PAGs;
4. standard FCI from the original source;
5. FCI+ insight and bounded-sparsity complexity;
6. source-paper-to-implementation map;
7. package architecture and end-user workflow;
8. exact-oracle and finite-sample validation;
9. established-software comparison;
10. Tennessee STAR data and analysis design;
11. randomized-arm descriptive evidence;
12. PAG and robustness results;
13. computational results;
14. defensible causal conclusions and limitations;
15. contributions, future work, and discussion.

The deck will use native, editable shapes, text, charts, and tables wherever
practical. Dense report tables will be redesigned rather than pasted. Each
slide will have one claim-level title and one main visual idea. Speaker notes
or a separate presenter-notes file will provide timing and explanation.

## Visual Design

The visual language will be restrained and academic:

- white or very light warm background;
- dark navy text;
- blue for standard FCI;
- orange for paper FCI+;
- teal for robust FCI+;
- gray or purple for external reference software;
- consistent PAG endpoint notation;
- minimal decorative elements;
- high contrast and projector-safe typography; and
- charts that state sample size, metric, and uncertainty directly.

The report and deck will reuse the same semantic color mapping, terminology,
and result values.

## Files and Deliverables

Expected principal outputs:

- `reports/fci_plus_star_report.tex`
- `output/pdf/fci_plus_star_report.pdf`
- `reports/figures/*.pdf` and any new report comparison figures
- `output/ppt/fci_plus_star_advisor_presentation.pptx`
- `output/ppt/fci_plus_star_advisor_presentation.pdf`
- `output/ppt/fci_plus_star_advisor_presentation_notes.md`
- supporting comparison data or scripts under `reports/` or
  `case_studies/tennessee_star/output/`

Existing generated data are reused when valid. New executable comparisons must
be reproducible from committed scripts or commands.

## Verification

Before delivery:

- verify all literature claims against primary sources;
- rerun relevant algorithm, comparison, and report tests;
- run Ruff formatting/checks and MyPy;
- compile LaTeX without unresolved references or citations;
- extract PDF text to check required sections and final result values;
- render every report page and inspect contact sheets plus changed pages;
- render every presentation slide and inspect montage and individual slides;
- run presentation overflow and out-of-bounds checks;
- verify the PPTX remains editable and the exported PDF matches it;
- confirm no temporary build artifacts are staged;
- review the final Git diff;
- commit intentionally and push `main`.

## Acceptance Criteria

The work is complete when:

- the PDF reads as a coherent academic paper rather than a concatenated
  software manual and case-study appendix;
- the two foundational sources receive materially deeper and correctly
  bounded analysis;
- concrete implementation choices are mapped to published concepts and tests;
- executed benchmarks are separated from documentation-only comparisons;
- the STAR causal conclusions follow the stated evidence hierarchy;
- the English PPT can support a 15–20 minute formal advisor presentation;
- all visuals are legible and professionally consistent;
- all quality checks pass; and
- the final report, deck, sources, figures, and comparison artifacts are pushed
  to the remote `main` branch.
