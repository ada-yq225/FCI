<!-- ppt-master-schema: design-spec/v1 -->
# FCI and FCI+ from Source Fidelity to Tennessee STAR - Design Spec

## I. Project Information

| Item | Value |
| --- | --- |
| Project Name | FCI and FCI+ from Source Fidelity to Tennessee STAR |
| Canvas Format | PPT 16:9, 1280 × 720 px |
| Page Count | 15 |
| Target Audience | Academic advisor and technically informed research committee |
| Communication Intent | Defend the project in sequence: original algorithms, independent implementation, known-truth validation, established-software comparison, and a bounded Tennessee STAR application |
| Desired Audience Outcome | Understand what was implemented, which claims are source-backed or empirically tested, why the robust workflow is useful, and which STAR conclusions are genuinely causal |
| Core Message / Ask / Action | Accept the project as an auditable independent implementation whose strongest empirical contribution is disciplined evidence separation rather than an overstated discovered DAG |
| Delivery Context | Formal 15–20 minute project-defense presentation with advisor questions |
| Artifact Afterlife | Editable PowerPoint, PDF handout, and speaker notes retained with the reproducible repository |
| Reading Mode | balanced |
| Content Strategy | A single evidence ladder connects source text to code, known-truth tests, software runs, and STAR interpretation while preserving the boundary between algorithm and application |
| Design Style | Data-journalism evidence brief with restrained academic typography and explicit provenance |
| Formula Policy | mixed |
| AI Image Acquisition Path | not applicable |
| Generation Mode | continuous |
| Spec Refinement | disabled |
| Created Date | 2026-07-26 |

## II. Canvas Specification

| Property | Value |
| --- | --- |
| Format | PPT 16:9 |
| Dimensions | 1280 × 720 px |
| viewBox | `0 0 1280 720` |
| Margins | 56 px left/right, 40 px top, 34 px bottom |
| Content Area | 1168 × 646 px inside the safe margins |

## III. Visual Theme

### Theme Style

- **Mode**: pyramid
- **Visual style**: data-journalism
- **Theme**: Evidence-first causal discovery, using stable semantic colors for each algorithm and evidence class
- **Tone**: Formal, precise, calm, skeptical of overclaiming, and visually decisive

### Color Scheme

| Role | HEX | Purpose |
| --- | --- | --- |
| Background | #F7F5EF | Warm paper canvas |
| Secondary background | #E9EDF5 | Quiet analytical panels |
| Primary | #183153 | Titles, structure, and high-authority claims |
| Accent | #2F6BFF | Standard FCI and source-aligned book logic |
| Secondary accent | #E58A2B | Paper FCI+ and sparse-search logic |
| Robust workflow | #1C9A8A | Robust application policy and stability evidence |
| External software | #7562A8 | Established-software comparisons |
| Positive evidence | #2E7D5B | Design-based or validated positive evidence |
| Caution | #B45F4A | Limitations, reversals, and unresolved sensitivity |
| Body text | #1A2433 | Main body text |
| Muted text | #5F6875 | Citations, labels, and caveats |
| Surface | #FFFFFF | Cards and data panels |
| Grid | #CBD2DC | Rules, axes, and separators |

## IV. Typography System

### Font Plan

| Role | Chinese | English | Fallback tail |
| --- | --- | --- | --- |
| Title | Georgia | Georgia | Times New Roman, serif |
| Body | Arial | Arial | Helvetica, sans-serif |
| Data | Consolas | Consolas | Courier New, monospace |
| Footnote | Arial | Arial | Helvetica, sans-serif |

- **Title stack**: Georgia, Times New Roman, serif
- **Body stack**: Arial, Helvetica, sans-serif
- **Data stack**: Consolas, Courier New, monospace
- **Footnote stack**: Arial, Helvetica, sans-serif
- **Role rationale**: Georgia distinguishes thesis-level claims; Arial keeps dense evidence readable; Consolas identifies code, metrics, and versioned artifacts.

### Font Size Hierarchy

| Purpose | Anchor Size (px) |
| --- | ---: |
| Body | 24 |
| Title | 42 |
| Subtitle | 32 |
| Lead | 30 |
| Annotation | 18 |
| Footnote | 16 |
| Data | 20 |

## V. Layout Principles

### Page Structure

- **Header area**: A short assertion title at upper left, with a small page/section marker at upper right.
- **Content area**: One dominant visual and no more than two supporting evidence regions; use white cards only where grouping materially improves comprehension.
- **Footer area**: A slim source line at left and page number at right; STAR slides also carry a compact evidence-class legend.

### Spacing Specification

| Element | Current Project |
| --- | --- |
| Safe margin | 56 px horizontal; 40 px top; 34 px bottom |
| Content block gap | 24 px |
| Icon-text gap | 12 px |

## VI. Icon Usage Specification

- **Primary bundled library**: none

| Purpose | Icon Path | Page |
| --- | --- | --- |

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Layout pattern | Crop Policy | Acquire Via | Status | Reference | text_policy | page_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## IX. Content Outline

### Part 1: Claim, target, and research questions

#### Slide 01 - Independent FCI and FCI+ implementation, then a disciplined STAR application

- **Audience move**: From seeing one software project to seeing a source-to-evidence research contribution
- **Layout**: Two horizontal evidence tracks converge into a compact hierarchy at right; the title and contribution statement dominate the upper third
- **Title**: Independent FCI and FCI+: From Source Fidelity to Tennessee STAR
- **Core message**: The project independently implements paper-aligned FCI and FCI+, validates them, and keeps randomized evidence separate from exploratory discovery.
- **Content**: Upper track: original sources → independent code → exact and simulated validation. Lower track: STAR randomized design → descriptive contrasts → cautious PAG diagnostics. Three lead claims: paper-aligned implementation; validation against known truth and established software; experiment first, PAG second.
- **Visualization**: Conceptual two-track evidence ribbon; not data-driven
- **Fact IDs**: F001, F004, F010
- **Cover impact**: A formal defense cover that states the complete contribution in one glance

#### Slide 02 - Latent variables and selection make one learned DAG the wrong target

- **Audience move**: From expecting a DAG to understanding why the correct output is a PAG
- **Layout**: Left causal schematic with a latent common cause and conditioned selection node; right endpoint key and three interpretation rules
- **Title**: Latent Variables and Selection Change the Target
- **Core message**: With latent confounding or selection, the estimand of discovery is an equivalence-class graph, not a unique causal DAG.
- **Content**: Show latent U affecting two observed variables and observation S receiving causes; map to a mixed ancestral graph and PAG. Define tail, arrowhead, and circle. State: adjacency is not a direct effect; circles are explicit uncertainty; sample CI testing is distinct from an oracle.
- **Visualization**: Conceptual causal graph and PAG endpoint legend; not data-driven
- **Fact IDs**: F002, F003

#### Slide 03 - Five auditable research questions

- **Audience move**: From a broad “which algorithm wins?” question to a sequence of answerable validation questions
- **Layout**: Five numbered cards arranged in theory, validation/software, and application bands
- **Title**: The Project Asks Five Auditable Questions
- **Core message**: Source fidelity, implementation traceability, known-truth recovery, fair software comparison, and bounded application conclusions must be judged separately.
- **Content**: 1. What do the sources assume and prove? 2. Where is each stage implemented and tested? 3. What happens when truth is known? 4. What comparisons across software are fair? 5. Which STAR findings are randomized, stable, or exploratory?

### Part 2: Original algorithms and implementation

#### Slide 04 - Standard FCI in Spirtes et al. (2000)

- **Audience move**: From recognizing the FCI name to understanding its historical schedule and guarantee boundary
- **Layout**: Six-stage pipeline across the middle; Possible-D-SEP path condition in a contrasting lower callout; theorem scope at right
- **Title**: Original FCI Closes Non-Local Separations through Possible-D-SEP
- **Core message**: Standard FCI repairs a local skeleton using a computable superset of D-SEP, then orients a partially identified ancestral structure.
- **Content**: Complete graph → PC-style adjacency search and sepsets → unshielded colliders → Possible-D-SEP in both endpoint orders → remove new independences and reset marks → orientation closure. Define the intermediate-triple condition. State that the 2000 result assumes Faithfulness and correct CI decisions and did not establish maximal informativeness; modern PAG orientation uses Zhang’s later rule set.
- **Visualization**: Conceptual algorithm flow; not data-driven
- **Fact IDs**: F001, F002, F003

#### Slide 05 - FCI+ in Claassen et al. (2013)

- **Audience move**: From seeing FCI+ as a faster black box to understanding its sparse logical characterization
- **Layout**: Four-layer algorithm schematic with a side formula card and an assumptions strip
- **Title**: FCI+ Replaces Broad Search with a Sparse Logical Characterization
- **Core message**: FCI+ recognizes candidate D-SEP links and builds hierarchy-derived separators, yielding a conditional polynomial query bound for sparse graphs.
- **Content**: Initial PC skeleton → augmented skeleton → D-SEP-link witness and recursive hierarchy → test/minimize separator, remove link, revisit candidates → established Zhang-rule orientation schedule. Show O(N^{2(k+2)}) and qualify it with fixed observed-MAG degree k, Faithfulness, and a constant-time exact CI oracle. Note that this is not a finite-sample runtime or accuracy guarantee.
- **Visualization**: Conceptual algorithm flow and formula; not data-driven
- **Fact IDs**: F004, F005, F006

#### Slide 06 - Source-to-code-to-test traceability

- **Audience move**: From trusting an implementation claim to seeing inspectable boundaries and evidence
- **Layout**: Three-column traceability map with source concept, implementation symbols, and test/artifact families
- **Title**: Every Paper Stage Is Inspectable, Testable, and Reusable
- **Core message**: The independent package exposes the FCI and FCI+ stages as modules and records the evidence needed to audit them.
- **Content**: FCI row: skeleton.py, pdsep.py, orientation.py, rules.py. FCI+ row: dsep.py and fci_plus.py with augmented skeleton, candidate links, hierarchy, and bounded endpoint bases. Audit row: PAG invariants, CI/orientation traces, sepset provenance, exports, and per-edge explanations. Include key symbols FCI.fit, possible_dsep, refine_skeleton_with_pdsep, build_augmented_skeleton, possible_dsep_links, hierarchy, refine_skeleton_with_fci_plus_dsep, and PAG.
- **Visualization**: Source-to-code-to-test mapping; not data-driven

#### Slide 07 - Paper profiles versus the robust application workflow

- **Audience move**: From confusing engineering policy with the original theorem to understanding the deliberate separation
- **Layout**: Two-column comparison ending in a balanced trade-off scale
- **Title**: Robust Settings Trade Specificity for Auditable Stability
- **Core message**: The paper profiles preserve source-aligned schedules; the robust profile is a separately labeled finite-sample policy that prefers conservative, stable output.
- **Content**: Paper FCI+: explicit k, literal sparse search, and the Zhang-rule orientation schedule. Robust workflow: stable skeleton, max-p sepsets, conservative colliders, restricted tail orientation, bounded searches, cyclic-order audit, and bootstrap support. Trade-off: more CI calls and more circles in exchange for fewer brittle directions. Explicitly state: engineering extension, not a new FCI+ theorem.
- **Visualization**: Policy comparison table and balance; one pure text-grid table
- **Native-ready**: no

### Part 3: Validation and software landscape

#### Slide 08 - Known-truth validation

- **Audience move**: From relying on real-data plausibility to seeing exact and simulated tests with known truth
- **Layout**: Left Figure 4(b) witness schematic; right two-level metric panel for exact-oracle and finite-sample evidence
- **Title**: Known Truth Separates Algorithmic Recovery from Sample Error
- **Core message**: Exact-oracle fixtures validate the implemented logic, while finite-sample simulations expose the cost and limits of robustness.
- **Content**: Show the Figure 4(b) false X—Y adjacency removed by separator {U,V,Z}; report 63 FCI+ versus 102 FCI logical CI queries on the encoded fixture. Separately label the fixed-alpha regression in which both profiles use α=0.001 across five families, three seeds, and N=2,500: skeleton F1 is 0.979 for both, exact endpoint-sensitive F1 increases from 0.536 to 0.703, and mean CI calls increase from 197.1 to 380.7. Add caveats that the repository derives the PAG target from the source MAG and that this regime is distinct from the robust automatic-alpha benchmark on Slide 10.
- **Visualization**: Data-driven exact-query and finite-sample comparison
- **Native-ready**: no
- **Fact IDs**: F006

#### Slide 09 - Established-software comparison

- **Audience move**: From asking for a market winner to understanding the executed evidence and the package’s auditable distinction
- **Layout**: Capability matrix with evidence-status badges and a lower fairness note
- **Title**: This Is an Auditability Comparison, Not a Leaderboard
- **Core message**: Established tools provide valuable reference executions, while the local package’s distinctive contribution is explicit paper profiles plus structured provenance and stability artifacts.
- **Content**: Rows: local fci_engine executed; R pcalg 2.7-12 executed; Python causal-learn executed standard FCI; Tetrad 7.6.10 documentation-only. Columns: FCI, public FCI+, CI interface, structured trace/provenance, stability audit, artifact export, evidence status. State endpoint encodings and default CI tests must be normalized before graph comparison.
- **Visualization**: Data-driven feature matrix from dated software inventory
- **Native-ready**: no
- **Fact IDs**: F007, F008, F009

#### Slide 10 - Matched known-truth software benchmark

- **Audience move**: From feature documentation to a fairer empirical comparison under a common synthetic target
- **Layout**: Algorithm rows against skeleton F1, endpoint-sensitive F1, CI calls, and elapsed time; skip badges remain visible
- **Title**: Matched Executions Show Trade-offs, Not Universal Dominance
- **Core message**: Under a shared known-truth benchmark, recovery quality and computational work must be read together, with unavailable or incompatible runs left explicit.
- **Content**: Compare local paper FCI, local paper FCI+, local robust FCI+, causal-learn FCI, and pcalg FCI+ using repeated N=2,500 cases. Use the committed aggregate values generated by the benchmark. Label versions, profile configuration, repeat count, and skip reasons. State that timing is not a cross-language speed contest.
- **Visualization**: Data-driven dot-and-range or compact bar comparison from software_benchmark_summary.csv
- **Native-ready**: no
- **Fact IDs**: F007, F008

### Part 4: Tennessee STAR application

#### Slide 11 - STAR cohort and identification boundary

- **Audience move**: From seeing a famous randomized experiment to understanding the analyzed cohorts and selection risks
- **Layout**: Cohort funnel with three analysis panels and a separate identification-boundary box
- **Title**: STAR Supplies Randomization—and Later-Outcome Selection Risk
- **Core message**: The randomized kindergarten assignment is the strongest design basis for an observed-arm contrast; later complete-case discovery panels are selection-sensitive diagnostics.
- **Content**: 11,601 raw records → 6,325 kindergarten assignment cohort across 79 schools → attrition panel 5,744/9 nodes, longitudinal panel 2,787/9 nodes, focused panel 2,976/8 nodes. Note that no temporal constraints were supplied to discovery. Distinguish the 1,000 school-cluster contrast bootstrap from the 12-resample PAG stability audit. State that missing outcomes, original blocks, switching, and compliance were not reconstructed, so the selected observed-arm contrast is not a fully identified selected-subset causal effect.
- **Visualization**: Data-driven cohort funnel
- **Native-ready**: no
- **Fact IDs**: F010

#### Slide 12 - Randomized-arm contrasts

- **Audience move**: From PAG interpretation to the strongest design-based causal evidence
- **Layout**: Forest plot occupying two thirds, with a concise design-based conclusion card at right
- **Title**: The Strongest Causal Signal Is the Randomized Kindergarten Contrast
- **Core message**: The design-supported observed kindergarten arm contrast is consistent with a small-class benefit, while the aide contrast is near zero at this precision; selected-subset causal identification remains incomplete.
- **Content**: Small − regular kindergarten observed-score contrast +13.90 [5.21, 22.07], described as design-supported and consistent with benefit but not a fully identified selected-subset causal effect. Aide − regular kindergarten score +0.31 [−7.40, 7.38]. Observed grade-3 small − regular +11.69 [3.74, 19.38], explicitly labeled descriptive follow-up among observed students. Include grade-3 observation-rate contrasts Small − Regular +1.54 pp [−1.73, 4.56] and Aide − Regular −2.05 pp [−5.35, 1.41], plus the 1,000 school-cluster bootstrap note.
- **Visualization**: Data-driven forest plot from star_descriptive_contrasts.csv
- **Native-ready**: no

#### Slide 13 - Structural stability and endpoint uncertainty

- **Audience move**: From asking for a directed causal graph to accepting stable adjacencies and unresolved directions as the more credible discovery result
- **Layout**: Order-audit heatmap left; chronology/endpoint comparison strip right; evidence legend across the top
- **Title**: In STAR, Stable Adjacencies Are More Credible than Directions
- **Core message**: The robust profile is preferred under the predeclared stability-and-chronology criterion, but its circles do not add causal identification.
- **Content**: Robust FCI+ reproduced the complete baseline PAG in all 26 tested cyclic-order refits and had zero audited temporal flags. Early achievement—later achievement and early achievement—observation adjacencies have 100% local bootstrap frequency. Show paper FCI+ and R FCI+ chronology reversals versus robust circles. Footnote that 26 tested shifts are not a theorem of order invariance and the focused class—grade-3 adjacency disappears at alpha 0.01.
- **Visualization**: Data-driven order-audit heatmap and endpoint strip
- **Native-ready**: no

#### Slide 14 - STAR computational comparison

- **Audience move**: From assuming fewer CI tests means stronger causal evidence to separating efficiency from identification
- **Layout**: Three small multiples for attrition, longitudinal, and focused panels with CI calls and median runtime
- **Title**: FCI+ Reduced CI Work—but Efficiency Is Not Evidential Strength
- **Core message**: Paper FCI+ reduced CI calls on all three STAR panels, yet query efficiency neither guarantees finite-sample accuracy nor strengthens a causal claim.
- **Content**: Paper FCI+ / standard FCI CI calls: 47% attrition, 35% longitudinal, 64% focused. Show attrition 793 vs 1,705; longitudinal 335 vs 960; focused 185 vs 291. Include robust and R pcalg FCI+ values and note that three-run wall-clock measurements are not a cross-language contest.
- **Visualization**: Data-driven grouped horizontal bars from star_benchmark.csv
- **Native-ready**: no

### Part 5: Conclusion

#### Slide 15 - Evidence hierarchy and bounded contribution

- **Audience move**: From remembering individual graphs and metrics to retaining the defensible causal conclusion and research contribution
- **Layout**: Four-rung evidence ladder left, three deliverable cards right, and a slim next-validation arrow along the bottom
- **Title**: The Contribution Is a Reproducible Bridge from Source Text to Cautious Evidence
- **Core message**: Randomized STAR evidence is primary; stable PAG adjacencies are supportive structure; sensitive directions and treatment-effect claims remain unresolved.
- **Content**: Design-based: the observed kindergarten arm contrast favors small over regular classes and is consistent with benefit under random assignment, but missing outcomes, blocks, switching, and compliance prevent a fully identified selected-subset causal-effect claim; no comparable aide advantage is detected. Structural: early achievement is stably connected to later achievement and later observation. Uncertain: later directions and the focused class—grade-3 adjacency are specification-sensitive. Not identified: no unique DAG, latent cause, adjustment set, or treatment-effect magnitude comes from the PAG. Deliverables: paper profiles and source map; exact/simulated/software audits; report, data artifacts, and advisor deck. Next: broader CI regimes, larger benchmark suite, formal randomized analysis, external replication.
- **Visualization**: Conceptual evidence ladder and deliverable summary; not data-driven
- **Closing impact**: Close with the project’s bounded contribution and a clear discussion invitation

## X. Speaker Notes Requirements

- **Filename**: match each SVG filename under `notes/`
- **Content**: Full English narration with timing, source locators, and explicit caveats separating oracle results, finite-sample simulations, documentation-only comparisons, randomized STAR contrasts, and exploratory PAG output
- **Total duration**: 18 minutes plus discussion
- **Notes style**: formal, concise, and technically transparent
- **Presentation purpose**: defend the implementation, report the evidence, and persuade the advisor that the project’s restraint and auditability are methodological strengths
