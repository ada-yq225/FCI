# 01_independent_fci_and_fci_plus

[Timing: 0:55]

This project makes one contribution across two deliberately separate tracks. On the algorithm track, I return to the original FCI and FCI+ sources, implement their stages independently in Python, and validate the implementation where the graphical truth is known. On the application track, I begin with the randomized design of Tennessee STAR, estimate transparent observed-arm contrasts, and use PAG discovery only as a secondary structural diagnostic. The evidence hierarchy on the right is the rule for the entire defense: the randomized assignment gives the observed-arm contrast its strongest design support, stable adjacencies can support a structural interpretation, and unresolved endpoints must remain uncertainty. Missing outcomes and unreconstructed blocks, switching, and compliance still limit selected-subset causal interpretation. The result is an auditable evidence chain, not a claim that observational data reveal one definitive causal DAG.

[Sources]
- `reports/research/fci_fci_plus_source_dossier.md`, sections “Spirtes, Glymour, and Scheines (2000)” and “Claassen, Mooij, and Heskes (2013).”
- `case_studies/tennessee_star/output/star_case_study_summary.json`, keys `cohorts`, `descriptives.contrasts`, and `robust_application_comparisons`.
- `reports/research/claim_evidence_matrix.csv`, claim IDs `FCI-STAGES`, `FCIPLUS-ALGO2`, `IMPL-AUDIT`, and `OUTPUT-SEMANTICS`.
[/Sources]

---

# 02_latent_variables_and_selection

[Timing: 1:00]

FCI is needed because the observed variables may be only a margin of the true causal system. In the schematic, an unobserved common cause affects X and Y, while conditioning on a selection variable can induce additional associations in the observed sample. After latent variables are marginalized and selection is represented, directed acyclic graphs over measured variables are generally the wrong output target. A partial ancestral graph, or PAG, instead represents an equivalence class of maximal ancestral graphs. A tail, arrowhead, or circle is an ancestral constraint, not an estimated direct effect; a circle explicitly reports that an endpoint is unresolved. I will also keep exact conditional-independence oracles separate from sample tests, because the source theorems assume the former while the STAR analysis must use the latter.

[Sources]
- Spirtes, Glymour, and Scheines, *Causation, Prediction, and Search*, 2nd ed. (2000), Chapter 6 §§6.3 and 6.5–6.7, printed pp. 129–145.
- Richardson and Spirtes (2002), *Annals of Statistics* 30(4), pp. 962–1030, DOI `10.1214/aos/1031689015`.
- `reports/research/claim_evidence_matrix.csv`, claim IDs `FCI-LATENT`, `PAG-MODERN`, and `OUTPUT-SEMANTICS`.
[/Sources]

---

# 03_five_auditable_questions

[Timing: 0:45]

The project is evaluated through five questions rather than one vague claim that an algorithm is best. First, what do Spirtes and colleagues in 2000 and Claassen and colleagues in 2013 actually assume and prove? Second, where is every source stage implemented and tested? Third, what happens on graphs where the correct PAG is known independently? Fourth, which comparisons with existing software are technically fair after normalizing endpoints, tests, and profiles? Fifth, which STAR findings arise from random assignment, which are structurally stable, and which remain exploratory? The order matters: source fidelity precedes validation, and validation precedes application interpretation. A visually attractive real-data graph cannot substitute for any earlier layer.

[Sources]
- `reports/research/claim_evidence_matrix.csv`, complete claim inventory and locator columns.
- `reports/research/fci_fci_plus_source_dossier.md`, sections “Source-to-implementation mapping,” “Claims that require finite-sample qualification,” and “Software-comparison evidence policy.”
- `case_studies/tennessee_star/output/star_case_study_summary.json`, key `evidence_hierarchy`.
[/Sources]

---

# 04_standard_fci_spirtes_2000

[Timing: 1:25]

The historical FCI schedule begins with a complete undirected graph, performs a PC-style adjacency search, stores separating sets, and orients unshielded colliders. The central non-local correction is Possible-D-SEP: for each endpoint order, FCI searches vertices reachable along paths whose intermediate triples are colliders or members of triangles. This computable set is a superset of the graphical D-SEP information that may contain a separating set missed by local neighborhoods. When a new independence is found, FCI removes the edge, resets marks, and runs its orientation phase again. Theorem 6.4 in the 2000 book is an oracle-scope soundness result under Faithfulness and correct conditional-independence decisions. It returns a partially oriented inducing-path graph; the same source explicitly did not establish that its historical rule schedule was maximally informative. Modern PAG orientation under Zhang’s later rule set is therefore a separate ingredient, not something retroactively attributed to the book.

[Sources]
- Spirtes, Glymour, and Scheines, *Causation, Prediction, and Search*, 2nd ed. (2000), Possible-D-SEP and Fast Causal Inference Algorithm, Chapter 6 §6.7, printed pp. 144–145.
- Spirtes et al. (2000), Theorem 6.4 and the immediately following paragraph, printed p. 145.
- Zhang (2008), *Artificial Intelligence* 172(16–17), §§3–4, pp. 1873–1896, DOI `10.1016/j.artint.2008.08.001`.
- `reports/research/claim_evidence_matrix.csv`, claim IDs `FCI-PDSEP`, `FCI-STAGES`, `FCI-SOUND`, and `FCI-INCOMPLETE`.
[/Sources]

---

# 05_fci_plus_claassen_2013

[Timing: 1:35]

FCI+ does not merely truncate Possible-D-SEP. Its contribution is a sparse logical characterization of the non-local edges that need attention. Algorithm 2 begins with a locally bounded PC skeleton, constructs an augmented skeleton whose temporary arrowheads encode particular dependence information, recognizes candidate D-SEP-link witnesses, and builds a recursive hierarchy from known separating sets and collider structure. It then tests and minimizes a targeted separator, removes a verified D-SEP link, revisits the candidate set, and finally applies the established Zhang-rule orientation schedule. Claassen and colleagues derive a query bound of O of N to the power two times k plus four, equivalently O of N to the two times open-parenthesis k plus two close-parenthesis. That statement requires a fixed degree bound k for the observed maximal ancestral graph, Faithfulness, and a constant-time exact oracle. It is not a finite-sample accuracy theorem and not a wall-clock guarantee. The supplement abstract contains a different exponent, but the main paper and the supplement’s theorem body support the exponent shown here, so the report documents the discrepancy rather than silently choosing one.

[Sources]
- Claassen, Mooij, and Heskes (2013), “Learning Sparse Causal Models is not NP-hard,” Algorithm 2, p. 178, and complexity analysis, p. 179.
- Claassen, Mooij, and Heskes proof supplement (2014), Theorem 1 and proof, PDF pp. 10–11; compare the arXiv abstract.
- `reports/research/claim_evidence_matrix.csv`, claim IDs `FCIPLUS-ORACLE`, `FCIPLUS-SPARSITY`, `FCIPLUS-AUGMENT`, `FCIPLUS-HIERARCHY`, `FCIPLUS-REVISIT`, `FCIPLUS-ALGO2`, `FCIPLUS-COMPLEXITY`, and `FCIPLUS-EXPONENT-CAVEAT`.
[/Sources]

---

# 06_source_code_test_traceability

[Timing: 1:10]

The implementation claim is inspectable at module and symbol level. Standard FCI is divided across the skeleton, Possible-D-SEP, orientation, and rule modules; the public fit method calls concrete functions such as possible_dsep and refine_skeleton_with_pdsep. FCI+ is separated into D-SEP and algorithm modules that expose the augmented skeleton, candidate D-SEP links, recursive hierarchy, and FCI+ skeleton refinement. These paths are checked by exact-oracle recovery fixtures, unit tests for the path and hierarchy logic, and regression tests for the historical and Zhang-rule orientation schedules. A third row is intentionally labeled as engineering rather than paper content: the PAG and result objects retain CI and orientation traces, the source of every separating set, per-edge explanations, invariants, stability audits, and exportable artifacts. The linked dossier contains twenty source-to-code-to-test rows. The package therefore offers simple fci and fci_plus entry points while preserving enough internal evidence for a user to audit how the result was obtained.

[Sources]
- `reports/research/fci_fci_plus_source_dossier.md`, table “Source-to-implementation mapping,” 20 data rows.
- `src/fci_engine/discovery/fci.py::FCI.fit`, `src/fci_engine/discovery/pdsep.py::possible_dsep`, and `src/fci_engine/discovery/pdsep.py::refine_skeleton_with_pdsep`.
- `src/fci_engine/discovery/dsep.py::build_augmented_skeleton`, `possible_dsep_links`, `hierarchy`, and `refine_skeleton_with_fci_plus_dsep`.
- `tests/test_pdsep.py`, `tests/test_fci_plus.py`, `tests/test_published_reference_graphs.py`, and `tests/test_result_exports.py`.
[/Sources]

---

# 07_paper_vs_robust_profiles

[Timing: 1:05]

The package keeps paper fidelity and practical robustness as distinct configurations. The paper FCI+ profile asks the user for k and follows the source-aligned sparse-search policy, including first-found separating sets and the standard Zhang-rule orientation path. The practical profile is a finite-sample engineering policy: it stabilizes the skeleton, selects maximum-p-value separating sets, treats collider evidence conservatively, restricts tail orientation, and adds cyclic-order and bootstrap diagnostics. This choice deliberately trades more conditional-independence work and more circle endpoints for stable skeletons and fewer brittle directions. It is not described as the paper’s FCI+, and it does not create a new correctness or completeness theorem. In STAR, I prefer it only under a predeclared stability-and-chronology criterion; another application could rationally prefer the literal paper profile.

[Sources]
- `src/fci_engine/config.py::FCIPlusConfig.paper` and `src/fci_engine/config.py::FCIPlusConfig.practical`.
- `reports/research/claim_evidence_matrix.csv`, claim IDs `IMPL-PAPER-FCIPLUS`, `IMPL-ROBUST`, and `FCIPLUS-FINITE`.
- `tests/test_robust_application_profile.py::test_practical_profile_improves_seeded_finite_sample_endpoint_recovery`.
[/Sources]

---

# 08_known_truth_validation

[Timing: 1:25]

Known-truth validation separates implementation logic from sampling error. On the left is the graphical fixture based on Figure 4(b) of the FCI+ paper: the source supplies a maximal ancestral graph and exhibits the separator U, V, Z that removes a false X–Y adjacency. The repository independently specifies the corresponding PAG target rather than claiming that the paper printed one. With an exact oracle, both implemented pipelines recover their targets, and FCI+ uses sixty-three logical CI queries compared with one hundred two for FCI on this fixture. The right panel is a separate fixed-alpha regression: both paper and robust profiles use alpha zero point zero zero one across five graph families, three seeds, and two thousand five hundred observations. Skeleton F1 is zero point nine seven nine for both; endpoint-sensitive exact-edge F1 rises from zero point five three six to zero point seven zero three, while mean CI work rises from one hundred ninety-seven point one to three hundred eighty point seven. These values are not the automatic-alpha software benchmark on the next slides. Fewer oracle queries therefore cannot be converted into an automatic finite-sample superiority claim.

[Sources]
- Claassen, Mooij, and Heskes (2013), §3.2 and Figure 4(b), pp. 175–176.
- `tests/test_published_reference_graphs.py::test_fci_plus_removes_figure4_link_only_in_hierarchical_dsep_stage`.
- `tests/test_robust_application_profile.py::test_practical_profile_improves_seeded_finite_sample_endpoint_recovery`, using `src/fci_engine/simulation/oracle_cases.py::realistic_oracle_cases(n_repeats=3, n_samples=2500)` with each case’s fixed `alpha=0.001`.
- `reports/research/claim_evidence_matrix.csv`, claim IDs `FCIPLUS-FIG4` and `FCIPLUS-FINITE`.
[/Sources]

---

# 09_established_software_comparison

[Timing: 1:10]

The software comparison distinguishes executed evidence from documentation. The local package, R pcalg version two point seven dash twelve, and causal-learn version zero point one point four point seven were executed. Tetrad version seven point six point ten was reviewed through its official documentation only because this environment did not include the required Java runtime. Pcalg provides public FCI and FCI-plus entry points; causal-learn provides standard FCI but no inspected public FCI-plus entry point; and Tetrad documents multiple FCI-family methods but was not benchmarked here. The local package’s distinctive feature is not a universal accuracy ranking. It is the combination of explicit paper profiles with structured CI and orientation traces, separating-set provenance, order and bootstrap audits, edge explanations, and artifact exports. Raw endpoint matrices, default tests, and selection conventions differ across packages, so comparisons require normalization before any exact graph statement is fair.

[Sources]
- `reports/data/software_feature_matrix.csv`, rows `fci_engine`, `pcalg`, `causal-learn`, and `Tetrad`, evidence date 2026-07-26.
- `reports/data/software_landscape.json`, per-tool `evidence_kind`, version, execution status, and official locators.
- `reports/research/claim_evidence_matrix.csv`, claim IDs `SW-PCALG`, `SW-CAUSALLEARN`, and `SW-TETRAD`.
[/Sources]

---

# 10_matched_software_benchmark

[Timing: 1:20]

This matched benchmark executes five labeled configurations on five synthetic graph families, with three repeats and two thousand five hundred samples per case. All seventy-five requested runs completed. The dots show semantic edge F1 and endpoint accuracy on a common scale, while the right column reports CI calls where the API exposes them. The local paper FCI-plus profile uses about one hundred ninety-seven calls on average, compared with about four hundred fifty for historical FCI, and its recovery is close to the causal-learn and pcalg reference executions after normalization. The robust workflow reaches semantic F1 about zero point nine eight five and endpoint accuracy about zero point eight four, but it uses about four hundred two CI calls. The critical qualification is that robust uses the package’s automatic alpha, which equals zero point zero one at this sample size, whereas the paper and reference rows use zero point zero zero one. This is therefore evidence for a complete practical workflow, not an isolated claim that one algorithm definition dominates. Pcalg timing also includes R process startup, so wall-clock values are recorded for reproducibility rather than treated as a cross-language speed contest.

[Sources]
- `reports/data/software_benchmark_summary.csv`, all five algorithm rows; columns `n_requested`, `n_completed`, `mean_semantic_edge_f1`, `mean_endpoint_accuracy`, `mean_ci_test_count`, `alpha_policy`, and `effective_alpha`.
- `reports/data/software_benchmark_cases.csv`, 75 executed case rows.
- `reports/generate_software_comparison.py::build_algorithm_specs` and `::summarize_rows`.
[/Sources]

---

# 11_star_cohort_and_identification

[Timing: 1:10]

The official STAR record has eleven thousand six hundred one student rows. The analysis reconstructs a kindergarten assignment cohort of six thousand three hundred twenty-five students across seventy-nine schools, then creates three distinct discovery panels. The attrition panel contains five thousand seven hundred forty-four observations and nine variables, the longitudinal complete-case panel contains two thousand seven hundred eighty-seven observations and nine variables, and the focused treatment panel contains two thousand nine hundred seventy-six observations and eight variables. These panels answer different diagnostic questions and are not pooled into one causal analysis. The identification boundary is explicit: randomized kindergarten assignment supports the primary design-based arm comparison, while later complete-case panels can reflect conditioning and attrition. Because missing outcomes, original blocks, switching, and compliance were not reconstructed, the selected observed-arm contrast is not presented as a fully identified selected-subset causal effect. No temporal constraints were supplied to discovery, so chronology is used only as an external audit. The arm-contrast intervals use one thousand school-cluster resamples; the one hundred school-resample PAG bootstrap is an adjacency-stability diagnostic and is never presented as a treatment-effect procedure.

[Sources]
- `case_studies/tennessee_star/output/star_case_study_summary.json`, keys `cohorts.raw_rows`, `cohorts.kindergarten_rows`, `cohorts.kindergarten_schools`, and `cohorts.panels`.
- `case_studies/tennessee_star/study.py::prepare_study` and `::save_processed_panels`.
- Tennessee STAR Dataverse dataset, DOI `10.7910/DVN/SIWH9F`.
- `case_studies/tennessee_star/study.py::_cluster_bootstrap_arm_metrics` and `::cluster_bootstrap_adjacencies`.
[/Sources]

---

# 12_star_randomized_contrasts

[Timing: 1:30]

The strongest design-supported result here is the kindergarten observed-arm contrast. Small-class students with observed kindergarten outcomes score thirteen point nine points above regular-class students, with a school-cluster bootstrap interval from five point two one to twenty-two point zero seven. Random assignment makes this contrast consistent with a small-class benefit, but missing outcomes, original blocks, switching, and compliance were not reconstructed, so I do not treat it as a fully identified selected-subset causal effect. The regular-plus-aide arm differs from regular by only zero point three one points, with an interval from negative seven point four zero to seven point three eight, so there is no detectable aide advantage at this precision. Among students with observed grade-three outcomes, the small-minus-regular score contrast is eleven point six nine points with an interval from three point seven four to nineteen point three eight, but that is a descriptive follow-up in a selected observed sample. The corresponding grade-three observation-rate contrasts are plus one point five four percentage points for small versus regular, with an interval from negative one point seven three to four point five six, and negative two point zero five percentage points for aide versus regular, with an interval from negative five point three five to one point four one. Neither observation-rate interval excludes zero.

[Sources]
- `case_studies/tennessee_star/output/star_descriptive_contrasts.csv`, rows `kindergarten_score`, `grade3_score`, and `grade3_observed_rate` for both arm comparisons.
- `case_studies/tennessee_star/output/star_case_study_summary.json`, key `descriptives.contrasts`.
- `case_studies/tennessee_star/study.py::_cluster_bootstrap_arm_metrics`, 1,000 school-cluster resamples.
[/Sources]

---

# 13_star_structural_stability

[Timing: 1:25]

The discovery evidence is more credible at the adjacency level than at the endpoint-direction level. Under cyclic column shifts, robust FCI+ reproduces its entire baseline PAG output in all twenty-six tested refits: nine of nine for attrition, nine of nine for longitudinal data, and eight of eight for the focused panel, with zero recorded temporal flags. Those twenty-six shifts are a finite audit, not a theorem of order invariance. The paper profile and standard FCI are less stable on some panels, and in the longitudinal graph both paper FCI+ and the executed pcalg FCI+ orient later achievement toward kindergarten achievement, which conflicts with chronology. The robust profile leaves the endpoints as circles instead. That is calibrated abstention, not extra identification. Early achievement to later achievement and early achievement to later observation are present in every local bootstrap sample, so those adjacencies deserve more confidence than their directions. The focused class-to-grade-three edge remains especially fragile because it disappears at alpha zero point zero one despite high baseline bootstrap adjacency frequency.

[Sources]
- `case_studies/tennessee_star/output/star_python_order_audit.csv`, all panel/profile exact-reproduction counts and temporal-flag counts.
- `case_studies/tennessee_star/output/star_bootstrap_adjacencies.csv`, 100% local bootstrap adjacency rows.
- `case_studies/tennessee_star/output/star_sensitivity.csv`, focused class-to-grade-three adjacency at `alpha=0.01`.
- `case_studies/tennessee_star/output/star_pcalg_runs.csv`, longitudinal endpoint comparison for `pcalg_fci_plus`.
[/Sources]

---

# 14_star_computational_comparison

[Timing: 1:05]

FCI+ reduces conditional-independence work on each STAR panel. In the attrition panel, paper FCI+ uses seven hundred ninety-three tests compared with one thousand seven hundred five for FCI, or forty-seven percent as many. In the longitudinal panel the comparison is three hundred thirty-five versus nine hundred sixty, or thirty-five percent. In the focused panel it is one hundred eighty-five versus two hundred ninety-one, or sixty-four percent. The runtime strip reports the same three-run medians in the order FCI, paper FCI+, robust FCI+, and R pcalg FCI+: attrition is five point six six, one point eight two, two point six one, and ten point three nine seconds; longitudinal is two point one four, zero point three three, zero point five five, and one point six five; focused is zero point three three, zero point one six, zero point two five, and zero point eight eight. These are implementation measurements, not a language leaderboard. Computational efficiency is useful engineering evidence; it does not increase the causal strength of an endpoint or identify a treatment effect.

[Sources]
- `case_studies/tennessee_star/output/star_benchmark.csv`, all 12 panel-by-algorithm rows; columns `ci_tests` and `median_elapsed_seconds`.
- `case_studies/tennessee_star/output/star_case_study_summary.json`, key `runs`.
- `case_studies/tennessee_star/run_case_study.py`, three-run benchmark loop and timing scope.
[/Sources]

---

# 15_evidence_hierarchy_and_conclusion

[Timing: 1:00]

The final conclusion follows the evidence ladder. At the design-based level, the observed kindergarten arm contrast favors small over regular classes and is consistent with benefit under random assignment, while no comparable aide advantage is detected at the available precision. Because missing outcomes, blocks, switching, and compliance were not reconstructed, this is not a fully identified selected-subset causal effect. At the structural level, early achievement is stably adjacent to later achievement and later observation. At the uncertain level, later endpoint directions and the focused class-to-grade-three adjacency depend on specification. The PAG does not identify a unique DAG, a particular latent cause, an adjustment set, or a treatment-effect magnitude. The completed contribution is the bridge across these layers: source-aligned FCI and FCI-plus profiles, a concrete source-to-code-to-test map, exact and simulated validation, executed software audits, and a reproducible STAR report with data, figures, slides, and notes. The next validation priorities are broader conditional-independence regimes, larger benchmarks, a formal reconstruction of the STAR randomized analysis, and external replication.

[Sources]
- `case_studies/tennessee_star/output/star_descriptive_contrasts.csv`, kindergarten score contrasts and grade-three descriptive follow-up rows.
- `case_studies/tennessee_star/output/star_python_order_audit.csv`, robust 26-of-26 exact reproduction audit.
- `case_studies/tennessee_star/output/star_sensitivity.csv` and `star_bootstrap_adjacencies.csv`, specification-sensitive versus stable adjacencies.
- `reports/research/claim_evidence_matrix.csv`, claim ID `OUTPUT-SEMANTICS`.
[/Sources]
