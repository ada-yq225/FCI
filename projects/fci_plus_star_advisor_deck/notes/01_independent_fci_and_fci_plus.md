[Timing: 0:55]

This project makes one contribution across two deliberately separate tracks. On the algorithm track, I return to the original FCI and FCI+ sources, implement their stages independently in Python, and validate the implementation where the graphical truth is known. On the application track, I begin with the randomized design of Tennessee STAR, estimate transparent observed-arm contrasts, and use PAG discovery only as a secondary structural diagnostic. The evidence hierarchy on the right is the rule for the entire defense: the randomized assignment gives the observed-arm contrast its strongest design support, stable adjacencies can support a structural interpretation, and unresolved endpoints must remain uncertainty. Missing outcomes and unreconstructed blocks, switching, and compliance still limit selected-subset causal interpretation. The result is an auditable evidence chain, not a claim that observational data reveal one definitive causal DAG.

[Sources]
- `reports/research/fci_fci_plus_source_dossier.md`, sections “Spirtes, Glymour, and Scheines (2000)” and “Claassen, Mooij, and Heskes (2013).”
- `case_studies/tennessee_star/output/star_case_study_summary.json`, keys `cohorts`, `descriptives.contrasts`, and `robust_application_comparisons`.
- `reports/research/claim_evidence_matrix.csv`, claim IDs `FCI-STAGES`, `FCIPLUS-ALGO2`, `IMPL-AUDIT`, and `OUTPUT-SEMANTICS`.
[/Sources]