[Timing: 1:25]

Known-truth validation separates implementation logic from sampling error. On the left is the graphical fixture based on Figure 4(b) of the FCI+ paper: the source supplies a maximal ancestral graph and exhibits the separator U, V, Z that removes a false X–Y adjacency. The repository independently specifies the corresponding PAG target rather than claiming that the paper printed one. With an exact oracle, both implemented pipelines recover their targets, and FCI+ uses sixty-three logical CI queries compared with one hundred two for FCI on this fixture. The right panel is a separate fixed-alpha regression: both paper and robust profiles use alpha zero point zero zero one across five graph families, three seeds, and two thousand five hundred observations. Skeleton F1 is zero point nine seven nine for both; endpoint-sensitive exact-edge F1 rises from zero point five three six to zero point seven zero three, while mean CI work rises from one hundred ninety-seven point one to three hundred eighty point seven. These values are not the automatic-alpha software benchmark on the next slides. Fewer oracle queries therefore cannot be converted into an automatic finite-sample superiority claim.

[Sources]
- Claassen, Mooij, and Heskes (2013), §3.2 and Figure 4(b), pp. 175–176.
- `tests/test_published_reference_graphs.py::test_fci_plus_removes_figure4_link_only_in_hierarchical_dsep_stage`.
- `tests/test_robust_application_profile.py::test_practical_profile_improves_seeded_finite_sample_endpoint_recovery`, using `src/fci_engine/simulation/oracle_cases.py::realistic_oracle_cases(n_repeats=3, n_samples=2500)` with each case’s fixed `alpha=0.001`.
- `reports/research/claim_evidence_matrix.csv`, claim IDs `FCIPLUS-FIG4` and `FCIPLUS-FINITE`.
[/Sources]