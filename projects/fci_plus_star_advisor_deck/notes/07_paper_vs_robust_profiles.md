[Timing: 1:05]

The package keeps paper fidelity and practical robustness as distinct configurations. The paper FCI+ profile asks the user for k and follows the source-aligned sparse-search policy, including first-found separating sets and the standard Zhang-rule orientation path. The practical profile is a finite-sample engineering policy: it stabilizes the skeleton, selects maximum-p-value separating sets, treats collider evidence conservatively, restricts tail orientation, and adds cyclic-order and bootstrap diagnostics. This choice deliberately trades more conditional-independence work and more circle endpoints for stable skeletons and fewer brittle directions. It is not described as the paper’s FCI+, and it does not create a new correctness or completeness theorem. In STAR, I prefer it only under a predeclared stability-and-chronology criterion; another application could rationally prefer the literal paper profile.

[Sources]
- `src/fci_engine/config.py::FCIPlusConfig.paper` and `src/fci_engine/config.py::FCIPlusConfig.practical`.
- `reports/research/claim_evidence_matrix.csv`, claim IDs `IMPL-PAPER-FCIPLUS`, `IMPL-ROBUST`, and `FCIPLUS-FINITE`.
- `tests/test_robust_application_profile.py::test_practical_profile_improves_seeded_finite_sample_endpoint_recovery`.
[/Sources]