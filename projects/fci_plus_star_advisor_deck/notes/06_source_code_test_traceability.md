[Timing: 1:10]

The implementation claim is inspectable at module and symbol level. Standard FCI is divided across the skeleton, Possible-D-SEP, orientation, and rule modules; the public fit method calls concrete functions such as possible_dsep and refine_skeleton_with_pdsep. FCI+ is separated into D-SEP and algorithm modules that expose the augmented skeleton, candidate D-SEP links, recursive hierarchy, and FCI+ skeleton refinement. These paths are checked by exact-oracle recovery fixtures, unit tests for the path and hierarchy logic, and regression tests for the historical and Zhang-rule orientation schedules. A third row is intentionally labeled as engineering rather than paper content: the PAG and result objects retain CI and orientation traces, the source of every separating set, per-edge explanations, invariants, stability audits, and exportable artifacts. The linked dossier contains twenty source-to-code-to-test rows. The package therefore offers simple fci and fci_plus entry points while preserving enough internal evidence for a user to audit how the result was obtained.

[Sources]
- `reports/research/fci_fci_plus_source_dossier.md`, table “Source-to-implementation mapping,” 20 data rows.
- `src/fci_engine/discovery/fci.py::FCI.fit`, `src/fci_engine/discovery/pdsep.py::possible_dsep`, and `src/fci_engine/discovery/pdsep.py::refine_skeleton_with_pdsep`.
- `src/fci_engine/discovery/dsep.py::build_augmented_skeleton`, `possible_dsep_links`, `hierarchy`, and `refine_skeleton_with_fci_plus_dsep`.
- `tests/test_pdsep.py`, `tests/test_fci_plus.py`, `tests/test_published_reference_graphs.py`, and `tests/test_result_exports.py`.
[/Sources]