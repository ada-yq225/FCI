# Logbook

## 15 August 2026 — Development history merged into this repository

Since the start of the project I have been developing the code and the
report in a personal repository, <https://github.com/ada-yq225/FCI>,
and did not push my day-to-day work to this course repository. That was
an oversight on my part, and I am correcting it today.

The complete, unmodified development history has now been merged into
this repository (merge commit `fa75e15`): 63 commits authored between
22 May 2026 and 15 August 2026. Commit hashes, author dates, and commit
messages are exactly as they were written during development, so the
commit graph in this repository now documents the actual step-by-step
progression of the work. The history can be inspected with
`git log --graph` or on the GitHub commits page.

The open inactivity issues (#1–#14) reflect that my activity was
happening in the personal repository during that period; I have left
them open, in accordance with the course rules.

From today onwards, all work is committed directly to this repository,
and this logbook will be kept up to date.

## Development record, commit by commit

The record below describes every commit in the merged development
history, oldest first. Each description corresponds to the actual diff
of that commit, which can be inspected in this repository.

### Phase 1 — Core algorithm package (22 May)

- `a81d797` (22 May) Initialize the `fci_engine` package: packaging
  metadata, module layout, and empty subpackage skeletons.
- `833fee5` (22 May) Implement the PAG data structures: the endpoint
  mark representation and the `PAG` class with adjacency, mutation
  validation, and query methods.
- `f2913b9` (22 May) Add the conditional-independence test framework:
  the `CITest` protocol, structured test results, and the Fisher-Z
  test.
- `e17ede0` (22 May) Add the canonical CI query cache with hit counting
  and tabular input validation.
- `f83ea65` (22 May) Implement the PC-style initial skeleton search.
- `b1267b3` (22 May) Record separating sets and orient unshielded
  colliders from them.
- `fc9669a` (22 May) Implement standard FCI's Possible-D-Sep skeleton
  refinement.
- `a529d98` (22 May) Add the first FCI orientation rule set.

### Phase 2 — Algorithm hardening and first benchmarks (26–30 May)

- `0e819ee` (26 May) Broad expansion of the project in one large
  commit: the unit-test suite, example scripts, and supporting
  modules were added alongside algorithm fixes (167 files).
- `171cb87` (26 May) Improve the FCI+ estimator and its sparse D-SEP
  stage.
- `5fdcf73` (26 May) Improve skeleton stability and remove generated
  files that had been committed by mistake.
- `8e302ac` (26 May) Improve collider-orientation stability and CI test
  performance.
- `01e702f` (26 May) Prefer the strongest (maximum p-value) separating
  set at a given depth instead of the first found.
- `a6e8954` (26 May) Add exact-oracle diagnostics and a conservative
  orientation option, with the MAG simulation modules.
- `7598bea` (26 May) Show R `pcalg` output inside the visual benchmark
  report.
- `6ed513e` (26 May) Focus the visual benchmark on the `pcalg`
  comparison.
- `bf051ee` (27 May) Add the leaf-tail orientation strategy for FCI+.
- `16e1aa9` (29 May) Add the robust orientation profile combining
  conservative colliders with the leaf-tail schedule.
- `cb68483` (29 May) Improve benchmark report readability.
- `40415bf` (29 May) Keep the benchmark graph comparison in one row.
- `50742b0` (30 May) Remove non-English text from the project.
- `933eb45` (30 May) Improve robustness defaults and the README.

### Phase 3 — Compatibility and interactive reporting (31 May – 5 June)

- `709067e` (31 May) Expand supported Python versions (3.9–3.13).
- `9a17c39` (31 May) Add interactive per-edge explanations to the
  benchmark report.
- `e72b58b` (1 June) Add the user-facing interactive HTML report
  export on `FCIResult`.
- `651ca73` (1 June) Polish interactive report usability.
- `6e2dfa3` (1 June) Simplify the interactive report layout.
- `9402d02` (1 June) Fix the aggregate benchmark report layout.
- `0bf2eff` (1 June) Modernize the interactive report UI and add a
  custom-data demo script.
- `6846b10` (1 June) Show all benchmark cases in the visual report.
- `58b2afd` (5 June) Translate the custom-data demo text to English.

### Phase 4 — FCI+ source fidelity (26–29 June)

- `84810b8` (26 June) Strengthen FCI+ fidelity to the published
  algorithm: D-SEP candidate recognition and hierarchy behaviour, with
  regression tests.
- `34129ac` (29 June) Improve the advisor-facing benchmark report.

### Phase 5 — Paper alignment and the Tennessee STAR study (15–16 July)

- `376df49` (15 July) Align the FCI+ implementation with the published
  Algorithm 2: literal paired endpoint-base enumeration and profile
  separation.
- `d0b3fb0` (15 July) Refresh the README validation evidence.
- `6b488cd` (16 July) Harden both algorithms and the release checks:
  strict typing, packaging validation, and broad test additions
  (64 files).
- `e2c50df` (16 July) Fix Python 3.9 MyPy compatibility in CI.
- `b524618` (16 July) Add the Tennessee STAR application case study:
  official Dataverse data with SHA-256 verification, cohort coding,
  three analysis panels, discovery suite, and the standalone HTML
  report.

### Phase 6 — Three-algorithm analysis, research report, and advisor deck (26–27 July)

- `6b80604` (26 July) Add the three-algorithm STAR analysis with an
  independently executed R `pcalg::fciPlus` reference and matched
  G-square semantics.
- `995fc82` (26 July) Harden validation and regression coverage.
- `088f9ae` (26 July) Align the FCI algorithms and stabilize CI.
- `fb11be2` (26 July) Improve the robust FCI+ application workflow:
  cyclic-order audit, cluster bootstrap, and the robust profile in the
  case study.
- `41f3a14` (26 July) Write the design document for the research report
  and advisor deck.
- `587039d` (26 July) Write the implementation plan for the research
  report and advisor deck.
- `2f07639` (26 July) Ignore local git worktrees.
- `3c9ba76` (26 July) Add the source-verified FCI/FCI+ research dossier
  with claim-evidence locators.
- `4a56a8f` (26 July) Repair research evidence locators.
- `13580f8` (26 July) Benchmark established FCI software (causal-learn
  and `pcalg`) on committed known-truth cases.
- `1541a27` (26 July) Preserve external benchmark failures instead of
  silently skipping them.
- `c45dfa0` (26 July) Add algorithm and software comparison figures.
- `bc18f47` (26 July) Keep research figures fully vector.
- `dec3fe2` (26 July) Rewrite the full LaTeX research report.
- `9064b74` (26 July) Harden the report's evidence claims.
- `bff28b2` (26 July) Add the English advisor presentation sources
  (slides, notes, and chart verification materials).
- `511f48f` (26 July) Format the report validation code.
- `26ac9dc` (27 July) Install the report plotting dependency in CI.
- `d8b243a` (27 July) Pin the Python 3.9 plotting toolchain in CI.
- `4f0a60b` (27 July) Fix report and deck visual layout issues.

### Phase 7 — Advisor package completion (10 August)

- `56f348e` (10 Aug) Complete the FCI+ STAR advisor package: final
  report build, presentation exports, and figure refresh.
- `90d7e33` (10 Aug) Fix CI formatting and STAR type checks.

### Phase 8 — Report figure quality and condensed report (15 August)

- `1de2c5b` (15 Aug) Improve STAR report PAG edge routing: separated
  node ports, curved detours around collinear nodes, wider tier
  spacing, and routing regression tests.
- `4121496` (15 Aug) Reuse the same routing geometry in the PDF report
  figures so the HTML and PDF views cannot drift apart.
- `7457796` (15 Aug) Add the condensed 20-page research report
  alongside the full report.
