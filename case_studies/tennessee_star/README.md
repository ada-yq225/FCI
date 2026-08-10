# Tennessee STAR Application

This directory is the second stage of the project: an applied case study built
on top of the completed `fci_engine` package.

The separation is intentional:

- `src/fci_engine/` contains the general FCI and FCI+ algorithms.
- `case_studies/tennessee_star/` contains STAR data acquisition, coding,
  cohort choices, resampling, domain audits, and the visual report.
- The algorithm package never imports this case study.

## Data source

The committed data are from:

> C. M. Achilles, Helen Pate Bain, Fred Bellott, Jayne Boyd-Zaharias,
> Jeremy Finn, John Folger, John Johnston, and Elizabeth Word (2008),
> "Tennessee's Student Teacher Achievement Ratio (STAR) project",
> Harvard Dataverse, V1,
> [doi:10.7910/DVN/SIWH9F](https://doi.org/10.7910/DVN/SIWH9F).

The Dataverse record is released under CC0 1.0. The local case study includes:

- a deterministic gzip of the official 11,601 × 379 student-level tab export;
- the official 146-page STAR user guide;
- [SOURCE.json](data/raw/SOURCE.json), with file IDs and SHA-256 hashes.

The experiment and its three class-size arms are documented in the
[official Tennessee STAR technical report](https://eric.ed.gov/?id=ED328356).

Run the downloader to replace or verify the local copies:

```bash
PYTHONPATH=src python -m case_studies.tennessee_star.download_data --force
```

## Analysis design

The analysis starts from the 6,325 students with a kindergarten STAR class
assignment in 79 schools. It builds three panels:

| Panel | Purpose |
| --- | --- |
| `attrition` | Relate observed kindergarten characteristics and achievement to whether both grade-3 scores are observed. |
| `longitudinal` | Discover structure between kindergarten and grade-3 achievement among complete cases. |
| `focused_treatment` | Examine the kindergarten class / grade-3 achievement relation without allowing kindergarten achievement to be used as a separator. |

The raw data mix categorical variables, counts, and test scores. The primary
analysis therefore:

1. preserves naturally categorical variables;
2. collapses sparse race categories into `Other`;
3. bins age, teacher experience, and achievement;
4. applies the discrete likelihood-ratio G-square CI test;
5. uses the paper profile for standard FCI;
6. uses the Claassen et al. paper profile for FCI+ with `k=3`;
7. runs CRAN `pcalg::fciPlus` as an independent R reference;
8. separately runs `fci_plus(..., profile="practical")` as the recommended
   finite-sample application result;
9. audits every local algorithm under every cyclic variable ordering;
10. resamples whole kindergarten schools 100 times for adjacency stability.

The paper-profile runs answer “did the implementation follow and reproduce the
published algorithms?” The robust application run answers “which conclusions
survive finite-sample safeguards?” They are intentionally separate. The robust
profile uses stable depth-wise skeleton updates, strongest-at-depth separating
sets, conservative colliders, and cautious orientation. It does not modify the
paper implementation and does not add domain-forced arrows.

The R runner supplies a compact-table G-square function matching
`src/fci_engine/ci/discrete.py`. It intentionally does not use the default
`pcalg::disCItest` minimum-sample shortcut, because that shortcut can classify
high-dimensional contingency tables as independent without calculating the
same statistic used by the Python implementations. The `pcalg` API also does
not expose the local paper profile's explicit `k=3` bound, so its result is an
independent implementation comparison rather than an identical internal
search schedule.

The main threshold is `alpha=0.05`. A sensitivity table repeats the focused
treatment analysis at `alpha=0.01`, with three versus four quantile bins, and
with FCI+ sparsity bounds `k=2,3,4`. These are bounded analysis settings, not
claims that the unknown STAR MAG has a known maximum degree.

## Reproduce the report

Install R and CRAN `pcalg`, then run:

```bash
Rscript case_studies/tennessee_star/pcalg_reference.R
PYTHONPATH=src python -m case_studies.tennessee_star.run_case_study
```

To refresh the R results and the Python report in one command:

```bash
PYTHONPATH=src python -m case_studies.tennessee_star.run_case_study \
  --refresh-pcalg \
  --rscript /path/to/Rscript
```

The command writes:

- `output/star_case_study_report.html`: standalone visual report;
- `output/star_case_study_summary.json`: complete machine-readable result;
- `output/star_benchmark.csv`: runtime and CI-test comparison;
- `output/star_pag_edges.csv`: every learned PAG edge;
- `output/star_bootstrap_adjacencies.csv`: school-bootstrap frequencies;
- `output/star_python_order_audit.csv`: all cyclic-order refits for the three
  local analysis profiles;
- `output/star_sensitivity.csv`: alpha/binning/FCI+ `k` sensitivity;
- `output/star_descriptive_contrasts.csv`: randomized-arm summaries.
- `output/star_pcalg_runs.csv`: R and `pcalg` versions, timings, and CI calls;
- `output/star_pcalg_edges.csv`: every R `pcalg::fciPlus` PAG edge;
- `output/star_pcalg_order_audit.csv`: cyclic variable-order audit.

The exact numeric FCI inputs are written under `data/processed/`.

## Interpretation boundary

The report deliberately presents three different forms of evidence:

- randomized-arm score contrasts, used as the external experimental reference;
- self-implemented FCI/FCI+ PAGs;
- an independently executed R `pcalg::fciPlus` PAG.

The three paper-comparison algorithms return an identical focused-treatment
PAG, but do not fully agree on the attrition and longitudinal endpoint
orientations. The R implementation gives the temporally sensible
`K_Achievement --> Grade3_Observed` relation in the attrition panel, while both
FCI+ implementations reverse the kindergarten/grade-3 achievement chronology
in the longitudinal panel.

The separate robust FCI+ application result is exactly invariant across all
cyclic column orders in all three panels, retains each main target adjacency,
and contains no fully directed temporal reversal. Its main target edges are
`o-o`: the improvement comes from refusing to assert unstable directions, not
from discovering stronger causal identification. Consequently, the project
uses the robust profile for applied skeleton-level conclusions but does not
label any empirical PAG as universally true. Randomized design evidence
remains strongest for the treatment effect, and cross-implementation,
cross-order skeleton agreement is more credible than
implementation-specific arrowheads.

The PAG does not estimate a class-size treatment effect. A bidirected edge is
not automatically proof of latent confounding, particularly after restricting
the analysis to students with observed grade-3 outcomes. Backward temporal
arrows, alpha sensitivity, sparse contingency tables, and school/classroom
clustering are reported as audit limitations instead of being hidden.
