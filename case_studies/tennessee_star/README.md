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
7. resamples whole kindergarten schools for adjacency stability.

The main threshold is `alpha=0.05`. A sensitivity table repeats the focused
treatment analysis at `alpha=0.01` and with three versus four quantile bins.

## Reproduce the report

```bash
PYTHONPATH=src python -m case_studies.tennessee_star.run_case_study
```

The command writes:

- `output/star_case_study_report.html`: standalone visual report;
- `output/star_case_study_summary.json`: complete machine-readable result;
- `output/star_benchmark.csv`: runtime and CI-test comparison;
- `output/star_pag_edges.csv`: every learned PAG edge;
- `output/star_bootstrap_adjacencies.csv`: school-bootstrap frequencies;
- `output/star_sensitivity.csv`: alpha/binning sensitivity;
- `output/star_descriptive_contrasts.csv`: randomized-arm summaries.

The exact numeric FCI inputs are written under `data/processed/`.

## Interpretation boundary

The report deliberately presents two different forms of evidence:

- randomized-arm score contrasts, used as the external experimental reference;
- FCI/FCI+ PAGs, used to demonstrate structure discovery under possible latent
  confounding and selection.

The PAG does not estimate a class-size treatment effect. A bidirected edge is
not automatically proof of latent confounding, particularly after restricting
the analysis to students with observed grade-3 outcomes. Backward temporal
arrows, alpha sensitivity, sparse contingency tables, and school/classroom
clustering are reported as audit limitations instead of being hidden.
