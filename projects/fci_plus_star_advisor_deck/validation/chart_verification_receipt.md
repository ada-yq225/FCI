# Chart Verification Receipt

The page list was derived from `design_spec.md` §IX. Fixed-layout tables and
conceptual diagrams were excluded because their coordinates are not
value-mapped. All numeric values below come from committed project artifacts.

- `08_known_truth_validation.svg` | type=composite horizontal bars |
  mode=decomposable-calc | query scale=0–129.7 in the displayed comparison
  zone | FCI+ / FCI width ratio `273 / 442 = 0.6176`, matching
  `63 / 102 = 0.6176` | fixed-alpha exact-edge-F1 widths preserve
  `.536 / .703`; both-profile skeleton F1 is `0.979487`, displayed as
  `.979` | svg=updated
- `10_matched_software_benchmark.svg` | type=two-series dot plot |
  mode=direct-calc | scale=.70–1.00 | all ten positions recomputed from
  `software_benchmark_summary.csv`; maximum absolute coordinate difference
  from the rounded SVG positions is under 1.2 px | svg=unchanged
- `11_star_cohort_and_identification.svg` | type=funnel |
  mode=formula-verify | first-stage bottom width =
  `672 × 6,325 / 11,601 = 366.3 px`; SVG uses 366 px | the three analysis
  panels are categorical branches rather than a continued proportional
  funnel | svg=updated
- `12_star_randomized_contrasts.svg` | type=forest plot |
  mode=direct-calc | score scale=−10 to +25 over x=365–805 | all four point
  estimates and eight confidence-interval endpoints were recalculated from
  `star_descriptive_contrasts.csv`; SVG coordinates are rounded to the
  nearest pixel | visible observation-rate contrasts also reconcile to
  `+1.54 pp [−1.73, 4.56]` and `−2.05 pp [−5.35, 1.41]` | svg=updated
- `13_star_structural_stability.svg` | type=heatmap |
  mode=manual-verify | exact-match cells checked against
  `star_python_order_audit.csv`: FCI `2/9, 8/9, 6/8`; paper FCI+
  `9/9, 5/9, 6/8`; robust FCI+ `9/9, 9/9, 8/8` | colors follow
  exact/high/low match status consistently | svg=updated
- `14_star_computational_comparison.svg` | type=grouped horizontal bars |
  mode=decomposable-calc | scale=0–2,000 over x=300–1,180, or 0.44 px per CI
  call | all twelve widths recomputed from `star_benchmark.csv`; maximum
  absolute difference from rounded SVG width is under 0.4 px | the compact
  runtime strip reconciles all twelve `median_elapsed_seconds` values after
  two-decimal rounding | svg=updated

`09_established_software_comparison.svg` is a fixed-grid feature matrix:
values drive cell labels, not geometry, so it is outside coordinate
verification. The chart-marker is retained for downstream provenance.
