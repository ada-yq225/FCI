[Timing: 1:05]

FCI+ reduces conditional-independence work on each STAR panel. In the attrition panel, paper FCI+ uses seven hundred ninety-three tests compared with one thousand seven hundred five for FCI, or forty-seven percent as many. In the longitudinal panel the comparison is three hundred thirty-five versus nine hundred sixty, or thirty-five percent. In the focused panel it is one hundred eighty-five versus two hundred ninety-one, or sixty-four percent. The runtime strip reports the same three-run medians in the order FCI, paper FCI+, robust FCI+, and R pcalg FCI+: attrition is five point four four, one point seven nine, two point eight four, and ten point three nine seconds; longitudinal is two point one five, zero point three four, zero point five six, and one point six five; focused is zero point three two, zero point one six, zero point two five, and zero point eight eight. These are implementation measurements, not a language leaderboard. Computational efficiency is useful engineering evidence; it does not increase the causal strength of an endpoint or identify a treatment effect.

[Sources]
- `case_studies/tennessee_star/output/star_benchmark.csv`, all 12 panel-by-algorithm rows; columns `ci_tests` and `median_elapsed_seconds`.
- `case_studies/tennessee_star/output/star_case_study_summary.json`, key `runs`.
- `case_studies/tennessee_star/run_case_study.py`, three-run benchmark loop and timing scope.
[/Sources]