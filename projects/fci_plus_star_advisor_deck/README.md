# FCI and FCI+ Advisor-Defense Deck

This PPT Master project contains the editable source for the 15-slide English
advisor presentation, *Independent FCI and FCI+: From Source Fidelity to
Tennessee STAR*.

The narrative keeps four evidence layers separate:

1. the FCI algorithm in Spirtes, Glymour, and Scheines (2000);
2. the FCI+ algorithm in Claassen, Mooij, and Heskes (2013);
3. the independent Python implementation, known-truth tests, and established
   software comparison; and
4. the Tennessee STAR application, where randomized contrasts remain primary
   and PAG output is interpreted as an exploratory structural audit.

## Stable deliverables

The repository publishes:

- `output/ppt/fci_plus_star_advisor_presentation.pptx` — editable native
  DrawingML PowerPoint;
- `output/ppt/fci_plus_star_advisor_presentation.pdf` — visually matched PDF
  handout;
- `output/ppt/fci_plus_star_advisor_presentation_notes.md` — approximately
  18 minutes of English speaker notes.

## Authoring structure

- `design_spec.md` and `spec_lock.md` define the approved 16:9 design and
  execution contract.
- `svg_output/` contains the hand-authored slide sources.
- `svg_final/` contains self-contained visual-preview SVGs.
- `notes/` contains the complete narration and one file per slide.
- `sources/` contains the dated paper, software, benchmark, and STAR evidence
  used by the deck, together with the exact LaTeX report source synchronized
  at the final export.
- `validation/chart_verification_receipt.md` records the data-coordinate
  checks for every value-mapped chart.

Transient live-preview logs, timestamped exports, backups, and raster QA
renders are intentionally excluded from version control.

## Verification

The final deck passed:

- the PPT Master first-page and 15-slide SVG quality gates;
- chart-coordinate verification against the committed CSV artifacts;
- native PPTX postflight with 15 slides and no warning categories;
- strict no-merge SVG-to-PPTX export, preserving authored line breaks and
  preventing PowerPoint paragraph reflow;
- slide-canvas overflow testing;
- visual review of every rendered PPTX slide and the final 15-page PDF; and
- one-to-one correspondence between the 15 slides and 15 note sections.
