# FCI+ Research Report and Advisor Deck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the English LaTeX/PDF research report with deeper
paper-level analysis, implementation traceability, and established-software
comparisons, then create and verify an editable 15-slide English advisor
presentation and push every final artifact to `main`.

**Architecture:** Primary-source facts, repository facts, and experiment
results are first consolidated into auditable research and comparison
artifacts. The report and presentation consume those same facts so their
claims, values, terminology, and evidence hierarchy remain synchronized.
Paper-aligned FCI/FCI+ implementations stay distinct from the robust
application profile.

**Tech Stack:** Python 3.9+, LaTeX/latexmk, matplotlib, pandas, `fci_engine`,
`causal-learn`, R `pcalg`, PPT Master SVG-to-PPTX workflow, Poppler, pypdf,
Ruff, MyPy, Pytest, Git.

## Global Constraints

- All report and presentation content is English.
- The presentation targets 15–20 minutes and approximately 15 slides.
- Primary literature claims must be verified against original sources.
- Executed benchmarks and documentation-derived feature comparisons must be
  labeled separately.
- Literal paper profiles must not be modified to achieve better STAR results.
- The robust profile is an application workflow, not a replacement definition
  of FCI+.
- PAG output is not a treatment-effect estimate and must not be described as a
  unique DAG.
- Final PDFs and PPTX must pass visual inspection before delivery.
- All committed files must be reproducible or clearly identified as generated
  final artifacts.

---

### Task 1: Build the primary-source and software-evidence dossier

**Files:**
- Create: `reports/research/fci_fci_plus_source_dossier.md`
- Create: `reports/research/software_landscape.json`
- Create: `reports/research/claim_evidence_matrix.csv`
- Test: `tests/test_research_report_artifacts.py`

**Interfaces:**
- Consumes: Spirtes et al. (2000), Claassen et al. (2013), official
  `pcalg`, `causal-learn`, and Tetrad documentation, and repository symbols.
- Produces: verified claims and capability records consumed by Tasks 3–5.

- [ ] **Step 1: Write the failing artifact-contract test**

```python
def test_research_dossier_has_required_primary_source_sections() -> None:
    text = (ROOT / "reports/research/fci_fci_plus_source_dossier.md").read_text()
    for heading in (
        "## Spirtes, Glymour, and Scheines (2000)",
        "## Claassen, Mooij, and Heskes (2013)",
        "## Source-to-implementation mapping",
        "## Claims that require finite-sample qualification",
    ):
        assert heading in text


def test_software_landscape_separates_executed_and_documented_evidence() -> None:
    payload = json.loads(
        (ROOT / "reports/research/software_landscape.json").read_text()
    )
    assert payload["as_of"] == "2026-07-26"
    assert {row["tool"] for row in payload["tools"]} >= {
        "fci_engine",
        "pcalg",
        "causal-learn",
        "Tetrad",
    }
    assert all(row["evidence_kind"] in {"executed", "documentation"} for row in payload["tools"])
```

- [ ] **Step 2: Run the contract test and confirm the artifacts are missing**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_research_report_artifacts.py -q
```

Expected: failure because the dossier and software-landscape files do not yet
exist.

- [ ] **Step 3: Search and verify the primary sources**

Use the academic-research workflow to verify:

- the standard FCI algorithm, Possible-D-SEP definition, completeness scope,
  and assumptions in *Causation, Prediction, and Search*, second edition;
- the augmented skeleton, D-SEP links, hierarchical removal procedure,
  sparsity parameter, and polynomial-complexity claim in Claassen et al.
  (2013);
- the exact theorem/algorithm/figure locators used in the report.

Record for every non-trivial claim:

```csv
claim_id,claim,source,locator,evidence_kind,repository_symbol,validation_artifact
```

No claim enters the paper if its source and locator are empty.

- [ ] **Step 4: Verify current software capabilities from primary documentation**

Capture dated capability records for:

- `fci_engine`;
- CRAN `pcalg::fciPlus`;
- `causal-learn` FCI;
- Tetrad FCI-family algorithms.

Each JSON row must include:

```json
{
  "tool": "causal-learn",
  "version": "locally observed or documentation version",
  "as_of": "2026-07-26",
  "evidence_kind": "executed",
  "algorithms": ["FCI"],
  "ci_tests": ["fisherz", "chisq", "gsq", "kci"],
  "audit_exports": false,
  "source_urls": ["official documentation URL"],
  "comparison_limits": ["No local FCI+ implementation was identified."]
}
```

- [ ] **Step 5: Map source concepts to exact implementation symbols**

Inspect and record at least:

- `src/fci_engine/discovery/fci.py::FCI`;
- `src/fci_engine/discovery/fci_plus.py::FCIPlus`;
- `src/fci_engine/discovery/skeleton.py`;
- `src/fci_engine/discovery/pdsep.py`;
- `src/fci_engine/discovery/dsep.py`;
- `src/fci_engine/discovery/orientation.py`;
- `src/fci_engine/discovery/rules.py`;
- `src/fci_engine/graph/pag.py::PAG`;
- exact-oracle and published-reference tests.

Label every non-paper option as an engineering extension.

- [ ] **Step 6: Run the artifact-contract test**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_research_report_artifacts.py -q
```

Expected: pass.

- [ ] **Step 7: Commit the research dossier**

```bash
git add reports/research tests/test_research_report_artifacts.py
git commit -m "docs: add verified FCI research dossier"
```

---

### Task 2: Generate reproducible established-software comparison artifacts

**Files:**
- Create: `reports/generate_software_comparison.py`
- Create: `reports/data/software_benchmark_cases.csv`
- Create: `reports/data/software_benchmark_summary.csv`
- Create: `reports/data/software_feature_matrix.csv`
- Modify: `tests/test_research_report_artifacts.py`

**Interfaces:**
- Consumes:
  `fci_engine.simulation.realistic_oracle_cases`,
  `fci_engine.metrics.benchmark.run_oracle_benchmark`, and
  `aggregate_benchmark_results`.
- Produces: deterministic CSV tables consumed by report figures, LaTeX tables,
  and presentation slides.

- [ ] **Step 1: Add a failing generator test**

```python
def test_software_comparison_generator_writes_complete_tables(tmp_path: Path) -> None:
    outputs = generate_software_comparison(
        output_dir=tmp_path,
        repeats=1,
        samples=600,
        include_pcalg=False,
    )
    summary = pd.read_csv(outputs["summary"])
    assert {
        "fci_engine.fci",
        "fci_engine.fci_plus",
        "fci_engine.fci_plus.robust",
        "causal-learn.fci.fisherz",
    } <= set(summary["algorithm"])
    assert {
        "mean_skeleton_f1",
        "mean_exact_edge_f1",
        "mean_semantic_edge_f1",
        "mean_endpoint_accuracy",
        "mean_elapsed_seconds",
    } <= set(summary.columns)
```

- [ ] **Step 2: Confirm the test fails**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_research_report_artifacts.py::test_software_comparison_generator_writes_complete_tables \
  -q
```

Expected: import failure because the generator does not exist.

- [ ] **Step 3: Implement the comparison generator**

Define:

```python
def generate_software_comparison(
    *,
    output_dir: Path,
    repeats: int = 3,
    samples: int = 2_500,
    include_pcalg: bool = True,
) -> dict[str, Path]:
    """Run matched known-truth cases and write long-form and aggregate CSVs."""
```

Required behavior:

- construct `realistic_oracle_cases(n_repeats=repeats, n_samples=samples)`;
- run local FCI, paper FCI+, robust FCI+, causal-learn FCI, and optional pcalg;
- preserve skip reasons instead of dropping unavailable tools;
- write one row per case/algorithm and one aggregate row per algorithm;
- include software versions and the exact CI-test method;
- keep pcalg FCI+ and causal-learn FCI labeled as different algorithm families;
- never rank documentation-only tools by empirical metrics.

- [ ] **Step 4: Write the feature matrix from the verified landscape**

Generate columns:

```text
tool,language,standard_fci,fci_plus,latent_confounding,selection_bias,
custom_ci,order_audit,bootstrap_workflow,orientation_trace,sepset_provenance,
artifact_export,executed_here,evidence_date
```

- [ ] **Step 5: Run the focused test**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_research_report_artifacts.py -q
```

Expected: pass.

- [ ] **Step 6: Run the full comparison**

Run:

```bash
PYTHONPATH=src .venv/bin/python reports/generate_software_comparison.py \
  --repeats 3 \
  --samples 2500 \
  --output-dir reports/data
```

Expected: committed long-form, aggregate, and feature-matrix CSV files with
explicit skip records if an optional external implementation is unavailable.

- [ ] **Step 7: Inspect the results before writing claims**

Check:

```bash
python3 - <<'PY'
import pandas as pd
for path in (
    "reports/data/software_benchmark_cases.csv",
    "reports/data/software_benchmark_summary.csv",
    "reports/data/software_feature_matrix.csv",
):
    frame = pd.read_csv(path)
    print(path, frame.shape)
    print(frame.to_string(index=False))
PY
```

Expected: every number later used in prose is visible in these tables.

- [ ] **Step 8: Commit the executed comparison**

```bash
git add reports/generate_software_comparison.py reports/data \
  tests/test_research_report_artifacts.py
git commit -m "research: benchmark established FCI software"
```

---

### Task 3: Add synchronized publication figures and report guards

**Files:**
- Modify: `reports/generate_report_figures.py`
- Create: `reports/figures/fci_fci_plus_workflow.pdf`
- Create: `reports/figures/source_implementation_map.pdf`
- Create: `reports/figures/software_benchmark_comparison.pdf`
- Create: `reports/figures/software_feature_comparison.pdf`
- Modify: `tests/test_research_report_artifacts.py`

**Interfaces:**
- Consumes: Task 1 evidence matrix, Task 2 benchmark tables, and committed STAR
  outputs.
- Produces: vector figures used by both report and presentation.

- [ ] **Step 1: Add failing figure-output assertions**

```python
def test_required_research_figures_exist_and_are_nonempty() -> None:
    for name in (
        "fci_fci_plus_workflow.pdf",
        "source_implementation_map.pdf",
        "software_benchmark_comparison.pdf",
        "software_feature_comparison.pdf",
    ):
        path = ROOT / "reports/figures" / name
        assert path.stat().st_size > 10_000
```

- [ ] **Step 2: Confirm the figure test fails**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_research_report_artifacts.py::test_required_research_figures_exist_and_are_nonempty \
  -q
```

Expected: failure because the new PDFs do not exist.

- [ ] **Step 3: Extend the figure generator**

Use the shared semantic palette:

```python
COLORS = {
    "fci": "#2F6BFF",
    "fci_plus": "#E58A2B",
    "robust": "#1C9A8A",
    "external": "#7562A8",
    "neutral": "#687386",
}
```

Create:

- a process figure contrasting standard FCI and FCI+ stages;
- a source-to-code traceability figure;
- a benchmark chart with exact, semantic, skeleton, endpoint, and runtime
  metrics;
- a feature comparison that visually distinguishes executed evidence from
  documentation-only evidence.

All figures must be vector PDF, use no internal title, and have readable labels
at report width.

- [ ] **Step 4: Generate and test figures**

Run:

```bash
PYTHONPATH=src .venv/bin/python reports/generate_report_figures.py
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_research_report_artifacts.py -q
```

Expected: pass.

- [ ] **Step 5: Render the new figures for visual inspection**

Run:

```bash
mkdir -p tmp/pdfs/research-figures
for figure in \
  fci_fci_plus_workflow \
  source_implementation_map \
  software_benchmark_comparison \
  software_feature_comparison; do
  pdftoppm -png -singlefile -r 160 \
    "reports/figures/${figure}.pdf" \
    "tmp/pdfs/research-figures/${figure}"
done
```

Expected: no clipped labels, overlapping legends, unreadable text, or
inconsistent colors.

- [ ] **Step 6: Commit figures and guards**

```bash
git add reports/generate_report_figures.py reports/figures \
  tests/test_research_report_artifacts.py
git commit -m "docs: add FCI research comparison figures"
```

---

### Task 4: Rewrite and verify the LaTeX research report

**Files:**
- Modify: `reports/fci_plus_star_report.tex`
- Modify: `output/pdf/fci_plus_star_report.pdf`
- Modify: `tests/test_research_report_artifacts.py`

**Interfaces:**
- Consumes: Tasks 1–3 and existing STAR summary/CSV artifacts.
- Produces: the final thesis-style PDF and source.

- [ ] **Step 1: Add failing structural and claim guards**

```python
def test_latex_report_contains_rewritten_research_structure() -> None:
    tex = (ROOT / "reports/fci_plus_star_report.tex").read_text()
    for title in (
        r"\\chapter{Standard FCI in Spirtes et al. (2000)}",
        r"\\chapter{FCI+ in Claassen et al. (2013)}",
        r"\\chapter{Independent Python Implementation}",
        r"\\chapter{Comparison with Established Software}",
        r"\\chapter{Causal Interpretation and Evidence Hierarchy}",
    ):
        assert title in tex
    assert "documentation-only comparison" in tex
    assert "does not estimate a treatment effect" in tex
```

- [ ] **Step 2: Confirm the report guard fails**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_research_report_artifacts.py::test_latex_report_contains_rewritten_research_structure \
  -q
```

Expected: failure because the old chapter structure remains.

- [ ] **Step 3: Rewrite title, abstract, executive summary, and introduction**

Use the five-sentence abstract pattern:

1. contribution;
2. scientific difficulty;
3. method;
4. validation design;
5. strongest quantitative and STAR result.

State the one-sentence contribution consistently:

> This project independently implements paper-aligned FCI and FCI+, validates
> them against exact graphical oracles and established software, and separates
> algorithm fidelity from a robust finite-sample application workflow.

- [ ] **Step 4: Write dedicated source-paper chapters**

For both original sources:

- explain definitions and assumptions before algorithm steps;
- include pseudocode or stage tables;
- identify oracle versus finite-sample scope;
- state complexity conditions exactly;
- cite definition, theorem, algorithm, or figure locators from Task 1;
- include a “not established by the source” paragraph.

- [ ] **Step 5: Rewrite the implementation chapter**

Include the concept-to-code-to-test matrix and concrete public API. Explain
where the implementation is literal, where engineering structure is neutral,
and where the practical profile deliberately changes finite-sample policy.

- [ ] **Step 6: Add established-software comparison**

Present:

- executed known-truth metrics for local methods, causal-learn, and pcalg when
  available;
- the existing matched STAR pcalg experiment;
- a dated feature matrix including Tetrad;
- fairness constraints and endpoint/CI-test convention differences;
- improvements as auditable capabilities, not unsupported superiority claims.

- [ ] **Step 7: Preserve and tighten the STAR application**

Keep algorithm and application chapters separate. Use the evidence hierarchy
to distinguish:

- randomized class-arm effects;
- robust skeleton findings;
- unresolved endpoints;
- unstable implementation-specific orientations;
- effects of attrition and complete-case selection.

- [ ] **Step 8: Compile the report**

Run:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -output-directory=output/pdf \
  reports/fci_plus_star_report.tex
```

Expected: successful PDF generation with no undefined citations or references.

- [ ] **Step 9: Run automated PDF checks**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
from pypdf import PdfReader

path = Path("output/pdf/fci_plus_star_report.pdf")
reader = PdfReader(path)
text = "\n".join(page.extract_text() or "" for page in reader.pages)
required = (
    "Standard FCI in Spirtes et al. (2000)",
    "FCI+ in Claassen et al. (2013)",
    "Independent Python Implementation",
    "Comparison with Established Software",
    "Causal Interpretation and Evidence Hierarchy",
)
assert all(item in text for item in required)
assert "??" not in text
print("pages", len(reader.pages))
PY
```

- [ ] **Step 10: Render and visually inspect every page**

Run:

```bash
mkdir -p tmp/pdfs/fci-plus-rewritten
pdftoppm -png -r 120 output/pdf/fci_plus_star_report.pdf \
  tmp/pdfs/fci-plus-rewritten/page
```

Create contact sheets, inspect all pages, then inspect changed dense pages at
full resolution. Repair orphan headings, clipped tables, bad page breaks,
unreadable charts, and overfull boxes before continuing.

- [ ] **Step 11: Run report tests and commit**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_research_report_artifacts.py \
  tests/test_star_case_study.py -q
```

Then:

```bash
git add reports/fci_plus_star_report.tex output/pdf/fci_plus_star_report.pdf \
  tests/test_research_report_artifacts.py
git commit -m "docs: rewrite FCI+ research report"
```

---

### Task 5: Generate the English advisor presentation with PPT Master

**Files:**
- Create: `projects/fci_plus_star_advisor_deck/`
- Create: `output/ppt/fci_plus_star_advisor_presentation.pptx`
- Create: `output/ppt/fci_plus_star_advisor_presentation.pdf`
- Create: `output/ppt/fci_plus_star_advisor_presentation_notes.md`

**Interfaces:**
- Consumes: final report, Task 1 research dossier, Task 2 comparison data, and
  Task 3 figures.
- Produces: editable PowerPoint, PDF export, and presenter notes.

- [ ] **Step 1: Initialize the PPT Master Generate-PPTX project**

Run:

```bash
python3 /Users/yq225/.codex/skills/ppt-master/scripts/project_manager.py \
  init fci_plus_star_advisor_deck --format ppt169
```

Import the final report source, research dossier, comparison tables, and STAR
summary using the project manager.

- [ ] **Step 2: Apply delegated Strategist decisions**

Because the user explicitly delegated all design decisions, document one
consolidated decision state:

- audience: academic advisor;
- purpose: formal 15–20 minute defense;
- 15 slides;
- free-design 16:9 canvas;
- restrained academic style;
- navy typography with blue/orange/teal/purple method colors;
- no external decorative imagery;
- report figures and native data charts only;
- mixed formula policy;
- continuous generation;
- no additional spec-refinement gate.

Create the complete project `design_spec.md` and `spec_lock.md` from the PPT
Master references, then run:

```bash
python3 /Users/yq225/.codex/skills/ppt-master/scripts/project_manager.py \
  validate projects/fci_plus_star_advisor_deck
```

- [ ] **Step 3: Create the exact 15-slide roster**

Use the slide sequence from the design specification. Each slide must define:

- one claim-level title;
- one audience move;
- one main visual;
- concise visible copy;
- source records for non-trivial claims;
- native-ready data chart metadata where applicable.

- [ ] **Step 4: Start live preview and author slide 1**

Run:

```bash
python3 /Users/yq225/.codex/skills/ppt-master/scripts/svg_editor/server.py \
  projects/fci_plus_star_advisor_deck --live --daemon
```

Hand-author `P01.svg`, then run the unfiltered first-page gate:

```bash
python3 /Users/yq225/.codex/skills/ppt-master/scripts/svg_quality_checker.py \
  projects/fci_plus_star_advisor_deck --stage first-page --json
```

Classify the gate signal, repair all blocking issues in one pass, and rerun
once.

- [ ] **Step 5: Hand-author slides 2–15**

Create every remaining SVG sequentially without generator scripts. Maintain:

- title size at least 35 pt;
- body size at least 16 pt;
- one main visual idea per slide;
- consistent color semantics;
- no unsupported claims;
- no dense report paragraphs;
- source footer and speaker-note provenance.

- [ ] **Step 6: Run final SVG and chart quality gates**

Run the final checker once, unfiltered:

```bash
python3 /Users/yq225/.codex/skills/ppt-master/scripts/svg_quality_checker.py \
  projects/fci_plus_star_advisor_deck --stage final --json
```

Because the deck contains data charts, run the PPT Master `verify-charts`
stage before export. Resolve every blocking issue.

- [ ] **Step 7: Write speaker notes**

Create `notes/total.md` grounded in every information-bearing element. Target
approximately:

- 30 seconds for title;
- 45–75 seconds for background and algorithm slides;
- 60–90 seconds for validation and STAR result slides;
- 45 seconds for conclusions.

Include `[Sources]` blocks for external claims and assets.

- [ ] **Step 8: Export serially**

Run each command separately:

```bash
python3 /Users/yq225/.codex/skills/ppt-master/scripts/total_md_split.py \
  projects/fci_plus_star_advisor_deck
```

```bash
python3 /Users/yq225/.codex/skills/ppt-master/scripts/finalize_svg.py \
  projects/fci_plus_star_advisor_deck
```

```bash
python3 /Users/yq225/.codex/skills/ppt-master/scripts/svg_to_pptx.py \
  projects/fci_plus_star_advisor_deck
```

Copy the timestamped passed export to the stable
`output/ppt/fci_plus_star_advisor_presentation.pptx` path and copy
`notes/total.md` to the stable notes path.

- [ ] **Step 9: Render and inspect all slides**

Render the PPTX and create a montage. Inspect every slide individually at full
size. Run overflow/out-of-bounds checks and repair the owning SVG when needed.
Export the final deck to
`output/ppt/fci_plus_star_advisor_presentation.pdf` and confirm its slide count
matches the PPTX.

- [ ] **Step 10: Commit presentation artifacts**

```bash
git add projects/fci_plus_star_advisor_deck \
  output/ppt/fci_plus_star_advisor_presentation.pptx \
  output/ppt/fci_plus_star_advisor_presentation.pdf \
  output/ppt/fci_plus_star_advisor_presentation_notes.md
git commit -m "docs: add FCI+ advisor presentation"
```

---

### Task 6: Independent review and full verification

**Files:**
- Modify as required by review findings.

**Interfaces:**
- Consumes: final report, deck, source dossier, raw experiment outputs.
- Produces: review-approved, internally consistent release artifacts.

- [ ] **Step 1: Run a fresh claim-verification review**

Give a fresh reviewer only:

- final PDF text;
- source dossier and claim matrix;
- software benchmark CSVs;
- STAR summary and contrast CSVs.

Require a table of every numerical or comparative claim and its exact source.
Fix any unsupported or mismatched statement.

- [ ] **Step 2: Run independent paper and deck quality reviews**

Review for:

- source fidelity;
- distinction between theory and finite-sample practice;
- fair external-software comparison;
- causal overclaiming;
- narrative clarity;
- advisor-level presentation flow;
- visual legibility.

- [ ] **Step 3: Run the complete repository checks**

Run:

```bash
python3 -m ruff check src tests examples case_studies reports
python3 -m ruff format --check src tests examples case_studies reports
.venv/bin/python -m mypy
.venv/bin/python -m mypy case_studies/tennessee_star
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Expected: every command exits zero.

- [ ] **Step 4: Build and install the wheel**

Run:

```bash
python3 -m build --wheel --no-isolation --outdir /tmp/fci-report-dist
python3 -m venv /tmp/fci-report-wheel-smoke
/tmp/fci-report-wheel-smoke/bin/python -m pip install --no-deps \
  /tmp/fci-report-dist/fci_engine-0.1.0-py3-none-any.whl
```

Then run an installed-package FCI+ fit from `/tmp`.

- [ ] **Step 5: Recompile and rerender final artifacts after all fixes**

No completion claim may rely on a render made before the last source change.
Recompile the report, export the PPTX/PDF, rerun PDF text checks, render every
page/slide, and inspect the latest images.

---

### Task 7: Final Git audit and push

**Files:**
- All final report, source, figure, comparison, presentation, notes, and design
  artifacts.

**Interfaces:**
- Consumes: verified working tree.
- Produces: clean, synchronized local and remote `main`.

- [ ] **Step 1: Audit repository cleanliness**

Run:

```bash
git status --short
git diff --check
git diff --stat
git log -5 --oneline
```

Exclude LaTeX auxiliary files, temporary slide renders, virtual environments,
and caches.

- [ ] **Step 2: Stage only intended files**

Use explicit paths rather than `git add -A`. Inspect:

```bash
git diff --cached --stat
git diff --cached --check
```

- [ ] **Step 3: Commit the final integration fixes**

```bash
git commit -m "docs: finalize FCI+ research paper and presentation"
```

- [ ] **Step 4: Push and verify synchronization**

```bash
git push origin main
git status -sb
git log -1 --oneline --decorate
```

Expected: `HEAD`, local `main`, and `origin/main` point to the same final
commit, with no uncommitted changes.
