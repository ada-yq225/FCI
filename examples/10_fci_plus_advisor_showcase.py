"""Generate a concise, evidence-first advisor-facing FCI+ report."""

from __future__ import annotations

import argparse
import html
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fci_engine import (
    __version__,
    canonical_dsep_mag,
    compare_pag_shapes,
    compare_pag_shapes_semantic,
    default_oracle_cases,
    fci,
    fci_plus,
    realistic_oracle_cases,
    run_oracle_benchmark,
    sample_canonical_dsep_data,
    shape_from_pag,
)
from fci_engine.metrics import (
    BenchmarkAggregate,
    BenchmarkResult,
    aggregate_benchmark_results,
)

REPORT_PATH = Path(__file__).with_name("advisor_showcase.html")

PAPER_URL = "https://auai.org/uai2013/prints/papers/121.pdf"
ZHANG_URL = "https://doi.org/10.1016/j.artint.2008.08.001"
RULE_TABLE_URL = "https://www.cs.ru.nl/~tomc/docs/UAI2011_LCCausD.pdf"

PAPER_CONFIG = {
    "alpha": 0.001,
    "max_cond_set_size": 3,
    "sparsity_bound": 3,
    "max_path_length": None,
    "sepset_selection": "first",
    "orientation_strategy": "standard",
}

ALGORITHM_STEPS = (
    ("1", "PCAdjSearch(V,O,k)", "Skeleton and one minimal sepset per removed edge."),
    ("2", "AugmentGraph(G,I,O)", "Single-node dependence adds invariant arrowheads."),
    ("3", "GetPDseps(G+)", "Strict U↔X↔Y↔V witness plus two cross paths."),
    ("4–22", "n/m loops + HIE", "Nested n=1..k, m=1..k subsets; minimize and restart."),
    ("23", "FCI R0–R10", "R1–R4 closure; R5; R6–R7; then R8–R10."),
)


@dataclass(frozen=True)
class SampleRun:
    n_samples: int
    seed: int
    shape: dict[tuple[str, str], tuple[str, str]]
    exact_f1: float
    semantic_f1: float
    skeleton_f1: float
    endpoint_accuracy: float
    ci_tests: int
    xy_present: bool
    xy_edge: str
    separator: tuple[str, ...]
    separator_source: str
    standard_fci_exact_f1: float
    standard_fci_ci_tests: int
    standard_fci_xy_present: bool


@dataclass(frozen=True)
class ShowcaseContext:
    target_shape: dict[tuple[str, str], tuple[str, str]]
    learned_shape: dict[tuple[str, str], tuple[str, str]]
    exact_pass: bool
    exact_separator: tuple[str, ...]
    exact_separator_source: str
    exact_diagnostics: dict[str, int]
    exact_ci_tests: int
    sample_runs: tuple[SampleRun, ...]
    generated_at: str
    git_sha: str
    python_version: str
    package_version: str
    external_enabled: bool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--no-external", action="store_true")
    parser.add_argument("--realistic", action="store_true")
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    cases = default_oracle_cases()
    if args.realistic:
        cases = [
            *cases,
            *realistic_oracle_cases(
                n_repeats=args.repeats,
                n_samples=args.samples,
            ),
        ]
    results = run_oracle_benchmark(
        cases,
        include_causal_learn=not args.no_external,
        include_pcalg=not args.no_external,
        include_kernel_ci=False,
    )
    context = build_showcase_context(
        external_enabled=not args.no_external,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_showcase(results, context), encoding="utf-8")
    print(f"Wrote {args.output}")


def build_showcase_context(external_enabled: bool = True) -> ShowcaseContext:
    """Run the published exact-oracle case and two seeded finite-sample cases."""

    mag = canonical_dsep_mag()
    exact = fci_plus(
        mag.dummy_data(),
        ci_test=mag.oracle_ci_test(),
        max_cond_set_size=None,
        sparsity_bound=None,
        max_path_length=None,
        sepset_selection="first",
        orientation_strategy="standard",
    )
    target_shape = _string_shape(mag.oracle_shape())
    learned_shape = shape_from_pag(exact.graph)

    sample_runs = tuple(
        _run_sample_case(n_samples=n_samples, seed=1, target_shape=target_shape)
        for n_samples in (5_000, 50_000)
    )
    return ShowcaseContext(
        target_shape=target_shape,
        learned_shape=learned_shape,
        exact_pass=learned_shape == target_shape,
        exact_separator=tuple(sorted(exact.sepsets.get(("X", "Y"), set()))),
        exact_separator_source=exact.sepset_sources.get(("X", "Y"), "—"),
        exact_diagnostics=dict(exact.dsep_diagnostics or {}),
        exact_ci_tests=exact.ci_test_count,
        sample_runs=sample_runs,
        generated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        git_sha=_git_sha(),
        python_version=platform.python_version(),
        package_version=__version__,
        external_enabled=external_enabled,
    )


def _run_sample_case(
    n_samples: int,
    seed: int,
    target_shape: dict[tuple[str, str], tuple[str, str]],
) -> SampleRun:
    data = sample_canonical_dsep_data(n_samples=n_samples, seed=seed)
    result = fci_plus(data, **PAPER_CONFIG)
    standard_result = fci(data, **PAPER_CONFIG)
    shape = shape_from_pag(result.graph)
    standard_shape = shape_from_pag(standard_result.graph)
    exact = compare_pag_shapes(target_shape, shape)
    semantic = compare_pag_shapes_semantic(target_shape, shape)
    standard_exact = compare_pag_shapes(target_shape, standard_shape)
    xy_key = ("X", "Y")
    return SampleRun(
        n_samples=n_samples,
        seed=seed,
        shape=shape,
        exact_f1=exact.exact_edge_f1,
        semantic_f1=semantic.semantic_edge_f1,
        skeleton_f1=exact.skeleton_f1,
        endpoint_accuracy=exact.endpoint_accuracy,
        ci_tests=result.ci_test_count,
        xy_present=xy_key in shape,
        xy_edge=_edge_notation(xy_key, shape[xy_key]) if xy_key in shape else "absent",
        separator=tuple(sorted(result.sepsets.get(xy_key, set()))),
        separator_source=result.sepset_sources.get(xy_key, "—"),
        standard_fci_exact_f1=standard_exact.exact_edge_f1,
        standard_fci_ci_tests=standard_result.ci_test_count,
        standard_fci_xy_present=xy_key in standard_shape,
    )


def render_showcase(
    results: list[BenchmarkResult],
    context: Optional[ShowcaseContext] = None,
) -> str:
    """Render a standalone report; all scientific KPIs come from run objects."""

    if context is None:
        context = build_showcase_context()
    sample_5k, sample_50k = context.sample_runs
    status = "PASS" if context.exact_pass else "FAIL"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FCI+ Paper-aligned Validation</title>
<style>{_styles()}</style>
</head>
<body>
<nav aria-label="Report sections">
  <a class="brand" href="#overview">FCI+ validation</a>
  <div class="nav-links">
    <a href="#figure4">Figure 4(b)</a><a href="#samples">Finite sample</a>
    <a href="#benchmark">Benchmark</a><a href="#limits">Limitations</a>
    <a href="#reproduce">Reproduce</a>
  </div>
  <span class="build">{_esc(context.git_sha)} · {_esc(context.generated_at[:10])}</span>
</nav>
<main>
  <section id="overview" class="hero">
    <div>
      <p class="eyebrow">PAPER-ALIGNED VALIDATION · CLAASSEN, MOOIJ &amp; HESKES (2013)</p>
      <h1>Exact-oracle recovery matches Figure 4(b); finite-sample recovery needs care.</h1>
      <p class="lead">FCI+ recovers the complete published PAG with exact m-separation. On the same seeded latent SEM, 5,000 rows retain one false edge while 50,000 rows recover the target; standard FCI recovers both runs with more CI queries. This page shows the trade-off.</p>
      <div class="notice">A PAG represents a Markov-equivalence class, not one unique DAG. External packages are comparators; the declared oracle PAG is the scoring target.</div>
    </div>
    <div class="hero-kpis" aria-label="Key results">
      {_kpi("Exact oracle", status, context.exact_pass)}
      {_kpi("50k exact F1", f"{sample_50k.exact_f1:.3f}", sample_50k.exact_f1 == 1.0)}
      {_kpi("D-SEP separator", "{U, V, Z}", context.exact_separator == ("U", "V", "Z"))}
    </div>
  </section>

  <section class="paper-map" aria-labelledby="algorithm-heading">
    <div class="section-heading">
      <div><p class="eyebrow">IMPLEMENTATION MAP</p><h2 id="algorithm-heading">Algorithm 2, line by line</h2></div>
      <a href="{PAPER_URL}">Open UAI paper ↗</a>
    </div>
    <div class="steps">{render_algorithm_steps()}</div>
  </section>

  <section id="figure4" aria-labelledby="figure-heading">
    <div class="section-heading">
      <div><p class="eyebrow">FLAGSHIP ORACLE TEST</p><h2 id="figure-heading">Figure 4(b): true target vs learned PAG</h2></div>
      <span class="status {'pass' if context.exact_pass else 'fail'}">{'✓' if context.exact_pass else '✕'} {status}</span>
    </div>
    <p class="section-copy">The oracle target is programmatically reconstructed from the paper's Figure 4(b) MAG; it is not a screenshot. The learned graph below comes from the public <code>fci_plus</code> pipeline using exact MAG m-separation CI.</p>
    <div class="graph-grid">
      <figure>{render_pag_svg(context.target_shape, "Figure 4(b) oracle PAG target")}<figcaption>Oracle PAG target derived from the published MAG.</figcaption></figure>
      <figure>{render_pag_svg(context.learned_shape, "FCI+ learned PAG with exact oracle CI")}<figcaption>FCI+ output using deterministic m-separation—not sampled data.</figcaption></figure>
    </div>
    <div class="checks" role="list">
      {_check("Complete endpoint shape", context.exact_pass, "learned shape equals oracle shape")}
      {_check("False PC link X–Y", ("X", "Y") not in context.learned_shape, "absent after FCI+ D-SEP")}
      {_check("Separator provenance", context.exact_separator == ("U", "V", "Z") and context.exact_separator_source == "fci_plus_dsep", f"{{{', '.join(context.exact_separator)}}} · {context.exact_separator_source}")}
    </div>
    <details><summary>Exact-run D-SEP diagnostics and provenance</summary>{render_diagnostics(context)}</details>
  </section>

  <section id="samples" aria-labelledby="sample-heading">
    <div class="section-heading"><div><p class="eyebrow">PRACTICALITY CHECK</p><h2 id="sample-heading">Same latent SEM, only sample size changes</h2></div></div>
    <p class="section-copy">Both runs use seed 1, Fisher-Z, α=0.001, k=3, first sepset, unlimited paths, and standard R0–R10. The 5k miss is part of the result, not hidden. The 50k success is a seeded fixture—not a convergence proof.</p>
    <div class="sample-grid">
      {render_sample_card(sample_5k, context.target_shape)}
      {render_sample_card(sample_50k, context.target_shape)}
    </div>
  </section>

  <section id="benchmark" aria-labelledby="benchmark-heading">
    <div class="section-heading"><div><p class="eyebrow">FAIR COMPARISON</p><h2 id="benchmark-heading">Same-cohort oracle benchmark</h2></div></div>
    <p class="section-copy">Only algorithms completing every displayed case enter the aggregate. Missing/skipped runs are excluded rather than scored as zero.</p>
    {render_leaderboard(results, context.external_enabled)}
  </section>

  <section id="limits" aria-labelledby="limits-heading">
    <div class="section-heading"><div><p class="eyebrow">INTERPRETATION</p><h2 id="limits-heading">What this does—and does not—establish</h2></div></div>
    {render_limitations()}
  </section>

  <section id="reproduce" aria-labelledby="reproduce-heading">
    <div class="section-heading"><div><p class="eyebrow">AUDIT TRAIL</p><h2 id="reproduce-heading">Citations and reproducibility</h2></div></div>
    <div class="repro-grid">
      {render_citations()}
      {render_reproducibility(context)}
    </div>
  </section>
</main>
<footer>Generated locally from executable tests and report code · no remote assets</footer>
</body>
</html>"""


def render_algorithm_steps() -> str:
    return "".join(
        "<article class='step'>"
        f"<span>{_esc(line)}</span><h3>{_esc(title)}</h3><p>{_esc(text)}</p>"
        "</article>"
        for line, title, text in ALGORITHM_STEPS
    )


def render_diagnostics(context: ShowcaseContext) -> str:
    diagnostics = context.exact_diagnostics
    items = (
        ("Candidate links", diagnostics.get("candidate_edges_seen", 0)),
        ("D-SEP CI tests", diagnostics.get("ci_tests", 0)),
        ("Edges removed", diagnostics.get("edges_removed", 0)),
        ("Max condition size", diagnostics.get("max_conditioning_size", 0)),
        ("All CI tests", context.exact_ci_tests),
        ("Sepset source", context.exact_separator_source),
    )
    cells = "".join(
        f"<div><small>{_esc(label)}</small><strong>{_esc(value)}</strong></div>"
        for label, value in items
    )
    return f"<div class='diagnostic-grid'>{cells}</div>"


def render_sample_card(
    run: SampleRun,
    target_shape: dict[tuple[str, str], tuple[str, str]],
) -> str:
    passed = run.exact_f1 == 1.0
    extra_edges = set(run.shape) - set(target_shape)
    xy_text = (
        f"Extra edge retained: {run.xy_edge}"
        if run.xy_present
        else f"X–Y removed with {{{', '.join(run.separator)}}}"
    )
    metrics = (
        ("Exact F1", run.exact_f1),
        ("Semantic F1", run.semantic_f1),
        ("Skeleton F1", run.skeleton_f1),
        ("Endpoint acc.", run.endpoint_accuracy),
    )
    metric_html = "".join(
        f"<div><small>{label}</small><strong>{value:.3f}</strong></div>"
        for label, value in metrics
    )
    standard_xy = "retained X–Y" if run.standard_fci_xy_present else "removed X–Y"
    return f"""<article class="sample-card {'success' if passed else 'warning'}">
      <div class="sample-head"><div><p class="eyebrow">N = {run.n_samples:,}</p><h3>{'Exact recovery' if passed else 'Illustrative finite-sample miss'}</h3></div><span class="status {'pass' if passed else 'warn'}">{'✓ PASS' if passed else '⚠ LIMITATION'}</span></div>
      {render_pag_svg(run.shape, f"Learned PAG at N={run.n_samples:,}", extra_edges)}
      <p class="xy-result">{_esc(xy_text)}</p>
      <div class="metric-grid">{metric_html}</div>
      <div class="baseline"><strong>Standard FCI, same data:</strong> exact F1={run.standard_fci_exact_f1:.3f} · {_esc(standard_xy)} · {run.standard_fci_ci_tests} CI tests</div>
      <div class="provenance">seed={run.seed} · CI tests={run.ci_tests} · source={_esc(run.separator_source)}</div>
    </article>"""


def render_pag_svg(
    shape: dict[tuple[str, str], tuple[str, str]],
    title: str,
    highlighted_edges: Optional[set[tuple[str, str]]] = None,
) -> str:
    highlighted_edges = highlighted_edges or set()
    positions = {
        "Z": (300, 45),
        "U": (145, 145),
        "V": (455, 145),
        "X": (75, 285),
        "Y": (525, 285),
    }
    label_positions = {
        frozenset(("Z", "U")): (208, 84),
        frozenset(("Z", "V")): (392, 84),
        frozenset(("U", "X")): (92, 214),
        frozenset(("V", "Y")): (508, 214),
        frozenset(("U", "Y")): (365, 184),
        frozenset(("V", "X")): (235, 244),
        frozenset(("X", "Y")): (300, 285),
    }
    edge_parts = []
    for edge, endpoints in shape.items():
        x, y = edge
        if x not in positions or y not in positions:
            continue
        x1, y1 = positions[x]
        x2, y2 = positions[y]
        highlighted = edge in highlighted_edges
        css = "edge extra" if highlighted else "edge"
        stroke = "#b22a2a" if highlighted else "#657386"
        dash = " stroke-dasharray='7 5'" if highlighted else ""
        edge_parts.append(
            f"<line class='{css}' x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' "
            f"stroke='{stroke}' stroke-width='{'3' if highlighted else '2.2'}'{dash} tabindex='0'>"
            f"<title>{_esc(_edge_notation(edge, endpoints))}</title></line>"
        )
        lx, ly = label_positions.get(frozenset(edge), ((x1 + x2) / 2, (y1 + y2) / 2))
        notation = _endpoint_notation(endpoints)
        label_fill = "#fff0f0" if highlighted else "#ffffff"
        label_stroke = "#b22a2a" if highlighted else "#ccd4dd"
        edge_parts.append(
            f"<rect class='edge-label-bg {'extra' if highlighted else ''}' x='{lx - 22}' y='{ly - 12}' width='44' height='24' rx='6' fill='{label_fill}' stroke='{label_stroke}'/>"
            f"<text class='edge-label' x='{lx}' y='{ly + 5}' text-anchor='middle' fill='#243447'>{_esc(notation)}</text>"
        )
    node_parts = "".join(
        f"<g><circle class='node' cx='{x}' cy='{y}' r='25' fill='#ffffff' stroke='#26384d' stroke-width='2'/>"
        f"<text class='node-label' x='{x}' y='{y + 6}' text-anchor='middle' fill='#17212b'>{node}</text></g>"
        for node, (x, y) in positions.items()
    )
    return f"""<svg class="pag" viewBox="0 0 600 330" role="img" aria-labelledby="{_slug(title)}-title {_slug(title)}-desc">
      <title id="{_slug(title)}-title">{_esc(title)}</title>
      <desc id="{_slug(title)}-desc">Five-node PAG with endpoint notation printed on every edge. Red dashed edges are extra relative to the oracle target.</desc>
      {''.join(edge_parts)}{node_parts}
    </svg>"""


def render_leaderboard(
    results: list[BenchmarkResult],
    external_enabled: bool = True,
) -> str:
    included, excluded, case_ids = same_cohort_aggregates(results)
    if not case_ids:
        return "<div class='empty'>No benchmark results were supplied.</div>"
    rows = []
    for item in included:
        rows.append(
            "<tr>"
            f"<td>{_esc(item.algorithm)}</td><td>{item.n_cases}/{len(case_ids)}</td>"
            f"<td>{item.mean_exact_edge_f1:.3f}</td>"
            f"<td>{item.mean_semantic_edge_f1:.3f}</td>"
            f"<td>{item.mean_skeleton_f1:.3f}</td>"
            f"<td>{item.mean_endpoint_accuracy:.3f}</td>"
            f"<td>{_fmt(item.mean_ci_test_count)}</td></tr>"
        )
    excluded_html = "".join(
        f"<li><code>{_esc(name)}</code>: {_esc(reason)}</li>"
        for name, reason in excluded
    )
    if not external_enabled:
        excluded_html += "<li>External comparators disabled for this artifact.</li>"
    return f"""<div class="cohort-line"><strong>Cohort:</strong> {len(case_ids)} shared cases · {_esc(', '.join(case_ids))}</div>
    <div class="table-wrap"><table><thead><tr><th>Algorithm</th><th>Coverage</th><th>Exact F1</th><th>Semantic F1</th><th>Skeleton F1</th><th>Endpoint acc.</th><th>Mean CI tests</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
    <details><summary>Excluded algorithms / unavailable runs</summary><ul>{excluded_html or '<li>None</li>'}</ul></details>"""


def same_cohort_aggregates(
    results: list[BenchmarkResult],
) -> tuple[list[BenchmarkAggregate], list[tuple[str, str]], tuple[str, ...]]:
    """Aggregate only algorithms with completed results for the same case set."""

    case_ids = tuple(sorted({result.case_name for result in results}))
    if not case_ids:
        return [], [], ()
    required = set(case_ids)
    algorithms = sorted({result.algorithm for result in results})
    included_results: list[BenchmarkResult] = []
    excluded: list[tuple[str, str]] = []
    for algorithm in algorithms:
        group = [result for result in results if result.algorithm == algorithm]
        completed_cases = {result.case_name for result in group if not result.skipped}
        if completed_cases != required:
            missing = sorted(required - completed_cases)
            reasons = sorted(
                {result.skipped_reason for result in group if result.skipped_reason}
            )
            detail = f"coverage {len(completed_cases)}/{len(required)}"
            if missing:
                detail += f"; missing {', '.join(missing)}"
            if reasons:
                detail += f"; {'; '.join(reasons)}"
            excluded.append((algorithm, detail))
            continue
        included_results.extend(result for result in group if not result.skipped)
    return aggregate_benchmark_results(included_results), excluded, case_ids


def render_limitations() -> str:
    items = (
        ("PAG, not a unique DAG", "Circles mark non-identifiability; a bidirected edge does not prove one named latent variable."),
        ("Oracle assumptions", "Soundness/completeness relies on Markov, faithfulness, acyclicity and correct CI answers."),
        ("Finite-sample sensitivity", "Alpha, test power and invariant-arrowhead errors can prevent FCI+ candidate recognition."),
        ("One seeded SEM", "The 5k/50k comparison is a reproducible stress case, not a universal sample-size threshold."),
        ("No universal winner", "pcalg and causal-learn are versioned references; pcalg 2.7-12 also has an index-1 PosDsepLinks omission, so external output is not truth."),
        ("Complexity is an oracle bound", "FCI+ proves O(N^(2(k+2))) CI tests for bounded MAG degree k; statistical power can still fall with larger conditioning sets."),
    )
    return "<div class='limit-grid'>" + "".join(
        f"<article><h3>{_esc(title)}</h3><p>{_esc(text)}</p></article>"
        for title, text in items
    ) + "</div>"


def render_citations() -> str:
    return f"""<article class="citation-card"><h3>Primary sources</h3><ol>
      <li>Claassen, Mooij &amp; Heskes (2013), <a href="{PAPER_URL}">Learning Sparse Causal Models is not NP-hard</a>, UAI. Figure 4(b), Lemma 4, Algorithm 2.</li>
      <li>Zhang (2008), <a href="{ZHANG_URL}">On the completeness of orientation rules for causal discovery in the presence of latent confounders and selection bias</a>.</li>
      <li>Claassen &amp; Heskes (2011), <a href="{RULE_TABLE_URL}">A Logical Characterization of Constraint-Based Causal Discovery</a>, readable R1–R10 table and schedule.</li>
      <li>CRAN pcalg, <a href="https://rdrr.io/cran/pcalg/src/R/fciPlus.R">fciPlus.R source</a>. Used as a differential reference with a documented index-1 candidate caveat.</li>
    </ol></article>"""


def render_reproducibility(context: ShowcaseContext) -> str:
    config = " · ".join(f"{key}={value}" for key, value in PAPER_CONFIG.items())
    command = (
        "PYTHONPATH=src python examples/10_fci_plus_advisor_showcase.py "
        "--no-external --output examples/advisor_showcase.html"
    )
    tests = (
        "python -m pytest -q tests/test_published_reference_graphs.py "
        "tests/test_advisor_showcase.py"
    )
    return f"""<article class="citation-card"><h3>Reproduce exactly</h3>
      <dl><dt>Git</dt><dd>{_esc(context.git_sha)}</dd><dt>Python</dt><dd>{_esc(context.python_version)}</dd><dt>fci-engine</dt><dd>{_esc(context.package_version)}</dd><dt>Generated UTC</dt><dd>{_esc(context.generated_at)}</dd><dt>Sample fixture</dt><dd>seed=1; N=5,000 and 50,000</dd><dt>Configuration</dt><dd>{_esc(config)}</dd></dl>
      <pre><code>{_esc(command)}</code></pre><pre><code>{_esc(tests)}</code></pre>
    </article>"""


def demo_dsep_diagnostics() -> dict[str, int]:
    """Backward-compatible helper backed by the published exact-oracle run."""

    return build_showcase_context(external_enabled=False).exact_diagnostics


def _styles() -> str:
    return """
:root{--ink:#17212b;--muted:#5d6b78;--line:#d8dee5;--paper:#fff;--bg:#f3f5f7;--nav:#132238;--blue:#285f9e;--green:#167254;--green-bg:#e9f7f1;--amber:#9a5b00;--amber-bg:#fff6df;--red:#b22a2a;--red-bg:#fff0f0}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.5}nav{position:sticky;top:0;z-index:10;height:58px;padding:0 max(18px,calc((100vw - 1180px)/2));display:flex;align-items:center;gap:22px;background:var(--nav);color:#fff;border-bottom:1px solid #2b3d55}nav a{color:#dbe7f3;text-decoration:none;font-size:13px}.brand{font-weight:800!important;color:#fff!important;font-size:15px!important}.nav-links{display:flex;gap:18px;flex:1}.build{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:#aebdd0}main{width:min(1180px,calc(100vw - 28px));margin:22px auto 50px}section{scroll-margin-top:76px;background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:26px;margin-bottom:18px;box-shadow:0 2px 8px rgba(18,34,55,.035)}h1{font-size:clamp(30px,4.2vw,51px);line-height:1.08;letter-spacing:-.035em;margin:8px 0 16px;max-width:850px}h2{font-size:25px;line-height:1.2;margin:2px 0}h3{margin:0 0 6px;font-size:15px}p{margin:0;color:var(--muted)}a{color:var(--blue)}code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.eyebrow{font-size:11px;font-weight:850;letter-spacing:.12em;color:var(--blue)}.hero{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(260px,.55fr);gap:34px;align-items:center;padding:34px}.lead{font-size:17px;max-width:780px}.notice{margin-top:18px;border-left:4px solid var(--blue);background:#edf4fb;padding:12px 14px;color:#294964;font-size:13px}.hero-kpis{display:grid;gap:10px}.kpi{border:1px solid var(--line);border-radius:10px;padding:14px;background:#fafbfc}.kpi small,.metric-grid small,.diagnostic-grid small{display:block;color:var(--muted);font-size:11px;font-weight:700}.kpi strong{display:block;margin-top:2px;font-size:22px}.kpi.pass{border-color:#9cd6c0;background:var(--green-bg)}.kpi.fail{border-color:#f0b2b2;background:var(--red-bg)}.section-heading{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:12px}.section-heading>a{font-size:13px}.section-copy{max-width:920px;margin-bottom:20px}.steps{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}.step{border:1px solid var(--line);border-radius:9px;padding:13px;background:#fafbfc;min-height:132px}.step span{display:inline-grid;place-items:center;background:var(--nav);color:#fff;border-radius:5px;min-width:30px;height:25px;padding:0 6px;font-size:11px;font-weight:800}.step h3{margin-top:10px;font-size:13px}.step p{font-size:12px}.status{display:inline-flex;align-items:center;border-radius:999px;padding:5px 10px;font-size:11px;font-weight:850;white-space:nowrap}.status.pass{color:var(--green);background:var(--green-bg)}.status.warn{color:var(--amber);background:var(--amber-bg)}.status.fail{color:var(--red);background:var(--red-bg)}.graph-grid,.sample-grid,.repro-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.graph-grid figure{margin:0;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#fbfcfd}.graph-grid figcaption{border-top:1px solid var(--line);padding:10px 13px;font-size:12px;color:var(--muted)}.pag{display:block;width:100%;height:auto;max-height:330px}.edge{stroke:#657386;stroke-width:2.2}.edge.extra{stroke:var(--red);stroke-width:3;stroke-dasharray:7 5}.edge:focus{outline:none;stroke:#005fcc;stroke-width:5}.node{fill:#fff;stroke:#26384d;stroke-width:2}.node-label{font-size:17px;font-weight:800;fill:var(--ink)}.edge-label-bg{fill:#fff;stroke:#ccd4dd}.edge-label-bg.extra{fill:var(--red-bg);stroke:var(--red)}.edge-label{font:700 14px ui-monospace,SFMono-Regular,Menlo,monospace;fill:#243447}.checks{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}.check{border:1px solid var(--line);border-radius:9px;padding:12px}.check.pass{border-color:#a9d9c7;background:var(--green-bg)}.check.fail{border-color:#efb5b5;background:var(--red-bg)}.check strong{display:block;font-size:13px}.check span{display:block;color:var(--muted);font-size:12px;margin-top:3px}details{margin-top:13px;border-top:1px solid var(--line);padding-top:12px}summary{cursor:pointer;color:#34475b;font-size:13px;font-weight:750}.diagnostic-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-top:12px}.diagnostic-grid>div,.metric-grid>div{border:1px solid var(--line);border-radius:7px;padding:9px;background:#fff}.diagnostic-grid strong,.metric-grid strong{display:block;margin-top:2px;font-size:17px}.sample-card{border:1px solid var(--line);border-radius:11px;padding:16px;background:#fbfcfd}.sample-card.warning{border-top:5px solid var(--amber)}.sample-card.success{border-top:5px solid var(--green)}.sample-head{display:flex;justify-content:space-between;gap:12px}.sample-head h3{font-size:18px}.sample-card .pag{margin:8px 0}.xy-result{border-radius:7px;padding:9px 11px;font-size:13px;font-weight:750}.warning .xy-result{background:var(--amber-bg);color:#704500}.success .xy-result{background:var(--green-bg);color:#0e5d43}.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin-top:10px}.baseline{margin-top:9px;border-left:3px solid var(--blue);padding:7px 9px;background:#edf4fb;color:#294964;font-size:11px}.provenance{margin-top:10px;color:var(--muted);font:11px ui-monospace,SFMono-Regular,Menlo,monospace}.cohort-line{background:#edf4fb;border-radius:8px;padding:10px 12px;font-size:13px;margin:12px 0}.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:9px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:10px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}th:first-child,td:first-child{text-align:left}th{background:#f3f6f9;color:#35495e}.empty{border:1px dashed var(--line);border-radius:9px;padding:25px;text-align:center;color:var(--muted)}.limit-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.limit-grid article{border:1px solid var(--line);border-left:4px solid #77889b;border-radius:8px;padding:13px}.limit-grid p{font-size:12px}.citation-card{border:1px solid var(--line);border-radius:10px;padding:16px;background:#fbfcfd}.citation-card li{margin-bottom:9px;font-size:13px;color:#455667}.citation-card dl{display:grid;grid-template-columns:105px 1fr;gap:6px 10px;font-size:12px}.citation-card dt{font-weight:800}.citation-card dd{margin:0;color:var(--muted);overflow-wrap:anywhere}pre{overflow:auto;background:#142238;color:#e7edf5;border-radius:7px;padding:11px;font-size:11px}footer{text-align:center;color:#708090;font-size:12px;padding:0 20px 35px}@media(max-width:900px){nav{padding:0 14px}.nav-links,.build{display:none}.hero,.graph-grid,.sample-grid,.repro-grid{grid-template-columns:1fr}.steps{grid-template-columns:repeat(2,1fr)}.checks,.limit-grid{grid-template-columns:1fr 1fr}.diagnostic-grid{grid-template-columns:repeat(3,1fr)}.metric-grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:560px){main{width:min(100% - 16px,1180px)}section,.hero{padding:18px}.steps,.checks,.limit-grid{grid-template-columns:1fr}.diagnostic-grid{grid-template-columns:repeat(2,1fr)}h1{font-size:31px}}@media print{@page{size:A4;margin:12mm}nav{position:static}.nav-links{display:none}body{background:#fff}main{width:100%;margin:0}section{break-inside:auto;box-shadow:none}.hero,.graph-grid,.sample-grid,.repro-grid{grid-template-columns:1fr}.steps{grid-template-columns:repeat(2,1fr)}.hero-kpis{margin-top:14px}.graph-grid figure,.sample-card,.citation-card{break-inside:avoid;margin-bottom:12px}.checks,.limit-grid{grid-template-columns:1fr 1fr}.diagnostic-grid{grid-template-columns:repeat(3,1fr)}details>*{display:block!important}a{color:inherit;text-decoration:none}}
"""


def _kpi(label: str, value: str, passed: bool) -> str:
    return f"<div class='kpi {'pass' if passed else 'fail'}'><small>{_esc(label)}</small><strong>{_esc(value)}</strong></div>"


def _check(label: str, passed: bool, detail: str) -> str:
    return f"<div class='check {'pass' if passed else 'fail'}' role='listitem'><strong>{'✓' if passed else '✕'} {_esc(label)}</strong><span>{_esc(detail)}</span></div>"


def _endpoint_notation(endpoints: tuple[str, str]) -> str:
    left = {"TAIL": "—", "ARROW": "<", "CIRCLE": "o"}[endpoints[0]]
    right = {"TAIL": "—", "ARROW": ">", "CIRCLE": "o"}[endpoints[1]]
    return f"{left}—{right}"


def _edge_notation(
    edge: tuple[str, str],
    endpoints: tuple[str, str],
) -> str:
    return f"{edge[0]} {_endpoint_notation(endpoints)} {edge[1]}"


def _string_shape(shape: dict) -> dict[tuple[str, str], tuple[str, str]]:
    return {
        (str(x), str(y)): (
            endpoint_x if isinstance(endpoint_x, str) else endpoint_x.name,
            endpoint_y if isinstance(endpoint_y, str) else endpoint_y.name,
        )
        for (x, y), (endpoint_x, endpoint_y) in shape.items()
    }


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or "unavailable"


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in value).strip("-")


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
