"""Standalone visual report renderer for the Tennessee STAR case study."""

from __future__ import annotations

import html
import json
import math
from typing import Any

from case_studies.tennessee_star.study import TEMPORAL_TIERS

ARM_COLORS = {
    "Regular": "#667085",
    "Regular + aide": "#d97706",
    "Small": "#047857",
}
ALGORITHM_LABELS = {
    "fci": "Standard FCI",
    "fci_plus": "FCI+",
}
PANEL_LABELS = {
    "attrition": "Attrition / observation",
    "longitudinal": "Longitudinal achievement",
    "focused_treatment": "Focused treatment-outcome",
}


def render_report(payload: dict[str, Any]) -> str:
    """Return a self-contained HTML report."""

    run_index = {(run["panel"], run["algorithm"]): run for run in payload["runs"]}
    descriptive = payload["descriptives"]
    small_k = _contrast(
        descriptive,
        metric="kindergarten_score",
        comparison="Small - Regular",
    )
    small_g3 = _contrast(
        descriptive,
        metric="grade3_score",
        comparison="Small - Regular",
    )
    aide_k = _contrast(
        descriptive,
        metric="kindergarten_score",
        comparison="Regular + aide - Regular",
    )
    focused_plus = run_index[("focused_treatment", "fci_plus")]
    focused_edge = _find_edge(
        focused_plus,
        "K_Class",
        "Grade3_Achievement",
    )
    attrition_class_edge = _find_edge(
        run_index[("attrition", "fci_plus")],
        "K_Class",
        "Grade3_Observed",
    )
    fastest_ratio = min(
        comparison["fci_plus_runtime_ratio"]
        for comparison in payload["comparisons"].values()
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tennessee STAR - FCI and FCI+ Case Study</title>
<style>
  :root {{
    color-scheme: light;
    --ink: #172033;
    --muted: #667085;
    --line: #d0d5dd;
    --soft-line: #eaecf0;
    --paper: #ffffff;
    --canvas: #f3f6fa;
    --green: #047857;
    --green-soft: #ecfdf3;
    --blue: #2563eb;
    --blue-soft: #eff6ff;
    --amber: #b45309;
    --amber-soft: #fffbeb;
    --red: #b42318;
    --red-soft: #fef3f2;
  }}
  * {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    margin: 0;
    background: var(--canvas);
    color: var(--ink);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system,
      BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  header {{
    background: linear-gradient(135deg, #102a43 0%, #164e63 60%, #047857 100%);
    color: #fff;
    padding: 42px max(24px, calc((100vw - 1420px) / 2)) 34px;
  }}
  header h1 {{
    font-size: clamp(28px, 4vw, 44px);
    line-height: 1.08;
    margin: 0 0 12px;
    max-width: 980px;
  }}
  header p {{
    color: #dbeafe;
    line-height: 1.65;
    margin: 0;
    max-width: 980px;
  }}
  nav {{
    align-items: center;
    background: #fff;
    border-bottom: 1px solid var(--line);
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    padding: 12px max(24px, calc((100vw - 1420px) / 2));
    position: sticky;
    top: 0;
    z-index: 10;
  }}
  nav a {{
    color: #344054;
    font-size: 13px;
    font-weight: 700;
    text-decoration: none;
  }}
  nav a:hover {{ color: var(--green); }}
  main {{
    margin: 24px auto 48px;
    max-width: 1420px;
    padding: 0 20px;
  }}
  section {{
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 12px;
    margin-bottom: 20px;
    padding: 22px;
  }}
  h2 {{ font-size: 22px; margin: 0 0 8px; }}
  h3 {{ font-size: 16px; margin: 0 0 8px; }}
  p {{ color: var(--muted); line-height: 1.6; }}
  a {{ color: var(--blue); }}
  code {{
    background: #f2f4f7;
    border-radius: 4px;
    color: #344054;
    padding: 2px 5px;
  }}
  .eyebrow {{
    color: var(--green);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .08em;
    margin-bottom: 5px;
    text-transform: uppercase;
  }}
  .section-copy {{ margin: 0 0 18px; max-width: 980px; }}
  .metric-grid {{
    display: grid;
    gap: 12px;
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }}
  .metric {{
    background: #fbfcfe;
    border: 1px solid var(--soft-line);
    border-radius: 9px;
    min-height: 112px;
    padding: 14px;
  }}
  .metric-label {{
    color: var(--muted);
    font-size: 12px;
    font-weight: 700;
  }}
  .metric-value {{
    font-size: 28px;
    font-weight: 800;
    margin: 5px 0 3px;
  }}
  .metric-context {{ color: var(--muted); font-size: 12px; line-height: 1.4; }}
  .callout {{
    border-left: 4px solid var(--green);
    background: var(--green-soft);
    color: #065f46;
    line-height: 1.6;
    margin-top: 16px;
    padding: 12px 14px;
  }}
  .warning {{
    border-left-color: var(--amber);
    background: var(--amber-soft);
    color: #92400e;
  }}
  .danger {{
    border-left-color: var(--red);
    background: var(--red-soft);
    color: #912018;
  }}
  .flow {{
    align-items: stretch;
    display: grid;
    gap: 10px;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    margin-top: 16px;
  }}
  .flow-step {{
    background: #fbfcfe;
    border: 1px solid var(--soft-line);
    border-radius: 9px;
    padding: 14px;
    position: relative;
  }}
  .flow-step:not(:last-child)::after {{
    color: #98a2b3;
    content: "→";
    font-size: 22px;
    position: absolute;
    right: -18px;
    top: 42%;
    z-index: 2;
  }}
  .flow-count {{ font-size: 24px; font-weight: 800; }}
  .flow-name {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
  .chart-grid {{
    display: grid;
    gap: 18px;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    margin-top: 18px;
  }}
  .chart-panel {{
    border-top: 1px solid var(--soft-line);
    padding-top: 14px;
  }}
  .chart-panel svg {{ height: auto; width: 100%; }}
  .caption {{
    color: var(--muted);
    font-size: 12px;
    line-height: 1.45;
    margin-top: 8px;
  }}
  .table-wrap {{ overflow-x: auto; }}
  table {{
    border-collapse: collapse;
    font-size: 13px;
    width: 100%;
  }}
  th, td {{
    border-bottom: 1px solid var(--soft-line);
    padding: 9px 10px;
    text-align: left;
    vertical-align: top;
  }}
  th {{
    color: #475467;
    font-size: 11px;
    letter-spacing: .04em;
    text-transform: uppercase;
  }}
  .positive {{ color: var(--green); font-weight: 750; }}
  .muted {{ color: var(--muted); }}
  .pag-grid {{
    display: grid;
    gap: 18px;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    margin-top: 16px;
  }}
  .pag-card {{
    border: 1px solid var(--soft-line);
    border-radius: 10px;
    min-width: 0;
    padding: 12px;
  }}
  .pag-header {{
    align-items: center;
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
  }}
  .badge {{
    background: #f2f4f7;
    border-radius: 999px;
    color: #475467;
    font-size: 11px;
    font-weight: 700;
    padding: 4px 8px;
  }}
  .pag-card svg {{ height: auto; width: 100%; }}
  .pag-edge {{ cursor: pointer; }}
  .pag-edge:hover .edge-line,
  .pag-edge:focus .edge-line {{ stroke: var(--blue); stroke-width: 3; }}
  .pag-node rect {{ fill: #fff; stroke: #667085; }}
  .pag-node text {{ fill: #172033; font-size: 11px; font-weight: 700; }}
  .edge-detail {{
    background: #f8fafc;
    border-radius: 7px;
    color: #475467;
    font-size: 12px;
    line-height: 1.5;
    margin-top: 8px;
    min-height: 43px;
    padding: 8px 10px;
  }}
  .performance-grid {{
    display: grid;
    gap: 14px;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    margin-top: 16px;
  }}
  .performance-card {{
    border: 1px solid var(--soft-line);
    border-radius: 9px;
    padding: 14px;
  }}
  .ratio-row {{
    display: grid;
    gap: 8px;
    grid-template-columns: 80px 1fr 54px;
    margin-top: 10px;
  }}
  .ratio-track {{
    align-self: center;
    background: #eaecf0;
    border-radius: 999px;
    height: 9px;
    overflow: hidden;
  }}
  .ratio-fill {{ background: var(--green); height: 100%; }}
  .two-column {{
    display: grid;
    gap: 18px;
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }}
  .finding {{
    border-top: 1px solid var(--soft-line);
    padding: 12px 0;
  }}
  .finding:first-child {{ border-top: 0; padding-top: 0; }}
  .finding strong {{ display: block; margin-bottom: 4px; }}
  footer {{
    color: var(--muted);
    font-size: 12px;
    padding: 0 20px 36px;
    text-align: center;
  }}
  @media (max-width: 1000px) {{
    .metric-grid, .flow, .chart-grid, .performance-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .pag-grid, .two-column {{ grid-template-columns: 1fr; }}
    .flow-step::after {{ display: none; }}
  }}
  @media (max-width: 640px) {{
    header {{ padding: 30px 20px; }}
    main {{ padding: 0 10px; }}
    section {{ padding: 16px; }}
    .metric-grid, .flow, .chart-grid, .performance-grid {{
      grid-template-columns: 1fr;
    }}
    nav {{ position: static; }}
  }}
</style>
</head>
<body>
<header>
  <div class="eyebrow" style="color:#a7f3d0">Applied case study - separate from the algorithm package</div>
  <h1>Tennessee STAR causal-structure discovery with self-implemented FCI and FCI+</h1>
  <p>
    A reproducible application to the randomized class-size experiment. The report
    keeps three layers separate: descriptive randomized-arm evidence, data-only PAG
    discovery, and researcher interpretation of selection, discretization, and
    temporal plausibility.
  </p>
</header>
<nav>
  <a href="#design">Design</a>
  <a href="#descriptive">Descriptive evidence</a>
  <a href="#pag">PAG results</a>
  <a href="#stability">Stability</a>
  <a href="#performance">Performance</a>
  <a href="#assessment">Self-assessment</a>
  <a href="#reproducibility">Reproducibility</a>
</nav>
<main>
  <section id="design">
    <div class="eyebrow">Cohort and method</div>
    <h2>Application design</h2>
    <p class="section-copy">
      The source contains 11,601 students who participated in at least one
      experimental year. The analysis starts from the 6,325 students assigned
      to a STAR class in kindergarten across 79 schools: small classes of
      approximately 13-17 students, regular classes of 22-25, or regular
      classes with a full-time aide. All discovery variables are discretized
      and tested with the likelihood-ratio G-square CI test.
    </p>
    <div class="metric-grid">
      <div class="metric"><div class="metric-label">Original student records</div><div class="metric-value">{payload["cohorts"]["raw_rows"]:,}</div><div class="metric-context">Official Dataverse student file</div></div>
      <div class="metric"><div class="metric-label">Kindergarten randomized cohort</div><div class="metric-value">{payload["cohorts"]["kindergarten_rows"]:,}</div><div class="metric-context">{payload["cohorts"]["kindergarten_schools"]} kindergarten schools</div></div>
      <div class="metric"><div class="metric-label">Primary CI threshold</div><div class="metric-value">α = {payload["configuration"]["alpha"]:.2f}</div><div class="metric-context">Paper profiles; G-square CI</div></div>
      <div class="metric"><div class="metric-label">FCI+ sparsity bound</div><div class="metric-value">k = {payload["configuration"]["fci_plus_k"]}</div><div class="metric-context">Claassen et al. Algorithm 2 profile</div></div>
    </div>
    {_render_flow(payload)}
    <div class="callout">
      The algorithm code remains under <code>src/fci_engine</code>. This case
      study owns every STAR-specific choice: cohort construction, coding,
      discretization, school-cluster resampling, and interpretation.
    </div>
  </section>

  <section id="descriptive">
    <div class="eyebrow">Randomized-arm reference</div>
    <h2>What the observed outcomes say before causal discovery</h2>
    <p class="section-copy">
      These are descriptive arm comparisons, with 95% intervals from a
      kindergarten-school cluster bootstrap. They are not produced by FCI and
      provide an external design-based reference for evaluating the PAG output.
    </p>
    <div class="chart-grid">
      <div class="chart-panel">
        <h3>End-of-kindergarten score</h3>
        {_render_arm_chart(descriptive, "kindergarten_score", minimum=880, maximum=950, decimals=1)}
      </div>
      <div class="chart-panel">
        <h3>Grade-3 score among observed students</h3>
        {_render_arm_chart(descriptive, "grade3_score", minimum=1220, maximum=1280, decimals=1)}
      </div>
      <div class="chart-panel">
        <h3>Both grade-3 scores observed</h3>
        {_render_arm_chart(descriptive, "grade3_observed_rate", minimum=0.35, maximum=0.57, decimals=1, percent=True)}
      </div>
    </div>
    {_render_contrast_table(descriptive)}
    <div class="callout">
      Small classes exceeded regular classes by
      <strong>{small_k["estimate"]:.1f} points</strong> at kindergarten
      (95% cluster interval {small_k["ci_low"]:.1f} to {small_k["ci_high"]:.1f})
      and <strong>{small_g3["estimate"]:.1f} points</strong> at grade 3 among
      observed students ({small_g3["ci_low"]:.1f} to {small_g3["ci_high"]:.1f}).
      The kindergarten regular-with-aide contrast was only
      {aide_k["estimate"]:.1f} points.
    </div>
  </section>

  <section id="pag">
    <div class="eyebrow">Algorithm output</div>
    <h2>Learned Partial Ancestral Graphs</h2>
    <p class="section-copy">
      Click any edge to see its endpoint notation. The graphs are data-only:
      no temporal or randomization constraints were supplied. Therefore,
      backward-looking arrows are retained and audited rather than silently
      corrected.
    </p>
    {_render_panel_graphs(run_index, "attrition")}
    {_render_panel_graphs(run_index, "longitudinal")}
    {_render_panel_graphs(run_index, "focused_treatment")}
    <div class="callout warning">
      In the focused complete-case panel, both algorithms learned
      <strong>{_esc(focused_edge or "no class-outcome edge")}</strong>.
      Because kindergarten assignment was randomized, this bidirected mark
      should not be read literally as proof of a pre-treatment latent
      confounder. Complete-case selection, school clustering, CI-test
      approximation, and omitted post-randomization variables can produce the
      same observed independence pattern.
    </div>
  </section>

  <section id="stability">
    <div class="eyebrow">School-cluster resampling</div>
    <h2>Which adjacencies survive resampling?</h2>
    <p class="section-copy">
      Schools, not individual rows, were resampled to preserve within-school
      dependence. Frequencies describe adjacency stability only; they do not
      validate endpoint direction or remove systematic selection bias.
    </p>
    {_render_stability_table(run_index, payload["configuration"]["cluster_bootstraps"])}
    <h3 style="margin-top:22px">Focused treatment-outcome sensitivity</h3>
    {_render_sensitivity_table(payload["sensitivity"])}
  </section>

  <section id="performance">
    <div class="eyebrow">Same data, same CI test</div>
    <h2>FCI versus FCI+ computational work</h2>
    <p class="section-copy">
      Each full-data fit used identical panel rows, G-square tests, α, and
      endpoint rules. Standard FCI used its paper profile; FCI+ used the
      paper-aligned bounded-degree profile with k =
      {payload["configuration"]["fci_plus_k"]}.
    </p>
    {_render_performance(payload, run_index)}
    <div class="callout">
      Across these small 8-9 node panels, FCI+ used fewer CI tests in every
      comparison and reached as little as <strong>{fastest_ratio:.0%}</strong>
      of standard FCI's median runtime. This is an empirical application result,
      not a proof of asymptotic complexity. The polynomial FCI+ guarantee
      requires faithfulness and a true maximum MAG degree no larger than the
      supplied k; neither condition can be verified from STAR alone.
    </div>
  </section>

  <section id="assessment">
    <div class="eyebrow">Researcher self-assessment</div>
    <h2>What can and cannot be concluded</h2>
    <div class="two-column">
      <div>
        <h3>Supported conclusions</h3>
        <div class="finding"><strong>Small-class achievement advantage is visible descriptively.</strong><p>The randomized-arm summaries show higher kindergarten and grade-3 composite scores for the small-class arm, while the aide arm is close to the regular-class arm.</p></div>
        <div class="finding"><strong>Grade-3 observation is selective.</strong><p>Both PAG methods connect observation status to socioeconomic/contextual or kindergarten-achievement nodes, while {_esc(attrition_class_edge or "no direct K_Class-Grade3_Observed adjacency")} is retained in the primary attrition fit.</p></div>
        <div class="finding"><strong>Achievement persistence is stronger than a direct treatment edge in the full longitudinal panel.</strong><p>Kindergarten and grade-3 achievement remain adjacent, whereas class assignment is separated after the early achievement node is included.</p></div>
        <div class="finding"><strong>FCI+ reduces search work on this application.</strong><p>It achieves the largest savings on the attrition and longitudinal panels, where Possible-D-SEP style search has more work to do.</p></div>
      </div>
      <div>
        <h3>Limits and audit failures</h3>
        <div class="finding"><strong>The PAG is not an effect estimate.</strong><p>Neither FCI nor FCI+ estimates the number of score points caused by a smaller class. The arm contrasts come from the experiment's assignment, not from PAG endpoints.</p></div>
        <div class="finding"><strong>Endpoint direction is not temporally trustworthy without prior knowledge.</strong><p>{_render_temporal_flag_summary(payload)} These flags are reported as failed plausibility checks, not rewritten after the fact.</p></div>
        <div class="finding"><strong>The treatment-outcome edge is specification-sensitive.</strong><p>The sensitivity table shows whether the edge survives changes in α and discretization. A conclusion that appears only at one setting is not treated as robust.</p></div>
        <div class="finding"><strong>Mixed educational data are approximated as discrete.</strong><p>Quantile bins make G-square applicable, but bin boundaries discard information and high-order contingency tables can have sparse expected counts. Students are also clustered within schools and classrooms.</p></div>
      </div>
    </div>
    <div class="callout danger">
      Final assessment: the application demonstrates that the implementation
      can discover and audit PAG structure on a real randomized experiment, and
      that FCI+ materially reduces CI-test work. It does <strong>not</strong>
      establish a unique causal DAG, prove latent confounding, or replace the
      experiment's design-based estimate of the class-size effect.
    </div>
  </section>

  <section id="reproducibility">
    <div class="eyebrow">Data provenance and command</div>
    <h2>Reproduce the case study</h2>
    <div class="table-wrap">
      <table>
        <tbody>
          <tr><th>Dataset</th><td>Tennessee Student/Teacher Achievement Ratio (STAR), Harvard Dataverse V1</td></tr>
          <tr><th>Persistent ID</th><td><a href="{_esc(payload["source"]["doi"])}">{_esc(payload["source"]["doi"])}</a></td></tr>
          <tr><th>Official study report</th><td><a href="https://eric.ed.gov/?id=ED328356">Tennessee STAR Technical Report, 1985-1990</a></td></tr>
          <tr><th>Algorithm references</th><td><a href="https://www.cs.cmu.edu/afs/cs.cmu.edu/project/learn-43/lib/photoz/.g/web/.g/group/group2/g/opera/g/scottd/fullbook.pdf">Spirtes et al. FCI</a>; <a href="https://www.auai.org/uai2013/prints/papers/121.pdf">Claassen et al. FCI+</a></td></tr>
          <tr><th>License</th><td>{_esc(payload["source"]["license"])}</td></tr>
          <tr><th>Student records</th><td>{payload["cohorts"]["raw_rows"]:,} rows × 379 variables</td></tr>
          <tr><th>Primary configuration</th><td>G-square, α={payload["configuration"]["alpha"]}, FCI paper profile, FCI+ paper profile k={payload["configuration"]["fci_plus_k"]}</td></tr>
          <tr><th>Command</th><td><code>PYTHONPATH=src python -m case_studies.tennessee_star.run_case_study</code></td></tr>
        </tbody>
      </table>
    </div>
    <p class="caption">
      Data codes were checked against the official STAR user guide. Generated
      CSV and JSON tables contain the exact inputs and outputs used by this
      report.
    </p>
  </section>
</main>
<footer>
  Generated locally from the committed STAR application code and self-implemented
  <code>fci_engine</code> package. No remote scripts or chart assets.
</footer>
<script>
document.querySelectorAll(".pag-edge").forEach((edge) => {{
  edge.addEventListener("click", () => {{
    const detail = document.getElementById(edge.dataset.target);
    if (detail) {{
      detail.textContent = edge.dataset.explanation;
    }}
  }});
  edge.addEventListener("keydown", (event) => {{
    if (event.key === "Enter" || event.key === " ") {{
      event.preventDefault();
      edge.click();
    }}
  }});
}});
</script>
</body>
</html>
"""
    return "\n".join(line.rstrip() for line in document.splitlines()) + "\n"


def _render_flow(payload: dict[str, Any]) -> str:
    panels = payload["cohorts"]["panels"]
    return f"""
    <div class="flow">
      <div class="flow-step"><div class="flow-count">{payload["cohorts"]["raw_rows"]:,}</div><div class="flow-name">Participated in at least one STAR year</div></div>
      <div class="flow-step"><div class="flow-count">{payload["cohorts"]["kindergarten_rows"]:,}</div><div class="flow-name">Kindergarten class assignment observed</div></div>
      <div class="flow-step"><div class="flow-count">{panels["attrition"]["rows"]:,}</div><div class="flow-name">Attrition panel with kindergarten outcome</div></div>
      <div class="flow-step"><div class="flow-count">{panels["longitudinal"]["rows"]:,}</div><div class="flow-name">Longitudinal complete-case panel</div></div>
    </div>
    """


def _render_arm_chart(
    descriptive: dict[str, Any],
    metric: str,
    *,
    minimum: float,
    maximum: float,
    decimals: int,
    percent: bool = False,
) -> str:
    width = 410
    height = 245
    left = 74
    right = 16
    top = 18
    bottom = 46
    plot_width = width - left - right
    plot_height = height - top - bottom
    rows = descriptive["arms"][metric]
    arms = ["Regular", "Regular + aide", "Small"]

    def y(value: float) -> float:
        return top + plot_height * (maximum - value) / (maximum - minimum)

    grid = []
    for fraction in (0.0, 0.5, 1.0):
        value = minimum + fraction * (maximum - minimum)
        yy = y(value)
        label = f"{value * 100:.0f}%" if percent else f"{value:.0f}"
        grid.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" '
            f'y2="{yy:.1f}" stroke="#e4e7ec"/>'
            f'<text x="{left - 8}" y="{yy + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#667085">{label}</text>'
        )

    marks = []
    slot = plot_width / len(arms)
    for index, arm in enumerate(arms):
        item = rows[arm]
        x = left + slot * index + slot / 2
        estimate = item["estimate"]
        low = item["ci_low"]
        high = item["ci_high"]
        yy = y(estimate)
        low_y = y(low)
        high_y = y(high)
        label = f"{estimate * 100:.1f}%" if percent else f"{estimate:.{decimals}f}"
        marks.append(
            f'<line x1="{x:.1f}" y1="{high_y:.1f}" x2="{x:.1f}" '
            f'y2="{low_y:.1f}" stroke="{ARM_COLORS[arm]}" stroke-width="2"/>'
            f'<line x1="{x - 7:.1f}" y1="{high_y:.1f}" x2="{x + 7:.1f}" '
            f'y2="{high_y:.1f}" stroke="{ARM_COLORS[arm]}" stroke-width="2"/>'
            f'<line x1="{x - 7:.1f}" y1="{low_y:.1f}" x2="{x + 7:.1f}" '
            f'y2="{low_y:.1f}" stroke="{ARM_COLORS[arm]}" stroke-width="2"/>'
            f'<circle cx="{x:.1f}" cy="{yy:.1f}" r="6" '
            f'fill="{ARM_COLORS[arm]}"><title>{_esc(arm)}: {label}</title></circle>'
            f'<text x="{x:.1f}" y="{yy - 11:.1f}" text-anchor="middle" '
            f'font-size="11" font-weight="700" fill="#172033">{label}</text>'
            f'<text x="{x:.1f}" y="{height - 18}" text-anchor="middle" '
            f'font-size="10" fill="#475467">{_short_arm(arm)}</text>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_esc(metric)} by kindergarten class arm">'
        + "".join(grid)
        + "".join(marks)
        + "</svg>"
        + '<div class="caption">Point estimate and 95% school-cluster bootstrap interval.</div>'
    )


def _render_contrast_table(descriptive: dict[str, Any]) -> str:
    labels = {
        "kindergarten_score": "Kindergarten score",
        "grade3_score": "Grade-3 score",
        "grade3_observed_rate": "Grade-3 observed rate",
    }
    rows = []
    for contrast in descriptive["contrasts"]:
        percent = contrast["metric"] == "grade3_observed_rate"
        factor = 100 if percent else 1
        suffix = " pp" if percent else " points"
        rows.append(
            "<tr>"
            f"<td>{_esc(labels[contrast['metric']])}</td>"
            f"<td>{_esc(contrast['comparison'])}</td>"
            f"<td>{contrast['estimate'] * factor:+.2f}{suffix}</td>"
            f"<td>{contrast['ci_low'] * factor:+.2f} to "
            f"{contrast['ci_high'] * factor:+.2f}{suffix}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap" style="margin-top:18px"><table>'
        "<thead><tr><th>Outcome</th><th>Contrast</th><th>Estimate</th>"
        "<th>95% cluster interval</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _render_panel_graphs(
    run_index: dict[tuple[str, str], dict[str, Any]],
    panel: str,
) -> str:
    cards = []
    for algorithm in ("fci", "fci_plus"):
        run = run_index[(panel, algorithm)]
        graph_id = f"{panel}-{algorithm}".replace("_", "-")
        cards.append(
            '<div class="pag-card">'
            '<div class="pag-header">'
            f"<h3>{_esc(ALGORITHM_LABELS[algorithm])}</h3>"
            f'<span class="badge">{run["edges"]} edges · '
            f"{run['ci_tests']:,} CI tests</span>"
            "</div>"
            f"{_render_pag_svg(run, graph_id)}"
            f'<div class="edge-detail" id="detail-{graph_id}">'
            "Select an edge to inspect its PAG endpoint meaning.</div>"
            "</div>"
        )
    return (
        '<div style="margin-top:22px">'
        f"<h3>{_esc(PANEL_LABELS[panel])}</h3>"
        '<div class="pag-grid">' + "".join(cards) + "</div></div>"
    )


def _render_pag_svg(run: dict[str, Any], graph_id: str) -> str:
    nodes = run["node_names"]
    positions = _node_positions(nodes)
    width = 720
    height = 430
    node_width = 136.0
    node_height = 36.0
    edge_parts = []
    for index, edge in enumerate(run["pag_edges"]):
        x = edge["x"]
        y = edge["y"]
        x1, y1 = positions[x]
        x2, y2 = positions[y]
        start, end = _trimmed_segment(
            (x1, y1),
            (x2, y2),
            node_width / 2 + 5,
            node_height / 2 + 5,
        )
        endpoint_x = edge["endpoint_x"]
        endpoint_y = edge["endpoint_y"]
        edge_text = edge["edge"]
        explanation = _edge_explanation(
            edge_text,
            endpoint_x,
            endpoint_y,
        )
        target = f"detail-{graph_id}"
        edge_parts.append(
            f'<g class="pag-edge" tabindex="0" role="button" '
            f'data-target="{target}" '
            f'data-explanation="{_attr(explanation)}">'
            f"<title>{_esc(explanation)}</title>"
            f'<line class="edge-hit" x1="{start[0]:.1f}" y1="{start[1]:.1f}" '
            f'x2="{end[0]:.1f}" y2="{end[1]:.1f}" stroke="transparent" '
            f'stroke-width="14"/>'
            f'<line class="edge-line" x1="{start[0]:.1f}" y1="{start[1]:.1f}" '
            f'x2="{end[0]:.1f}" y2="{end[1]:.1f}" stroke="#667085" '
            f'stroke-width="1.8"/>'
            f"{_endpoint_mark(start, end, endpoint_x)}"
            f"{_endpoint_mark(end, start, endpoint_y)}"
            "</g>"
        )

    node_parts = []
    for node in nodes:
        x, y = positions[node]
        label = node.replace("_", " ")
        node_parts.append(
            f'<g class="pag-node"><rect x="{x - node_width / 2:.1f}" '
            f'y="{y - node_height / 2:.1f}" width="{node_width:.1f}" '
            f'height="{node_height:.1f}" rx="7"/>'
            f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle">'
            f"{_esc(label)}</text></g>"
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_esc(ALGORITHM_LABELS[run["algorithm"]])} PAG">'
        + "".join(edge_parts)
        + "".join(node_parts)
        + "</svg>"
    )


def _node_positions(nodes: list[str]) -> dict[str, tuple[float, float]]:
    tier_nodes: dict[int, list[str]] = {}
    for node in nodes:
        tier_nodes.setdefault(TEMPORAL_TIERS.get(node, 1), []).append(node)
    x_positions = {0: 90.0, 1: 300.0, 2: 505.0, 3: 650.0}
    positions = {}
    for tier, values in tier_nodes.items():
        ordered = sorted(values)
        if len(ordered) == 1:
            ys = [215.0]
        else:
            step = 340.0 / (len(ordered) - 1)
            ys = [45.0 + step * index for index in range(len(ordered))]
        for node, y in zip(ordered, ys):
            positions[node] = (x_positions.get(tier, 300.0), y)
    return positions


def _trimmed_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    horizontal_radius: float,
    vertical_radius: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = math.hypot(dx, dy) or 1.0
    ux = dx / distance
    uy = dy / distance
    scale = min(
        horizontal_radius / max(abs(ux), 1e-9),
        vertical_radius / max(abs(uy), 1e-9),
    )
    return (
        (start[0] + ux * scale, start[1] + uy * scale),
        (end[0] - ux * scale, end[1] - uy * scale),
    )


def _endpoint_mark(
    point: tuple[float, float],
    other: tuple[float, float],
    endpoint: str,
) -> str:
    x, y = point
    dx = x - other[0]
    dy = y - other[1]
    length = math.hypot(dx, dy) or 1.0
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    if endpoint == "CIRCLE":
        return (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#fff" '
            f'stroke="#667085" stroke-width="1.8"/>'
        )
    if endpoint == "TAIL":
        return (
            f'<line x1="{x - px * 6:.1f}" y1="{y - py * 6:.1f}" '
            f'x2="{x + px * 6:.1f}" y2="{y + py * 6:.1f}" '
            f'stroke="#667085" stroke-width="2"/>'
        )
    if endpoint == "ARROW":
        base_x = x - ux * 12
        base_y = y - uy * 12
        points = (
            f"{x:.1f},{y:.1f} "
            f"{base_x + px * 5:.1f},{base_y + py * 5:.1f} "
            f"{base_x - px * 5:.1f},{base_y - py * 5:.1f}"
        )
        return f'<polygon points="{points}" fill="#667085"/>'
    return ""


def _edge_explanation(edge: str, endpoint_x: str, endpoint_y: str) -> str:
    meanings = {
        "CIRCLE": "unresolved endpoint",
        "TAIL": "ancestral tail",
        "ARROW": "arrowhead excluding ancestry in the reverse direction",
    }
    return (
        f"{edge}. Left endpoint: {meanings.get(endpoint_x, endpoint_x)}; "
        f"right endpoint: {meanings.get(endpoint_y, endpoint_y)}. "
        "This is a PAG relation, not a direct-effect estimate."
    )


def _render_stability_table(
    run_index: dict[tuple[str, str], dict[str, Any]],
    bootstraps: int,
) -> str:
    targets = {
        "attrition": [
            ("K_Class", "Grade3_Observed"),
            ("Free_Lunch", "Grade3_Observed"),
            ("K_Achievement", "Grade3_Observed"),
        ],
        "longitudinal": [
            ("K_Class", "Grade3_Achievement"),
            ("K_Achievement", "Grade3_Achievement"),
            ("Free_Lunch", "Grade3_Achievement"),
        ],
        "focused_treatment": [
            ("K_Class", "Grade3_Achievement"),
        ],
    }
    rows = []
    for panel, pairs in targets.items():
        for x, y in pairs:
            for algorithm in ("fci", "fci_plus"):
                run = run_index[(panel, algorithm)]
                edge = _find_edge(run, x, y)
                frequency = _bootstrap_frequency(run, x, y)
                rows.append(
                    "<tr>"
                    f"<td>{_esc(PANEL_LABELS[panel])}</td>"
                    f"<td>{_esc(x.replace('_', ' '))} - "
                    f"{_esc(y.replace('_', ' '))}</td>"
                    f"<td>{_esc(ALGORITHM_LABELS[algorithm])}</td>"
                    f"<td>{_esc(edge or 'No full-data adjacency')}</td>"
                    f"<td>{frequency:.0%}</td>"
                    "</tr>"
                )
    return (
        f'<div class="caption">{bootstraps} school-cluster bootstrap fits per '
        "panel and algorithm.</div>"
        '<div class="table-wrap"><table><thead><tr><th>Panel</th>'
        "<th>Target pair</th><th>Algorithm</th><th>Full-data PAG</th>"
        "<th>Adjacency frequency</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _render_sensitivity_table(rows: list[dict[str, Any]]) -> str:
    rendered = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{row['bins']}</td>"
            f"<td>{row['alpha']:.2f}</td>"
            f"<td>{_esc(ALGORITHM_LABELS[row['algorithm']])}</td>"
            f"<td>{'Yes' if row['adjacent'] else 'No'}</td>"
            f"<td>{_esc(row['edge'] or '-')}</td>"
            f"<td>{row['ci_tests']:,}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Quantile bins</th>'
        "<th>α</th><th>Algorithm</th><th>K class adjacent to grade 3?</th>"
        "<th>Endpoint marks</th><th>CI tests</th></tr></thead><tbody>"
        + "".join(rendered)
        + "</tbody></table></div>"
    )


def _render_performance(
    payload: dict[str, Any],
    run_index: dict[tuple[str, str], dict[str, Any]],
) -> str:
    cards = []
    for panel in ("attrition", "longitudinal", "focused_treatment"):
        fci_run = run_index[(panel, "fci")]
        plus_run = run_index[(panel, "fci_plus")]
        comparison = payload["comparisons"][panel]
        runtime_ratio = comparison["fci_plus_runtime_ratio"]
        test_ratio = comparison["fci_plus_ci_test_ratio"]
        cards.append(
            '<div class="performance-card">'
            f"<h3>{_esc(PANEL_LABELS[panel])}</h3>"
            f'<div class="ratio-row"><span class="muted">Runtime</span>'
            f'<div class="ratio-track"><div class="ratio-fill" '
            f'style="width:{min(runtime_ratio, 1) * 100:.1f}%"></div></div>'
            f"<strong>{runtime_ratio:.0%}</strong></div>"
            f'<div class="ratio-row"><span class="muted">CI tests</span>'
            f'<div class="ratio-track"><div class="ratio-fill" '
            f'style="width:{min(test_ratio, 1) * 100:.1f}%"></div></div>'
            f"<strong>{test_ratio:.0%}</strong></div>"
            f'<div class="caption">FCI+ / FCI. Median time '
            f"{plus_run['median_elapsed_seconds']:.3f}s vs "
            f"{fci_run['median_elapsed_seconds']:.3f}s; "
            f"{plus_run['ci_tests']:,} vs {fci_run['ci_tests']:,} CI tests. "
            f"Skeleton Jaccard {comparison['skeleton_jaccard']:.2f}.</div>"
            "</div>"
        )
    return '<div class="performance-grid">' + "".join(cards) + "</div>"


def _render_temporal_flag_summary(payload: dict[str, Any]) -> str:
    flags = []
    for run in payload["runs"]:
        for flag in run["temporal_flags"]:
            flags.append(
                f"{ALGORITHM_LABELS[run['algorithm']]} "
                f"({PANEL_LABELS[run['panel']]}): {flag}"
            )
    if not flags:
        return "No fully directed temporal reversals were found."
    return (
        f"The temporal audit found {len(flags)} backward directed "
        f"{'edge' if len(flags) == 1 else 'edges'}: "
        + "; ".join(_esc(flag) for flag in flags)
        + "."
    )


def _contrast(
    descriptive: dict[str, Any],
    *,
    metric: str,
    comparison: str,
) -> dict[str, Any]:
    return next(
        item
        for item in descriptive["contrasts"]
        if item["metric"] == metric and item["comparison"] == comparison
    )


def _find_edge(run: dict[str, Any], x: str, y: str) -> str | None:
    target = frozenset((x, y))
    for edge in run["pag_edges"]:
        if frozenset((edge["x"], edge["y"])) == target:
            return str(edge["edge"])
    return None


def _bootstrap_frequency(run: dict[str, Any], x: str, y: str) -> float:
    target = frozenset((x, y))
    for edge in run["bootstrap_adjacencies"]:
        if frozenset((edge["x"], edge["y"])) == target:
            return float(edge["frequency"])
    return 0.0


def _short_arm(arm: str) -> str:
    return {
        "Regular": "Regular",
        "Regular + aide": "Regular+aide",
        "Small": "Small",
    }[arm]


def _esc(value: object) -> str:
    return html.escape(str(value))


def _attr(value: object) -> str:
    return html.escape(str(value), quote=True)


def payload_json(payload: dict[str, Any]) -> str:
    """Return stable pretty JSON for the generated summary artifact."""

    return json.dumps(payload, indent=2, sort_keys=True)
