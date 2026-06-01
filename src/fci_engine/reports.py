"""Interactive HTML reports for fitted FCI results."""

from __future__ import annotations

import html
import math
from typing import TYPE_CHECKING

from fci_engine.result import EdgeExplanation

if TYPE_CHECKING:
    from fci_engine.result import FCIResult


def render_interactive_report(
    result: "FCIResult",
    title: str = "FCI Interactive PAG Report",
) -> str:
    """Return a standalone interactive HTML report for a fitted FCI result."""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
  :root {{
    color-scheme: light;
    --ink: #101827;
    --muted: #667085;
    --line: #d0d5dd;
    --panel: #ffffff;
    --band: #f3f6fa;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
      "Segoe UI", sans-serif;
    background: var(--band);
    color: var(--ink);
  }}
  header {{
    padding: 28px 36px 20px;
    background: #ffffff;
    border-bottom: 1px solid var(--line);
  }}
  main {{
    width: min(1320px, calc(100vw - 36px));
    margin: 24px auto 40px;
  }}
  h1 {{ margin: 0 0 8px; font-size: 28px; line-height: 1.15; }}
  h2 {{ margin: 0 0 14px; font-size: 18px; }}
  p {{ margin: 0; color: var(--muted); line-height: 1.5; }}
  section {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    margin-bottom: 18px;
    padding: 18px;
  }}
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 10px;
  }}
  .metric-card {{
    border: 1px solid #eaecf0;
    border-radius: 8px;
    padding: 10px;
    background: #fbfcfe;
  }}
  .metric-label {{
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 4px;
  }}
  .metric-value {{
    font-size: 18px;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
  }}
  .report-layout {{
    display: grid;
    grid-template-columns: minmax(520px, 1.1fr) minmax(360px, 0.9fr);
    gap: 18px;
    align-items: start;
  }}
  .graph-wrap {{
    width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }}
  .graph-svg {{
    display: block;
    width: 100%;
    min-width: 620px;
    height: auto;
  }}
  .graph-edge {{
    cursor: pointer;
    outline: none;
  }}
  .edge-hit {{
    stroke: transparent;
    stroke-width: 18;
    pointer-events: stroke;
  }}
  .edge-line,
  .edge-endpoint {{
    pointer-events: none;
    transition: stroke 120ms ease, stroke-width 120ms ease, fill 120ms ease;
  }}
  .graph-edge:hover .edge-line,
  .graph-edge:focus .edge-line {{
    stroke-width: 3.8;
  }}
  .graph-edge.is-selected .edge-line {{
    stroke: #0f172a;
    stroke-width: 4.2;
  }}
  .graph-edge.is-selected .edge-endpoint {{
    stroke: #0f172a;
    fill: #0f172a;
  }}
  .graph-edge.is-selected circle.edge-endpoint {{
    fill: #ffffff;
  }}
  .edge-explainer {{
    border: 1px solid #d0d5dd;
    border-radius: 8px;
    background: #fbfcfe;
    padding: 12px;
  }}
  .edge-explainer-title {{
    font-size: 15px;
    font-weight: 800;
    margin-bottom: 8px;
  }}
  .edge-explainer-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
  }}
  .edge-explainer-item {{
    min-width: 0;
  }}
  .edge-explainer-item.wide {{
    grid-column: 1 / -1;
  }}
  .edge-explainer-label {{
    color: var(--muted);
    font-size: 11px;
    font-weight: 700;
    margin-bottom: 3px;
    text-transform: uppercase;
  }}
  .edge-explainer-value {{
    font-size: 13px;
    line-height: 1.45;
    overflow-wrap: anywhere;
  }}
  .table-scroll {{
    width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    border: 1px solid #eaecf0;
    border-radius: 8px;
  }}
  table {{
    width: 100%;
    min-width: 760px;
    border-collapse: collapse;
    font-size: 13px;
  }}
  th, td {{
    text-align: left;
    border-bottom: 1px solid #eaecf0;
    padding: 8px 10px;
    vertical-align: top;
  }}
  th {{ color: #475467; font-weight: 700; background: #f9fafb; }}
  .caption {{
    color: var(--muted);
    font-size: 13px;
    margin-top: 8px;
  }}
  .edge-modal-backdrop[hidden] {{ display: none; }}
  .edge-modal-backdrop {{
    position: fixed;
    inset: 0;
    z-index: 50;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
    background: rgba(16, 24, 39, 0.48);
  }}
  .edge-modal {{
    width: min(820px, 100%);
    max-height: calc(100vh - 48px);
    overflow: auto;
    border: 1px solid #d0d5dd;
    border-radius: 8px;
    background: #ffffff;
    box-shadow: 0 24px 70px rgba(16, 24, 39, 0.24);
    padding: 16px;
  }}
  .edge-modal-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 12px;
  }}
  .edge-modal-title {{
    font-size: 16px;
    font-weight: 800;
  }}
  .edge-modal-close {{
    border: 1px solid #d0d5dd;
    border-radius: 6px;
    background: #ffffff;
    color: #101827;
    cursor: pointer;
    font: inherit;
    font-size: 13px;
    font-weight: 700;
    padding: 6px 10px;
  }}
  @media (max-width: 980px) {{
    main {{ width: min(100vw - 24px, 900px); }}
    .summary-grid, .report-layout, .edge-explainer-grid {{
      grid-template-columns: 1fr;
    }}
  }}
</style>
</head>
<body>
<header>
  <h1>{_esc(title)}</h1>
  <p>Interactive PAG report for a fitted {_esc(result.algorithm)} result.
  Click any edge to see endpoint meanings, skeleton evidence, orientation
  reasoning, and diagnostic trace summaries.</p>
</header>
<main>
  <section>
    <h2>Run Summary</h2>
    {render_summary_cards(result)}
  </section>
  <section class="report-layout">
    <div>
      <h2>Learned PAG</h2>
      <div class="graph-wrap">{render_pag_svg(result)}</div>
      <p class="caption">Green means a retained PAG edge. The report explains
      learned evidence, not ground truth, unless a separate reference graph is
      provided by the user.</p>
    </div>
    {render_edge_explainer()}
  </section>
  <section>
    <h2>Edge Table</h2>
    {render_edge_table(result)}
  </section>
</main>
{render_edge_modal()}
{render_interaction_script()}
</body>
</html>
"""


def render_summary_cards(result: "FCIResult") -> str:
    values = [
        ("Algorithm", result.algorithm),
        ("Nodes", str(len(result.graph.nodes))),
        ("Edges", str(len(result.graph.edges()))),
        ("CI tests", str(result.ci_test_count)),
        ("Orientation events", str(len(result.orientation_trace))),
    ]
    return (
        "<div class='summary-grid'>"
        + "".join(
            "<div class='metric-card'>"
            f"<div class='metric-label'>{_esc(label)}</div>"
            f"<div class='metric-value'>{_esc(value)}</div>"
            "</div>"
            for label, value in values
        )
        + "</div>"
    )


def render_pag_svg(result: "FCIResult", width: int = 760, height: int = 560) -> str:
    positions = _circle_layout(list(result.graph.nodes), width, height)
    edge_parts = []
    for x, y, endpoint_x, endpoint_y in result.graph.to_edge_list():
        explanation = result.explain_edge(str(x), str(y))
        metadata = _edge_metadata(explanation)
        x1, y1 = positions[str(x)]
        x2, y2 = positions[str(y)]
        start, end = _trimmed_line((x1, y1), (x2, y2), 50)
        edge_parts.append(
            "<g class='graph-edge' tabindex='0' role='button' "
            f"aria-label='{_esc(metadata['aria_label'])}' "
            f"data-edge='{_esc(metadata['edge'])}' "
            f"data-status='{_esc(metadata['status'])}' "
            f"data-endpoint-meaning='{_esc(metadata['endpoint_meaning'])}' "
            f"data-reasoning='{_esc(metadata['reasoning'])}' "
            f"data-skeleton-evidence='{_esc(metadata['skeleton_evidence'])}' "
            f"data-orientation-evidence='{_esc(metadata['orientation_evidence'])}'>"
            f"<title>{_esc(metadata['aria_label'])}</title>"
            f'<line class="edge-hit" x1="{start[0]:.1f}" y1="{start[1]:.1f}" '
            f'x2="{end[0]:.1f}" y2="{end[1]:.1f}"/>'
            f'<line class="edge-line" x1="{start[0]:.1f}" y1="{start[1]:.1f}" '
            f'x2="{end[0]:.1f}" y2="{end[1]:.1f}" '
            'stroke="#047857" stroke-width="2.4"/>'
            f"{_endpoint_svg(endpoint_x.name, start, end, '#047857')}"
            f"{_endpoint_svg(endpoint_y.name, end, start, '#047857')}"
            "</g>"
        )

    node_parts = []
    for node in result.graph.nodes:
        x, y = positions[str(node)]
        node_parts.append(_node_svg(str(node), x, y))

    empty_text = ""
    if not edge_parts:
        empty_text = (
            f'<text x="{width / 2:.1f}" y="{height / 2:.1f}" text-anchor="middle" '
            'font-size="14" fill="#667085">No retained PAG edges.</text>'
        )

    return (
        f'<svg class="graph-svg" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="Learned PAG">'
        '<rect width="100%" height="100%" rx="8" fill="#ffffff" '
        'stroke="#d0d5dd"/>'
        f"{''.join(edge_parts)}"
        f"{''.join(node_parts)}"
        f"{empty_text}"
        "</svg>"
    )


def render_edge_explainer() -> str:
    return (
        "<div class='edge-explainer'>"
        "<div class='edge-explainer-title'>Click an edge to explain it.</div>"
        "<div class='edge-explainer-grid'>"
        f"{_explainer_item('Edge', 'Select an edge')}"
        f"{_explainer_item('Status', 'No edge selected')}"
        f"{_explainer_item('Endpoint meaning', 'No edge selected', wide=True)}"
        f"{_explainer_item('Reasoning', 'No edge selected', wide=True)}"
        f"{_explainer_item('Skeleton evidence', 'No edge selected', wide=True)}"
        f"{_explainer_item('Orientation evidence', 'No edge selected', wide=True)}"
        "</div>"
        "<p class='caption'>This explanation is generated from deterministic "
        "PAG semantics, recorded CI/sepset evidence, and orientation-rule "
        "events. It is not generated by a language model.</p>"
        "</div>"
    )


def render_edge_table(result: "FCIResult") -> str:
    rows = []
    for x, y in result.graph.edges():
        explanation = result.explain_edge(str(x), str(y))
        metadata = _edge_metadata(explanation)
        frequency = (
            ""
            if explanation.bootstrap_frequency is None
            else f"{explanation.bootstrap_frequency:.3f}"
        )
        rows.append(
            "<tr>"
            f"<td>{_esc(metadata['edge'])}</td>"
            f"<td>{_esc(metadata['status'])}</td>"
            f"<td>{_esc(frequency)}</td>"
            f"<td>{_esc(metadata['skeleton_evidence'])}</td>"
            f"<td>{_esc(metadata['orientation_evidence'])}</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='5'>No retained PAG edges.</td></tr>")
    return (
        "<div class='table-scroll'><table>"
        "<thead><tr><th>Edge</th><th>Status</th><th>Bootstrap frequency</th>"
        "<th>Skeleton evidence</th><th>Orientation evidence</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_edge_modal() -> str:
    return (
        "<div class='edge-modal-backdrop' hidden>"
        "<div class='edge-modal' role='dialog' aria-modal='true' "
        "aria-labelledby='edge-modal-title'>"
        "<div class='edge-modal-header'>"
        "<div id='edge-modal-title' class='edge-modal-title'>"
        "Selected edge explanation</div>"
        "<button type='button' class='edge-modal-close'>Close</button>"
        "</div>"
        "<div class='edge-explainer-grid'>"
        f"{_explainer_item('Edge', 'Select an edge')}"
        f"{_explainer_item('Status', 'No edge selected')}"
        f"{_explainer_item('Endpoint meaning', 'No edge selected', wide=True)}"
        f"{_explainer_item('Reasoning', 'No edge selected', wide=True)}"
        f"{_explainer_item('Skeleton evidence', 'No edge selected', wide=True)}"
        f"{_explainer_item('Orientation evidence', 'No edge selected', wide=True)}"
        "</div>"
        "</div>"
        "</div>"
    )


def render_interaction_script() -> str:
    return """<script>
(function () {
  function setText(root, selector, value) {
    var node = root.querySelector(selector);
    if (node) {
      node.textContent = value || "";
    }
  }

  function fillExplanation(root, edgeNode) {
    setText(root, ".edge-explainer-title", "Selected edge explanation");
    setText(root, ".edge-modal-title", "Selected edge explanation");
    setText(root, ".explain-edge", edgeNode.dataset.edge);
    setText(root, ".explain-status", edgeNode.dataset.status);
    setText(root, ".explain-endpoint-meaning", edgeNode.dataset.endpointMeaning);
    setText(root, ".explain-reasoning", edgeNode.dataset.reasoning);
    setText(root, ".explain-skeleton-evidence", edgeNode.dataset.skeletonEvidence);
    setText(root, ".explain-orientation-evidence", edgeNode.dataset.orientationEvidence);
  }

  function openModal(edgeNode) {
    var modal = document.querySelector(".edge-modal-backdrop");
    if (!modal) {
      return;
    }
    fillExplanation(modal, edgeNode);
    modal.hidden = false;
    var closeButton = modal.querySelector(".edge-modal-close");
    if (closeButton) {
      closeButton.focus();
    }
  }

  function closeModal() {
    var modal = document.querySelector(".edge-modal-backdrop");
    if (modal) {
      modal.hidden = true;
    }
  }

  function explainEdge(edgeNode) {
    document.querySelectorAll(".graph-edge.is-selected").forEach(function (node) {
      node.classList.remove("is-selected");
    });
    edgeNode.classList.add("is-selected");

    var panel = document.querySelector(".edge-explainer");
    if (panel) {
      fillExplanation(panel, edgeNode);
    }
    openModal(edgeNode);
  }

  document.querySelectorAll(".graph-edge").forEach(function (edgeNode) {
    edgeNode.addEventListener("click", function () {
      explainEdge(edgeNode);
    });
    edgeNode.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        explainEdge(edgeNode);
      }
    });
  });

  document.querySelectorAll(".edge-modal-close").forEach(function (button) {
    button.addEventListener("click", closeModal);
  });
  document.querySelectorAll(".edge-modal-backdrop").forEach(function (modal) {
    modal.addEventListener("click", function (event) {
      if (event.target === modal) {
        closeModal();
      }
    });
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      closeModal();
    }
  });
}());
</script>"""


def _edge_metadata(explanation: EdgeExplanation) -> dict[str, str]:
    endpoints = (explanation.endpoint_x or "NONE", explanation.endpoint_y or "NONE")
    edge_text = explanation.edge_repr or f"{explanation.x} ... {explanation.y}"
    endpoint_meaning = _endpoint_meaning(explanation.x, explanation.y, endpoints)
    skeleton_evidence = _skeleton_evidence(explanation)
    orientation_evidence = _orientation_evidence(explanation)
    reasoning = (
        f"{edge_text} is present because the adjacency search did not record a "
        f"separating set that removed {explanation.x} and {explanation.y}. "
        f"{_orientation_summary(explanation)}"
    )
    if explanation.bootstrap_frequency is not None:
        reasoning += (
            f" Bootstrap stability kept this edge with frequency "
            f"{explanation.bootstrap_frequency:.3f}."
        )
    return {
        "edge": edge_text,
        "status": _endpoint_status_text(endpoints),
        "endpoint_meaning": endpoint_meaning,
        "reasoning": reasoning,
        "skeleton_evidence": skeleton_evidence,
        "orientation_evidence": orientation_evidence,
        "aria_label": f"{edge_text}. {endpoint_meaning}",
    }


def _endpoint_status_text(endpoints: tuple[str, str]) -> str:
    return f"{endpoints[0]} at left endpoint, {endpoints[1]} at right endpoint"


def _endpoint_meaning(x: str, y: str, endpoints: tuple[str, str]) -> str:
    relation = _relation_meaning(x, y, endpoints)
    return (
        relation
        + " "
        + _single_endpoint_meaning(x, y, endpoints[0])
        + " "
        + _single_endpoint_meaning(y, x, endpoints[1])
    )


def _relation_meaning(x: str, y: str, endpoints: tuple[str, str]) -> str:
    if endpoints == ("CIRCLE", "CIRCLE"):
        return (
            f"{x} o-o {y} means FCI kept an adjacency but did not identify either "
            "ancestral direction from the available conditional independence "
            "constraints."
        )
    if endpoints == ("CIRCLE", "ARROW"):
        return (
            f"{x} o-> {y} means the endpoint at {y} is an arrowhead. This rules "
            f"out {y} being an ancestor of {x}; the endpoint at {x} remains "
            "unidentified."
        )
    if endpoints == ("ARROW", "CIRCLE"):
        return (
            f"{x} <-o {y} means the endpoint at {x} is an arrowhead. This rules "
            f"out {x} being an ancestor of {y}; the endpoint at {y} remains "
            "unidentified."
        )
    if endpoints == ("TAIL", "ARROW"):
        return (
            f"{x} --> {y} means the learned PAG supports {x} as an ancestor or "
            f"cause candidate of {y}, and rules out {y} as an ancestor of {x}."
        )
    if endpoints == ("ARROW", "TAIL"):
        return (
            f"{x} <-- {y} means the learned PAG supports {y} as an ancestor or "
            f"cause candidate of {x}, and rules out {x} as an ancestor of {y}."
        )
    if endpoints == ("ARROW", "ARROW"):
        return (
            f"{x} <-> {y} means both endpoints are arrowheads. In a PAG this is "
            "compatible with latent confounding or another ancestral-graph "
            "constraint that rules out either observed variable being an ancestor "
            "of the other."
        )
    if endpoints == ("TAIL", "TAIL"):
        return (
            f"{x} --- {y} is an undirected tail-tail edge, usually associated "
            "with selection-bias style constraints or background knowledge."
        )
    return f"{x}-{y} has mixed endpoint marks; read each endpoint independently."


def _single_endpoint_meaning(node: str, other: str, endpoint: str) -> str:
    if endpoint == "ARROW":
        return (
            f"The arrowhead at {node} says {node} is not an ancestor of {other} "
            "in every graph represented by this PAG."
        )
    if endpoint == "TAIL":
        return (
            f"The tail at {node} points the edge out of {node}, supporting "
            f"{node} as an ancestor or cause candidate of {other}."
        )
    if endpoint == "CIRCLE":
        return (
            f"The circle at {node} is deliberately unresolved: the data and "
            "orientation rules did not justify replacing it with a tail or "
            "arrowhead."
        )
    return f"The endpoint at {node} is inactive."


def _skeleton_evidence(explanation: EdgeExplanation) -> str:
    if explanation.sepset is not None:
        return (
            "A separating set is recorded for this pair: "
            f"{explanation.sepset}. This usually explains a removed edge; "
            "because this edge is still present, inspect the configured pipeline "
            "and later refinement steps."
        )
    text = (
        "No separating set was recorded for this node pair, so the skeleton "
        "search did not accept a conditional independence query that would "
        "remove this adjacency under the configured search limits."
    )
    if explanation.ci_tests:
        p_values = [float(event["p_value"]) for event in explanation.ci_tests]
        max_p = max(p_values)
        text += (
            f" The report recorded {len(explanation.ci_tests)} direct CI queries "
            f"for this pair; the largest p-value was {max_p:.4g}."
        )
    else:
        text += " No pair-specific CI trace event is available in this result."
    return text


def _orientation_evidence(explanation: EdgeExplanation) -> str:
    if not explanation.orientation_events:
        return (
            "No endpoint-changing orientation event was recorded for this edge. "
            "Any circle endpoints remain unresolved because no rule justified a "
            "tail or arrowhead; if the edge has definite endpoints, they may have "
            "come from initialization, background knowledge, or an external result "
            "without rule-level trace."
        )
    pieces = []
    for event in explanation.orientation_events[:5]:
        rule = str(event.get("rule", "unknown"))
        reason = str(event.get("reason", "")).strip()
        pieces.append(_single_rule_reasoning(rule, reason))
    if len(explanation.orientation_events) > 5:
        pieces.append(
            f"{len(explanation.orientation_events) - 5} additional orientation "
            "events are omitted from this short explanation."
        )
    return " ".join(pieces)


def _orientation_summary(explanation: EdgeExplanation) -> str:
    if explanation.orientation_events:
        return (
            "The endpoint directions are explained by recorded orientation-rule "
            "events shown in the orientation evidence field."
        )
    return (
        "No recorded orientation rule changed this edge, so unresolved circles "
        "are intentional and definite endpoints should be interpreted from the "
        "final PAG marks shown on the edge."
    )


def _single_rule_reasoning(rule: str, reason: str) -> str:
    if rule.startswith("orient_unshielded_colliders"):
        text = (
            "The collider rule placed arrowheads into the middle node of an "
            "unshielded triple because the separating-set evidence supported a "
            "collider interpretation."
        )
    elif rule == "R1":
        text = (
            "Rule R1 oriented a tail to avoid creating a new unshielded collider "
            "that would contradict the separating sets."
        )
    elif rule == "R2":
        text = (
            "Rule R2 propagated ancestry along an existing directed or partially "
            "directed path, forcing the endpoint to agree with that path."
        )
    elif rule == "R3":
        text = (
            "Rule R3 used a triangle-style PAG pattern where two adjacent "
            "constraints force the remaining endpoint."
        )
    elif rule == "R4":
        text = (
            "Rule R4 used a discriminating path, where the path structure makes "
            "a triple's collider or non-collider status identifiable."
        )
    elif rule in {"R5", "R6", "R7"}:
        text = (
            f"Rule {rule} is a selection-bias orientation rule that propagates "
            "tail information through uncovered circle-path patterns."
        )
    elif rule in {"R8", "R9", "R10"}:
        text = (
            f"Rule {rule} prevents an orientation that would create an invalid "
            "ancestral cycle or conflict with an uncovered potentially directed "
            "path."
        )
    elif rule == "background_knowledge":
        text = "Background knowledge forced this endpoint orientation."
    else:
        text = f"{rule} changed an endpoint according to the PAG rules."
    if reason:
        text += f" Recorded trigger: {reason}."
    return text


def _explainer_item(label: str, value: str, wide: bool = False) -> str:
    class_name = "explain-" + label.lower().replace(" ", "-")
    wide_class = " wide" if wide else ""
    return (
        f"<div class='edge-explainer-item{wide_class}'>"
        f"<div class='edge-explainer-label'>{_esc(label)}</div>"
        f"<div class='edge-explainer-value {class_name}'>{_esc(value)}</div>"
        "</div>"
    )


def _circle_layout(
    nodes: list[str],
    width: int,
    height: int,
) -> dict[str, tuple[float, float]]:
    center_x = width / 2
    center_y = height / 2 + 12
    radius = min(width, height) * 0.38
    positions = {}
    for index, node in enumerate(nodes):
        angle = -math.pi / 2 + 2 * math.pi * index / max(len(nodes), 1)
        positions[node] = (
            center_x + radius * math.cos(angle),
            center_y + radius * math.sin(angle),
        )
    return positions


def _node_svg(node: str, x: float, y: float) -> str:
    lines = _label_lines(str(node))
    max_len = max(len(line) for line in lines)
    box_width = min(122, max(70, 8 * max_len + 20))
    box_height = 26 + 14 * len(lines)
    top = y - box_height / 2
    left = x - box_width / 2
    text_start = y - (len(lines) - 1) * 7 + 4
    text_lines = []
    for index, line in enumerate(lines):
        text_lines.append(
            f'<tspan x="{x:.1f}" y="{text_start + index * 14:.1f}">'
            f"{_esc(line)}</tspan>"
        )
    return (
        f"<g><title>{_esc(node)}</title>"
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{box_width:.1f}" '
        f'height="{box_height:.1f}" rx="7" fill="#f8fafc" '
        'stroke="#344054" stroke-width="1.4"/>'
        f'<text x="{x:.1f}" y="{text_start:.1f}" text-anchor="middle" '
        'font-size="11" font-weight="700" fill="#101827">'
        f"{''.join(text_lines)}</text></g>"
    )


def _label_lines(label: str) -> list[str]:
    words = label.replace("_", " ").replace("-", " ").split()
    if not words:
        return [label]
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= 13:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) == 1:
            break
    if current and len(lines) < 2:
        lines.append(current)
    return [_ellipsis(line, 13) for line in lines[:2]]


def _ellipsis(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _trimmed_line(
    start: tuple[float, float],
    end: tuple[float, float],
    radius: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = math.hypot(dx, dy)
    if distance == 0:
        return start, end
    ux = dx / distance
    uy = dy / distance
    return (
        (start[0] + ux * radius, start[1] + uy * radius),
        (end[0] - ux * radius, end[1] - uy * radius),
    )


def _endpoint_svg(
    endpoint: str,
    point: tuple[float, float],
    toward: tuple[float, float],
    color: str,
) -> str:
    dx = point[0] - toward[0]
    dy = point[1] - toward[1]
    distance = math.hypot(dx, dy)
    if distance == 0:
        return ""
    ux = dx / distance
    uy = dy / distance
    px = -uy
    py = ux
    if endpoint == "CIRCLE":
        return (
            f'<circle class="edge-endpoint" cx="{point[0]:.1f}" '
            f'cy="{point[1]:.1f}" r="5" fill="#ffffff" stroke="{color}" '
            'stroke-width="1.8"/>'
        )
    if endpoint == "ARROW":
        base_x = point[0] - ux * 13
        base_y = point[1] - uy * 13
        p1 = (base_x + px * 5, base_y + py * 5)
        p2 = (base_x - px * 5, base_y - py * 5)
        return (
            f'<polygon class="edge-endpoint" points="{point[0]:.1f},{point[1]:.1f} '
            f'{p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" '
            f'fill="{color}"/>'
        )
    if endpoint == "TAIL":
        p1 = (point[0] + px * 7, point[1] + py * 7)
        p2 = (point[0] - px * 7, point[1] - py * 7)
        return (
            f'<line class="edge-endpoint" x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" '
            f'x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" stroke="{color}" '
            'stroke-width="2.4"/>'
        )
    return ""


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)
