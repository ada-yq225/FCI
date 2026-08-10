"""Vector-only contracts for publication figures."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "reports" / "figures"
RESEARCH_FIGURES = (
    "fci_fci_plus_workflow.pdf",
    "source_implementation_map.pdf",
    "software_benchmark_comparison.pdf",
    "software_feature_comparison.pdf",
)


def test_research_figure_pdfs_do_not_embed_raster_images() -> None:
    image_xobject = re.compile(rb"/Subtype\s*/Image\b")

    for name in RESEARCH_FIGURES:
        path = FIGURE_DIR / name
        assert not image_xobject.search(
            path.read_bytes()
        ), f"{name} contains a raster /Image XObject"
