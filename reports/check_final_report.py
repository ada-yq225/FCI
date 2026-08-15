"""Check a final-report LaTeX source against the ESE IRP limits.

Rules encoded from https://ese-msc.github.io/irp/deliverables/written-reports/:

- at most 5,000 countable words (a 5% tolerance exists for tool variation);
- at most 10 figures and tables combined;
- abstract of approximately 200 words or fewer;
- PDF no larger than 8 MB, named ``deliverables/<username>-final-report.pdf``.

Excluded from the word count: the title page, acknowledgements, table of
contents, section titles, text inside figures and tables (captions, legends,
axis labels), footnotes, equations, the references section, and appendices.
In-text citations count as words.

This is an approximation of the course's counting rules; treat a result close
to the limit as a warning, not a guarantee.

Usage::

    python reports/check_final_report.py path/to/report.tex [path/to/report.pdf]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

WORD_LIMIT = 5_000
WORD_TOLERANCE = 1.05
FLOAT_LIMIT = 10
ABSTRACT_LIMIT = 200
PDF_LIMIT_BYTES = 8 * 4_194_304 // 4  # 8 MB, matching .github/test_irp.py

EXCLUDED_ENVIRONMENTS = (
    "figure",
    "figure*",
    "table",
    "table*",
    "longtable",
    "tabular",
    "tabularx",
    "thebibliography",
    "equation",
    "equation*",
    "align",
    "align*",
    "titlepage",
    "tableofcontents",
)


def _strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", " ", tex)


def _drop_environment(tex: str, name: str) -> str:
    escaped = re.escape(name)
    pattern = rf"\\begin\{{{escaped}\}}.*?\\end\{{{escaped}\}}"
    return re.sub(pattern, " ", tex, flags=re.DOTALL)


def _extract_abstract(tex: str) -> str:
    match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        tex,
        flags=re.DOTALL,
    )
    return match.group(1) if match else ""


def _drop_preamble_and_backmatter(tex: str) -> str:
    body = tex.split(r"\begin{document}", 1)[-1]
    for marker in (r"\begin{thebibliography}", r"\bibliography{", r"\appendix"):
        body = body.split(marker, 1)[0]
    return body


def _to_plain_words(tex: str) -> list[str]:
    text = tex
    for name in EXCLUDED_ENVIRONMENTS:
        text = _drop_environment(text, name)
    # Section titles are excluded; the running text is not.
    text = re.sub(r"\\(?:sub)*section\*?\{[^{}]*\}", " ", text)
    text = re.sub(r"\\paragraph\*?\{[^{}]*\}", " ", text)
    text = re.sub(r"\\footnote\{[^{}]*\}", " ", text)
    text = re.sub(r"\\maketitle|\\tableofcontents|\\listoffigures", " ", text)
    # A citation counts as one word per bracketed reference.
    text = re.sub(r"\\[a-zA-Z]*cite[a-zA-Z]*\s*(\[[^\]]*\])?\{[^{}]*\}", "CITE", text)
    text = re.sub(r"\\(?:c|C)ref\{[^{}]*\}", "REF", text)
    # Inline math counts approximately as one word.
    text = re.sub(r"\$[^$]*\$", " MATH ", text)
    text = re.sub(r"\\\((?:.|\n)*?\\\)", " MATH ", text)
    text = re.sub(r"\\\[(?:.|\n)*?\\\]", " ", text)
    # Remaining commands: keep their braced text, drop the command itself.
    for _ in range(4):
        text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?\{([^{}]*)\}", r"\2", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = re.sub(r"[{}~&]", " ", text)
    return [word for word in text.split() if any(ch.isalnum() for ch in word)]


def check(tex_path: Path, pdf_path: Path | None) -> int:
    tex = _strip_comments(tex_path.read_text(encoding="utf-8"))

    abstract_words = len(_to_plain_words(_extract_abstract(tex)))
    body = _drop_preamble_and_backmatter(tex)
    body_words = len(_to_plain_words(body))

    figures = len(re.findall(r"\\begin\{figure\*?\}", tex))
    tables = len(re.findall(r"\\begin\{(?:table\*?|longtable)\}", tex))

    failures = 0

    def report(label: str, value: str, ok: bool, warn: bool = False) -> None:
        nonlocal failures
        status = "PASS" if ok else ("WARN" if warn else "FAIL")
        if not ok and not warn:
            failures += 1
        print(f"  [{status}] {label}: {value}")

    print(f"Checking {tex_path}")
    hard_limit = int(WORD_LIMIT * WORD_TOLERANCE)
    report(
        "countable words (approximate)",
        f"{body_words} (limit {WORD_LIMIT}, tolerance {hard_limit})",
        body_words <= WORD_LIMIT,
        warn=WORD_LIMIT < body_words <= hard_limit,
    )
    report(
        "figures + tables",
        f"{figures} + {tables} = {figures + tables} (limit {FLOAT_LIMIT})",
        figures + tables <= FLOAT_LIMIT,
    )
    report(
        "abstract words",
        f"{abstract_words} (limit ~{ABSTRACT_LIMIT})",
        abstract_words <= ABSTRACT_LIMIT,
    )

    if pdf_path is not None:
        size = pdf_path.stat().st_size
        report(
            "PDF size",
            f"{size / 1_048_576:.1f} MB (limit 8 MB)",
            size <= PDF_LIMIT_BYTES,
        )
        expected = "yq225-final-report.pdf"
        report(
            "PDF filename",
            pdf_path.name,
            pdf_path.name == expected,
        )

    print(
        "  Reminder: title page must include university/department/course, "
        "title, full name,\n  the IRP repository URL, all supervisors, and "
        "the submission month and year."
    )
    return 1 if failures else 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    tex_path = Path(sys.argv[1])
    pdf_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    return check(tex_path, pdf_path)


if __name__ == "__main__":
    raise SystemExit(main())
