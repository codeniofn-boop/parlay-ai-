#!/usr/bin/env python3
"""
publish_report.py
=================

Turns the weekly report JSON into a self-contained web page.

``report_template.html`` holds the page (styles, markup, and a browser port of
the staking maths) with one placeholder, ``/*__REPORT_DATA__*/{}``, where the
report JSON is injected. The result is static: no server, no build step, and
it works from a double-click, from GitHub Pages (``docs/index.html``), or as a
Claude artifact.

Two output shapes:

* **document** (default) – a complete HTML document for GitHub Pages or local
  viewing: ``python3 publish_report.py`` -> ``docs/index.html``.
* **fragment** – the same page without the ``<html>/<head>/<body>`` skeleton,
  for hosts that wrap the content themselves: ``--fragment``.

``run_weekly_report.py`` calls :func:`build_html` after every weekly run, so
the page is always in step with ``weekly_parlay_report.txt``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Optional, Sequence

__all__ = ["PublishError", "build_html", "publish", "TEMPLATE_FILENAME", "DEFAULT_HTML_OUT"]

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_FILENAME = "report_template.html"
DEFAULT_JSON_IN = "weekly_parlay_report.json"
DEFAULT_HTML_OUT = os.path.join("docs", "index.html")
PLACEHOLDER = "/*__REPORT_DATA__*/{}"
HEAD_MARKER = "<!--HEAD-END-->"


class PublishError(RuntimeError):
    """Template or report problems that make the page impossible to build."""


def _load_template(path: Optional[str] = None) -> str:
    candidates = [path] if path else [TEMPLATE_FILENAME, os.path.join(HERE, TEMPLATE_FILENAME)]
    for c in candidates:
        if c and os.path.isfile(c):
            with open(c, "r", encoding="utf-8") as fh:
                text = fh.read()
            if PLACEHOLDER not in text:
                raise PublishError(f"{c} has no {PLACEHOLDER} placeholder")
            if HEAD_MARKER not in text:
                raise PublishError(f"{c} has no {HEAD_MARKER} marker")
            return text
    raise PublishError(f"Template not found: {path or TEMPLATE_FILENAME}")


def _safe_json(data: Dict[str, Any]) -> str:
    """JSON that is safe inside a <script> block (no '</script>' breakouts)."""
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return text.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def build_html(report: Dict[str, Any], fragment: bool = False, template_path: Optional[str] = None) -> str:
    """Render the page from a report dict (``WeeklyReport.to_dict()`` shape)."""
    if not isinstance(report, dict) or "config" not in report or "sections" not in report:
        raise PublishError("report must be a dict with 'config' and 'sections' (weekly_parlay_report.json)")
    template = _load_template(template_path)
    filled = template.replace(PLACEHOLDER, "/*__REPORT_DATA__*/" + _safe_json(report), 1)
    head, body = filled.split(HEAD_MARKER, 1)
    if fragment:
        return head.rstrip() + "\n" + body.lstrip()
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
        + head.strip() + "\n</head>\n<body>\n" + body.strip() + "\n</body>\n</html>\n"
    )


def publish(json_in: str = DEFAULT_JSON_IN, html_out: str = DEFAULT_HTML_OUT, fragment: bool = False,
            template_path: Optional[str] = None) -> str:
    """Read the report JSON, write the page; returns the absolute output path."""
    if not os.path.isfile(json_in):
        raise PublishError(f"Report JSON not found: {json_in} (run run_weekly_report.py first)")
    try:
        with open(json_in, "r", encoding="utf-8") as fh:
            report = json.load(fh)
    except json.JSONDecodeError as exc:
        raise PublishError(f"{json_in} is not valid JSON: {exc}") from exc
    html = build_html(report, fragment=fragment, template_path=template_path)
    abs_out = os.path.abspath(html_out)
    try:
        os.makedirs(os.path.dirname(abs_out) or ".", exist_ok=True)
        with open(abs_out, "w", encoding="utf-8") as fh:
            fh.write(html)
    except OSError as exc:
        raise PublishError(f"Could not write {abs_out}: {exc}") from exc
    logger.info("Wrote %s (%d bytes)", abs_out, len(html.encode("utf-8")))
    return abs_out


def _selftest() -> int:
    failures = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal failures
        print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
        if not cond:
            failures += 1

    print("publish_report self-test")
    report = {"generated_at": "2026-10-07T12:00:00", "config": {"week": 5, "season": 2026, "report_date": "2026-10-07",
              "bankroll": 1000, "staking": {"mode": "kelly", "kelly_multiplier": 0.25, "max_stake_pct": 0.05},
              "top_n_per_group": 5, "leg_groups": [2, 3], "portfolio_cap_pct": 0.15, "source_label": "test </script> label"},
              "sections": {"2-LEG": []}, "passed": [], "rejected": []}
    doc = build_html(report)
    check(doc.startswith("<!doctype html>") and "</html>" in doc, "document output has a skeleton")
    check("<\\/script>" in doc and "</script> label" not in doc, "script breakout escaped")
    check(doc.count("<title>") == 1 and "Parlay Report Card" in doc, "title present once")
    frag = build_html(report, fragment=True)
    check(not frag.startswith("<!doctype") and "<html" not in frag and frag.lstrip().startswith("<title>"), "fragment has no skeleton and starts with the title")
    check(HEAD_MARKER not in doc and HEAD_MARKER not in frag and PLACEHOLDER not in doc, "markers consumed")
    try:
        build_html({"nope": 1})
        check(False, "bad report rejected")
    except PublishError:
        check(True, "bad report rejected")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        jp = os.path.join(tmp, "r.json")
        with open(jp, "w", encoding="utf-8") as fh:
            json.dump(report, fh)
        out = publish(jp, os.path.join(tmp, "site", "index.html"))
        check(os.path.getsize(out) > 5000, "publish writes a page into a new folder")
        try:
            publish(os.path.join(tmp, "missing.json"), os.path.join(tmp, "x.html"))
            check(False, "missing JSON raises")
        except PublishError:
            check(True, "missing JSON raises PublishError")
    print(f"\n{'ALL TESTS PASSED' if failures == 0 else f'{failures} TEST(S) FAILED'}")
    return 0 if failures == 0 else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="publish_report", description="Build the weekly parlay web page from the report JSON.")
    p.add_argument("--json", dest="json_in", default=DEFAULT_JSON_IN, help="Report JSON written by the weekly run")
    p.add_argument("--out", dest="html_out", default=DEFAULT_HTML_OUT, help="Output HTML path")
    p.add_argument("--template", default=None, help="Template path (default: report_template.html)")
    p.add_argument("--fragment", action="store_true", help="Emit the page without the html/head/body skeleton")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.selftest:
        return _selftest()
    try:
        out = publish(args.json_in, args.html_out, fragment=args.fragment, template_path=args.template)
    except PublishError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
