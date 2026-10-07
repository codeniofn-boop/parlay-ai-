#!/usr/bin/env python3
"""
run_weekly_report.py
====================

Zero-argument launcher for the weekly parlay report. Built for running from an
editor's Run button (IDLE, VS Code, PyCharm) or by double-clicking, so no
Terminal commands are needed.

How it works
------------
1. Reads ``pipeline_config.json`` from the same folder (bankroll, staking
   mode, week, ...). Missing keys fall back to safe defaults, and a missing
   file simply means "use all defaults".
2. Calls :func:`weekly_reporter.run_weekly_report`, which collects tickets
   from ``parlay_finder.py`` (or the JSON / demo fallback), stakes them via
   ``staking_engine.py`` and writes ``weekly_parlay_report.txt``.
3. Prints the report and, when ``open_when_done`` is true, opens the text
   file in the system's default viewer (TextEdit on macOS).

Every failure is caught and explained in plain language instead of a
traceback, so the launcher is safe to hand to a non-technical operator.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict

# Make sibling modules importable no matter which folder the editor uses as
# its working directory, and run relative paths from the script's folder.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.chdir(HERE)

CONFIG_FILENAME = "pipeline_config.json"

DEFAULTS: Dict[str, Any] = {
    "week": None,
    "bankroll": 1000.0,
    "mode": "kelly",
    "kelly_multiplier": 0.25,
    "flat_pct": 0.01,
    "flat_dollars": None,
    "max_stake_pct": 0.05,
    "min_edge": 0.0,
    "portfolio_cap": 0.15,
    "source": "finder",
    "input": None,
    "top_n": 5,
    "output": "weekly_parlay_report.txt",
    "json_out": "weekly_parlay_report.json",
    "open_when_done": True,
}


def load_config(path: str = CONFIG_FILENAME) -> Dict[str, Any]:
    """Merge ``pipeline_config.json`` over the defaults; tolerate a missing file."""
    config = dict(DEFAULTS)
    if not os.path.isfile(path):
        print(f"(no {path} found; using built-in defaults)")
        return config
    try:
        with open(path, "r", encoding="utf-8") as fh:
            user = json.load(fh)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"{path} is not valid JSON ({exc}). Fix the typo or delete the file to use defaults."
        )
    if not isinstance(user, dict):
        raise SystemExit(f"{path} must contain a JSON object of settings.")
    for key, value in user.items():
        if key.startswith("_"):
            continue  # comments such as "_comment"
        if key not in DEFAULTS:
            print(f"warning: ignoring unknown setting '{key}' in {path}")
            continue
        config[key] = value
    return config


def open_file(path: str) -> None:
    """Open ``path`` with the OS default application; never raise."""
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.run(["open", path], check=False)
        elif system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", path], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:  # pragma: no cover - best effort only
        print(f"(could not auto-open the report: {exc})")


def main() -> int:
    try:
        from staking_engine import StakingConfig, StakingInputError
        from weekly_reporter import ReporterError, estimate_nfl_week, run_weekly_report
    except ImportError as exc:
        print("ERROR: staking_engine.py and weekly_reporter.py must sit in the same folder as this script.")
        print(f"Details: {exc}")
        return 1

    try:
        cfg = load_config()
    except SystemExit as exc:
        print(f"ERROR: {exc}")
        return 1

    today = _dt.date.today()
    week = cfg["week"]
    if week is None:
        week = estimate_nfl_week(today)
        print(f"Week not set in {CONFIG_FILENAME}; using upcoming week {week} based on today's date.")

    try:
        staking = StakingConfig(
            mode=cfg["mode"],
            flat_pct=float(cfg["flat_pct"]),
            flat_dollars=None if cfg["flat_dollars"] in (None, "") else float(cfg["flat_dollars"]),
            kelly_multiplier=float(cfg["kelly_multiplier"]),
            max_stake_pct=float(cfg["max_stake_pct"]),
            min_edge=float(cfg["min_edge"]),
        )
        portfolio_cap = cfg["portfolio_cap"]
        portfolio_cap = float(portfolio_cap) if portfolio_cap not in (None, "") and float(portfolio_cap) > 0 else None

        print(f"Generating report for NFL Week {week} with a ${float(cfg['bankroll']):,.2f} bankroll "
              f"({staking.mode.label})...")
        report = run_weekly_report(
            week=int(week),
            bankroll=float(cfg["bankroll"]),
            staking=staking,
            source=str(cfg["source"]),
            input_path=cfg["input"] or None,
            output_path=str(cfg["output"]),
            json_path=cfg["json_out"] or None,
            top_n=int(cfg["top_n"]),
            portfolio_cap_pct=portfolio_cap,
            report_date=today,
        )
    except (ReporterError, StakingInputError) as exc:
        print(f"ERROR: {exc}")
        return 1
    except (TypeError, ValueError) as exc:
        print(f"ERROR: a value in {CONFIG_FILENAME} has the wrong type: {exc}")
        return 1

    print()
    print(report.text)
    out_path = os.path.abspath(str(cfg["output"]))
    print(f"Saved: {out_path}")
    if cfg["json_out"]:
        print(f"Saved: {os.path.abspath(str(cfg['json_out']))}")
    if cfg["open_when_done"]:
        open_file(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
