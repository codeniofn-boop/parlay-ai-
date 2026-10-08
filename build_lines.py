#!/usr/bin/env python3
"""
build_lines.py
==============

Expands a short, one-row-per-game ``week_inputs.csv`` into the full
``lines.csv`` that ``parlay_finder.py`` reads (six rows per game: both sides
of the spread, total and moneyline, each with a model probability).

Maintaining 15 rows a week is realistic; maintaining 90 is not.

``week_inputs.csv`` columns (header names are case-insensitive)::

    week, date, away, home,
    home_spread,            e.g. -8.5 when the home team is favoured by 8.5, +1.5 when the away team is
    home_spread_price,      American price on the home side of the spread (blank -> -110)
    away_spread_price,      American price on the away side (blank -> -110)
    total,                  the over/under number
    over_price, under_price,                       (blank -> -110)
    away_ml, home_ml,       American moneylines
    model_home_margin,      the model's projected final margin, home minus away (e.g. ESPN FPI "Cowboys by 11.4" -> +11.4 if Dallas is home)
    model_home_win_prob,    the model's probability the home team wins, 0-1 or 0-100
    model_total             optional: the model's projected total points; blank -> totals get no edge
    home_team_total, away_team_total            optional team total lines (e.g. 24.5)
    home_tt_over_price, home_tt_under_price,    optional prices for them (blank -> -110)
    away_tt_over_price, away_tt_under_price
    source                  optional free text recorded in lines.csv notes

Model probabilities per side are derived as::

    P(home covers) = 1 - Phi((spread_point - model_home_margin) / 13.5)   spread_point = -home_spread
    P(home wins)   = model_home_win_prob (or Phi(model_home_margin / 13.5) when blank)
    P(over)        = 1 - Phi((total - model_total) / 10)                 only when model_total is given
    P(team over)   = 1 - Phi((team_total - model_team_points) / 8.4)     model_team_points = (model_total +/- margin) / 2,
                                                                         only when BOTH model_total and margin are given

13.5 and 10 are the long-run standard deviations of NFL final margins and
totals. When a model number is missing, the side's ``model_prob`` is left
blank so the finder falls back to the de-vigged market price (no edge),
which is the honest default.

Usage::

    python3 build_lines.py                      # week_inputs.csv -> lines.csv
    python3 build_lines.py --inputs my.csv --out lines.csv --week 5
    python3 build_lines.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["BuildLinesError", "normal_cdf", "expand_game", "build_lines_rows", "build_lines_csv",
           "INPUTS_FILENAME", "LINES_FILENAME", "LINES_COLUMNS"]

logger = logging.getLogger(__name__)

INPUTS_FILENAME = "week_inputs.csv"
LINES_FILENAME = "lines.csv"
LINES_COLUMNS = ("week", "away", "home", "market", "selection", "american_odds", "model_prob", "notes")
MARGIN_SD = 13.5
TOTAL_SD = 10.0
TEAM_SD = math.sqrt((TOTAL_SD ** 2 + MARGIN_SD ** 2) / 4.0)
DEFAULT_PRICE = -110


class BuildLinesError(ValueError):
    """Raised for a malformed week_inputs.csv row."""


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _num(value: Any, name: str, default: Optional[float] = None, line: int = 0) -> Optional[float]:
    text = "" if value is None else str(value).strip().replace("+", "")
    if text == "":
        return default
    try:
        return float(text)
    except ValueError as exc:
        raise BuildLinesError(f"line {line}: {name} must be a number, got '{value}'") from exc


def _price(value: Any, name: str, line: int) -> int:
    p = _num(value, name, float(DEFAULT_PRICE), line)
    assert p is not None
    p_int = int(round(p))
    if -100 < p_int < 100:
        raise BuildLinesError(f"line {line}: {name} must be <= -100 or >= +100, got {p_int}")
    return p_int


def _prob(value: Any, name: str, line: int) -> Optional[float]:
    p = _num(value, name, None, line)
    if p is None:
        return None
    if p > 1.0:
        p /= 100.0
    if not 0.0 < p < 1.0:
        raise BuildLinesError(f"line {line}: {name} must be between 0 and 1 (or 0-100), got {value}")
    return p


def _fmt_line(x: float) -> str:
    return f"{x:+g}"


def expand_game(row: Dict[str, Any], line: int = 0) -> List[Dict[str, Any]]:
    """Turn one game row into six lines.csv rows (model_prob may be blank)."""
    r = {(k or "").strip().lower(): (v if v is not None else "") for k, v in row.items()}
    try:
        week = int(float(str(r["week"]).strip()))
    except (KeyError, ValueError) as exc:
        raise BuildLinesError(f"line {line}: week is required and must be a number") from exc
    away, home = str(r.get("away", "")).strip(), str(r.get("home", "")).strip()
    if not away or not home:
        raise BuildLinesError(f"line {line}: away and home are required")

    home_spread = _num(r.get("home_spread"), "home_spread", None, line)
    total = _num(r.get("total"), "total", None, line)
    away_ml = _num(r.get("away_ml"), "away_ml", None, line)
    home_ml = _num(r.get("home_ml"), "home_ml", None, line)
    margin = _num(r.get("model_home_margin"), "model_home_margin", None, line)
    home_win = _prob(r.get("model_home_win_prob"), "model_home_win_prob", line)
    model_total = _num(r.get("model_total"), "model_total", None, line)
    source = str(r.get("source", "")).strip()
    notes = source or ""

    if home_win is None and margin is not None:
        home_win = normal_cdf(margin / MARGIN_SD)

    out: List[Dict[str, Any]] = []

    def add(market: str, selection: str, odds: int, prob: Optional[float]) -> None:
        out.append({"week": week, "away": away, "home": home, "market": market, "selection": selection,
                    "american_odds": odds, "model_prob": "" if prob is None else round(prob, 4), "notes": notes})

    if home_spread is not None:
        spread_point = -home_spread
        p_home_cover = None if margin is None else 1.0 - normal_cdf((spread_point - margin) / MARGIN_SD)
        add("spread", f"{home} {_fmt_line(home_spread)}", _price(r.get("home_spread_price"), "home_spread_price", line), p_home_cover)
        add("spread", f"{away} {_fmt_line(-home_spread)}", _price(r.get("away_spread_price"), "away_spread_price", line),
            None if p_home_cover is None else 1.0 - p_home_cover)
    if total is not None:
        p_over = None if model_total is None else 1.0 - normal_cdf((total - model_total) / TOTAL_SD)
        add("total", f"Over {total:g}", _price(r.get("over_price"), "over_price", line), p_over)
        add("total", f"Under {total:g}", _price(r.get("under_price"), "under_price", line), None if p_over is None else 1.0 - p_over)
    if away_ml is not None and home_ml is not None:
        add("moneyline", f"{home} ML", _price(home_ml, "home_ml", line), home_win)
        add("moneyline", f"{away} ML", _price(away_ml, "away_ml", line), None if home_win is None else 1.0 - home_win)
    for team, key, sign in ((home, "home", +1), (away, "away", -1)):
        tt_line = _num(r.get(f"{key}_team_total"), f"{key}_team_total", None, line)
        if tt_line is None:
            continue
        p_over = None
        if model_total is not None and margin is not None:
            p_over = 1.0 - normal_cdf((tt_line - (model_total + sign * margin) / 2.0) / TEAM_SD)
        add("team_total", f"{team} Over {tt_line:g}", _price(r.get(f"{key}_tt_over_price"), f"{key}_tt_over_price", line), p_over)
        add("team_total", f"{team} Under {tt_line:g}", _price(r.get(f"{key}_tt_under_price"), f"{key}_tt_under_price", line),
            None if p_over is None else 1.0 - p_over)
    if not out:
        raise BuildLinesError(f"line {line}: {away} @ {home} has no spread, total or moneyline")
    return out


def build_lines_rows(inputs_path: str, week: Optional[int] = None) -> List[Dict[str, Any]]:
    if not os.path.isfile(inputs_path):
        raise BuildLinesError(f"Input file not found: {inputs_path}")
    rows: List[Dict[str, Any]] = []
    with open(inputs_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise BuildLinesError(f"{inputs_path} is empty")
        needed = {"week", "away", "home"}
        if needed - {f.strip().lower() for f in reader.fieldnames}:
            raise BuildLinesError(f"{inputs_path} needs at least the columns: week, away, home")
        for i, raw in enumerate(reader, 2):
            if not any((v or "").strip() for v in raw.values()):
                continue  # blank line
            expanded = expand_game(raw, i)
            if week is None or expanded[0]["week"] == week:
                rows.extend(expanded)
    if not rows:
        raise BuildLinesError(f"No games in {inputs_path}" + (f" for week {week}" if week else ""))
    return rows


def build_lines_csv(inputs_path: str = INPUTS_FILENAME, out_path: str = LINES_FILENAME, week: Optional[int] = None) -> str:
    rows = build_lines_rows(inputs_path, week)
    abs_out = os.path.abspath(out_path)
    with open(abs_out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(LINES_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d line rows for %d game(s) to %s", len(rows), len(rows) // 6 or 1, abs_out)
    return abs_out


def _selftest() -> int:
    failures = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal failures
        print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
        if not cond:
            failures += 1

    print("build_lines self-test")
    row = {"week": "5", "away": "Tampa Bay Buccaneers", "home": "Dallas Cowboys", "home_spread": "-8.5",
           "total": "47.5", "away_ml": "+360", "home_ml": "-470", "model_home_margin": "11.4",
           "model_home_win_prob": "87", "source": "test"}
    rows = expand_game(row, 2)
    check(len(rows) == 6, "six rows per game")
    sel = {r["selection"]: r for r in rows}
    check("Dallas Cowboys -8.5" in sel and "Tampa Bay Buccaneers +8.5" in sel, "spread selections formatted")
    check(sel["Dallas Cowboys -8.5"]["american_odds"] == -110 and sel["Over 47.5"]["american_odds"] == -110, "blank prices default to -110")
    check(sel["Dallas Cowboys ML"]["american_odds"] == -470 and sel["Tampa Bay Buccaneers ML"]["american_odds"] == 360, "moneylines carried")
    p_cover = sel["Dallas Cowboys -8.5"]["model_prob"]
    expected = 1 - normal_cdf((8.5 - 11.4) / 13.5)
    check(abs(p_cover - round(expected, 4)) < 1e-9 and 0.58 < p_cover < 0.59, f"cover prob from margin ({p_cover})")
    check(abs(sel["Tampa Bay Buccaneers +8.5"]["model_prob"] + p_cover - 1) < 1e-3, "spread sides complementary")
    check(sel["Dallas Cowboys ML"]["model_prob"] == 0.87 and sel["Tampa Bay Buccaneers ML"]["model_prob"] == 0.13, "win prob accepts 0-100")
    check(sel["Over 47.5"]["model_prob"] == "" and sel["Under 47.5"]["model_prob"] == "", "totals blank without model_total")
    rows2 = expand_game({**row, "model_total": "52", "model_home_win_prob": "", "home_team_total": "29.5", "away_team_total": "18.5"}, 3)
    sel2 = {r["selection"]: r for r in rows2}
    check(sel2["Over 47.5"]["model_prob"] > 0.6, "model_total gives an over probability")
    check(len(rows2) == 10 and sel2["Dallas Cowboys Over 29.5"]["american_odds"] == -110
          and 0.5 < sel2["Dallas Cowboys Over 29.5"]["model_prob"] < 0.7
          and abs(sel2["Tampa Bay Buccaneers Under 18.5"]["model_prob"] + sel2["Tampa Bay Buccaneers Over 18.5"]["model_prob"] - 1) < 1e-3,
          "team totals expand with model probabilities from total and margin")
    check(abs(sel2["Dallas Cowboys ML"]["model_prob"] - round(normal_cdf(11.4 / 13.5), 4)) < 1e-9, "win prob derived from margin when blank")
    try:
        expand_game({"week": "5", "away": "A", "home": "B"}, 4)
        check(False, "game with no markets rejected")
    except BuildLinesError:
        check(True, "game with no markets rejected")
    try:
        expand_game({**row, "home_ml": "-50"}, 5)
        check(False, "bad price rejected")
    except BuildLinesError:
        check(True, "bad price rejected")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "week_inputs.csv")
        with open(inp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
            w.writerow({**row, "week": "6"})
        out = build_lines_csv(inp, os.path.join(tmp, "lines.csv"))
        with open(out, newline="", encoding="utf-8") as fh:
            got = list(csv.DictReader(fh))
        check(len(got) == 12 and set(got[0].keys()) == set(LINES_COLUMNS), "CSV round trip with both weeks")
        check(len(build_lines_rows(inp, week=6)) == 6, "week filter")
        # The finder must accept the output
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import parlay_finder
        sides = parlay_finder.load_lines_csv(out, week=5)
        check(len(sides) == 6 and any(abs(s.model_prob - 0.87) < 1e-9 for s in sides), "parlay_finder loads the built file")

    print(f"\n{'ALL TESTS PASSED' if failures == 0 else f'{failures} TEST(S) FAILED'}")
    return 0 if failures == 0 else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="build_lines", description="Expand week_inputs.csv into lines.csv for parlay_finder.")
    p.add_argument("--inputs", default=INPUTS_FILENAME)
    p.add_argument("--out", default=LINES_FILENAME)
    p.add_argument("--week", type=int, default=None, help="Only this week (default: all weeks in the file)")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.selftest:
        return _selftest()
    try:
        out = build_lines_csv(args.inputs, args.out, args.week)
    except BuildLinesError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
