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

Player props come from ``props_inputs.csv`` beside ``week_inputs.csv``, one
row per prop copied from your sportsbook (header names are case-insensitive)::

    week, away, home,        the game, spelled exactly as in week_inputs.csv
    player,                  e.g. Dak Prescott
    market,                  any label or alias from markets.json: Passing Yards, pass yds,
                             Receptions, Anytime TD, Rush + Rec Yards, ...
    line,                    the number for over/under props (blank for Anytime TD / First TD)
    over_price, under_price, American prices (blank -> -110)
    yes_price, no_price,     for yes/no props such as Anytime TD (no_price optional)
    position, notes          optional

Every over/under prop becomes an Over row and an Under row; a yes/no prop
becomes a Yes row and, when priced, a No row. ``model_prob`` for props is
filled by ``prop_model.py`` when it is present; otherwise it stays blank
and the prop carries no edge.

Usage::

    python3 build_lines.py                      # week_inputs.csv (+ props_inputs.csv) -> lines.csv
    python3 build_lines.py --inputs my.csv --props my_props.csv --out lines.csv --week 5
    python3 build_lines.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import market_registry
from market_registry import RegistryError, format_prop_selection

__all__ = ["BuildLinesError", "normal_cdf", "expand_game", "expand_prop", "build_lines_rows", "build_props_rows",
           "build_lines_csv", "INPUTS_FILENAME", "PROPS_FILENAME", "LINES_FILENAME", "LINES_COLUMNS"]

logger = logging.getLogger(__name__)

INPUTS_FILENAME = "week_inputs.csv"
PROPS_FILENAME = "props_inputs.csv"
LINES_FILENAME = "lines.csv"
LINES_COLUMNS = ("week", "away", "home", "market", "selection", "american_odds", "model_prob",
                 "player", "player_id", "team", "position", "blocked", "model_note", "notes")
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


def expand_prop(row: Dict[str, Any], line: int = 0, reg: Optional[market_registry.Registry] = None) -> List[Dict[str, Any]]:
    """Turn one props_inputs.csv row into its lines.csv rows (Over + Under, or Yes [+ No]).

    ``model_prob`` is left blank here; ``prop_model.py`` fills it in when it
    is available, so the raw sportsbook numbers stay visible in lines.csv.
    """
    reg = reg or market_registry.registry()
    r = {(k or "").strip().lower(): (v if v is not None else "") for k, v in row.items()}
    try:
        week = int(float(str(r["week"]).strip()))
    except (KeyError, ValueError) as exc:
        raise BuildLinesError(f"line {line}: week is required and must be a number") from exc
    away, home = str(r.get("away", "")).strip(), str(r.get("home", "")).strip()
    player = " ".join(str(r.get("player", "")).split())
    if not away or not home or not player:
        raise BuildLinesError(f"line {line}: away, home and player are required")
    market_text = str(r.get("market", "")).strip()
    key = reg.resolve(market_text)
    if key is None or not reg.is_prop(key):
        known = ", ".join(pm.label for pm in reg.props.values())
        raise BuildLinesError(f"line {line}: unknown prop market '{market_text}'. Known prop markets: {known} (see markets.json)")
    pm = reg.get(key)
    assert pm is not None
    position = str(r.get("position", "")).strip().upper()
    notes = str(r.get("notes", "") or r.get("source", "")).strip()
    team = str(r.get("team", "")).strip()
    if team:
        low = team.lower()
        match = next((t for t in (away, home) if t.lower() == low or t.lower().endswith(" " + low)), None)
        if match is None:
            raise BuildLinesError(f"line {line}: team '{team}' for {player} is neither {away} nor {home}")
        team = match
    out: List[Dict[str, Any]] = []

    def add(selection: str, odds: int) -> None:
        out.append({"week": week, "away": away, "home": home, "market": key, "selection": selection, "american_odds": odds,
                    "model_prob": "", "player": player, "team": team, "position": position, "blocked": "", "model_note": "", "notes": notes})

    if pm.is_yes_no:
        yes = next((r.get(k) for k in ("yes_price", "over_price", "price") if str(r.get(k, "") or "").strip() != ""), None)
        if yes is None:
            raise BuildLinesError(f"line {line}: {pm.label} for {player} needs a yes_price (the price to score)")
        add(format_prop_selection(pm, player, "Yes", None), _price(yes, "yes_price", line))
        no = next((r.get(k) for k in ("no_price", "under_price") if str(r.get(k, "") or "").strip() != ""), None)
        if no is not None:
            add(format_prop_selection(pm, player, "No", None), _price(no, "no_price", line))
    else:
        ln = _num(r.get("line"), "line", None, line)
        if ln is None:
            raise BuildLinesError(f"line {line}: {pm.label} for {player} needs a line (e.g. 264.5)")
        add(format_prop_selection(pm, player, "Over", ln), _price(r.get("over_price"), "over_price", line))
        add(format_prop_selection(pm, player, "Under", ln), _price(r.get("under_price"), "under_price", line))
    return out


def build_props_rows(props_path: str, week: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read props_inputs.csv -> lines.csv rows (no model probabilities yet)."""
    if not os.path.isfile(props_path):
        raise BuildLinesError(f"Props file not found: {props_path}")
    try:
        reg = market_registry.registry()
    except RegistryError as exc:
        raise BuildLinesError(str(exc)) from exc
    rows: List[Dict[str, Any]] = []
    with open(props_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise BuildLinesError(f"{props_path} is empty")
        needed = {"week", "away", "home", "player", "market"}
        missing = needed - {f.strip().lower() for f in reader.fieldnames}
        if missing:
            raise BuildLinesError(f"{props_path} needs the columns: {', '.join(sorted(needed))} (missing {', '.join(sorted(missing))})")
        for i, raw in enumerate(reader, 2):
            if not any((v or "").strip() for v in raw.values()):
                continue
            expanded = expand_prop(raw, i, reg)
            if week is None or expanded[0]["week"] == week:
                rows.extend(expanded)
    return rows


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


def model_prop_rows(prop_rows: List[Dict[str, Any]], inputs_path: str, week: Optional[int] = None,
                    offline: Optional[bool] = None) -> Dict[str, int]:
    """Fill ``model_prob`` on prop rows through ``prop_model.py`` (EXPERIMENTAL). Never raises.

    When the model or its data is unavailable the rows keep a blank
    ``model_prob`` (no edge) and a ``blocked`` reason, and a warning says why.
    ``offline`` defaults to the EDGEBOOK_OFFLINE environment variable.
    """
    counts = {"projected": 0, "blocked": 0, "missing": len(prop_rows)}
    if not prop_rows:
        counts["missing"] = 0
        return counts
    if offline is None:
        offline = os.environ.get("EDGEBOOK_OFFLINE", "").strip().lower() in ("1", "true", "yes")
    try:
        import prop_model
    except ImportError as exc:
        logger.warning("prop_model.py not available (%s); player props carry no model probability", exc)
        for r in prop_rows:
            r["blocked"] = "no projection (prop_model.py missing)"
        return counts
    try:
        contexts = prop_model.contexts_from_inputs(inputs_path, week)
        counts = prop_model.fill_model_probs(prop_rows, contexts, offline=offline)
    except prop_model.PropModelError as exc:
        logger.warning("Prop model unavailable (%s); player props carry no model probability", exc)
        for r in prop_rows:
            if not r.get("model_prob"):
                r["blocked"] = f"no projection ({exc})"
        counts = {"projected": 0, "blocked": len(prop_rows), "missing": 0}
    return counts


LOG_FILENAME = "lines_log.csv"
LOG_COLUMNS = ("logged_at", "week", "away", "home", "player", "market", "selection", "american_odds", "model_prob", "blocked", "notes")


def log_prop_lines(prop_rows: List[Dict[str, Any]], log_path: str, now: Optional[str] = None) -> int:
    """Append each prop side to a timestamped log unless its price and model probability are unchanged.

    The log is the pipeline's own history of the lines you entered (no paid
    data): it records a new row whenever a line's price or the model's
    probability moves, so later weeks can be checked against what was
    actually offered and when. Returns the number of rows appended.
    """
    if not prop_rows:
        return 0
    now = now or _dt.datetime.now().isoformat(timespec="seconds")
    latest: Dict[Tuple[str, str, str, str], Tuple[str, str]] = {}
    if os.path.isfile(log_path):
        with open(log_path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                key = (str(r.get("week", "")), r.get("away", ""), r.get("home", ""), r.get("selection", ""))
                latest[key] = (str(r.get("american_odds", "")), str(r.get("model_prob", "")))
    new_rows: List[Dict[str, Any]] = []
    for r in prop_rows:
        key = (str(r["week"]), r["away"], r["home"], r["selection"])
        state = (str(r["american_odds"]), str(r.get("model_prob", "")))
        if latest.get(key) == state:
            continue
        latest[key] = state
        new_rows.append({"logged_at": now, "week": r["week"], "away": r["away"], "home": r["home"], "player": r.get("player", ""),
                         "market": r["market"], "selection": r["selection"], "american_odds": r["american_odds"],
                         "model_prob": r.get("model_prob", ""), "blocked": r.get("blocked", ""), "notes": r.get("notes", "")})
    if new_rows:
        write_header = not os.path.isfile(log_path) or os.path.getsize(log_path) == 0
        with open(log_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(LOG_COLUMNS), restval="")
            if write_header:
                writer.writeheader()
            writer.writerows(new_rows)
    return len(new_rows)


def build_lines_csv(inputs_path: str = INPUTS_FILENAME, out_path: str = LINES_FILENAME, week: Optional[int] = None,
                    props_path: Optional[str] = None, include_props: bool = True, model_props: bool = True,
                    log_path: Optional[str] = None, log_props: bool = True) -> str:
    """week_inputs.csv (+ props_inputs.csv beside it, or ``props_path``) -> lines.csv.

    Prop rows get their ``model_prob`` from ``prop_model.py`` unless
    ``model_props`` is False (or the model cannot run, in which case they are
    written with a ``blocked`` reason and no edge). Every prop side is also
    appended to ``lines_log.csv`` beside the props file (``log_path``) whenever
    its price or probability changed, timestamped.
    """
    rows = build_lines_rows(inputs_path, week)
    games = {(r["week"], r["away"], r["home"]) for r in rows}
    n_props = 0
    model_counts: Dict[str, int] = {}
    if include_props:
        if props_path is None:
            candidate = os.path.join(os.path.dirname(os.path.abspath(inputs_path)), PROPS_FILENAME)
            props_path = candidate if os.path.isfile(candidate) else None
        if props_path:
            prop_rows = build_props_rows(props_path, week)
            for pr in prop_rows:
                if (pr["week"], pr["away"], pr["home"]) not in games:
                    raise BuildLinesError(f"{os.path.basename(props_path)}: {pr['player']} is listed for {pr['away']} @ {pr['home']} "
                                          f"(week {pr['week']}), which is not a game in {os.path.basename(inputs_path)}; "
                                          f"team names must match that file exactly")
            if model_props and prop_rows:
                model_counts = model_prop_rows(prop_rows, inputs_path, week)
            if log_props and prop_rows:
                log_file = log_path or os.path.join(os.path.dirname(os.path.abspath(props_path)), LOG_FILENAME)
                try:
                    appended = log_prop_lines(prop_rows, log_file)
                    if appended:
                        logger.info("Logged %d new/changed prop side(s) to %s", appended, log_file)
                except OSError as exc:
                    logger.warning("Could not write %s: %s", log_file, exc)
            rows.extend(prop_rows)
            n_props = len(prop_rows)
    abs_out = os.path.abspath(out_path)
    with open(abs_out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(LINES_COLUMNS), restval="")
        writer.writeheader()
        writer.writerows(rows)
    detail = ""
    if n_props:
        detail = f" including {n_props} player-prop side(s)"
        if model_counts:
            detail += (f" ({model_counts.get('projected', 0)} projected by the experimental prop model, "
                       f"{model_counts.get('blocked', 0)} blocked)")
    logger.info("Wrote %d line rows for %d game(s)%s to %s", len(rows), len(games), detail, abs_out)
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

    # Player props from props_inputs.csv
    prow = {"week": "5", "away": "Tampa Bay Buccaneers", "home": "Dallas Cowboys", "player": "Dak Prescott", "market": "pass yds",
            "line": "264.5", "over_price": "-115", "under_price": "-105", "position": "QB", "notes": "example"}
    prows = expand_prop(prow, 2)
    check(len(prows) == 2 and prows[0]["selection"] == "Dak Prescott Over 264.5 Passing Yards" and prows[1]["american_odds"] == -105
          and prows[0]["market"] == "passing_yards" and prows[0]["player"] == "Dak Prescott" and prows[0]["model_prob"] == "",
          "prop row expands to Over/Under with the alias resolved and model_prob blank")
    trow = {"week": "5", "away": "Tampa Bay Buccaneers", "home": "Dallas Cowboys", "player": "CeeDee Lamb", "market": "Anytime TD", "yes_price": "-130"}
    trows = expand_prop(trow, 3)
    check(len(trows) == 1 and trows[0]["selection"] == "CeeDee Lamb Anytime TD" and trows[0]["american_odds"] == -130, "yes/no prop expands to one Yes row")
    check(len(expand_prop({**trow, "no_price": "+100"}, 4)) == 2 and expand_prop({**trow, "yes_price": "", "over_price": "-120"}, 4)[0]["american_odds"] == -120,
          "No price adds the No row; over_price is accepted as the Yes price")
    check(expand_prop({**prow, "over_price": "", "under_price": ""}, 5)[0]["american_odds"] == -110, "blank prop prices default to -110")
    for bad, why in (({**prow, "market": "elephants"}, "unknown prop market"), ({**prow, "line": ""}, "missing line"),
                     ({**trow, "yes_price": ""}, "missing yes price"), ({**prow, "player": ""}, "missing player"),
                     ({**prow, "market": "spread"}, "game market in the props file")):
        try:
            expand_prop(bad, 9)
            check(False, f"{why} rejected")
        except BuildLinesError:
            check(True, f"{why} rejected")
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "week_inputs.csv")
        with open(inp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        props = os.path.join(tmp, "props_inputs.csv")
        fields = ["week", "away", "home", "player", "market", "line", "over_price", "under_price", "yes_price", "no_price", "position", "notes"]
        with open(props, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, restval="")
            w.writeheader()
            w.writerow(prow)
            w.writerow(trow)
        out = build_lines_csv(inp, os.path.join(tmp, "lines.csv"))
        with open(out, newline="", encoding="utf-8") as fh:
            got = list(csv.DictReader(fh))
        check(len(got) == 9 and set(got[0].keys()) == set(LINES_COLUMNS) and got[6]["player"] == "Dak Prescott" and got[0]["player"] == "",
              "props beside week_inputs.csv are appended to lines.csv with the full column set")
        sides = parlay_finder.load_lines_csv(out, week=5)
        dak = next(s for s in sides if s.player == "Dak Prescott" and s.direction == "Over")
        check(len(sides) == 9 and sum(1 for s in sides if s.is_prop) == 3 and 0.5 < dak.fair_prob < 0.55 and dak.position == "QB",
              "parlay_finder loads the prop sides with a two-way de-vig")
        out2 = build_lines_csv(inp, os.path.join(tmp, "lines2.csv"), include_props=False)
        with open(out2, newline="", encoding="utf-8") as fh:
            check(len(list(csv.DictReader(fh))) == 6, "include_props=False skips the props file")
        # The timestamped prop line log: one row per side, re-runs add nothing, a moved price adds rows
        log_file = os.path.join(tmp, LOG_FILENAME)
        with open(log_file, newline="", encoding="utf-8") as fh:
            logged = list(csv.DictReader(fh))
        check(len(logged) == 3 and set(logged[0].keys()) == set(LOG_COLUMNS) and logged[0]["logged_at"] and logged[0]["selection"].startswith("Dak Prescott"),
              "prop sides are logged with a timestamp on the first build")
        build_lines_csv(inp, os.path.join(tmp, "lines.csv"))
        with open(log_file, newline="", encoding="utf-8") as fh:
            check(len(list(csv.DictReader(fh))) == 3, "an unchanged rebuild appends nothing to the log")
        with open(props, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, restval="")
            w.writeheader()
            w.writerow({**prow, "over_price": "-125"})
            w.writerow(trow)
        build_lines_csv(inp, os.path.join(tmp, "lines.csv"))
        with open(log_file, newline="", encoding="utf-8") as fh:
            logged = list(csv.DictReader(fh))
        check(len(logged) == 4 and logged[-1]["american_odds"] == "-125" and logged[-1]["selection"].startswith("Dak Prescott Over"),
              "a moved price appends only the side that changed")
        with open(props, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, restval="")
            w.writeheader()
            w.writerow(prow)
            w.writerow(trow)
        build_lines_csv(inp, os.path.join(tmp, "lines4.csv"), log_props=False)
        with open(log_file, newline="", encoding="utf-8") as fh:
            check(len(list(csv.DictReader(fh))) == 4, "log_props=False leaves the log alone")
        with open(props, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, restval="")
            w.writerow({**prow, "home": "Nowhere Team"})
        try:
            build_lines_csv(inp, os.path.join(tmp, "lines3.csv"))
            check(False, "prop for a game missing from week_inputs.csv rejected")
        except BuildLinesError as exc:
            check("Nowhere Team" in str(exc), "prop for a game missing from week_inputs.csv rejected with the team name")

    print(f"\n{'ALL TESTS PASSED' if failures == 0 else f'{failures} TEST(S) FAILED'}")
    return 0 if failures == 0 else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="build_lines", description="Expand week_inputs.csv into lines.csv for parlay_finder.")
    p.add_argument("--inputs", default=INPUTS_FILENAME)
    p.add_argument("--props", default=None, help=f"Player props CSV (default: {PROPS_FILENAME} beside --inputs when it exists)")
    p.add_argument("--no-props", action="store_true", dest="no_props", help="Ignore props_inputs.csv")
    p.add_argument("--no-model", action="store_true", dest="no_model", help="Do not run the prop model (props carry no edge)")
    p.add_argument("--no-log", action="store_true", dest="no_log", help=f"Do not append prop lines to {LOG_FILENAME}")
    p.add_argument("--out", default=LINES_FILENAME)
    p.add_argument("--week", type=int, default=None, help="Only this week (default: all weeks in the file)")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.selftest:
        return _selftest()
    try:
        out = build_lines_csv(args.inputs, args.out, args.week, props_path=args.props, include_props=not args.no_props,
                              model_props=not args.no_model, log_props=not args.no_log)
    except BuildLinesError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
