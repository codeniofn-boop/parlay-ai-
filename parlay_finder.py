#!/usr/bin/env python3
"""
parlay_finder.py
================

Candidate parlay generator for the NFL analytics pipeline.

Each week this module produces a ranked list of 2-leg and 3-leg parlay
tickets, every one carrying a **model true probability** and the **book
price**, in exactly the dictionary shape ``weekly_reporter.py`` consumes.

Where the lines come from (``source``)
--------------------------------------
``sim`` (default)
    A deterministic simulated NFL season. 32 real franchises get hidden
    power ratings; each week a schedule is drawn, the "market" posts spreads,
    totals and moneylines with vig, and the "model" produces its own
    estimates. The model's estimate is noisier-or-sharper than the market
    depending on ``SimConfig`` (the default assumes a modest informational
    edge, which is the premise being tested). The hidden truth is kept on the
    game objects **only** for ``backtester.py`` to simulate final scores; the
    finder never looks at it when choosing legs.
``csv``
    A spreadsheet of real lines (``lines.csv``). Columns::

        week,away,home,market,selection,american_odds[,model_prob]

    ``market`` is ``spread`` / ``total`` / ``moneyline``; ``selection`` is the
    human line text, e.g. ``Buffalo Bills -3.5``, ``Over 44.5``,
    ``Kansas City Chiefs ML``. When ``model_prob`` is blank the de-vigged
    market probability is used, which (correctly) yields no edge.
``api``
    Live lines from The Odds API (https://the-odds-api.com, v4). Requires an
    API key in ``ODDS_API_KEY`` or ``finder.api_key`` in
    ``pipeline_config.json``. Model probabilities for live lines are merged
    from an optional ``model_probs.csv`` (``matchup,selection,model_prob``);
    without it the de-vigged market probability is used.

Selection logic
---------------
1. Every market side becomes a candidate leg with ``edge = p_model * D - 1``.
2. Legs whose edge clears ``min_leg_edge`` are kept, best ``max_candidate_legs``.
3. Legs are combined into 2- and 3-leg tickets from *different games*
   (same-game combos are excluded by default because their legs are
   correlated and the independence assumption would overstate ``p_true``).
4. Tickets are ranked by parlay edge, diversified so no single leg appears
   on more than ``max_tickets_per_leg`` tickets, and the top ``top_n`` per
   leg count are returned.

Integration
-----------
::

    import parlay_finder
    tickets = parlay_finder.find_parlays(week=6)               # list[dict]
    # or, inside backtester.py, against pre-simulated games:
    season = parlay_finder.simulate_season(2026, seed=7)
    tickets = parlay_finder.find_parlays(week=6, games=season[6])

CLI: ``python3 parlay_finder.py --week 6 --out parlays_week_06.json`` or
``python3 parlay_finder.py --selftest``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import itertools
import json
import logging
import math
import os
import random
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from staking_engine import StakingInputError, american_to_decimal, decimal_to_american

__all__ = [
    "FinderError",
    "FinderConfig",
    "SimConfig",
    "MarketSide",
    "SimulatedGame",
    "NFL_TEAMS",
    "normal_cdf",
    "devig_two_way",
    "simulate_season",
    "simulate_week_games",
    "sides_from_games",
    "load_lines_csv",
    "load_model_probs_csv",
    "parse_odds_api_response",
    "fetch_lines_from_odds_api",
    "build_parlays",
    "rank_score",
    "find_parlays",
    "load_finder_config",
    "resolve_data_path",
    "lines_file_has_week",
    "write_parlays_json",
]

__version__ = "1.0.0"

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "pipeline_config.json"

# ---------------------------------------------------------------------------
# League constants
# ---------------------------------------------------------------------------

NFL_TEAMS: Tuple[str, ...] = (
    "Buffalo Bills", "Miami Dolphins", "New England Patriots", "New York Jets",
    "Baltimore Ravens", "Cincinnati Bengals", "Cleveland Browns", "Pittsburgh Steelers",
    "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Tennessee Titans",
    "Denver Broncos", "Kansas City Chiefs", "Las Vegas Raiders", "Los Angeles Chargers",
    "Dallas Cowboys", "New York Giants", "Philadelphia Eagles", "Washington Commanders",
    "Chicago Bears", "Detroit Lions", "Green Bay Packers", "Minnesota Vikings",
    "Atlanta Falcons", "Carolina Panthers", "New Orleans Saints", "Tampa Bay Buccaneers",
    "Arizona Cardinals", "Los Angeles Rams", "San Francisco 49ers", "Seattle Seahawks",
)

# Empirical NFL dispersion: final margin ~ N(expected, 13.5), total ~ N(expected, 10).
MARGIN_SD = 13.5
TOTAL_SD = 10.0
HOME_FIELD_ADVANTAGE = 2.0
STANDARD_VIG = 0.045  # two-way overround on spreads/totals/moneylines (~ -110/-110)


class FinderError(RuntimeError):
    """Unrecoverable finder failure (bad CSV, API error, no games)."""


# ---------------------------------------------------------------------------
# Maths helpers
# ---------------------------------------------------------------------------


def normal_cdf(x: float) -> float:
    """Standard normal CDF via the error function (no SciPy needed)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _round_half(x: float) -> float:
    """Round to the nearest 0.5, the granularity books post lines at."""
    return round(x * 2.0) / 2.0


def implied_from_american(american: int) -> float:
    return 1.0 / american_to_decimal(american)


def american_from_implied(implied: float) -> int:
    """Convert a vig-inclusive probability to American odds rounded to 5."""
    implied = min(0.985, max(0.015, implied))
    if implied > 0.5:
        raw = -100.0 * implied / (1.0 - implied)
    else:
        raw = 100.0 * (1.0 - implied) / implied
    rounded = int(round(raw / 5.0) * 5)
    if -100 < rounded < 100:
        rounded = 100 if rounded >= 0 else -105
    return rounded


def devig_two_way(odds_a: int, odds_b: int) -> Tuple[float, float]:
    """Remove the bookmaker margin from a two-sided market (multiplicative).

    Returns the fair probabilities ``(p_a, p_b)`` that sum to 1.
    """
    ia, ib = implied_from_american(odds_a), implied_from_american(odds_b)
    total = ia + ib
    if total <= 0:
        raise FinderError("Cannot de-vig a market with non-positive implied total")
    return ia / total, ib / total


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MarketSide:
    """One priceable side of a market, i.e. one potential parlay leg."""

    week: int
    away: str
    home: str
    market: str            # "spread" | "total" | "moneyline"
    selection: str         # "Buffalo Bills -3.5" / "Over 44.5" / "Kansas City Chiefs ML"
    american_odds: int
    model_prob: float      # the pipeline's probability estimate (what we bet on)
    fair_prob: float       # de-vigged market probability (the market's opinion)
    line: Optional[float] = None
    true_prob: Optional[float] = None   # SIM ONLY: hidden truth for the backtester
    source: str = ""

    @property
    def matchup(self) -> str:
        return f"{self.away} @ {self.home}"

    @property
    def game_key(self) -> Tuple[int, str, str]:
        return (self.week, self.away, self.home)

    @property
    def decimal_odds(self) -> float:
        return american_to_decimal(self.american_odds)

    @property
    def implied_prob(self) -> float:
        return 1.0 / self.decimal_odds

    @property
    def edge(self) -> float:
        """Model edge per $1: ``p_model * D - 1``."""
        return self.model_prob * self.decimal_odds - 1.0

    def to_leg_dict(self) -> Dict[str, Any]:
        """Leg dictionary in the weekly_reporter contract (no hidden truth)."""
        return {
            "matchup": self.matchup,
            "away": self.away,
            "home": self.home,
            "selection": self.selection,
            "market": self.market,
            "line": self.line,
            "american_odds": self.american_odds,
            "p_true": round(self.model_prob, 6),
            "fair_prob": round(self.fair_prob, 6),
            "implied_prob": round(self.implied_prob, 6),
            "leg_edge": round(self.edge, 6),
        }


@dataclass
class SimConfig:
    """Knobs for the simulated league.

    How the market and the model relate to the hidden truth ``T``::

        market = T + e_market                       e_market ~ N(0, market_noise)
        model  = market + skill * (T - market) + e  e        ~ N(0, model_noise)

    The model therefore starts from the market number (it shares the public
    information) and corrects a fraction ``model_skill`` of the market's error
    using private information, plus its own noise. ``model_skill = 0`` is a
    genuine null: the model is the market plus noise, so legs it likes have
    no true edge on average. ``model_skill = 1`` with zero noise is an oracle.
    The defaults give a modest, realistic edge; the backtester's ``--no-edge``
    flag runs the null for comparison.
    """

    rating_sd: float = 6.0            # spread of team power ratings (points)
    scoring_mean: float = 22.5        # average points scored per team per game
    scoring_sd: float = 2.5
    home_field: float = HOME_FIELD_ADVANTAGE
    market_noise_margin: float = 1.0  # market error vs truth, spread points
    market_noise_total: float = 1.5   # market error vs truth, total points
    model_skill: float = 0.5          # fraction of the market's error the model corrects
    model_noise_margin: float = 0.75  # model's own noise, spread points
    model_noise_total: float = 1.0    # model's own noise, total points
    vig: float = STANDARD_VIG
    games_per_week: int = 16
    bye_weeks: Tuple[int, int] = (5, 14)   # inclusive range with 4 teams on bye

    def __post_init__(self) -> None:
        if not 0.0 <= self.model_skill <= 1.0:
            raise FinderError("model_skill must be within [0, 1]")
        for name in ("market_noise_margin", "market_noise_total", "model_noise_margin", "model_noise_total"):
            if getattr(self, name) < 0:
                raise FinderError(f"{name} cannot be negative")

    def no_edge(self) -> "SimConfig":
        """Copy where the model has no private information (null hypothesis)."""
        return SimConfig(**{**asdict(self), "model_skill": 0.0})


@dataclass
class SimulatedGame:
    """A simulated matchup with posted lines, model numbers and hidden truth."""

    week: int
    away: str
    home: str
    # Hidden truth (home-minus-away expected margin, expected total points)
    true_margin: float
    true_total: float
    # What the market posted
    home_spread: float          # e.g. -3.5 means home favoured by 3.5
    total_line: float
    spread_odds_home: int
    spread_odds_away: int
    total_odds_over: int
    total_odds_under: int
    ml_odds_home: int
    ml_odds_away: int
    # What the model believes
    model_margin: float
    model_total: float

    @property
    def matchup(self) -> str:
        return f"{self.away} @ {self.home}"

    # --- probability helpers -------------------------------------------------

    @staticmethod
    def _p_margin_over(mean: float, threshold: float) -> float:
        """P(actual margin > threshold) with margin ~ N(mean, MARGIN_SD)."""
        return 1.0 - normal_cdf((threshold - mean) / MARGIN_SD)

    @staticmethod
    def _p_total_over(mean: float, threshold: float) -> float:
        return 1.0 - normal_cdf((threshold - mean) / TOTAL_SD)

    def sides(self, source: str = "sim") -> List[MarketSide]:
        """All six market sides with model, fair and true probabilities."""
        spread_point = -self.home_spread  # home covers when margin > spread_point
        fair_home_cover, fair_away_cover = devig_two_way(self.spread_odds_home, self.spread_odds_away)
        fair_over, fair_under = devig_two_way(self.total_odds_over, self.total_odds_under)
        fair_home_ml, fair_away_ml = devig_two_way(self.ml_odds_home, self.ml_odds_away)

        model_home_cover = self._p_margin_over(self.model_margin, spread_point)
        true_home_cover = self._p_margin_over(self.true_margin, spread_point)
        model_over = self._p_total_over(self.model_total, self.total_line)
        true_over = self._p_total_over(self.true_total, self.total_line)
        model_home_win = self._p_margin_over(self.model_margin, 0.0)
        true_home_win = self._p_margin_over(self.true_margin, 0.0)

        common = dict(week=self.week, away=self.away, home=self.home, source=source)
        return [
            MarketSide(market="spread", selection=f"{self.home} {self.home_spread:+g}", line=self.home_spread,
                       american_odds=self.spread_odds_home, model_prob=model_home_cover,
                       fair_prob=fair_home_cover, true_prob=true_home_cover, **common),
            MarketSide(market="spread", selection=f"{self.away} {-self.home_spread:+g}", line=-self.home_spread,
                       american_odds=self.spread_odds_away, model_prob=1.0 - model_home_cover,
                       fair_prob=fair_away_cover, true_prob=1.0 - true_home_cover, **common),
            MarketSide(market="total", selection=f"Over {self.total_line:g}", line=self.total_line,
                       american_odds=self.total_odds_over, model_prob=model_over,
                       fair_prob=fair_over, true_prob=true_over, **common),
            MarketSide(market="total", selection=f"Under {self.total_line:g}", line=self.total_line,
                       american_odds=self.total_odds_under, model_prob=1.0 - model_over,
                       fair_prob=fair_under, true_prob=1.0 - true_over, **common),
            MarketSide(market="moneyline", selection=f"{self.home} ML", line=None,
                       american_odds=self.ml_odds_home, model_prob=model_home_win,
                       fair_prob=fair_home_ml, true_prob=true_home_win, **common),
            MarketSide(market="moneyline", selection=f"{self.away} ML", line=None,
                       american_odds=self.ml_odds_away, model_prob=1.0 - model_home_win,
                       fair_prob=fair_away_ml, true_prob=1.0 - true_home_win, **common),
        ]

    def simulate_final_score(self, rng: random.Random) -> Tuple[int, int]:
        """Draw a final score ``(away_score, home_score)`` from the hidden truth.

        Margin and total are drawn jointly so that spread, total and
        moneyline legs from the same game resolve consistently.
        """
        margin = rng.gauss(self.true_margin, MARGIN_SD)
        total = max(0.0, rng.gauss(self.true_total, TOTAL_SD))
        home = max(0, int(round((total + margin) / 2.0)))
        away = max(0, int(round((total - margin) / 2.0)))
        return away, home

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Simulated league
# ---------------------------------------------------------------------------


def _season_rng(season: int, seed: int, tag: str = "") -> random.Random:
    return random.Random(f"{season}:{seed}:{tag}")


def _team_strengths(season: int, seed: int, cfg: SimConfig) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Hidden power rating (points vs average) and scoring mean per team."""
    rng = _season_rng(season, seed, "ratings")
    ratings = {t: max(-12.0, min(12.0, rng.gauss(0.0, cfg.rating_sd))) for t in NFL_TEAMS}
    scoring = {t: max(15.0, min(32.0, rng.gauss(cfg.scoring_mean, cfg.scoring_sd))) for t in NFL_TEAMS}
    return ratings, scoring


def _price_two_way(fair_a: float, vig: float) -> Tuple[int, int]:
    """Add overround symmetrically and convert both sides to American odds."""
    fair_b = 1.0 - fair_a
    return american_from_implied(fair_a + vig / 2.0), american_from_implied(fair_b + vig / 2.0)


def simulate_week_games(
    season: int,
    week: int,
    seed: int = 7,
    cfg: Optional[SimConfig] = None,
) -> List[SimulatedGame]:
    """Deterministically simulate one week's slate (lines, model, hidden truth)."""
    cfg = cfg or SimConfig()
    if week < 1:
        raise FinderError(f"week must be >= 1, got {week}")
    ratings, scoring = _team_strengths(season, seed, cfg)
    rng = _season_rng(season, seed, f"week{week}")

    teams = list(NFL_TEAMS)
    rng.shuffle(teams)
    n_games = cfg.games_per_week
    if cfg.bye_weeks[0] <= week <= cfg.bye_weeks[1]:
        n_games = max(1, cfg.games_per_week - 2)  # four teams rest
    games: List[SimulatedGame] = []
    for i in range(n_games):
        away, home = teams[2 * i], teams[2 * i + 1]
        true_margin = ratings[home] - ratings[away] + cfg.home_field
        true_total = scoring[home] + scoring[away] + rng.gauss(0.0, 1.5)

        market_margin = true_margin + rng.gauss(0.0, cfg.market_noise_margin)
        market_total = true_total + rng.gauss(0.0, cfg.market_noise_total)
        # The model shares the market's information and corrects part of its error.
        model_margin = (market_margin + cfg.model_skill * (true_margin - market_margin)
                        + rng.gauss(0.0, cfg.model_noise_margin))
        model_total = (market_total + cfg.model_skill * (true_total - market_total)
                       + rng.gauss(0.0, cfg.model_noise_total))

        home_spread = -_round_half(market_margin)
        total_line = _round_half(market_total)

        # Prices: fair probability at the posted number from the market's own
        # view, plus vig. Rounding residue naturally produces -105/-115 shading.
        fair_home_cover = SimulatedGame._p_margin_over(market_margin, -home_spread)
        spread_home, spread_away = _price_two_way(fair_home_cover, cfg.vig)
        fair_over = SimulatedGame._p_total_over(market_total, total_line)
        over_odds, under_odds = _price_two_way(fair_over, cfg.vig)
        fair_home_win = SimulatedGame._p_margin_over(market_margin, 0.0)
        ml_home, ml_away = _price_two_way(fair_home_win, cfg.vig)

        games.append(SimulatedGame(
            week=week, away=away, home=home,
            true_margin=true_margin, true_total=true_total,
            home_spread=home_spread, total_line=total_line,
            spread_odds_home=spread_home, spread_odds_away=spread_away,
            total_odds_over=over_odds, total_odds_under=under_odds,
            ml_odds_home=ml_home, ml_odds_away=ml_away,
            model_margin=model_margin, model_total=model_total,
        ))
    return games


def simulate_season(
    season: int,
    seed: int = 7,
    weeks: int = 18,
    cfg: Optional[SimConfig] = None,
) -> Dict[int, List[SimulatedGame]]:
    """Every week's games for a season, keyed by week number."""
    return {w: simulate_week_games(season, w, seed, cfg) for w in range(1, weeks + 1)}


def sides_from_games(games: Iterable[SimulatedGame], source: str = "sim") -> List[MarketSide]:
    out: List[MarketSide] = []
    for g in games:
        out.extend(g.sides(source))
    return out


# ---------------------------------------------------------------------------
# Real lines: CSV
# ---------------------------------------------------------------------------

_SPREAD_RE = re.compile(r"^(?P<team>.+?)\s+(?P<line>[+-]\s?\d+(?:\.\d+)?)$")
_TOTAL_RE = re.compile(r"^(?P<dir>over|under)\s+(?P<line>\d+(?:\.\d+)?)$", re.IGNORECASE)
_ML_RE = re.compile(r"^(?P<team>.+?)\s+(?:ML|moneyline|money\s*line)$", re.IGNORECASE)


def parse_selection(selection: str, market_hint: str = "") -> Tuple[str, Optional[str], Optional[float]]:
    """Classify a selection string -> ``(market, team_or_direction, line)``.

    ``"Buffalo Bills -3.5"`` -> ``("spread", "Buffalo Bills", -3.5)``
    ``"Over 44.5"``          -> ``("total", "Over", 44.5)``
    ``"Chiefs ML"``          -> ``("moneyline", "Chiefs", None)``
    """
    text = selection.strip()
    m = _TOTAL_RE.match(text)
    if m:
        return "total", m.group("dir").title(), float(m.group("line"))
    m = _ML_RE.match(text)
    if m:
        return "moneyline", m.group("team").strip(), None
    m = _SPREAD_RE.match(text)
    if m:
        return "spread", m.group("team").strip(), float(m.group("line").replace(" ", ""))
    hint = market_hint.strip().lower()
    if hint in ("moneyline", "ml", "h2h"):
        return "moneyline", text, None
    raise FinderError(f"Cannot parse selection '{selection}'")


def _to_int_odds(value: Any) -> int:
    try:
        text = str(value).strip().replace("+", "")
        odds = int(float(text))
    except (TypeError, ValueError) as exc:
        raise FinderError(f"Invalid american odds '{value}'") from exc
    if -100 < odds < 100:
        raise FinderError(f"American odds must be <= -100 or >= +100, got {odds}")
    return odds


def load_lines_csv(path: str, week: Optional[int] = None, source: str = "csv") -> List[MarketSide]:
    """Load real lines from a CSV; see the module docstring for the columns."""
    if not os.path.isfile(path):
        raise FinderError(f"Lines file not found: {path}")
    rows: List[Dict[str, str]] = []
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise FinderError(f"{path} is empty")
            required = {"week", "away", "home", "market", "selection", "american_odds"}
            missing = required - {f.strip().lower() for f in reader.fieldnames}
            if missing:
                raise FinderError(f"{path} is missing columns: {', '.join(sorted(missing))}")
            for raw in reader:
                rows.append({(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()})
    except OSError as exc:
        raise FinderError(f"Could not read {path}: {exc}") from exc

    # First pass: parse rows, group two-way markets for de-vigging.
    parsed: List[Dict[str, Any]] = []
    for i, r in enumerate(rows, 2):  # line numbers for error messages (header = 1)
        try:
            wk = int(float(r["week"]))
        except ValueError as exc:
            raise FinderError(f"{path} line {i}: bad week '{r['week']}'") from exc
        if week is not None and wk != week:
            continue
        market_hint = r.get("market", "")
        market, _, line = parse_selection(r["selection"], market_hint)
        if market_hint and market_hint.lower() not in (market, "ml", "h2h"):
            logger.warning("%s line %d: market '%s' disagrees with selection '%s'; using %s",
                           path, i, market_hint, r["selection"], market)
        odds = _to_int_odds(r["american_odds"])
        mp = r.get("model_prob", "")
        model_prob: Optional[float] = None
        if mp not in ("", None):
            try:
                model_prob = float(mp)
            except ValueError as exc:
                raise FinderError(f"{path} line {i}: bad model_prob '{mp}'") from exc
            if model_prob > 1.0:
                model_prob /= 100.0
            if not 0.0 < model_prob < 1.0:
                raise FinderError(f"{path} line {i}: model_prob must be in (0, 1), got {model_prob}")
        parsed.append(dict(week=wk, away=r["away"], home=r["home"], market=market,
                           selection=r["selection"], line=line, odds=odds, model_prob=model_prob))

    # Second pass: de-vig per (game, market, line) group when both sides exist.
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for p in parsed:
        key = (p["week"], p["away"], p["home"], p["market"], abs(p["line"]) if p["line"] is not None else None)
        groups.setdefault(key, []).append(p)

    sides: List[MarketSide] = []
    missing_model = 0
    for key, members in groups.items():
        if len(members) == 2:
            fa, fb = devig_two_way(members[0]["odds"], members[1]["odds"])
            fairs = [fa, fb]
        else:
            fairs = [implied_from_american(m["odds"]) for m in members]  # best effort
        for p, fair in zip(members, fairs):
            model_prob = p["model_prob"]
            if model_prob is None:
                model_prob = fair
                missing_model += 1
            sides.append(MarketSide(week=p["week"], away=p["away"], home=p["home"], market=p["market"],
                                    selection=p["selection"], american_odds=p["odds"], model_prob=model_prob,
                                    fair_prob=fair, line=p["line"], source=source))
    if missing_model:
        logger.warning("%d side(s) in %s had no model_prob; using de-vigged market probability (zero edge)",
                       missing_model, path)
    if not sides:
        raise FinderError(f"No lines for week {week} in {path}" if week else f"No usable rows in {path}")
    return sides


def load_model_probs_csv(path: str) -> Dict[Tuple[str, str], float]:
    """``matchup,selection,model_prob`` -> {(matchup, selection): prob}."""
    if not os.path.isfile(path):
        raise FinderError(f"Model probabilities file not found: {path}")
    out: Dict[Tuple[str, str], float] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for i, raw in enumerate(reader, 2):
            r = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
            try:
                p = float(r["model_prob"])
            except (KeyError, ValueError) as exc:
                raise FinderError(f"{path} line {i}: bad or missing model_prob") from exc
            if p > 1.0:
                p /= 100.0
            out[(r.get("matchup", "").lower(), r.get("selection", "").lower())] = p
    return out


def apply_model_probs(sides: List[MarketSide], probs: Dict[Tuple[str, str], float]) -> int:
    """Overwrite ``model_prob`` where a matching entry exists; returns count applied."""
    applied = 0
    for s in sides:
        key = (s.matchup.lower(), s.selection.lower())
        if key in probs:
            s.model_prob = probs[key]
            applied += 1
    return applied


# ---------------------------------------------------------------------------
# Real lines: The Odds API (v4)
# ---------------------------------------------------------------------------

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"


def fetch_odds_api_json(api_key: str, bookmakers: Optional[str] = None, timeout: float = 20.0) -> Any:
    """GET the raw JSON for NFL h2h/spreads/totals. Network errors -> FinderError."""
    if not api_key:
        raise FinderError("The Odds API needs an api_key (ODDS_API_KEY env var or finder.api_key)")
    params = {"apiKey": api_key, "regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american"}
    if bookmakers:
        params["bookmakers"] = bookmakers
    url = ODDS_API_URL + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            remaining = resp.headers.get("x-requests-remaining")
            if remaining is not None:
                logger.info("Odds API requests remaining this month: %s", remaining)
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise FinderError(f"Odds API returned HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise FinderError(f"Could not reach the Odds API: {exc.reason}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FinderError(f"Odds API returned unreadable JSON: {exc}") from exc


def parse_odds_api_response(
    events: Any,
    week: int,
    preferred_bookmaker: Optional[str] = None,
    horizon_days: float = 8.0,
    now: Optional[_dt.datetime] = None,
) -> List[MarketSide]:
    """Turn the Odds API event list into MarketSides (de-vigged, zero-edge model)."""
    if not isinstance(events, list):
        raise FinderError("Odds API response is not a list of events")
    now = now or _dt.datetime.now(_dt.timezone.utc)
    sides: List[MarketSide] = []
    for ev in events:
        try:
            home, away = ev["home_team"], ev["away_team"]
            start = _dt.datetime.fromisoformat(str(ev["commence_time"]).replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError):
            logger.warning("Skipping malformed event: %r", str(ev)[:120])
            continue
        if not (now - _dt.timedelta(hours=6) <= start <= now + _dt.timedelta(days=horizon_days)):
            continue
        books = ev.get("bookmakers") or []
        if not books:
            continue
        book = next((b for b in books if preferred_bookmaker and b.get("key") == preferred_bookmaker), books[0])
        for mkt in book.get("markets") or []:
            key = mkt.get("key")
            outcomes = mkt.get("outcomes") or []
            if len(outcomes) != 2:
                continue
            try:
                o1, o2 = outcomes
                odds1, odds2 = _to_int_odds(o1["price"]), _to_int_odds(o2["price"])
            except (KeyError, FinderError):
                continue
            fair1, fair2 = devig_two_way(odds1, odds2)
            for o, odds, fair in ((o1, odds1, fair1), (o2, odds2, fair2)):
                name = str(o.get("name", ""))
                point = o.get("point")
                if key == "h2h":
                    market, selection, line = "moneyline", f"{name} ML", None
                elif key == "spreads" and point is not None:
                    market, selection, line = "spread", f"{name} {float(point):+g}", float(point)
                elif key == "totals" and point is not None:
                    market, selection, line = "total", f"{name.title()} {float(point):g}", float(point)
                else:
                    continue
                sides.append(MarketSide(week=week, away=away, home=home, market=market, selection=selection,
                                        american_odds=odds, model_prob=fair, fair_prob=fair, line=line,
                                        source=f"api:{book.get('key', 'book')}"))
    return sides


def fetch_lines_from_odds_api(week: int, api_key: str, bookmaker: Optional[str] = None) -> List[MarketSide]:
    sides = parse_odds_api_response(fetch_odds_api_json(api_key, bookmaker), week, bookmaker)
    if not sides:
        raise FinderError("The Odds API returned no NFL games inside the next 8 days")
    return sides


# ---------------------------------------------------------------------------
# Parlay construction
# ---------------------------------------------------------------------------


@dataclass
class FinderConfig:
    """All tunables for leg filtering and ticket construction."""

    source: str = "sim"                 # sim | csv | api
    lines_csv: str = "lines.csv"
    model_csv: Optional[str] = None     # optional model_probs.csv (csv/api sources)
    api_key: Optional[str] = None
    bookmaker: Optional[str] = None
    season: Optional[int] = None        # defaults to the current NFL season year
    seed: int = 7
    sim: SimConfig = field(default_factory=SimConfig)
    leg_sizes: Tuple[int, ...] = (2, 3)
    top_n: int = 10                     # tickets returned per leg size
    min_leg_edge: float = 0.02          # legs must beat this model edge (2% default)
    max_candidate_legs: int = 12        # cap on legs entering the combinatorics
    max_tickets_per_leg: int = 3        # diversification: a leg may appear on at most this many tickets in total
    max_tickets_per_game: int = 4       # diversification: any single game may affect at most this many tickets
    allow_same_game: bool = False       # correlated legs excluded by default
    markets: Tuple[str, ...] = ("spread", "total", "moneyline")
    max_leg_odds: int = 300             # skip legs priced longer than +300 (longshot over-confidence)
    rank_by: str = "growth"             # growth (edge^2 / (D-1), Kelly log-growth proxy) | edge
    auto_detect_lines: bool = True      # source "sim" switches to "csv" when lines_csv exists for the week

    def __post_init__(self) -> None:
        self.source = (self.source or "sim").lower()
        if self.source not in ("sim", "csv", "api"):
            raise FinderError(f"finder.source must be sim | csv | api, got '{self.source}'")
        self.rank_by = (self.rank_by or "growth").lower()
        if self.rank_by not in ("growth", "edge"):
            raise FinderError(f"finder.rank_by must be growth | edge, got '{self.rank_by}'")
        if self.top_n < 1:
            raise FinderError("finder.top_n must be >= 1")
        if any(n < 2 for n in self.leg_sizes):
            raise FinderError("leg sizes must be >= 2")
        if self.max_candidate_legs < max(self.leg_sizes):
            raise FinderError("max_candidate_legs must be >= the largest leg size")
        if isinstance(self.sim, dict):
            self.sim = SimConfig(**self.sim)
        if isinstance(self.leg_sizes, list):
            self.leg_sizes = tuple(int(n) for n in self.leg_sizes)
        if isinstance(self.markets, list):
            self.markets = tuple(str(m).lower() for m in self.markets)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FinderConfig":
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed and not k.startswith("_")})


def nfl_season_year(date: Optional[_dt.date] = None) -> int:
    date = date or _dt.date.today()
    return date.year if date.month >= 3 else date.year - 1


def resolve_data_path(path: str) -> str:
    """Find a data file in the working directory or next to this module."""
    if os.path.isabs(path) or os.path.isfile(path):
        return path
    beside = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    return beside if os.path.isfile(beside) else path


def lines_file_has_week(path: str, week: int) -> bool:
    """True when ``path`` exists and contains at least one row for ``week``."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or "week" not in {f.strip().lower() for f in reader.fieldnames}:
                return False
            for raw in reader:
                r = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
                try:
                    if int(float(r.get("week", ""))) == week:
                        return True
                except ValueError:
                    continue
    except OSError:
        return False
    return False


def load_finder_config(path: str = CONFIG_FILENAME) -> FinderConfig:
    """Read the ``"finder"`` section of pipeline_config.json (missing -> defaults)."""
    search = [path, os.path.join(os.path.dirname(os.path.abspath(__file__)), path)]
    for candidate in search:
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not read %s (%s); using finder defaults", candidate, exc)
                return FinderConfig()
            section = data.get("finder", {}) if isinstance(data, dict) else {}
            if not isinstance(section, dict):
                logger.warning("'finder' in %s is not an object; using defaults", candidate)
                return FinderConfig()
            return FinderConfig.from_dict(section)
    return FinderConfig()


def rank_score(p_true: float, decimal_odds: float, rank_by: str = "growth") -> float:
    """Ranking metric for a leg or ticket.

    ``edge``   -> ``p * D - 1`` (expected profit per $1; rewards long shots)
    ``growth`` -> ``edge * full_kelly = edge^2 / (D - 1)`` for +EV bets, the
                  expected log-growth proxy under Kelly sizing. It prefers a
                  solid edge at a short price over a thin-probability long
                  shot with the same raw edge, which is where model
                  over-confidence does the most damage.
    """
    edge = p_true * decimal_odds - 1.0
    if rank_by == "edge" or edge <= 0.0:
        return edge
    return edge * edge / (decimal_odds - 1.0)


def select_candidate_legs(sides: Sequence[MarketSide], cfg: FinderConfig) -> List[MarketSide]:
    """Keep +EV legs above the threshold and price cap, one side per market, best first."""
    best_per_market: Dict[Tuple[Any, ...], MarketSide] = {}
    for s in sides:
        if s.market not in cfg.markets:
            continue
        if s.edge <= cfg.min_leg_edge or s.american_odds > cfg.max_leg_odds:
            continue
        key = (s.game_key, s.market, abs(s.line) if s.line is not None else None)
        if key not in best_per_market or s.edge > best_per_market[key].edge:
            best_per_market[key] = s
    ranked = sorted(best_per_market.values(),
                    key=lambda s: (-rank_score(s.model_prob, s.decimal_odds, cfg.rank_by), -s.model_prob, s.selection))
    return ranked[: cfg.max_candidate_legs]


def _ticket_from_legs(legs: Sequence[MarketSide], week: int, ticket_id: str, source: str,
                      rank_by: str = "growth") -> Dict[str, Any]:
    p_true = 1.0
    decimal_odds = 1.0
    for leg in legs:
        p_true *= leg.model_prob
        decimal_odds *= leg.decimal_odds
    edge = p_true * decimal_odds - 1.0
    fair = 1.0
    for leg in legs:
        fair *= leg.fair_prob
    return {
        "ticket_id": ticket_id,
        "week": week,
        "n_legs": len(legs),
        "legs": [leg.to_leg_dict() for leg in legs],
        "p_true": round(p_true, 6),
        "fair_prob": round(fair, 6),
        "decimal_odds": round(decimal_odds, 6),
        "american_odds": decimal_to_american(decimal_odds),
        "edge": round(edge, 6),
        "rank_score": round(rank_score(p_true, decimal_odds, rank_by), 6),
        "notes": f"model {p_true:.1%} vs market fair {fair:.1%}; legs: "
                 + ", ".join(f"{leg.selection} ({leg.edge:+.1%})" for leg in legs),
        "source": source,
    }


def build_parlays(sides: Sequence[MarketSide], week: int, cfg: Optional[FinderConfig] = None) -> List[Dict[str, Any]]:
    """Combine candidate legs into ranked, diversified parlay ticket dicts."""
    cfg = cfg or FinderConfig()
    candidates = select_candidate_legs(sides, cfg)
    source = candidates[0].source if candidates else (sides[0].source if sides else cfg.source)
    if len(candidates) < min(cfg.leg_sizes):
        logger.info("Week %d: only %d +EV leg(s); no parlays possible", week, len(candidates))
        return []

    tickets: List[Dict[str, Any]] = []
    usage: Dict[Tuple[Any, ...], int] = {}       # leg -> tickets using it, across ALL sizes
    game_usage: Dict[Tuple[Any, ...], int] = {}  # game -> tickets depending on it, across ALL sizes
    for size in cfg.leg_sizes:
        combos: List[Tuple[float, Tuple[MarketSide, ...]]] = []
        for combo in itertools.combinations(candidates, size):
            if not cfg.allow_same_game and len({leg.game_key for leg in combo}) < size:
                continue
            p = 1.0
            d = 1.0
            for leg in combo:
                p *= leg.model_prob
                d *= leg.decimal_odds
            edge = p * d - 1.0
            if edge <= 0:
                continue
            combos.append((rank_score(p, d, cfg.rank_by), combo))
        combos.sort(key=lambda item: (-item[0], -math.prod(leg.model_prob for leg in item[1])))

        kept = 0
        for edge, combo in combos:
            keys = [(leg.game_key, leg.selection) for leg in combo]
            games = {leg.game_key for leg in combo}
            if any(usage.get(k, 0) >= cfg.max_tickets_per_leg for k in keys):
                continue
            if any(game_usage.get(g, 0) >= cfg.max_tickets_per_game for g in games):
                continue
            for k in keys:
                usage[k] = usage.get(k, 0) + 1
            for g in games:
                game_usage[g] = game_usage.get(g, 0) + 1
            kept += 1
            tickets.append(_ticket_from_legs(combo, week, f"W{week:02d}-{size}L-{kept:02d}", source, cfg.rank_by))
            if kept >= cfg.top_n:
                break
    return tickets


# ---------------------------------------------------------------------------
# Public entry point (probed by weekly_reporter.py)
# ---------------------------------------------------------------------------


def collect_sides(week: int, cfg: FinderConfig, games: Optional[Sequence[SimulatedGame]] = None) -> List[MarketSide]:
    """Gather every market side for the week from the configured source."""
    if games is not None:
        return sides_from_games(games, "sim")
    source = cfg.source
    lines_path = resolve_data_path(cfg.lines_csv)
    # week_inputs.csv (one row per game) is the source of truth when present:
    # regenerate the DEFAULT lines.csv beside it so hand edits never drift out
    # of sync. A lines_csv that points somewhere specific is never touched.
    uses_default_lines = os.path.basename(cfg.lines_csv) == cfg.lines_csv == "lines.csv"
    inputs_path = resolve_data_path("week_inputs.csv")
    if cfg.auto_detect_lines and uses_default_lines and os.path.isfile(inputs_path):
        try:
            from build_lines import BuildLinesError, build_lines_csv  # local import: optional helper module
            target = os.path.join(os.path.dirname(os.path.abspath(inputs_path)), "lines.csv")
            lines_path = build_lines_csv(inputs_path, target)
            logger.info("Rebuilt %s from %s", lines_path, inputs_path)
        except ImportError:
            logger.warning("week_inputs.csv found but build_lines.py is missing; using lines.csv as-is")
        except BuildLinesError as exc:
            logger.warning("week_inputs.csv could not be expanded (%s); using lines.csv as-is", exc)
    if source == "sim" and cfg.auto_detect_lines and os.path.isfile(lines_path):
        if lines_file_has_week(lines_path, week):
            logger.info("Found %s with week %d rows; using real lines instead of the simulated league", lines_path, week)
            source = "csv"
        else:
            logger.warning("%s exists but has no rows for week %d; using the simulated league", lines_path, week)
    if source == "sim":
        season = cfg.season or nfl_season_year()
        return sides_from_games(simulate_week_games(season, week, cfg.seed, cfg.sim), "sim")
    if source == "csv":
        sides = load_lines_csv(lines_path, week=week)
    else:  # api
        api_key = cfg.api_key or os.environ.get("ODDS_API_KEY", "")
        sides = fetch_lines_from_odds_api(week, api_key, cfg.bookmaker)
    model_path = resolve_data_path(cfg.model_csv) if cfg.model_csv else None
    if model_path is None and os.path.isfile(resolve_data_path("model_probs.csv")):
        model_path = resolve_data_path("model_probs.csv")  # drop-in companion file
    if model_path:
        applied = apply_model_probs(sides, load_model_probs_csv(model_path))
        logger.info("Applied %d model probabilities from %s", applied, model_path)
    return sides


def find_parlays(
    week: int,
    top_n: Optional[int] = None,
    bankroll: Optional[float] = None,
    games: Optional[Sequence[SimulatedGame]] = None,
    config: Optional[FinderConfig] = None,
    **overrides: Any,
) -> List[Dict[str, Any]]:
    """Return this week's ranked parlay tickets as reporter-ready dicts.

    ``bankroll`` is accepted for interface compatibility (the reporter passes
    it) but ticket selection is bankroll-independent by design; sizing is the
    staking engine's job.
    """
    cfg = config or load_finder_config()
    if top_n is not None or overrides:
        data = {**asdict(cfg), **({"top_n": int(top_n)} if top_n is not None else {}), **overrides}
        data["sim"] = cfg.sim if "sim" not in overrides else overrides["sim"]
        cfg = FinderConfig.from_dict(data)
    try:
        week = int(week)
    except (TypeError, ValueError) as exc:
        raise FinderError(f"week must be an integer, got {week!r}") from exc
    sides = collect_sides(week, cfg, games)
    tickets = build_parlays(sides, week, cfg)
    logger.info("Week %d: %d market sides -> %d parlay ticket(s) [%s]", week, len(sides), len(tickets), cfg.source)
    return tickets


def write_parlays_json(tickets: Sequence[Dict[str, Any]], path: str, week: int) -> str:
    """Save tickets in the JSON hand-off shape weekly_reporter accepts."""
    payload = {"week": week, "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
               "generated_by": f"parlay_finder.py v{__version__}", "parlays": list(tickets)}
    abs_path = os.path.abspath(path)
    try:
        with open(abs_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except OSError as exc:
        raise FinderError(f"Could not write {abs_path}: {exc}") from exc
    return abs_path


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _selftest() -> int:
    failures = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal failures
        print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
        if not cond:
            failures += 1

    print("parlay_finder self-test")
    check(abs(normal_cdf(0) - 0.5) < 1e-12 and abs(normal_cdf(1.96) - 0.975) < 1e-3, "normal_cdf")
    fa, fb = devig_two_way(-110, -110)
    check(abs(fa - 0.5) < 1e-12 and abs(fa + fb - 1) < 1e-12, "devig -110/-110 -> 50/50")
    fa, fb = devig_two_way(-150, +130)
    check(fa > fb and abs(fa + fb - 1) < 1e-12, "devig favourite > dog, sums to 1")
    check(american_from_implied(0.5238) == -110, "implied 52.38% -> -110")
    check(american_from_implied(0.40) == 150, "implied 40% -> +150")
    check(parse_selection("Buffalo Bills -3.5") == ("spread", "Buffalo Bills", -3.5), "parse spread")
    check(parse_selection("over 44.5") == ("total", "Over", 44.5), "parse total")
    check(parse_selection("Kansas City Chiefs ML") == ("moneyline", "Kansas City Chiefs", None), "parse ML")
    try:
        parse_selection("???")
        check(False, "unparseable selection raises")
    except FinderError:
        check(True, "unparseable selection raises FinderError")

    # Simulation determinism and sanity
    g1 = simulate_week_games(2026, 6, seed=7)
    g2 = simulate_week_games(2026, 6, seed=7)
    check([g.to_dict() for g in g1] == [g.to_dict() for g in g2], "simulation is deterministic")
    check(len(g1) == 14 and len(simulate_week_games(2026, 1, 7)) == 16, "bye weeks have 14 games, others 16")
    teams = [t for g in g1 for t in (g.away, g.home)]
    check(len(teams) == len(set(teams)), "no team plays twice in a week")
    sides = sides_from_games(g1)
    check(len(sides) == 6 * len(g1), "six sides per game")
    check(all(0 < s.model_prob < 1 and 0 < s.fair_prob < 1 and 0 < (s.true_prob or 0.5) < 1 for s in sides), "probabilities in (0,1)")
    for g in g1:
        hs = [s for s in g.sides() if s.market == "spread"]
        check(abs(hs[0].model_prob + hs[1].model_prob - 1) < 1e-12, "spread sides complementary")
        break
    rng = random.Random(1)
    scores = [g1[0].simulate_final_score(rng) for _ in range(2000)]
    mean_margin = sum(h - a for a, h in scores) / len(scores)
    check(abs(mean_margin - g1[0].true_margin) < 1.5, f"simulated margins centre on truth ({mean_margin:.2f} vs {g1[0].true_margin:.2f})")

    # Ticket construction
    cfg = FinderConfig(source="sim", seed=7, season=2026, top_n=5)
    tickets = find_parlays(6, config=cfg)
    check(len(tickets) > 0, f"finder produces tickets ({len(tickets)})")
    check(all(t["edge"] > 0 for t in tickets), "all tickets +EV under the model")
    check(all(t["n_legs"] in (2, 3) for t in tickets), "ticket sizes are 2 or 3")
    check(all(len({(l['matchup']) for l in t['legs']}) == t["n_legs"] for t in tickets), "no same-game legs by default")
    for size in (2, 3):
        group = [t for t in tickets if t["n_legs"] == size]
        scores = [t["rank_score"] for t in group]
        check(scores == sorted(scores, reverse=True) and len(group) <= 5, f"{size}-leg tickets ranked by growth score, <= top_n")
    check(all(l["american_odds"] <= 300 for t in tickets for l in t["legs"]), "no leg priced longer than +300 by default")
    check(rank_score(0.6, 1.8, "growth") > rank_score(0.2, 5.4, "growth") and
          abs(rank_score(0.6, 1.8, "edge") - rank_score(0.2, 5.4, "edge")) < 1e-12,
          "growth ranking prefers the short-priced bet with equal raw edge")
    by_edge = find_parlays(6, config=FinderConfig(source="sim", seed=7, season=2026, top_n=5, rank_by="edge"))
    e = [t["edge"] for t in by_edge if t["n_legs"] == 2]
    check(e == sorted(e, reverse=True), "rank_by=edge orders by raw edge")
    usage: Dict[Tuple[str, str], int] = {}
    for t in tickets:
        for l in t["legs"]:
            usage[(l["matchup"], l["selection"])] = usage.get((l["matchup"], l["selection"]), 0) + 1
    check(max(usage.values()) <= cfg.max_tickets_per_leg, "diversification cap respected across all ticket sizes")
    game_use: Dict[str, int] = {}
    for t in tickets:
        for g in {l["matchup"] for l in t["legs"]}:
            game_use[g] = game_use.get(g, 0) + 1
    check(max(game_use.values()) <= cfg.max_tickets_per_game, "per-game exposure cap respected")
    t0 = tickets[0]
    prod_p = math.prod(l["p_true"] for l in t0["legs"])
    prod_d = math.prod(american_to_decimal(l["american_odds"]) for l in t0["legs"])
    check(abs(t0["p_true"] - prod_p) < 1e-5 and abs(t0["decimal_odds"] - prod_d) < 1e-4, "ticket p_true/odds are leg products")
    check("true_prob" not in json.dumps(tickets), "hidden truth never leaks into tickets")
    check(find_parlays("6", top_n=2, config=cfg) and len([t for t in find_parlays("6", top_n=2, config=cfg) if t["n_legs"] == 2]) <= 2,
          "top_n override and string week accepted")

    # Same-game allowed produces >= as many combos
    cfg_sg = FinderConfig(source="sim", seed=7, season=2026, top_n=50, allow_same_game=True)
    check(len(find_parlays(6, config=cfg_sg)) >= len(find_parlays(6, config=FinderConfig(source="sim", seed=7, season=2026, top_n=50))),
          "allow_same_game widens the candidate pool")

    # Edge premise: legs the default model selects carry positive TRUE edge on
    # average, while the null model (no private information) selects legs
    # whose true edge is negative (roughly minus the vig).
    def mean_true_edge(sim: SimConfig) -> float:
        fc = FinderConfig(source="sim", sim=sim)
        edges: List[float] = []
        for seed in (1, 2, 3):
            for wk in range(1, 19):
                for leg in select_candidate_legs(sides_from_games(simulate_week_games(2026, wk, seed, sim)), fc):
                    edges.append((leg.true_prob or 0.0) * leg.decimal_odds - 1.0)
        return sum(edges) / len(edges)
    edge_true, null_true = mean_true_edge(SimConfig()), mean_true_edge(SimConfig().no_edge())
    check(edge_true > 0.01, f"default model's picks have positive true edge ({edge_true:+.2%})")
    check(null_true < 0.0, f"null model's picks have negative true edge ({null_true:+.2%})")
    try:
        SimConfig(model_skill=1.5)
        check(False, "model_skill > 1 rejected")
    except FinderError:
        check(True, "model_skill > 1 rejected")

    # CSV round trip
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "lines.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["week", "away", "home", "market", "selection", "american_odds", "model_prob"])
            w.writerow([6, "Kansas City Chiefs", "Buffalo Bills", "spread", "Buffalo Bills -3.5", -110, 0.56])
            w.writerow([6, "Kansas City Chiefs", "Buffalo Bills", "spread", "Kansas City Chiefs +3.5", -110, 0.44])
            w.writerow([6, "Kansas City Chiefs", "Buffalo Bills", "total", "Over 44.5", -110, 0.55])
            w.writerow([6, "Kansas City Chiefs", "Buffalo Bills", "total", "Under 44.5", -110, 0.45])
            w.writerow([6, "Dallas Cowboys", "Philadelphia Eagles", "moneyline", "Philadelphia Eagles ML", -150, 0.64])
            w.writerow([6, "Dallas Cowboys", "Philadelphia Eagles", "moneyline", "Dallas Cowboys ML", "+130", ""])
            w.writerow([6, "Green Bay Packers", "Detroit Lions", "spread", "Detroit Lions -3", -105, 0.55])
            w.writerow([6, "Green Bay Packers", "Detroit Lions", "spread", "Green Bay Packers +3", -115, 0.45])
            w.writerow([7, "Miami Dolphins", "New York Jets", "total", "Over 41.5", -110, 0.6])
        sides_csv = load_lines_csv(path, week=6)
        check(len(sides_csv) == 8, f"CSV loads week-6 rows only ({len(sides_csv)})")
        dal = next(s for s in sides_csv if s.selection == "Dallas Cowboys ML")
        check(abs(dal.model_prob - dal.fair_prob) < 1e-12 and 0.4 < dal.fair_prob < 0.46, "blank model_prob -> de-vigged fair prob")
        cfg_csv = FinderConfig(source="csv", lines_csv=path, top_n=5)
        t_csv = find_parlays(6, config=cfg_csv)
        check(len(t_csv) >= 1 and all(t["source"] == "csv" for t in t_csv), f"CSV source builds tickets ({len(t_csv)})")
        try:
            load_lines_csv(os.path.join(tmp, "missing.csv"))
            check(False, "missing CSV raises")
        except FinderError:
            check(True, "missing CSV raises FinderError")
        # model_probs merge
        mp = os.path.join(tmp, "model_probs.csv")
        with open(mp, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["matchup", "selection", "model_prob"])
            w.writerow(["Dallas Cowboys @ Philadelphia Eagles", "Dallas Cowboys ML", 0.50])
        applied = apply_model_probs(sides_csv, load_model_probs_csv(mp))
        check(applied == 1 and abs(dal.model_prob - 0.5) < 1e-12, "model_probs.csv overrides model_prob")
        # JSON hand-off
        out = write_parlays_json(tickets, os.path.join(tmp, "p.json"), 6)
        with open(out, encoding="utf-8") as fh:
            check(len(json.load(fh)["parlays"]) == len(tickets), "write_parlays_json round trip")
        # Drop-in auto-detection: a lines.csv in the working directory switches "sim" to "csv"
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            auto = find_parlays(6, config=FinderConfig(source="sim", seed=7, season=2026, lines_csv=path))
            check(auto and all(t["source"] == "csv" for t in auto), "lines.csv auto-detected for week 6")
            sim_again = find_parlays(8, config=FinderConfig(source="sim", seed=7, season=2026, lines_csv=path))
            check(sim_again and all(t["source"] == "sim" for t in sim_again), "no week-8 rows -> simulated league")
            off = find_parlays(6, config=FinderConfig(source="sim", seed=7, season=2026, lines_csv=path, auto_detect_lines=False))
            check(all(t["source"] == "sim" for t in off), "auto_detect_lines=False keeps the simulation")
        finally:
            os.chdir(cwd)
        # week_inputs.csv in the working directory regenerates lines.csv and is picked up
        os.chdir(tmp)
        try:
            with open("week_inputs.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["week", "away", "home", "home_spread", "total", "away_ml", "home_ml", "model_home_margin", "model_home_win_prob"])
                w.writerow([9, "Tampa Bay Buccaneers", "Dallas Cowboys", -8.5, 47.5, "+360", -470, 11.4, 87])
                w.writerow([9, "Cincinnati Bengals", "Miami Dolphins", 6.5, 42.5, -340, "+270", -9.1, 19])
                w.writerow([9, "Buffalo Bills", "Los Angeles Rams", -3, 54.5, "+136", -162, 2.4, 60])
            from_inputs = find_parlays(9, config=FinderConfig(source="sim", seed=7, season=2026, lines_csv="lines.csv"))
            check(os.path.isfile("lines.csv") and from_inputs and all(t["source"] == "csv" for t in from_inputs),
                  "week_inputs.csv -> lines.csv -> real-line tickets")
            check(any("Dallas Cowboys -8.5" in l["selection"] for t in from_inputs for l in t["legs"]),
                  "FPI-derived spread leg appears in tickets")
        finally:
            os.chdir(cwd)
        # finder config from pipeline_config.json
        cfgp = os.path.join(tmp, "pipeline_config.json")
        with open(cfgp, "w", encoding="utf-8") as fh:
            json.dump({"bankroll": 1000, "finder": {"source": "sim", "seed": 99, "top_n": 3, "_comment": "x"}}, fh)
        loaded = load_finder_config(cfgp)
        check(loaded.seed == 99 and loaded.top_n == 3 and loaded.source == "sim", "load_finder_config reads 'finder' section")

    # Odds API parser with a fixture in the real v4 shape
    now = _dt.datetime(2026, 10, 7, 12, 0, tzinfo=_dt.timezone.utc)
    fixture = [{
        "id": "abc", "sport_key": "americanfootball_nfl", "commence_time": "2026-10-11T17:00:00Z",
        "home_team": "Buffalo Bills", "away_team": "Kansas City Chiefs",
        "bookmakers": [{"key": "draftkings", "title": "DraftKings", "markets": [
            {"key": "h2h", "outcomes": [{"name": "Buffalo Bills", "price": -165}, {"name": "Kansas City Chiefs", "price": 140}]},
            {"key": "spreads", "outcomes": [{"name": "Buffalo Bills", "price": -110, "point": -3.5}, {"name": "Kansas City Chiefs", "price": -110, "point": 3.5}]},
            {"key": "totals", "outcomes": [{"name": "Over", "price": -108, "point": 44.5}, {"name": "Under", "price": -112, "point": 44.5}]},
        ]}],
    }, {"id": "old", "commence_time": "2026-09-01T17:00:00Z", "home_team": "A", "away_team": "B", "bookmakers": []},
       {"garbage": True}]
    api_sides = parse_odds_api_response(fixture, week=6, now=now)
    check(len(api_sides) == 6, f"Odds API fixture -> 6 sides ({len(api_sides)})")
    check(any(s.selection == "Buffalo Bills -3.5" for s in api_sides) and any(s.selection == "Over 44.5" for s in api_sides)
          and any(s.selection == "Kansas City Chiefs ML" for s in api_sides), "Odds API selections formatted")
    try:
        fetch_lines_from_odds_api(6, "")
        check(False, "empty API key raises")
    except FinderError:
        check(True, "empty API key raises FinderError")

    print(f"\n{'ALL TESTS PASSED' if failures == 0 else f'{failures} TEST(S) FAILED'}")
    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="parlay_finder", description="Generate ranked NFL parlay candidates for a week.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--week", type=int, default=None, help="NFL week (default: upcoming week from today's date)")
    p.add_argument("--source", choices=("sim", "csv", "api"), default=None, help="Override finder.source")
    p.add_argument("--lines", default=None, help="CSV of lines for --source csv")
    p.add_argument("--model-csv", dest="model_csv", default=None, help="Optional model_probs.csv to merge")
    p.add_argument("--season", type=int, default=None, help="Season year for the simulation")
    p.add_argument("--seed", type=int, default=None, help="Simulation seed")
    p.add_argument("--top-n", type=int, dest="top_n", default=None, help="Tickets per leg size")
    p.add_argument("--min-leg-edge", type=float, dest="min_leg_edge", default=None, help="Minimum per-leg edge")
    p.add_argument("--allow-same-game", action="store_true", help="Permit correlated same-game legs")
    p.add_argument("--out", default=None, help="Write tickets to this JSON file")
    p.add_argument("--quiet", action="store_true", help="Do not print the ticket table")
    p.add_argument("--selftest", action="store_true", help="Run built-in tests")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _print_tickets(tickets: Sequence[Dict[str, Any]]) -> None:
    if not tickets:
        print("No +EV parlays found this week.")
        return
    for t in tickets:
        print(f"{t['ticket_id']:<12} {t['n_legs']}L  {t['american_odds']:>+6}  "
              f"model {t['p_true']:.1%}  fair {t['fair_prob']:.1%}  edge {t['edge']:+.1%}")
        for leg in t["legs"]:
            print(f"    {leg['matchup']:<40} {leg['selection']:<26} ({leg['american_odds']:+d})  p={leg['p_true']:.1%}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.selftest:
        return _selftest()
    overrides: Dict[str, Any] = {}
    if args.source:
        overrides["source"] = args.source
    if args.lines:
        overrides["lines_csv"] = args.lines
    if args.model_csv:
        overrides["model_csv"] = args.model_csv
    if args.season is not None:
        overrides["season"] = args.season
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.min_leg_edge is not None:
        overrides["min_leg_edge"] = args.min_leg_edge
    if args.allow_same_game:
        overrides["allow_same_game"] = True

    week = args.week
    if week is None:
        from weekly_reporter import estimate_nfl_week  # local import avoids a cycle at module load
        week = estimate_nfl_week(_dt.date.today())
        print(f"No --week given; using upcoming week {week}.")
    try:
        tickets = find_parlays(week, top_n=args.top_n, **overrides)
        if args.out:
            print(f"Wrote {len(tickets)} ticket(s) to {write_parlays_json(tickets, args.out, week)}")
    except (FinderError, StakingInputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not args.quiet:
        _print_tickets(tickets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
