#!/usr/bin/env python3
"""
parlay_finder.py
================

Candidate parlay generator for the NFL analytics pipeline, tuned for the
highest attainable raw win rate.

Each week this module produces a ranked list of **2-leg** parlay tickets,
every one carrying a model true probability and the book price, in exactly
the dictionary shape ``weekly_reporter.py`` consumes.

Absolute optimisation rules (enforced in code, not just defaults)
-----------------------------------------------------------------
1. **Strict 2-leg limit.** ``MAX_LEGS = 2``. Any configuration asking for
   3+ legs is clamped back to 2 with a warning; 3-leg variations are never
   generated because every extra leg multiplies variance and degrades the
   stability of the raw win rate.
2. **Premium edge filter.** A leg qualifies only when
   ``P_true - P_implied >= min_prob_gap`` (default **0.06**) **and**
   ``P_true >= min_leg_prob`` (default **0.68**). ``P_implied`` is the
   book's vig-inclusive implied probability ``1 / decimal_odds`` by default
   (``edge_basis = "implied"``); switch to the de-vigged market probability
   with ``edge_basis = "fair"`` if you prefer the gap measured against the
   market's opinion rather than the price you pay.
3. **Anti-correlation check.** Two legs from the same game are allowed only
   when they are positively correlated (``same_game_policy =
   "positive_only"``), e.g. Home Moneyline + Home Team Total Over. Negative
   correlation traps (Home ML + Away Team Total Over, Favourite spread +
   Under, ...) and contradictory pairs (both sides of one market) are
   rejected outright. See :func:`leg_correlation` for the full matrix.

Where the lines come from (``source``)
--------------------------------------
``sim`` (default)
    A deterministic simulated NFL season. 32 real franchises get hidden
    power ratings; each week a schedule is drawn, the "market" posts
    spreads, totals, moneylines and team totals with vig, and the "model"
    forms its own view (see :class:`SimConfig`). The hidden truth stays on
    the game objects **only** for ``backtester.py`` to simulate final
    scores; the finder never looks at it.
``csv``
    Real lines from ``lines.csv``::

        week,away,home,market,selection,american_odds[,model_prob][,player,position,blocked,model_note]

    ``market`` is ``spread`` / ``total`` / ``moneyline`` / ``team_total`` or any
    player-prop key declared in ``markets.json`` (``passing_yards``,
    ``anytime_td``, ...). ``selection`` is the line text, e.g.
    ``Buffalo Bills -3.5``, ``Over 44.5``, ``Kansas City Chiefs ML``,
    ``Buffalo Bills Over 24.5``, ``Dak Prescott Over 264.5 Passing Yards``,
    ``CeeDee Lamb Anytime TD``. Prop rows name the player in the ``player``
    column; ``blocked`` carries a reason the side can never be bet (ruled
    out, no projection). A blank ``model_prob`` falls back to the de-vigged
    market probability, which (correctly) yields no edge. ``week_inputs.csv``
    (games) and ``props_inputs.csv`` (player props) beside the script are
    expanded into ``lines.csv`` automatically by ``build_lines.py``.
``api``
    Live lines from The Odds API v4 (``ODDS_API_KEY`` or ``finder.api_key``),
    with model probabilities merged from an optional ``model_probs.csv``.

Integration
-----------
::

    import parlay_finder
    tickets = parlay_finder.find_parlays(week=6)               # list[dict]
    season = parlay_finder.simulate_season(2026, seed=7)
    tickets = parlay_finder.find_parlays(week=6, games=season[6])

CLI: ``python3 parlay_finder.py --week 6 --explain`` shows why legs were
rejected; ``python3 parlay_finder.py --selftest`` runs the built-in tests.
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
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import correlation_rules
import market_registry
from market_registry import GAME_MARKETS, RegistryError
from staking_engine import StakingInputError, american_to_decimal, decimal_to_american

__all__ = [
    "FinderError", "FinderConfig", "SimConfig", "MarketSide", "SimulatedGame", "ParsedSelection",
    "NFL_TEAMS", "MAX_LEGS", "MARGIN_SD", "TOTAL_SD", "TEAM_SD", "MARKETS", "all_markets",
    "normal_cdf", "devig_two_way", "implied_from_american", "american_from_implied",
    "simulate_season", "simulate_week_games", "sides_from_games",
    "parse_selection", "load_lines_csv", "load_model_probs_csv", "apply_model_probs",
    "parse_odds_api_response", "fetch_lines_from_odds_api", "fetch_odds_api_json",
    "leg_passes", "explain_legs", "explain_by_market", "format_market_table", "short_reason", "slate_summary",
    "MarketRule", "ParlayList", "select_candidate_legs", "leg_correlation", "rank_score",
    "build_parlays", "find_parlays", "collect_sides", "load_finder_config",
    "resolve_data_path", "lines_file_has_week", "write_parlays_json", "nfl_season_year",
]

__version__ = "2.0.0"

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "pipeline_config.json"

# ---------------------------------------------------------------------------
# League constants and the absolute rules
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

MAX_LEGS = 2                       # Rule 1: never more than two legs on a ticket
DEFAULT_MIN_LEG_PROB = 0.68        # Rule 2: each leg must be a 68%+ proposition
DEFAULT_MIN_PROB_GAP = 0.06        # Rule 2: P_true - P_implied >= 6 points
DEFAULT_SAME_GAME_POLICY = "positive_only"   # Rule 3

# Empirical NFL dispersion: margin ~ N(exp, 13.5), total ~ N(exp, 10). A team's
# score is (total +/- margin) / 2, so its sd is sqrt((10^2 + 13.5^2) / 4).
MARGIN_SD = 13.5
TOTAL_SD = 10.0
TEAM_SD = math.sqrt((TOTAL_SD ** 2 + MARGIN_SD ** 2) / 4.0)
HOME_FIELD_ADVANTAGE = 2.0
STANDARD_VIG = 0.045

MARKETS: Tuple[str, ...] = GAME_MARKETS   # the four built-in game markets


def all_markets() -> Tuple[str, ...]:
    """Every market the pipeline knows: the game markets plus the player props declared in markets.json."""
    return market_registry.registry().market_keys


class FinderError(RuntimeError):
    """Unrecoverable finder failure (bad CSV, API error, bad configuration)."""


# ---------------------------------------------------------------------------
# Maths helpers
# ---------------------------------------------------------------------------


def normal_cdf(x: float) -> float:
    """Standard normal CDF via the error function (no SciPy needed)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _round_half(x: float) -> float:
    return round(x * 2.0) / 2.0


def implied_from_american(american: int) -> float:
    return 1.0 / american_to_decimal(american)


def american_from_implied(implied: float) -> int:
    """Vig-inclusive probability -> American odds rounded to the nearest 5."""
    implied = min(0.985, max(0.015, implied))
    raw = -100.0 * implied / (1.0 - implied) if implied > 0.5 else 100.0 * (1.0 - implied) / implied
    rounded = int(round(raw / 5.0) * 5)
    if -100 < rounded < 100:
        rounded = 100 if rounded >= 0 else -105
    return rounded


def devig_two_way(odds_a: int, odds_b: int) -> Tuple[float, float]:
    """Remove the margin from a two-sided market (multiplicative); sums to 1."""
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
    market: str                 # spread | total | moneyline | team_total
    selection: str              # "Buffalo Bills -3.5" / "Over 44.5" / "Chiefs ML" / "Bills Over 24.5"
    american_odds: int
    model_prob: float           # the pipeline's probability estimate (what we bet on)
    fair_prob: float            # de-vigged market probability (the market's opinion)
    line: Optional[float] = None
    team: Optional[str] = None       # team the side is about (None for game totals)
    direction: Optional[str] = None  # "Over" / "Under" for totals and team totals
    true_prob: Optional[float] = None   # SIM ONLY: hidden truth for the backtester
    source: str = ""
    player: Optional[str] = None        # player props: the player the line is about
    player_id: Optional[str] = None     # nflverse id when the prop model matched the player
    position: Optional[str] = None      # QB / RB / WR / TE when known
    blocked: Optional[str] = None       # reason this side can never be bet (ruled out, no projection, ...)
    model_note: str = ""                # one-line provenance of model_prob (projection, spread, factors)

    @property
    def matchup(self) -> str:
        return f"{self.away} @ {self.home}"

    @property
    def game_key(self) -> Tuple[int, str, str]:
        return (self.week, self.away, self.home)

    @property
    def is_prop(self) -> bool:
        return market_registry.registry().is_prop(self.market)

    @property
    def experimental(self) -> bool:
        """True for markets markets.json still flags as experimental (every prop until the backtest clears it)."""
        return market_registry.registry().experimental(self.market)

    @property
    def high_variance(self) -> bool:
        return market_registry.registry().high_variance(self.market)

    @property
    def market_label(self) -> str:
        return market_registry.registry().label(self.market)

    @property
    def market_key(self) -> Tuple[Any, ...]:
        """Identifies a two-sided market: game + market + team (team totals) + player (props) + number."""
        return (self.game_key, self.market, self.team if self.market == "team_total" else None, self.player,
                abs(self.line) if self.line is not None else None)

    @property
    def decimal_odds(self) -> float:
        return american_to_decimal(self.american_odds)

    @property
    def implied_prob(self) -> float:
        return 1.0 / self.decimal_odds

    @property
    def edge(self) -> float:
        """Model edge per $1 staked: ``p_model * D - 1``."""
        return self.model_prob * self.decimal_odds - 1.0

    def prob_gap(self, basis: str = "implied") -> float:
        """``P_true - P_implied`` (default) or ``P_true - P_fair``."""
        return self.model_prob - (self.fair_prob if basis == "fair" else self.implied_prob)

    def to_leg_dict(self) -> Dict[str, Any]:
        """Leg dictionary in the weekly_reporter contract (no hidden truth)."""
        return {
            "matchup": self.matchup, "away": self.away, "home": self.home,
            "selection": self.selection, "market": self.market, "line": self.line,
            "team": self.team, "direction": self.direction,
            "american_odds": self.american_odds,
            "p_true": round(self.model_prob, 6), "fair_prob": round(self.fair_prob, 6),
            "implied_prob": round(self.implied_prob, 6),
            "prob_gap": round(self.prob_gap("implied"), 6), "leg_edge": round(self.edge, 6),
            "player": self.player, "player_id": self.player_id, "position": self.position,
            "market_label": self.market_label, "is_prop": self.is_prop,
            "experimental": self.experimental, "high_variance": self.high_variance,
            "model_note": self.model_note,
        }


class ParsedSelection(NamedTuple):
    """Structured reading of a selection string (``player`` is set for player props only)."""

    market: str
    team: Optional[str]
    line: Optional[float]
    direction: Optional[str]
    player: Optional[str] = None


@dataclass
class SimConfig:
    """Knobs for the simulated league.

    ``market = T + e_market`` and ``model = market + skill * (T - market) + e``:
    the model shares the market's public information and corrects a fraction
    ``model_skill`` of its error with private information, plus noise.
    ``model_skill = 0`` is a genuine null (the backtester's ``--no-edge``).
    """

    rating_sd: float = 6.0
    scoring_mean: float = 22.5
    scoring_sd: float = 2.5
    home_field: float = HOME_FIELD_ADVANTAGE
    market_noise_margin: float = 1.0
    market_noise_total: float = 1.5
    model_skill: float = 0.5
    model_noise_margin: float = 0.75
    model_noise_total: float = 1.0
    vig: float = STANDARD_VIG
    games_per_week: int = 16
    bye_weeks: Tuple[int, int] = (5, 14)
    team_totals: bool = True            # post team totals (needed for the correlation rule)

    def __post_init__(self) -> None:
        if not 0.0 <= self.model_skill <= 1.0:
            raise FinderError("model_skill must be within [0, 1]")
        for name in ("market_noise_margin", "market_noise_total", "model_noise_margin", "model_noise_total"):
            if getattr(self, name) < 0:
                raise FinderError(f"{name} cannot be negative")

    def no_edge(self) -> "SimConfig":
        return SimConfig(**{**asdict(self), "model_skill": 0.0})


@dataclass
class SimulatedGame:
    """A simulated matchup with posted lines, model numbers and hidden truth."""

    week: int
    away: str
    home: str
    true_margin: float          # home minus away expected margin (hidden)
    true_total: float           # expected total points (hidden)
    home_spread: float          # e.g. -3.5 when the home team is favoured
    total_line: float
    spread_odds_home: int
    spread_odds_away: int
    total_odds_over: int
    total_odds_under: int
    ml_odds_home: int
    ml_odds_away: int
    model_margin: float
    model_total: float
    home_tt_line: Optional[float] = None    # team totals (None when not posted)
    away_tt_line: Optional[float] = None
    home_tt_odds_over: int = -110
    home_tt_odds_under: int = -110
    away_tt_odds_over: int = -110
    away_tt_odds_under: int = -110

    @property
    def matchup(self) -> str:
        return f"{self.away} @ {self.home}"

    @staticmethod
    def _p_over(mean: float, threshold: float, sd: float) -> float:
        return 1.0 - normal_cdf((threshold - mean) / sd)

    def sides(self, source: str = "sim") -> List[MarketSide]:
        """Every market side with model, fair and true probabilities."""
        spread_point = -self.home_spread
        f_home_cover, f_away_cover = devig_two_way(self.spread_odds_home, self.spread_odds_away)
        f_over, f_under = devig_two_way(self.total_odds_over, self.total_odds_under)
        f_home_ml, f_away_ml = devig_two_way(self.ml_odds_home, self.ml_odds_away)
        m_home_cover = self._p_over(self.model_margin, spread_point, MARGIN_SD)
        t_home_cover = self._p_over(self.true_margin, spread_point, MARGIN_SD)
        m_over = self._p_over(self.model_total, self.total_line, TOTAL_SD)
        t_over = self._p_over(self.true_total, self.total_line, TOTAL_SD)
        m_home_win = self._p_over(self.model_margin, 0.0, MARGIN_SD)
        t_home_win = self._p_over(self.true_margin, 0.0, MARGIN_SD)
        c = dict(week=self.week, away=self.away, home=self.home, source=source)
        out = [
            MarketSide(market="spread", selection=f"{self.home} {self.home_spread:+g}", line=self.home_spread, team=self.home,
                       american_odds=self.spread_odds_home, model_prob=m_home_cover, fair_prob=f_home_cover, true_prob=t_home_cover, **c),
            MarketSide(market="spread", selection=f"{self.away} {-self.home_spread:+g}", line=-self.home_spread, team=self.away,
                       american_odds=self.spread_odds_away, model_prob=1 - m_home_cover, fair_prob=f_away_cover, true_prob=1 - t_home_cover, **c),
            MarketSide(market="total", selection=f"Over {self.total_line:g}", line=self.total_line, direction="Over",
                       american_odds=self.total_odds_over, model_prob=m_over, fair_prob=f_over, true_prob=t_over, **c),
            MarketSide(market="total", selection=f"Under {self.total_line:g}", line=self.total_line, direction="Under",
                       american_odds=self.total_odds_under, model_prob=1 - m_over, fair_prob=f_under, true_prob=1 - t_over, **c),
            MarketSide(market="moneyline", selection=f"{self.home} ML", team=self.home,
                       american_odds=self.ml_odds_home, model_prob=m_home_win, fair_prob=f_home_ml, true_prob=t_home_win, **c),
            MarketSide(market="moneyline", selection=f"{self.away} ML", team=self.away,
                       american_odds=self.ml_odds_away, model_prob=1 - m_home_win, fair_prob=f_away_ml, true_prob=1 - t_home_win, **c),
        ]
        for team, line, o_over, o_under, sign in (
            (self.home, self.home_tt_line, self.home_tt_odds_over, self.home_tt_odds_under, +1),
            (self.away, self.away_tt_line, self.away_tt_odds_over, self.away_tt_odds_under, -1),
        ):
            if line is None:
                continue
            m_pts = (self.model_total + sign * self.model_margin) / 2.0
            t_pts = (self.true_total + sign * self.true_margin) / 2.0
            f_o, f_u = devig_two_way(o_over, o_under)
            m_o = self._p_over(m_pts, line, TEAM_SD)
            t_o = self._p_over(t_pts, line, TEAM_SD)
            out.append(MarketSide(market="team_total", selection=f"{team} Over {line:g}", line=line, team=team, direction="Over",
                                  american_odds=o_over, model_prob=m_o, fair_prob=f_o, true_prob=t_o, **c))
            out.append(MarketSide(market="team_total", selection=f"{team} Under {line:g}", line=line, team=team, direction="Under",
                                  american_odds=o_under, model_prob=1 - m_o, fair_prob=f_u, true_prob=1 - t_o, **c))
        return out

    def simulate_final_score(self, rng: random.Random) -> Tuple[int, int]:
        """Draw ``(away_score, home_score)`` jointly so every market grades consistently."""
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
    rng = _season_rng(season, seed, "ratings")
    ratings = {t: max(-12.0, min(12.0, rng.gauss(0.0, cfg.rating_sd))) for t in NFL_TEAMS}
    scoring = {t: max(15.0, min(32.0, rng.gauss(cfg.scoring_mean, cfg.scoring_sd))) for t in NFL_TEAMS}
    return ratings, scoring


def _price_two_way(fair_a: float, vig: float) -> Tuple[int, int]:
    return american_from_implied(fair_a + vig / 2.0), american_from_implied((1.0 - fair_a) + vig / 2.0)


def simulate_week_games(season: int, week: int, seed: int = 7, cfg: Optional[SimConfig] = None) -> List[SimulatedGame]:
    """Deterministically simulate one week's slate (lines, model view, hidden truth)."""
    cfg = cfg or SimConfig()
    if week < 1:
        raise FinderError(f"week must be >= 1, got {week}")
    ratings, scoring = _team_strengths(season, seed, cfg)
    rng = _season_rng(season, seed, f"week{week}")
    teams = list(NFL_TEAMS)
    rng.shuffle(teams)
    n_games = cfg.games_per_week
    if cfg.bye_weeks[0] <= week <= cfg.bye_weeks[1]:
        n_games = max(1, cfg.games_per_week - 2)
    games: List[SimulatedGame] = []
    for i in range(n_games):
        away, home = teams[2 * i], teams[2 * i + 1]
        true_margin = ratings[home] - ratings[away] + cfg.home_field
        true_total = scoring[home] + scoring[away] + rng.gauss(0.0, 1.5)
        market_margin = true_margin + rng.gauss(0.0, cfg.market_noise_margin)
        market_total = true_total + rng.gauss(0.0, cfg.market_noise_total)
        model_margin = market_margin + cfg.model_skill * (true_margin - market_margin) + rng.gauss(0.0, cfg.model_noise_margin)
        model_total = market_total + cfg.model_skill * (true_total - market_total) + rng.gauss(0.0, cfg.model_noise_total)

        home_spread = -_round_half(market_margin)
        total_line = _round_half(market_total)
        spread_home, spread_away = _price_two_way(SimulatedGame._p_over(market_margin, -home_spread, MARGIN_SD), cfg.vig)
        over_odds, under_odds = _price_two_way(SimulatedGame._p_over(market_total, total_line, TOTAL_SD), cfg.vig)
        ml_home, ml_away = _price_two_way(SimulatedGame._p_over(market_margin, 0.0, MARGIN_SD), cfg.vig)

        tt: Dict[str, Any] = {}
        if cfg.team_totals:
            home_pts = (market_total + market_margin) / 2.0
            away_pts = (market_total - market_margin) / 2.0
            tt["home_tt_line"], tt["away_tt_line"] = _round_half(home_pts), _round_half(away_pts)
            tt["home_tt_odds_over"], tt["home_tt_odds_under"] = _price_two_way(SimulatedGame._p_over(home_pts, tt["home_tt_line"], TEAM_SD), cfg.vig)
            tt["away_tt_odds_over"], tt["away_tt_odds_under"] = _price_two_way(SimulatedGame._p_over(away_pts, tt["away_tt_line"], TEAM_SD), cfg.vig)

        games.append(SimulatedGame(
            week=week, away=away, home=home, true_margin=true_margin, true_total=true_total,
            home_spread=home_spread, total_line=total_line,
            spread_odds_home=spread_home, spread_odds_away=spread_away,
            total_odds_over=over_odds, total_odds_under=under_odds,
            ml_odds_home=ml_home, ml_odds_away=ml_away,
            model_margin=model_margin, model_total=model_total, **tt,
        ))
    return games


def simulate_season(season: int, seed: int = 7, weeks: int = 18, cfg: Optional[SimConfig] = None) -> Dict[int, List[SimulatedGame]]:
    return {w: simulate_week_games(season, w, seed, cfg) for w in range(1, weeks + 1)}


def sides_from_games(games: Iterable[SimulatedGame], source: str = "sim") -> List[MarketSide]:
    out: List[MarketSide] = []
    for g in games:
        out.extend(g.sides(source))
    return out


# ---------------------------------------------------------------------------
# Selection parsing and real lines (CSV)
# ---------------------------------------------------------------------------

_TOTAL_RE = re.compile(r"^(?P<dir>over|under)\s+(?P<line>\d+(?:\.\d+)?)$", re.IGNORECASE)
_TEAM_TOTAL_RE = re.compile(r"^(?P<team>.+?)\s+(?:team\s+total\s+)?(?P<dir>over|under)\s+(?P<line>\d+(?:\.\d+)?)$", re.IGNORECASE)
_ML_RE = re.compile(r"^(?P<team>.+?)\s+(?:ML|moneyline|money\s*line)$", re.IGNORECASE)
_SPREAD_RE = re.compile(r"^(?P<team>.+?)\s+(?P<line>[+-]\s?\d+(?:\.\d+)?)$")
_TT_PREFIX_RE = re.compile(r"^team\s+total:?\s+", re.IGNORECASE)


def parse_selection(selection: str, market_hint: str = "") -> ParsedSelection:
    """Classify a selection string.

    ``"Buffalo Bills -3.5"``       -> ``("spread", "Buffalo Bills", -3.5, None)``
    ``"Over 44.5"``                -> ``("total", None, 44.5, "Over")``
    ``"Kansas City Chiefs ML"``    -> ``("moneyline", "Kansas City Chiefs", None, None)``
    ``"Buffalo Bills Over 24.5"``  -> ``("team_total", "Buffalo Bills", 24.5, "Over")``

    Player props are recognised from the labels in ``markets.json``::

        "Dak Prescott Over 264.5 Passing Yards" -> ("passing_yards", None, 264.5, "Over", "Dak Prescott")
        "CeeDee Lamb Anytime TD"                -> ("anytime_td", None, None, "Yes", "CeeDee Lamb")

    A prop whose text omits the market label (``"Dak Prescott Over 264.5"``) is
    still understood when ``market_hint`` names the prop market.
    """
    text = _TT_PREFIX_RE.sub("", " ".join(str(selection).split()))
    reg = market_registry.registry()
    prop = reg.parse_prop_selection(text)
    hint_key = reg.resolve(market_hint) if market_hint else None
    if prop is None and hint_key is not None and reg.is_prop(hint_key):
        pm = reg.get(hint_key)
        assert pm is not None
        if pm.is_yes_no:
            no = re.match(r"^(?P<player>.+?)\s+no$", text, re.IGNORECASE)
            player = (no.group("player") if no else text).strip()
            if player:
                prop = (hint_key, player, None, "No" if no else "Yes")
        else:
            m = _TEAM_TOTAL_RE.match(text)  # "<player> Over 264.5"
            if m:
                prop = (hint_key, m.group("team").strip(), float(m.group("line")), m.group("dir").title())
    if prop is not None:
        market, player, line, direction = prop
        return ParsedSelection(market, None, line, direction, player)
    m = _TOTAL_RE.match(text)
    if m:
        return ParsedSelection("total", None, float(m.group("line")), m.group("dir").title())
    m = _TEAM_TOTAL_RE.match(text)
    if m:
        return ParsedSelection("team_total", m.group("team").strip(), float(m.group("line")), m.group("dir").title())
    m = _ML_RE.match(text)
    if m:
        return ParsedSelection("moneyline", m.group("team").strip(), None, None)
    m = _SPREAD_RE.match(text)
    if m:
        return ParsedSelection("spread", m.group("team").strip(), float(m.group("line").replace(" ", "")), None)
    if market_hint.strip().lower() in ("moneyline", "ml", "h2h"):
        return ParsedSelection("moneyline", text, None, None)
    raise FinderError(f"Cannot parse selection '{selection}'")


def _to_int_odds(value: Any) -> int:
    try:
        odds = int(float(str(value).strip().replace("+", "")))
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
            missing = {"week", "away", "home", "market", "selection", "american_odds"} - {f.strip().lower() for f in reader.fieldnames}
            if missing:
                raise FinderError(f"{path} is missing columns: {', '.join(sorted(missing))}")
            for raw in reader:
                rows.append({(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()})
    except OSError as exc:
        raise FinderError(f"Could not read {path}: {exc}") from exc

    parsed: List[Dict[str, Any]] = []
    for i, r in enumerate(rows, 2):
        if not any(r.values()):
            continue
        try:
            wk = int(float(r["week"]))
        except ValueError as exc:
            raise FinderError(f"{path} line {i}: bad week '{r['week']}'") from exc
        if week is not None and wk != week:
            continue
        hint = r.get("market", "")
        ps = parse_selection(r["selection"], hint)
        reg = market_registry.registry()
        hint_key = reg.resolve(hint) if hint else None
        if hint_key is not None and reg.is_prop(hint_key) and not reg.is_prop(ps.market):
            raise FinderError(f"{path} line {i}: market '{hint}' is a player prop but the selection '{r['selection']}' names no "
                              f"player; write it as '<player> Over 264.5 {reg.label(hint_key)}' or fill the player column")
        if hint and hint_key != ps.market:
            logger.warning("%s line %d: market '%s' disagrees with selection '%s'; using %s", path, i, hint, r["selection"], ps.market)
        player = (r.get("player") or "").strip() or ps.player
        if player and not reg.is_prop(ps.market):
            raise FinderError(f"{path} line {i}: player '{player}' given for the game market '{ps.market}'")
        if reg.is_prop(ps.market) and not player:
            raise FinderError(f"{path} line {i}: prop market '{ps.market}' needs a player")
        team = ps.team
        if reg.is_prop(ps.market):
            team = (r.get("team") or "").strip() or None   # the player's team, from the prop model or the props file
        odds = _to_int_odds(r["american_odds"])
        model_prob: Optional[float] = None
        mp = r.get("model_prob", "")
        if mp not in ("", None):
            try:
                model_prob = float(mp)
            except ValueError as exc:
                raise FinderError(f"{path} line {i}: bad model_prob '{mp}'") from exc
            if model_prob > 1.0:
                model_prob /= 100.0
            if not 0.0 < model_prob < 1.0:
                raise FinderError(f"{path} line {i}: model_prob must be in (0, 1), got {model_prob}")
        parsed.append(dict(week=wk, away=r["away"], home=r["home"], market=ps.market, selection=r["selection"],
                           line=ps.line, team=team, direction=ps.direction, odds=odds, model_prob=model_prob,
                           player=player or None, player_id=(r.get("player_id") or "").strip() or None,
                           position=(r.get("position") or "").strip().upper() or None,
                           blocked=(r.get("blocked") or "").strip() or None, model_note=(r.get("model_note") or "").strip()))

    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for p in parsed:
        key = (p["week"], p["away"], p["home"], p["market"], p["team"] if p["market"] == "team_total" else None,
               p["player"], abs(p["line"]) if p["line"] is not None else None)
        groups.setdefault(key, []).append(p)

    sides: List[MarketSide] = []
    missing_model = 0
    for members in groups.values():
        if len(members) == 2:
            fairs = list(devig_two_way(members[0]["odds"], members[1]["odds"]))
        else:
            fairs = [implied_from_american(m["odds"]) for m in members]
        for p, fair in zip(members, fairs):
            model_prob = p["model_prob"]
            if model_prob is None:
                model_prob, missing_model = fair, missing_model + 1
            sides.append(MarketSide(week=p["week"], away=p["away"], home=p["home"], market=p["market"], selection=p["selection"],
                                    american_odds=p["odds"], model_prob=model_prob, fair_prob=fair, line=p["line"],
                                    team=p["team"], direction=p["direction"], source=source, player=p["player"],
                                    player_id=p["player_id"], position=p["position"], blocked=p["blocked"],
                                    model_note=p["model_note"]))
    if missing_model:
        logger.warning("%d side(s) in %s had no model_prob; using de-vigged market probability (zero edge)", missing_model, path)
    if not sides:
        raise FinderError(f"No lines for week {week} in {path}" if week else f"No usable rows in {path}")
    return sides


def load_model_probs_csv(path: str) -> Dict[Tuple[str, str], float]:
    """``matchup,selection,model_prob`` -> {(matchup, selection): prob}."""
    if not os.path.isfile(path):
        raise FinderError(f"Model probabilities file not found: {path}")
    out: Dict[Tuple[str, str], float] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for i, raw in enumerate(csv.DictReader(fh), 2):
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
    applied = 0
    for s in sides:
        key = (s.matchup.lower(), s.selection.lower())
        if key in probs:
            s.model_prob, applied = probs[key], applied + 1
    return applied


# ---------------------------------------------------------------------------
# Real lines: The Odds API (v4)
# ---------------------------------------------------------------------------

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"


def fetch_odds_api_json(api_key: str, bookmakers: Optional[str] = None, timeout: float = 20.0) -> Any:
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


def parse_odds_api_response(events: Any, week: int, preferred_bookmaker: Optional[str] = None,
                            horizon_days: float = 8.0, now: Optional[_dt.datetime] = None) -> List[MarketSide]:
    """Odds API events -> MarketSides (de-vigged, zero-edge model until merged)."""
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
            if key == "team_totals":
                by_team: Dict[str, List[Dict[str, Any]]] = {}
                for o in outcomes:
                    by_team.setdefault(str(o.get("description", "")), []).append(o)
                pairs = list(by_team.values())
            else:
                pairs = [outcomes] if len(outcomes) == 2 else []
            for pair in pairs:
                if len(pair) != 2:
                    continue
                try:
                    o1, o2 = pair
                    odds1, odds2 = _to_int_odds(o1["price"]), _to_int_odds(o2["price"])
                except (KeyError, FinderError):
                    continue
                fair1, fair2 = devig_two_way(odds1, odds2)
                for o, odds, fair in ((o1, odds1, fair1), (o2, odds2, fair2)):
                    name, point = str(o.get("name", "")), o.get("point")
                    if key == "h2h":
                        sides.append(MarketSide(week=week, away=away, home=home, market="moneyline", selection=f"{name} ML", team=name,
                                                american_odds=odds, model_prob=fair, fair_prob=fair, source=f"api:{book.get('key', 'book')}"))
                    elif key == "spreads" and point is not None:
                        sides.append(MarketSide(week=week, away=away, home=home, market="spread", selection=f"{name} {float(point):+g}",
                                                line=float(point), team=name, american_odds=odds, model_prob=fair, fair_prob=fair,
                                                source=f"api:{book.get('key', 'book')}"))
                    elif key == "totals" and point is not None:
                        sides.append(MarketSide(week=week, away=away, home=home, market="total", selection=f"{name.title()} {float(point):g}",
                                                line=float(point), direction=name.title(), american_odds=odds, model_prob=fair, fair_prob=fair,
                                                source=f"api:{book.get('key', 'book')}"))
                    elif key == "team_totals" and point is not None:
                        team = str(o.get("description", ""))
                        sides.append(MarketSide(week=week, away=away, home=home, market="team_total",
                                                selection=f"{team} {name.title()} {float(point):g}", line=float(point), team=team,
                                                direction=name.title(), american_odds=odds, model_prob=fair, fair_prob=fair,
                                                source=f"api:{book.get('key', 'book')}"))
    return sides


def fetch_lines_from_odds_api(week: int, api_key: str, bookmaker: Optional[str] = None) -> List[MarketSide]:
    sides = parse_odds_api_response(fetch_odds_api_json(api_key, bookmaker), week, bookmaker)
    if not sides:
        raise FinderError("The Odds API returned no NFL games inside the next 8 days")
    return sides


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FinderConfig:
    """All tunables. The three absolute rules live in the defaults and in validation."""

    source: str = "sim"
    lines_csv: str = "lines.csv"
    model_csv: Optional[str] = None
    api_key: Optional[str] = None
    bookmaker: Optional[str] = None
    season: Optional[int] = None
    seed: int = 7
    sim: SimConfig = field(default_factory=SimConfig)
    leg_sizes: Tuple[int, ...] = (2,)                 # Rule 1: clamped to MAX_LEGS
    top_n: int = 10
    min_leg_prob: float = DEFAULT_MIN_LEG_PROB        # Rule 2: P_true >= 0.68
    min_prob_gap: float = DEFAULT_MIN_PROB_GAP        # Rule 2: P_true - P_implied >= 0.06
    edge_basis: str = "implied"                       # implied (1/D, vig included) | fair (de-vigged)
    min_leg_edge: float = 0.0                         # extra multiplicative-edge floor (p*D-1)
    max_leg_odds: int = 300
    max_candidate_legs: int = 12
    max_tickets_per_leg: int = 3
    max_tickets_per_game: int = 4
    same_game_policy: str = DEFAULT_SAME_GAME_POLICY  # Rule 3: never | positive_only | any
    allow_same_game: Optional[bool] = None            # legacy alias: False -> never, True -> any
    markets: Tuple[str, ...] = ("game", "props")      # market keys, labels or aliases; wildcards game | props | all
    market_rules: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # per-market overrides of the rule-2 thresholds
    rank_by: str = "growth"
    auto_detect_lines: bool = True

    def rule_for(self, market: str) -> "MarketRule":
        """The rule-2 thresholds in force for ``market``: its market_rules entry, else the global values."""
        o = self.market_rules.get(market, {})
        reg = market_registry.registry()
        extra = float(reg.model.get("high_variance_extra_gap", 0.0) or 0.0) if reg.high_variance(market) else 0.0
        return MarketRule(min_leg_prob=float(o.get("min_leg_prob", self.min_leg_prob)),
                          min_prob_gap=float(o.get("min_prob_gap", self.min_prob_gap)),
                          max_leg_odds=int(o.get("max_leg_odds", self.max_leg_odds)),
                          min_leg_edge=float(o.get("min_leg_edge", self.min_leg_edge)), extra_gap=extra)

    def __post_init__(self) -> None:
        self.source = (self.source or "sim").lower()
        if self.source not in ("sim", "csv", "api"):
            raise FinderError(f"finder.source must be sim | csv | api, got '{self.source}'")
        if self.top_n < 1:
            raise FinderError("finder.top_n must be >= 1")
        if isinstance(self.sim, dict):
            self.sim = SimConfig(**self.sim)
        if isinstance(self.leg_sizes, (list, tuple)):
            self.leg_sizes = tuple(int(n) for n in self.leg_sizes)
        else:
            self.leg_sizes = (int(self.leg_sizes),)
        if any(n < 2 for n in self.leg_sizes):
            raise FinderError("leg sizes must be >= 2")
        dropped = tuple(n for n in self.leg_sizes if n > MAX_LEGS)
        if dropped:
            logger.warning("Rule 1: %s-leg parlays requested but the finder is capped at %d legs; ignoring them",
                           "/".join(str(n) for n in dropped), MAX_LEGS)
        self.leg_sizes = tuple(sorted({n for n in self.leg_sizes if n <= MAX_LEGS})) or (MAX_LEGS,)
        if self.max_candidate_legs < MAX_LEGS:
            raise FinderError("max_candidate_legs must be >= 2")
        if not 0.0 <= self.min_leg_prob < 1.0:
            raise FinderError("min_leg_prob must be within [0, 1)")
        if not -1.0 < self.min_prob_gap < 1.0:
            raise FinderError("min_prob_gap must be a probability difference, e.g. 0.06")
        self.edge_basis = (self.edge_basis or "implied").lower()
        if self.edge_basis not in ("implied", "fair"):
            raise FinderError("edge_basis must be implied | fair")
        self.rank_by = (self.rank_by or "growth").lower()
        if self.rank_by not in ("growth", "edge"):
            raise FinderError("rank_by must be growth | edge")
        if self.allow_same_game is not None:
            self.same_game_policy = "any" if self.allow_same_game else "never"
        self.same_game_policy = (self.same_game_policy or DEFAULT_SAME_GAME_POLICY).lower()
        if self.same_game_policy not in ("never", "positive_only", "any"):
            raise FinderError("same_game_policy must be never | positive_only | any")
        if isinstance(self.markets, str):
            self.markets = (self.markets,)
        try:
            self.markets = market_registry.registry().expand_markets(tuple(str(m) for m in self.markets))
        except RegistryError as exc:
            raise FinderError(f"finder.markets: {exc}") from exc
        self.market_rules = _normalize_market_rules(self.market_rules)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FinderConfig":
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed and not k.startswith("_")})


class MarketRule(NamedTuple):
    """Rule-2 thresholds for one market. ``extra_gap`` is the surcharge high-variance markets pay."""

    min_leg_prob: float
    min_prob_gap: float
    max_leg_odds: int
    min_leg_edge: float
    extra_gap: float = 0.0

    @property
    def required_gap(self) -> float:
        return self.min_prob_gap + self.extra_gap


_RULE_KEYS: Tuple[str, ...] = ("min_leg_prob", "min_prob_gap", "max_leg_odds", "min_leg_edge")


def _normalize_market_rules(raw: Any) -> Dict[str, Dict[str, float]]:
    """``finder.market_rules`` -> ``{canonical market: {setting: value}}``.

    Keys may be market keys, labels, aliases or the wildcards ``props`` /
    ``game`` / ``all``; an explicit market beats a wildcard. Unknown markets,
    unknown settings and out-of-range values raise :class:`FinderError`.
    """
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise FinderError("finder.market_rules must be an object keyed by market")
    reg = market_registry.registry()
    wildcard: Dict[str, Dict[str, float]] = {}
    explicit: Dict[str, Dict[str, float]] = {}
    for key, value in raw.items():
        if str(key).startswith("_"):
            continue
        if not isinstance(value, dict):
            raise FinderError(f"finder.market_rules.{key} must be an object such as {{\"min_leg_prob\": 0.6}}")
        rule: Dict[str, float] = {}
        for k, v in value.items():
            if str(k).startswith("_"):
                continue
            if k not in _RULE_KEYS:
                raise FinderError(f"finder.market_rules.{key}: unknown setting '{k}' (use {', '.join(_RULE_KEYS)})")
            try:
                rule[k] = float(v)
            except (TypeError, ValueError) as exc:
                raise FinderError(f"finder.market_rules.{key}.{k} must be a number, got {v!r}") from exc
        if "min_leg_prob" in rule and not 0.0 <= rule["min_leg_prob"] < 1.0:
            raise FinderError(f"finder.market_rules.{key}.min_leg_prob must be within [0, 1)")
        if "min_prob_gap" in rule and not -1.0 < rule["min_prob_gap"] < 1.0:
            raise FinderError(f"finder.market_rules.{key}.min_prob_gap must be a probability difference, e.g. 0.04")
        if "max_leg_odds" in rule:
            rule["max_leg_odds"] = float(int(rule["max_leg_odds"]))
        text = str(key).strip().lower()
        try:
            if text in ("props", "game", "all"):
                for m in reg.expand_markets([text]):
                    wildcard[m] = {**wildcard.get(m, {}), **rule}
            else:
                m = reg.resolve(text)
                if m is None:
                    raise FinderError(f"finder.market_rules: unknown market '{key}' (known: {', '.join(reg.market_keys)})")
                explicit[m] = {**explicit.get(m, {}), **rule}
        except RegistryError as exc:
            raise FinderError(f"finder.market_rules: {exc}") from exc
    out = dict(wildcard)
    for m, rule in explicit.items():
        out[m] = {**out.get(m, {}), **rule}
    return out


def nfl_season_year(date: Optional[_dt.date] = None) -> int:
    date = date or _dt.date.today()
    return date.year if date.month >= 3 else date.year - 1


def resolve_data_path(path: str) -> str:
    if os.path.isabs(path) or os.path.isfile(path):
        return path
    beside = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    return beside if os.path.isfile(beside) else path


def lines_file_has_week(path: str, week: int) -> bool:
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
    for candidate in (path, os.path.join(os.path.dirname(os.path.abspath(__file__)), path)):
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


# ---------------------------------------------------------------------------
# Leg filters (Rule 2) and the correlation matrix (Rule 3)
# ---------------------------------------------------------------------------


def leg_passes(s: MarketSide, cfg: FinderConfig) -> Tuple[bool, str]:
    """Apply the premium edge filter to one side -> ``(ok, reason_if_not)``."""
    if s.market not in cfg.markets:
        return False, "market not enabled"
    if s.blocked:
        return False, s.blocked
    rule = cfg.rule_for(s.market)
    if s.american_odds > rule.max_leg_odds:
        return False, f"priced longer than +{rule.max_leg_odds}"
    if s.model_prob < rule.min_leg_prob:
        return False, f"model probability below {rule.min_leg_prob * 100:g}%"
    if s.prob_gap(cfg.edge_basis) < rule.required_gap:
        return False, (f"gap over {cfg.edge_basis} probability below {rule.required_gap * 100:g}%"
                       + (" (high-variance market)" if rule.extra_gap else ""))
    if s.edge <= rule.min_leg_edge:
        return False, "edge not positive"
    return True, ""


def explain_legs(sides: Sequence[MarketSide], cfg: FinderConfig) -> Dict[str, int]:
    """Count sides by the first filter they fail (``"passed"`` for survivors)."""
    counts: Dict[str, int] = {}
    for s in sides:
        ok, reason = leg_passes(s, cfg)
        key = "passed" if ok else reason
        counts[key] = counts.get(key, 0) + 1
    return counts


def short_reason(reason: str) -> str:
    """Compact form of a leg_passes reason for tables."""
    r = reason.lower()
    if r.startswith("model probability below"):
        return "below floor"
    if r.startswith("gap over"):
        return "gap too small"
    if r.startswith("priced longer"):
        return "price too long"
    if r == "edge not positive":
        return "no edge"
    if r == "market not enabled":
        return "market off"
    return reason


def explain_by_market(sides: Sequence[MarketSide], cfg: FinderConfig) -> Dict[str, Dict[str, Any]]:
    """Per-market view of the rule-2 filter: how many sides, how many pass, under which thresholds, and why not.

    Returns ``{market: {label, is_prop, experimental, sides, passed, blocked, min_leg_prob,
    min_prob_gap, best_prob, best_gap, reasons}}`` in registry order (game markets first).
    ``best_prob`` / ``best_gap`` are the strongest unblocked side's numbers, the hint for
    tuning ``finder.market_rules``.
    """
    reg = market_registry.registry()
    out: Dict[str, Dict[str, Any]] = {}
    for s in sides:
        row = out.get(s.market)
        if row is None:
            rule = cfg.rule_for(s.market)
            row = out[s.market] = {
                "market": s.market, "label": reg.label(s.market), "is_prop": reg.is_prop(s.market),
                "experimental": reg.experimental(s.market), "enabled": s.market in cfg.markets,
                "sides": 0, "passed": 0, "blocked": 0,
                "min_leg_prob": rule.min_leg_prob, "min_prob_gap": rule.required_gap,
                "best_prob": None, "best_gap": None, "reasons": {},
            }
        row["sides"] += 1
        ok, reason = leg_passes(s, cfg)
        if ok:
            row["passed"] += 1
        else:
            row["reasons"][reason] = row["reasons"].get(reason, 0) + 1
            if s.blocked:
                row["blocked"] += 1
        if not s.blocked:
            gap = s.prob_gap(cfg.edge_basis)
            row["best_prob"] = s.model_prob if row["best_prob"] is None else max(row["best_prob"], s.model_prob)
            row["best_gap"] = gap if row["best_gap"] is None else max(row["best_gap"], gap)
    order = {m: i for i, m in enumerate(reg.market_keys)}
    return dict(sorted(out.items(), key=lambda kv: order.get(kv[0], 999)))


def format_market_table(by_market: Dict[str, Dict[str, Any]], width: int = 80) -> List[str]:
    """Render :func:`explain_by_market` as fixed-width text lines (fits an 80-column report)."""
    if not by_market:
        return ["  (no market sides this week)"]
    mw, sw, pw, rw, bw, gw = 18, 5, 4, 7, 6, 8
    reason_w = max(10, width - (2 + mw + 2 + sw + 2 + pw + 2 + rw + 2 + bw + 2 + gw + 2))

    def fit(text: str, w: int) -> str:
        text = str(text)
        return text if len(text) <= w else text[: max(0, w - 3)] + "..."

    lines = [f"  {'Market':<{mw}}  {'Sides':>{sw}}  {'Pass':>{pw}}  {'Rule':>{rw}}  {'Best P':>{bw}}  {'Best gap':>{gw}}  {'Top reason':<{reason_w}}".rstrip(),
             f"  {'-' * mw}  {'-' * sw}  {'-' * pw}  {'-' * rw}  {'-' * bw}  {'-' * gw}  {'-' * reason_w}"]
    for row in by_market.values():
        name = row["label"] + ("*" if row.get("experimental") else "")
        rule = f"{row['min_leg_prob'] * 100:g}/{row['min_prob_gap'] * 100:g}"
        best_p = f"{row['best_prob'] * 100:.1f}%" if row.get("best_prob") is not None else "n/a"
        best_g = f"{row['best_gap'] * 100:+.1f}pt" if row.get("best_gap") is not None else "n/a"
        if row["passed"] == row["sides"]:
            top = "all pass"
        else:
            reason, n = max(row["reasons"].items(), key=lambda kv: kv[1]) if row["reasons"] else ("", 0)
            top = f"{short_reason(reason)} ({n})"
        lines.append(f"  {fit(name, mw):<{mw}}  {row['sides']:>{sw}}  {row['passed']:>{pw}}  {rule:>{rw}}  {best_p:>{bw}}  {best_g:>{gw}}  {fit(top, reason_w)}".rstrip())
    lines.append("  Rule = leg probability floor % / edge gap points, from finder.market_rules in")
    lines.append("  pipeline_config.json (global min_leg_prob / min_prob_gap otherwise).")
    lines.append("  * = experimental prop model.")
    return lines


class ParlayList(list):
    """The finder's ticket list; ``slate_summary`` carries the per-market filter diagnostics."""

    slate_summary: Optional[Dict[str, Any]] = None


def slate_summary(sides: Sequence[MarketSide], cfg: FinderConfig, week: int) -> Dict[str, Any]:
    """Diagnostics for the week's slate: side counts, legs passing per market and the thresholds used."""
    by_market = explain_by_market(sides, cfg)
    candidates = select_candidate_legs(sides, cfg)
    return {
        "week": week, "source": sides[0].source if sides else cfg.source, "sides": len(sides),
        "passed": sum(r["passed"] for r in by_market.values()), "candidates": len(candidates),
        "markets": len(by_market), "edge_basis": cfg.edge_basis,
        "global_rule": {"min_leg_prob": cfg.min_leg_prob, "min_prob_gap": cfg.min_prob_gap},
        "by_market": by_market, "table": format_market_table(by_market),
    }


def rank_score(p_true: float, decimal_odds: float, rank_by: str = "growth") -> float:
    """``edge`` -> p*D-1; ``growth`` -> edge^2/(D-1), the Kelly log-growth proxy."""
    edge = p_true * decimal_odds - 1.0
    if rank_by == "edge" or edge <= 0.0:
        return edge
    return edge * edge / (decimal_odds - 1.0)


def select_candidate_legs(sides: Sequence[MarketSide], cfg: FinderConfig) -> List[MarketSide]:
    """Legs clearing Rule 2, one side per market, best ranked first."""
    best: Dict[Tuple[Any, ...], MarketSide] = {}
    for s in sides:
        if not leg_passes(s, cfg)[0]:
            continue
        key = s.market_key
        if key not in best or s.edge > best[key].edge:
            best[key] = s
    ranked = sorted(best.values(), key=lambda s: (-rank_score(s.model_prob, s.decimal_odds, cfg.rank_by), -s.model_prob, s.selection))
    return ranked[: cfg.max_candidate_legs]


def leg_correlation_detail(a: MarketSide, b: MarketSide) -> Tuple[str, str, str]:
    """Correlation class of two legs -> ``(label, rule_id, reason)``.

    Legs from different games are ``neutral``. Legs from the same game are
    classified by the table in ``correlation_rules.json`` (see
    ``correlation_rules.py``): ``positive``, ``negative``, ``neutral``,
    ``exclusive`` (both cannot win) or ``redundant`` (double-counts one
    outcome). The game-market rows reproduce the classic matrix:

    ============================  ==========  =========
    pair                          same team   other team
    ============================  ==========  =========
    spread + moneyline            positive    negative
    spread/ML + team total Over   positive    negative
    spread/ML + team total Under  negative    positive
    game total + team total       positive when both Over or both Under, else negative
    game total + spread/ML        neutral (direction depends on favourite status)
    both team totals              neutral
    same market, same side        positive (duplicate line); opposite side -> exclusive
    ============================  ==========  =========

    Player props add the rows the table documents (a quarterback with his
    receiver, a rusher with his team's side, a scorer with his team total, ...).
    """
    if a.game_key != b.game_key:
        return "neutral", "", "different games"
    try:
        return correlation_rules.evaluate(a, b)
    except correlation_rules.RuleError as exc:
        raise FinderError(str(exc)) from exc


def leg_correlation(a: MarketSide, b: MarketSide) -> str:
    """Correlation label of two legs (``positive`` / ``negative`` / ``neutral`` / ``exclusive`` / ``redundant``)."""
    return leg_correlation_detail(a, b)[0]


SGP_PRICING_NOTE = ("both legs come from one game: books price this as a same-game parlay, not at the independent "
                    "product shown here; confirm the payout at the book before placing it")


# ---------------------------------------------------------------------------
# Ticket construction
# ---------------------------------------------------------------------------


def _ticket_from_legs(legs: Sequence[MarketSide], week: int, ticket_id: str, source: str,
                      rank_by: str, correlation: Optional[str], correlation_reason: str = "") -> Dict[str, Any]:
    p_true = math.prod(leg.model_prob for leg in legs)
    decimal_odds = math.prod(leg.decimal_odds for leg in legs)
    fair = math.prod(leg.fair_prob for leg in legs)
    edge = p_true * decimal_odds - 1.0
    legs_text = ", ".join(f"{leg.selection} ({leg.model_prob:.0%}, +{leg.prob_gap('implied') * 100:.1f} pts)" for leg in legs)
    note = f"model {p_true:.1%} vs market fair {fair:.1%}; legs: {legs_text}"
    if correlation:
        note += (f"; same-game pair, {correlation} correlation ({correlation_reason}); joint probability shown as the independent "
                 f"product, a conservative floor; {SGP_PRICING_NOTE}")
    groups = {"prop" if leg.is_prop else "game" for leg in legs}
    experimental = any(leg.experimental for leg in legs)
    if experimental:
        note += "; EXPERIMENTAL prop model on " + ", ".join(leg.player or leg.selection for leg in legs if leg.experimental)
    return {
        "ticket_id": ticket_id, "week": week, "n_legs": len(legs),
        "legs": [leg.to_leg_dict() for leg in legs],
        "p_true": round(p_true, 6), "fair_prob": round(fair, 6),
        "decimal_odds": round(decimal_odds, 6), "american_odds": decimal_to_american(decimal_odds),
        "edge": round(edge, 6), "rank_score": round(rank_score(p_true, decimal_odds, rank_by), 6),
        "same_game": correlation is not None, "correlation": correlation,
        "correlation_reason": correlation_reason if correlation else None,
        "sgp_required": correlation is not None, "pricing_note": SGP_PRICING_NOTE if correlation else "",
        "market_group": groups.pop() if len(groups) == 1 else "mixed", "experimental": experimental,
        "notes": note, "source": source,
    }


def _combo_allowed(combo: Sequence[MarketSide], cfg: FinderConfig) -> Tuple[bool, Optional[str], str]:
    """Rule 3 for one combination -> ``(allowed, correlation_label_or_None, reason)``.

    Exclusive and redundant pairs are rejected under every policy; ``never``
    rejects every same-game pair; ``positive_only`` keeps positive pairs only.
    """
    label: Optional[str] = None
    reason = ""
    for a, b in itertools.combinations(combo, 2):
        if a.game_key != b.game_key:
            continue
        corr, _rule_id, why = leg_correlation_detail(a, b)
        if corr in ("exclusive", "redundant"):
            return False, corr, why
        if cfg.same_game_policy == "never":
            return False, corr, "same-game pairs are disabled (same_game_policy = never)"
        if cfg.same_game_policy == "positive_only" and corr != "positive":
            return False, corr, why
        label, reason = corr, why
    return True, label, reason


def build_parlays(sides: Sequence[MarketSide], week: int, cfg: Optional[FinderConfig] = None) -> List[Dict[str, Any]]:
    """Combine qualifying legs into ranked, diversified 2-leg tickets."""
    cfg = cfg or FinderConfig()
    candidates = select_candidate_legs(sides, cfg)
    source = candidates[0].source if candidates else (sides[0].source if sides else cfg.source)
    if len(candidates) < min(cfg.leg_sizes):
        logger.info("Week %d: %d qualifying leg(s); no parlays possible", week, len(candidates))
        return []

    tickets: List[Dict[str, Any]] = []
    usage: Dict[Tuple[Any, ...], int] = {}
    game_usage: Dict[Tuple[Any, ...], int] = {}
    for size in cfg.leg_sizes:
        combos: List[Tuple[float, Tuple[MarketSide, ...], Optional[str], str]] = []
        for combo in itertools.combinations(candidates, size):
            allowed, corr, why = _combo_allowed(combo, cfg)
            if not allowed:
                continue
            p = math.prod(leg.model_prob for leg in combo)
            d = math.prod(leg.decimal_odds for leg in combo)
            if p * d - 1.0 <= 0:
                continue
            combos.append((rank_score(p, d, cfg.rank_by), combo, corr, why))
        combos.sort(key=lambda item: (-item[0], -math.prod(leg.model_prob for leg in item[1])))
        kept = 0
        for _score, combo, corr, why in combos:
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
            tickets.append(_ticket_from_legs(combo, week, f"W{week:02d}-{size}L-{kept:02d}", source, cfg.rank_by, corr, why))
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
    uses_default_lines = os.path.basename(cfg.lines_csv) == cfg.lines_csv == "lines.csv"
    inputs_path = resolve_data_path("week_inputs.csv")
    if cfg.auto_detect_lines and uses_default_lines and os.path.isfile(inputs_path):
        try:
            from build_lines import BuildLinesError, build_lines_csv
            lines_path = build_lines_csv(inputs_path, os.path.join(os.path.dirname(os.path.abspath(inputs_path)), "lines.csv"))
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
        return sides_from_games(simulate_week_games(cfg.season or nfl_season_year(), week, cfg.seed, cfg.sim), "sim")
    if source == "csv":
        sides = load_lines_csv(lines_path, week=week)
    else:
        sides = fetch_lines_from_odds_api(week, cfg.api_key or os.environ.get("ODDS_API_KEY", ""), cfg.bookmaker)
    model_path = resolve_data_path(cfg.model_csv) if cfg.model_csv else None
    if model_path is None and os.path.isfile(resolve_data_path("model_probs.csv")):
        model_path = resolve_data_path("model_probs.csv")
    if model_path:
        logger.info("Applied %d model probabilities from %s", apply_model_probs(sides, load_model_probs_csv(model_path)), model_path)
    return sides


def find_parlays(week: int, top_n: Optional[int] = None, bankroll: Optional[float] = None,
                 games: Optional[Sequence[SimulatedGame]] = None, config: Optional[FinderConfig] = None,
                 **overrides: Any) -> List[Dict[str, Any]]:
    """Return this week's ranked 2-leg tickets as reporter-ready dicts.

    ``bankroll`` is accepted for interface compatibility; ticket selection is
    bankroll-independent by design (sizing is the staking engine's job).
    """
    cfg = config or load_finder_config()
    if top_n is not None or overrides:
        data = {**asdict(cfg), **({"top_n": int(top_n)} if top_n is not None else {}), **overrides}
        data["sim"] = overrides.get("sim", cfg.sim)
        cfg = FinderConfig.from_dict(data)
    try:
        week = int(week)
    except (TypeError, ValueError) as exc:
        raise FinderError(f"week must be an integer, got {week!r}") from exc
    sides = collect_sides(week, cfg, games)
    tickets = ParlayList(build_parlays(sides, week, cfg))
    tickets.slate_summary = slate_summary(sides, cfg, week)
    if tickets:
        logger.info("Week %d: %d market sides -> %d ticket(s) [%s]", week, len(sides), len(tickets), cfg.source)
    else:
        why = ", ".join(f"{v} {k}" for k, v in sorted(explain_legs(sides, cfg).items(), key=lambda kv: -kv[1]) if k != "passed")
        logger.info("Week %d: %d market sides -> no ticket cleared the filters (%s)", week, len(sides), why or "no sides")
    return tickets


def write_parlays_json(tickets: Sequence[Dict[str, Any]], path: str, week: int) -> str:
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
    check(american_from_implied(0.5238) == -110 and american_from_implied(0.40) == 150, "implied -> american")
    check(parse_selection("Buffalo Bills -3.5")[:4] == ("spread", "Buffalo Bills", -3.5, None), "parse spread")
    check(parse_selection("over 44.5")[:4] == ("total", None, 44.5, "Over"), "parse total")
    check(parse_selection("Kansas City Chiefs ML")[:4] == ("moneyline", "Kansas City Chiefs", None, None), "parse ML")
    check(parse_selection("Buffalo Bills Over 24.5")[:4] == ("team_total", "Buffalo Bills", 24.5, "Over"), "parse team total")
    check(parse_selection("Team Total: Buffalo Bills under 24.5")[:4] == ("team_total", "Buffalo Bills", 24.5, "Under"), "parse prefixed team total")
    try:
        parse_selection("???")
        check(False, "unparseable selection raises")
    except FinderError:
        check(True, "unparseable selection raises FinderError")

    # Player props: the selection grammar comes from markets.json, so a new prop type needs no parser change
    check(parse_selection("Dak Prescott Over 264.5 Passing Yards") == ("passing_yards", None, 264.5, "Over", "Dak Prescott"), "parse prop over/under")
    check(parse_selection("CeeDee Lamb Anytime TD") == ("anytime_td", None, None, "Yes", "CeeDee Lamb"), "parse yes/no prop")
    check(parse_selection("Dak Prescott Under 264.5", "passing_yards") == ("passing_yards", None, 264.5, "Under", "Dak Prescott"), "market hint supplies a missing prop label")
    check(parse_selection("CeeDee Lamb", "anytime td").player == "CeeDee Lamb" and parse_selection("CeeDee Lamb No", "anytime_td").direction == "No", "hinted yes/no prop")
    check(parse_selection("Buffalo Bills Over 24.5").player is None, "game markets carry no player")
    prop_side = MarketSide(week=5, away="A", home="H", market="receiving_yards", selection="X Over 70.5 Receiving Yards", american_odds=-110,
                           model_prob=0.6, fair_prob=0.5, line=70.5, direction="Over", player="X")
    prop_side2 = MarketSide(**{**asdict(prop_side), "player": "Y", "selection": "Y Over 70.5 Receiving Yards"})
    game_side = MarketSide(week=5, away="A", home="H", market="spread", selection="H -3.5", american_odds=-110, model_prob=0.6, fair_prob=0.5, line=-3.5, team="H")
    check(prop_side.is_prop and prop_side.experimental and not prop_side.high_variance and prop_side.market_label == "Receiving Yards", "prop side knows its registry flags")
    check(not game_side.is_prop and not game_side.experimental and game_side.market_label == "Spread", "game side is not experimental")
    check(prop_side.market_key != prop_side2.market_key, "market_key separates two players on the same prop")
    ld = prop_side.to_leg_dict()
    check(ld["player"] == "X" and ld["experimental"] is True and ld["market_label"] == "Receiving Yards" and ld["is_prop"] is True, "leg dict carries prop fields")
    blocked_side = MarketSide(**{**asdict(prop_side), "blocked": "ruled out (injury report: Out)"})
    ok_b, why_b = leg_passes(blocked_side, FinderConfig(min_leg_prob=0.0, min_prob_gap=0.0))
    check(not ok_b and "ruled out" in why_b, "blocked side fails the filter with its reason")
    check(FinderConfig().markets == all_markets() and "passing_yards" in FinderConfig(markets=["spread", "props"]).markets, "markets default to everything; 'props' wildcard expands")
    check(FinderConfig(markets=["Pass Yds", "ML"]).markets == ("passing_yards", "moneyline"), "market aliases resolve in config")
    try:
        FinderConfig(markets=["nope"])
        check(False, "unknown market in config raises")
    except FinderError:
        check(True, "unknown market in config raises FinderError")

    # Per-market thresholds (finder.market_rules): explicit beats wildcard, aliases resolve, validation
    mr = FinderConfig(market_rules={"props": {"min_leg_prob": 0.55, "min_prob_gap": 0.04}, "Receptions": {"min_leg_prob": 0.6},
                                    "_comment": "ignored"})
    check(mr.rule_for("passing_yards") == MarketRule(0.55, 0.04, 300, 0.0, 0.0), "props wildcard sets every prop market")
    check(mr.rule_for("receptions").min_leg_prob == 0.6 and mr.rule_for("receptions").min_prob_gap == 0.04, "explicit market beats the wildcard per setting")
    check(mr.rule_for("spread") == MarketRule(0.68, 0.06, 300, 0.0, 0.0), "game markets keep the global rule")
    check(mr.rule_for("first_td").extra_gap > 0 and mr.rule_for("first_td").required_gap > 0.04, "high-variance market pays an extra gap")
    check(FinderConfig.from_dict(asdict(mr)).rule_for("receptions") == mr.rule_for("receptions"), "market_rules survive an asdict round trip")
    for bad_rules, why in (({"nope": {"min_leg_prob": 0.5}}, "unknown market"), ({"props": {"floor": 0.5}}, "unknown setting"),
                           ({"props": {"min_leg_prob": 1.2}}, "out-of-range floor"), ({"props": "0.5"}, "non-object rule")):
        try:
            FinderConfig(market_rules=bad_rules)
            check(False, f"market_rules {why} rejected")
        except FinderError:
            check(True, f"market_rules {why} rejected")
    loose_prop = MarketSide(**{**asdict(prop_side), "model_prob": 0.60, "fair_prob": 0.50})
    loose_game = MarketSide(**{**asdict(game_side), "model_prob": 0.60, "fair_prob": 0.50})
    check(leg_passes(loose_prop, mr)[0] and not leg_passes(loose_game, mr)[0] and "68%" in leg_passes(loose_game, mr)[1],
          "a 60% prop passes its 55% rule while a 60% spread fails the 68% global rule")
    bm = explain_by_market([loose_prop, loose_game, blocked_side, prop_side2], mr)
    check(list(bm) == ["spread", "receiving_yards"] and bm["receiving_yards"]["sides"] == 3 and bm["receiving_yards"]["passed"] == 2
          and bm["receiving_yards"]["blocked"] == 1 and bm["spread"]["passed"] == 0 and bm["receiving_yards"]["min_leg_prob"] == 0.55,
          "explain_by_market counts sides, passes and blocks per market in registry order")
    check(abs(bm["receiving_yards"]["best_prob"] - 0.6) < 1e-9 and bm["spread"]["best_gap"] is not None, "best prob / gap recorded from unblocked sides")
    table = format_market_table(bm)
    check(all(len(line) <= 80 for line in table) and any("Receiving Yards*" in line for line in table) and any("below floor (1)" in line for line in table),
          "market table fits 80 columns, flags experimental markets and names the top reason")

    # Simulation
    g1 = simulate_week_games(2026, 6, seed=7)
    check([g.to_dict() for g in g1] == [g.to_dict() for g in simulate_week_games(2026, 6, seed=7)], "simulation is deterministic")
    check(len(g1) == 14 and len(simulate_week_games(2026, 1, 7)) == 16, "bye weeks have 14 games, others 16")
    sides = sides_from_games(g1)
    check(len(sides) == 10 * len(g1), "ten sides per game (incl. two team totals)")
    check(all(0 < s.model_prob < 1 and 0 < s.fair_prob < 1 and 0 < (s.true_prob or 0.5) < 1 for s in sides), "probabilities in (0,1)")
    tts = [s for s in g1[0].sides() if s.market == "team_total" and s.team == g1[0].home]
    check(abs(tts[0].model_prob + tts[1].model_prob - 1) < 1e-12, "team total sides complementary")
    rng = random.Random(1)
    scores = [g1[0].simulate_final_score(rng) for _ in range(3000)]
    mean_home = sum(h for _, h in scores) / len(scores)
    check(abs(mean_home - (g1[0].true_total + g1[0].true_margin) / 2) < 1.0, f"simulated home points centre on truth ({mean_home:.1f})")

    # Rule 3: correlation matrix
    def side(market: str, team: Optional[str], direction: Optional[str], line: Optional[float] = None) -> MarketSide:
        return MarketSide(week=1, away="A", home="H", market=market, selection="x", american_odds=-110, model_prob=0.5,
                          fair_prob=0.5, line=line, team=team, direction=direction)
    H_ml, A_ml = side("moneyline", "H", None), side("moneyline", "A", None)
    H_sp, A_sp = side("spread", "H", None, -3.5), side("spread", "A", None, 3.5)
    over, under = side("total", None, "Over", 44.5), side("total", None, "Under", 44.5)
    H_tt_o, H_tt_u = side("team_total", "H", "Over", 24.5), side("team_total", "H", "Under", 24.5)
    A_tt_o, A_tt_u = side("team_total", "A", "Over", 20.5), side("team_total", "A", "Under", 20.5)
    check(leg_correlation(H_ml, H_sp) == "positive" and leg_correlation(H_ml, A_sp) == "negative", "ML + spread")
    check(leg_correlation(H_ml, H_tt_o) == "positive" and leg_correlation(H_ml, H_tt_u) == "negative", "ML + own team total")
    check(leg_correlation(H_ml, A_tt_o) == "negative" and leg_correlation(H_ml, A_tt_u) == "positive", "ML + opponent team total")
    check(leg_correlation(over, H_tt_o) == "positive" and leg_correlation(over, H_tt_u) == "negative", "game total + team total")
    check(leg_correlation(over, H_ml) == "neutral" and leg_correlation(H_tt_o, A_tt_o) == "neutral", "neutral pairs")
    check(leg_correlation(H_sp, A_sp) == "exclusive" and leg_correlation(over, under) == "exclusive" and leg_correlation(H_ml, A_ml) == "exclusive", "exclusive pairs")

    # Rule 3 for player props comes from correlation_rules.json; the matched reason travels onto the ticket
    def prop(market: str, player: str, team: str, direction: str, line: Optional[float], p: float = 0.7) -> MarketSide:
        reg_ = market_registry.registry()
        pm_ = reg_.get(market)
        assert pm_ is not None
        sel = market_registry.format_prop_selection(pm_, player, direction, line)
        return MarketSide(week=1, away="A", home="H", market=market, selection=sel, american_odds=-110, model_prob=p, fair_prob=0.5,
                          line=line, direction=direction, player=player, team=team)
    qb_o = prop("passing_yards", "QB One", "H", "Over", 250.5)
    wr_o = prop("receiving_yards", "WR One", "H", "Over", 70.5)
    rb_o = prop("rushing_yards", "RB Two", "A", "Over", 60.5)
    check(leg_correlation(qb_o, wr_o) == "positive" and leg_correlation(rb_o, H_sp) == "negative" and leg_correlation(rb_o, A_ml) == "positive",
          "props: QB with his receiver positive; opponent's rusher against the favourite negative; rusher with his own side positive")
    qb_u = prop("passing_yards", "QB One", "H", "Under", 250.5, 0.3)
    rb_rr = prop("rush_rec_yards", "RB Two", "A", "Over", 80.5)
    check(leg_correlation(qb_o, qb_u) == "exclusive" and leg_correlation(rb_o, rb_rr) == "redundant", "same player over/under exclusive; nested stats redundant")
    label_d, rule_id, why_d = leg_correlation_detail(qb_o, wr_o)
    check(label_d == "positive" and rule_id == "qb-passing-with-receiver" and "receiv" in why_d, "correlation detail names the rule and its reason")
    check(leg_correlation(qb_o, prop("passing_yards", "QB Two", "A", "Over", 230.5)) == "neutral", "props across teams are neutral")
    for policy in ("positive_only", "any"):
        allowed_r, corr_r, _why_r = _combo_allowed((rb_o, rb_rr), FinderConfig(same_game_policy=policy))
        check(not allowed_r and corr_r == "redundant", f"redundant pairs are rejected under same_game_policy={policy}")
    allowed_n, _c, why_n = _combo_allowed((qb_o, wr_o), FinderConfig(same_game_policy="never"))
    check(not allowed_n and "never" in why_n, "same_game_policy=never rejects even positive prop pairs")
    sg_ticket = _ticket_from_legs((qb_o, wr_o), 1, "T", "csv", "growth", "positive", why_d)
    check(sg_ticket["sgp_required"] and sg_ticket["pricing_note"] and sg_ticket["correlation_reason"] == why_d and "same-game parlay" in sg_ticket["notes"],
          "same-game tickets carry the SGP pricing flag, note and reason")
    cross = _ticket_from_legs((qb_o, prop("passing_yards", "QB Two", "A", "Over", 230.5)), 1, "T2", "csv", "growth", None)
    check(not cross["sgp_required"] and cross["pricing_note"] == "" and cross["correlation_reason"] is None, "cross-game tickets carry no SGP flag")

    # Rule 1: clamp
    clamped = FinderConfig(leg_sizes=(2, 3, 4))
    check(clamped.leg_sizes == (2,), "3+ leg sizes clamped to 2")
    check(FinderConfig(leg_sizes=[3]).leg_sizes == (2,), "3-only request falls back to 2")

    # Rule 2 + rule 3 invariants under the strict defaults, across many weeks
    strict = FinderConfig(source="sim", seed=7, season=2026)
    strict_tickets: List[Dict[str, Any]] = []
    for seed in range(1, 9):
        for wk in range(1, 19):
            strict_tickets.extend(find_parlays(wk, games=simulate_week_games(2026, wk, seed), config=strict))
    check(all(t["n_legs"] == 2 for t in strict_tickets), f"all strict tickets are 2-leg ({len(strict_tickets)} found over 144 weeks)")
    check(all(l["p_true"] >= 0.68 - 1e-9 for t in strict_tickets for l in t["legs"]), "every strict leg >= 68% model probability")
    check(all(l["prob_gap"] >= 0.06 - 1e-9 for t in strict_tickets for l in t["legs"]), "every strict leg >= 6 points over implied")
    check(all(t["correlation"] in (None, "positive") for t in strict_tickets), "same-game strict tickets are positively correlated only")
    sides6 = sides_from_games(simulate_week_games(2026, 6, 7))
    ex = explain_legs(sides6, strict)
    check(sum(ex.values()) == len(sides6), "explain_legs accounts for every side")

    # Mechanics under relaxed thresholds (so there is enough material to test)
    relaxed = FinderConfig(source="sim", seed=7, season=2026, min_leg_prob=0.0, min_prob_gap=0.0, top_n=50)
    tickets = find_parlays(6, games=g1, config=relaxed)
    check(len(tickets) > 0 and all(t["edge"] > 0 for t in tickets), f"relaxed config builds +EV tickets ({len(tickets)})")
    scores_ = [t["rank_score"] for t in tickets]
    check(scores_ == sorted(scores_, reverse=True), "tickets ranked by growth score")
    sg = [t for t in tickets if t["same_game"]]
    check(all(t["correlation"] == "positive" for t in sg), f"positive_only: {len(sg)} same-game tickets, all positive")
    never = find_parlays(6, games=g1, config=FinderConfig(source="sim", min_leg_prob=0, min_prob_gap=0, top_n=50, same_game_policy="never"))
    check(all(not t["same_game"] for t in never), "same_game_policy=never excludes same-game pairs")
    anyp = find_parlays(6, games=g1, config=FinderConfig(source="sim", min_leg_prob=0, min_prob_gap=0, top_n=50, same_game_policy="any"))
    check(len(anyp) >= len(tickets) and all(t["correlation"] != "exclusive" for t in anyp), "same_game_policy=any still rejects exclusive pairs")
    legacy = FinderConfig(allow_same_game=False)
    check(legacy.same_game_policy == "never", "legacy allow_same_game=False -> never")
    usage: Dict[Tuple[str, str], int] = {}
    game_use: Dict[str, int] = {}
    for t in tickets:
        for l in t["legs"]:
            usage[(l["matchup"], l["selection"])] = usage.get((l["matchup"], l["selection"]), 0) + 1
        for g in {l["matchup"] for l in t["legs"]}:
            game_use[g] = game_use.get(g, 0) + 1
    check(max(usage.values()) <= relaxed.max_tickets_per_leg and max(game_use.values()) <= relaxed.max_tickets_per_game, "diversification caps")
    t0 = tickets[0]
    check(abs(t0["p_true"] - math.prod(l["p_true"] for l in t0["legs"])) < 1e-5, "ticket p_true is the leg product")
    check("true_prob" not in json.dumps(tickets), "hidden truth never leaks into tickets")

    # Edge premise of the simulated league (relaxed filters, measured on candidates)
    def mean_true_edge(sim: SimConfig) -> float:
        fc = FinderConfig(source="sim", sim=sim, min_leg_prob=0.0, min_prob_gap=0.0, min_leg_edge=0.02)
        edges: List[float] = []
        for seed in (1, 2, 3):
            for wk in range(1, 19):
                for leg in select_candidate_legs(sides_from_games(simulate_week_games(2026, wk, seed, sim)), fc):
                    edges.append((leg.true_prob or 0.0) * leg.decimal_odds - 1.0)
        return sum(edges) / len(edges)
    e_true, n_true = mean_true_edge(SimConfig()), mean_true_edge(SimConfig().no_edge())
    check(e_true > 0.01 and n_true < 0.0, f"model picks carry true edge ({e_true:+.2%}); null picks do not ({n_true:+.2%})")

    # CSV round trip with a team total and the premium filters
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "lines.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["week", "away", "home", "market", "selection", "american_odds", "model_prob"])
            w.writerow([6, "Kansas City Chiefs", "Buffalo Bills", "moneyline", "Buffalo Bills ML", -200, 0.74])
            w.writerow([6, "Kansas City Chiefs", "Buffalo Bills", "moneyline", "Kansas City Chiefs ML", "+170", 0.26])
            w.writerow([6, "Kansas City Chiefs", "Buffalo Bills", "team_total", "Buffalo Bills Over 23.5", -110, 0.70])
            w.writerow([6, "Kansas City Chiefs", "Buffalo Bills", "team_total", "Buffalo Bills Under 23.5", -110, 0.30])
            w.writerow([6, "Kansas City Chiefs", "Buffalo Bills", "team_total", "Kansas City Chiefs Over 20.5", -110, 0.70])
            w.writerow([6, "Kansas City Chiefs", "Buffalo Bills", "team_total", "Kansas City Chiefs Under 20.5", -110, 0.30])
            w.writerow([6, "Dallas Cowboys", "Philadelphia Eagles", "spread", "Philadelphia Eagles -3", -105, 0.69])
            w.writerow([6, "Dallas Cowboys", "Philadelphia Eagles", "spread", "Dallas Cowboys +3", -115, 0.31])
            w.writerow([6, "Green Bay Packers", "Detroit Lions", "moneyline", "Detroit Lions ML", -150, ""])
            w.writerow([6, "Green Bay Packers", "Detroit Lions", "moneyline", "Green Bay Packers ML", "+130", ""])
            w.writerow([7, "Miami Dolphins", "New York Jets", "total", "Over 41.5", -110, 0.7])
        sides_csv = load_lines_csv(path, week=6)
        check(len(sides_csv) == 10, f"CSV loads week-6 rows only ({len(sides_csv)})")
        det = next(s for s in sides_csv if s.selection == "Detroit Lions ML")
        check(abs(det.model_prob - det.fair_prob) < 1e-12, "blank model_prob -> de-vigged fair prob")
        t_csv = find_parlays(6, config=FinderConfig(source="csv", lines_csv=path, top_n=10))
        check(len(t_csv) >= 1 and all(t["source"] == "csv" for t in t_csv), f"CSV source builds strict tickets ({len(t_csv)})")
        summ = getattr(t_csv, "slate_summary", None)
        check(isinstance(summ, dict) and summ["sides"] == 10 and set(summ["by_market"]) == {"moneyline", "team_total", "spread"}
              and summ["by_market"]["moneyline"]["passed"] >= 1 and isinstance(summ["table"], list), "find_parlays attaches a per-market slate summary")
        sels = {tuple(sorted(l["selection"] for l in t["legs"])) for t in t_csv}
        check(("Buffalo Bills ML", "Buffalo Bills Over 23.5") in sels, "Home ML + Home team total Over accepted (positive)")
        check(("Buffalo Bills ML", "Kansas City Chiefs Over 20.5") not in sels, "Home ML + Away team total Over rejected (negative)")
        check(all(l["p_true"] >= 0.68 and l["prob_gap"] >= 0.06 for t in t_csv for l in t["legs"]), "CSV tickets obey rule 2")
        try:
            load_lines_csv(os.path.join(tmp, "missing.csv"))
            check(False, "missing CSV raises")
        except FinderError:
            check(True, "missing CSV raises FinderError")
        mp = os.path.join(tmp, "model_probs.csv")
        with open(mp, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["matchup", "selection", "model_prob"])
            w.writerow(["Green Bay Packers @ Detroit Lions", "Detroit Lions ML", 0.72])
        check(apply_model_probs(sides_csv, load_model_probs_csv(mp)) == 1 and abs(det.model_prob - 0.72) < 1e-12, "model_probs.csv overrides")
        # Player-prop rows: player column, two-way de-vig, single-sided yes/no, blocked sides, experimental tickets
        ppath = os.path.join(tmp, "props_lines.csv")
        with open(ppath, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["week", "away", "home", "market", "selection", "american_odds", "model_prob", "player", "team", "position", "blocked"])
            w.writerow([6, "Kansas City Chiefs", "Buffalo Bills", "passing_yards", "Josh Allen Over 249.5 Passing Yards", -115, 0.74, "Josh Allen", "Buffalo Bills", "QB", ""])
            w.writerow([6, "Kansas City Chiefs", "Buffalo Bills", "passing_yards", "Josh Allen Under 249.5 Passing Yards", -105, 0.26, "Josh Allen", "Buffalo Bills", "QB", ""])
            w.writerow([6, "Kansas City Chiefs", "Buffalo Bills", "anytime_td", "James Cook Anytime TD", -130, 0.70, "James Cook", "Buffalo Bills", "RB", ""])
            w.writerow([6, "Kansas City Chiefs", "Buffalo Bills", "receiving_yards", "Travis Kelce Over 60.5 Receiving Yards", -110, "", "Travis Kelce", "Kansas City Chiefs", "TE", "ruled out (injury report: Out)"])
            w.writerow([6, "Kansas City Chiefs", "Buffalo Bills", "receiving_yards", "Travis Kelce Under 60.5 Receiving Yards", -110, "", "Travis Kelce", "Kansas City Chiefs", "TE", "ruled out (injury report: Out)"])
            w.writerow([6, "Dallas Cowboys", "Philadelphia Eagles", "moneyline", "Philadelphia Eagles ML", -150, 0.75])
        psides = load_lines_csv(ppath, week=6)
        allen_o = next(s for s in psides if s.selection.startswith("Josh Allen Over"))
        allen_u = next(s for s in psides if s.selection.startswith("Josh Allen Under"))
        check(len(psides) == 6 and allen_o.player == "Josh Allen" and allen_o.position == "QB" and allen_o.is_prop and allen_o.team == "Buffalo Bills"
              and abs(allen_o.fair_prob + allen_u.fair_prob - 1) < 1e-9, "prop rows load with player, team, position and a two-way de-vig")
        t_any = find_parlays(6, config=FinderConfig(source="csv", lines_csv=ppath, top_n=10, same_game_policy="any"))
        sg = [t for t in t_any if t["same_game"]]
        check(bool(sg) and all(t["sgp_required"] and t["pricing_note"] and t["correlation_reason"] for t in sg),
              f"same-game prop tickets are flagged for SGP pricing with a reason ({len(sg)})")
        cook = next(s for s in psides if s.market == "anytime_td")
        check(cook.direction == "Yes" and cook.line is None and abs(cook.fair_prob - cook.implied_prob) < 1e-12, "single-sided yes/no prop uses the implied price as fair")
        kelce = [s for s in psides if s.player == "Travis Kelce"]
        check(len(kelce) == 2 and all(s.blocked and not leg_passes(s, FinderConfig(min_leg_prob=0, min_prob_gap=0))[0] for s in kelce), "blocked column flows to the side and the filter")
        t_props = find_parlays(6, config=FinderConfig(source="csv", lines_csv=ppath, top_n=10))
        check(bool(t_props) and any(t["market_group"] in ("prop", "mixed") for t in t_props)
              and all(t["experimental"] == any(l["is_prop"] for l in t["legs"]) for t in t_props), f"prop legs form tickets flagged experimental ({len(t_props)})")
        check(all("EXPERIMENTAL" in t["notes"] for t in t_props if t["experimental"]), "experimental tickets say so in their notes")
        bad = os.path.join(tmp, "bad_prop.csv")
        with open(bad, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["week", "away", "home", "market", "selection", "american_odds"])
            w.writerow([6, "A", "H", "passing_yards", "Over 249.5", -110])
        try:
            load_lines_csv(bad)
            check(False, "prop row without a player raises")
        except FinderError:
            check(True, "prop row without a player raises FinderError")
        out = write_parlays_json(t_csv, os.path.join(tmp, "p.json"), 6)
        with open(out, encoding="utf-8") as fh:
            check(len(json.load(fh)["parlays"]) == len(t_csv), "write_parlays_json round trip")
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            auto = find_parlays(6, config=FinderConfig(source="sim", seed=7, season=2026, lines_csv=path))
            check(auto and all(t["source"] == "csv" for t in auto), "lines.csv auto-detected for week 6")
            with open("week_inputs.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["week", "away", "home", "home_spread", "total", "away_ml", "home_ml", "model_home_margin", "model_home_win_prob"])
                w.writerow([9, "Tampa Bay Buccaneers", "Dallas Cowboys", -8.5, 47.5, "+360", -470, 14.0, 90])
                w.writerow([9, "Cincinnati Bengals", "Miami Dolphins", 6.5, 42.5, -340, "+270", -12.0, 12])
                w.writerow([9, "Buffalo Bills", "Los Angeles Rams", -3, 54.5, "+136", -162, 8.0, 76])
            from_inputs = find_parlays(9, config=FinderConfig(source="sim", seed=7, season=2026, lines_csv="lines.csv"))
            check(os.path.isfile("lines.csv") and from_inputs and all(t["source"] == "csv" for t in from_inputs), "week_inputs.csv -> lines.csv -> tickets")
        finally:
            os.chdir(cwd)
        cfgp = os.path.join(tmp, "pipeline_config.json")
        with open(cfgp, "w", encoding="utf-8") as fh:
            json.dump({"finder": {"source": "sim", "seed": 99, "top_n": 3, "leg_sizes": [2, 3], "_comment": "x"}}, fh)
        loaded = load_finder_config(cfgp)
        check(loaded.seed == 99 and loaded.top_n == 3 and loaded.leg_sizes == (2,), "load_finder_config reads 'finder' and clamps legs")

    # Odds API fixture (v4 shape, incl. a team_totals market)
    now = _dt.datetime(2026, 10, 7, 12, 0, tzinfo=_dt.timezone.utc)
    fixture = [{
        "id": "abc", "commence_time": "2026-10-11T17:00:00Z", "home_team": "Buffalo Bills", "away_team": "Kansas City Chiefs",
        "bookmakers": [{"key": "draftkings", "markets": [
            {"key": "h2h", "outcomes": [{"name": "Buffalo Bills", "price": -165}, {"name": "Kansas City Chiefs", "price": 140}]},
            {"key": "spreads", "outcomes": [{"name": "Buffalo Bills", "price": -110, "point": -3.5}, {"name": "Kansas City Chiefs", "price": -110, "point": 3.5}]},
            {"key": "totals", "outcomes": [{"name": "Over", "price": -108, "point": 44.5}, {"name": "Under", "price": -112, "point": 44.5}]},
            {"key": "team_totals", "outcomes": [{"name": "Over", "description": "Buffalo Bills", "price": -115, "point": 24.5},
                                                {"name": "Under", "description": "Buffalo Bills", "price": -105, "point": 24.5}]},
        ]}]}, {"id": "old", "commence_time": "2026-09-01T17:00:00Z", "home_team": "A", "away_team": "B", "bookmakers": []}, {"garbage": True}]
    api_sides = parse_odds_api_response(fixture, week=6, now=now)
    check(len(api_sides) == 8 and any(s.selection == "Buffalo Bills Over 24.5" for s in api_sides), f"Odds API fixture -> 8 sides incl. team total ({len(api_sides)})")
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
    p = argparse.ArgumentParser(prog="parlay_finder", description="Generate ranked 2-leg NFL parlay candidates for a week.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--week", type=int, default=None, help="NFL week (default: upcoming week from today's date)")
    p.add_argument("--source", choices=("sim", "csv", "api"), default=None)
    p.add_argument("--lines", default=None, help="CSV of lines for --source csv")
    p.add_argument("--model-csv", dest="model_csv", default=None)
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--top-n", type=int, dest="top_n", default=None)
    p.add_argument("--min-leg-prob", type=float, dest="min_leg_prob", default=None, help="Rule 2 floor (default 0.68)")
    p.add_argument("--min-prob-gap", type=float, dest="min_prob_gap", default=None, help="Rule 2 gap (default 0.06)")
    p.add_argument("--edge-basis", choices=("implied", "fair"), dest="edge_basis", default=None)
    p.add_argument("--same-game", choices=("never", "positive_only", "any"), dest="same_game_policy", default=None)
    p.add_argument("--prop-floor", type=float, dest="prop_floor", default=None, help="Rule 2 floor for every player-prop market (e.g. 0.55)")
    p.add_argument("--prop-gap", type=float, dest="prop_gap", default=None, help="Rule 2 gap for every player-prop market (e.g. 0.04)")
    p.add_argument("--explain", action="store_true", help="Show legs passing per market and why sides were rejected")
    p.add_argument("--out", default=None, help="Write tickets to this JSON file")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _print_tickets(tickets: Sequence[Dict[str, Any]]) -> None:
    if not tickets:
        print("No parlay cleared the filters this week (2-leg only, each leg >= 68% and >= 6 points over the price).")
        return
    for t in tickets:
        tag = f"  [same game, {t['correlation']}]" if t["same_game"] else ""
        print(f"{t['ticket_id']:<12} {t['n_legs']}L  {t['american_odds']:>+6}  model {t['p_true']:.1%}  fair {t['fair_prob']:.1%}  edge {t['edge']:+.1%}{tag}")
        for leg in t["legs"]:
            print(f"    {leg['matchup']:<40} {leg['selection']:<30} ({leg['american_odds']:+d})  p={leg['p_true']:.1%}  gap={leg['prob_gap'] * 100:+.1f}pts")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.selftest:
        return _selftest()
    overrides: Dict[str, Any] = {k: v for k, v in (
        ("source", args.source), ("lines_csv", args.lines), ("model_csv", args.model_csv), ("season", args.season),
        ("seed", args.seed), ("min_leg_prob", args.min_leg_prob), ("min_prob_gap", args.min_prob_gap),
        ("edge_basis", args.edge_basis), ("same_game_policy", args.same_game_policy)) if v is not None}
    week = args.week
    if week is None:
        from weekly_reporter import estimate_nfl_week
        week = estimate_nfl_week(_dt.date.today())
        print(f"No --week given; using upcoming week {week}.")
    try:
        cfg = load_finder_config()
        prop_rule = {k: v for k, v in (("min_leg_prob", args.prop_floor), ("min_prob_gap", args.prop_gap)) if v is not None}
        if overrides or args.top_n is not None or prop_rule:
            data = {**asdict(cfg), **overrides, **({"top_n": args.top_n} if args.top_n is not None else {})}
            data["sim"] = cfg.sim
            if prop_rule:
                data["market_rules"] = {**cfg.market_rules, "props": {**{k: v for k, v in cfg.market_rules.get("props", {}).items()}, **prop_rule}}
                # an explicit CLI prop rule must beat any per-market entries from the config file
                data["market_rules"] = {k: v for k, v in data["market_rules"].items() if not market_registry.registry().is_prop(k)}
                data["market_rules"]["props"] = prop_rule
            cfg = FinderConfig.from_dict(data)
        tickets = find_parlays(week, config=cfg)
        if args.explain and tickets.slate_summary:
            summ = tickets.slate_summary
            print(f"Week {week}: {summ['sides']} market sides across {summ['markets']} market(s); "
                  f"{summ['passed']} side(s) clear the filter, {summ['candidates']} candidate leg(s) after de-duplication")
            print("LEGS CLEARING THE FILTER BY MARKET")
            for line in summ["table"]:
                print(line)
            print("REASONS (all markets)")
            for reason, n in sorted((kv for kv in explain_legs(collect_sides(week, cfg), cfg).items() if kv[0] != "passed"), key=lambda kv: -kv[1]):
                print(f"  {n:>4}  {reason}")
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
