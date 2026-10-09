#!/usr/bin/env python3
"""
prop_model.py  —  EXPERIMENTAL player-prop projection model
==========================================================

Turns a sportsbook player-prop line ("Dak Prescott Over 264.5 Passing Yards")
into a model probability for the EdgeBook AI pipeline. FPI only produces
game-level numbers, so props need their own model; this is it, and it is
labelled EXPERIMENTAL everywhere until the backtest calibration gate in
``markets.json`` clears it.

Data (all free, standard-library download, cached under ``data_cache/nflverse``)
--------------------------------------------------------------------------------
* nflverse weekly player stats (``stats_player_week_<season>.csv``): per-game
  attempts / carries / targets and the yards, TDs, completions and
  interceptions that go with them, for the current season and the one before.
* nflverse schedules (``games.csv``): results plus closing spread and total,
  used for game script in the walk-forward calibration.
* nflverse injury reports (``injuries_<season>.csv``): players listed Out or
  Doubtful are blocked.

How a projection is built (per player, per market, per game)
------------------------------------------------------------
    mean = sum over the market's parts of
           usage  x  rate  x  opponent factor  x  game-script factor

* **usage** - recent-weighted per-game attempts, carries or targets
  (``model.decay`` per game back), shrunk toward last season's rate and, for
  players with no history, the positional average.
* **rate** - the stat per usage (yards per target, TDs per carry, ...),
  shrunk toward the league positional rate with ``model.rate_pseudo_counts``
  pseudo-usages, because efficiency is noisy.
* **opponent factor** - what the opponent's defence has allowed per game
  relative to the league, shrunk halfway toward 1.0 and clamped.
* **game-script factor** - pass volume rises for underdogs and rush volume
  for favourites (``model.script``), scaled by the posted total relative to a
  league-average game. The team's expected margin comes from the FPI numbers
  in ``week_inputs.csv`` when present, otherwise from the market spread.

The projection becomes a probability through the market's distribution
(``markets.json``): normal for passing yards, gamma for rushing and
receiving yards (skewed, zero-floored), negative binomial for receptions,
completions and attempts, Poisson for passing TDs and interceptions, and
``1 - exp(-lambda)`` for Anytime TD. Whole-number lines get an explicit push
probability and the side's probability is conditional on no push.

Calibration gate
----------------
``python3 prop_model.py --calibrate --season 2025`` replays a past season
week by week, projecting every qualifying player from the data available
*before* each week, and reports bias, dispersion, interval coverage and
hit rates at synthetic lines per market. Run it before trusting a market.

Standard library only. ``python3 prop_model.py --selftest`` runs the tests.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import logging
import math
import os
import random
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import market_registry
from market_registry import PropMarket, Registry, RegistryError

__all__ = [
    "PropModelError", "PropModel", "StatsDB", "PlayerGame", "GameContext", "Projection",
    "fetch_file", "load_player_games", "load_injuries", "load_schedule", "normalize_name",
    "team_abbr", "team_full_name", "tail_probabilities", "fill_model_probs", "calibrate",
    "NFLVERSE_BASE", "STAT_COLUMNS", "USAGE_COLUMNS", "TEAM_ABBR", "cache_dir",
]

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
NFLVERSE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"
FILE_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "stats": ("stats_player/stats_player_week_{season}.csv", "player_stats/stats_player_week_{season}.csv"),
    "games": ("schedules/games.csv",),
    "injuries": ("injuries/injuries_{season}.csv",),
}
STAT_COLUMNS: Tuple[str, ...] = ("attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions",
                                 "carries", "rushing_yards", "rushing_tds", "targets", "receptions", "receiving_yards",
                                 "receiving_tds")
USAGE_COLUMNS: Tuple[str, ...] = ("attempts", "carries", "targets")
POSITION_GROUPS: Dict[str, str] = {"QB": "QB", "RB": "RB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}

TEAM_ABBR: Dict[str, str] = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
_ABBR_ALIASES: Dict[str, str] = {"LAR": "LA", "OAK": "LV", "SD": "LAC", "STL": "LA", "WSH": "WAS", "JAC": "JAX"}
_FULL_BY_ABBR: Dict[str, str] = {v: k for k, v in TEAM_ABBR.items()}
_NICKNAME_BY_ABBR: Dict[str, str] = {v: k.split()[-1].lower() for k, v in TEAM_ABBR.items()}

DEFAULT_PARAMS: Dict[str, Any] = {
    "decay": 0.75, "prior_season_weight": 0.35, "min_games": 2, "usage_prior_games": 3, "position_prior_games": 1.0,
    "rate_pseudo_counts": {"attempts": 150, "carries": 60, "targets": 40}, "opponent_shrink": 0.5,
    "opponent_clamp": [0.75, 1.25],
    "script": {"pass_per_point": 0.012, "rush_per_point": 0.015, "total_elasticity": 0.5, "league_total": 45.0},
    "usage_play_type": {"attempts": "pass", "targets": "pass", "carries": "rush"},
    "first_td_team_factor": 0.5, "block_injury_status": ["Out", "Doubtful"],
    "td_points_per_td": 0.093,  # expected team TDs per expected team point (about 0.65 TD per 7 points)
}


class PropModelError(RuntimeError):
    """A projection could not be made (no data, player not found, ruled out, ...)."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def nfl_season_year(date: Optional[_dt.date] = None) -> int:
    date = date or _dt.date.today()
    return date.year if date.month >= 3 else date.year - 1


def cache_dir() -> str:
    return os.environ.get("EDGEBOOK_DATA_CACHE") or os.path.join(HERE, "data_cache", "nflverse")


_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def normalize_name(name: str) -> str:
    """``"A.J. Brown"`` / ``"Brian Thomas Jr."`` -> ``"aj brown"`` / ``"brian thomas"``."""
    s = str(name).lower().replace("’", "'")
    s = re.sub(r"[.'\-]", "", s)
    s = _SUFFIX_RE.sub(" ", s)
    return " ".join(s.split())


def team_abbr(team: str) -> Optional[str]:
    """Full name, nickname or abbreviation -> nflverse abbreviation (None when unknown)."""
    if not team:
        return None
    t = str(team).strip()
    if t in TEAM_ABBR:
        return TEAM_ABBR[t]
    u = t.upper()
    if u in _FULL_BY_ABBR:
        return u
    if u in _ABBR_ALIASES:
        return _ABBR_ALIASES[u]
    low = t.lower()
    for full, abbr in TEAM_ABBR.items():
        if low == full.lower() or low == full.split()[-1].lower():
            return abbr
    return None


def team_full_name(abbr: str) -> str:
    a = str(abbr).upper()
    return _FULL_BY_ABBR.get(_ABBR_ALIASES.get(a, a), abbr)


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Distributions (pure Python)
# ---------------------------------------------------------------------------


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def regularized_lower_gamma(a: float, x: float) -> float:
    """P(a, x) = gamma(a, x) / Gamma(a), series / continued fraction (Numerical Recipes gammp)."""
    if x <= 0.0:
        return 0.0
    if a <= 0.0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap, total, delta = a, 1.0 / a, 1.0 / a
        for _ in range(500):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-14:
                break
        return max(0.0, min(1.0, total * math.exp(-x + a * math.log(x) - gln)))
    # continued fraction for Q(a, x), then P = 1 - Q
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    q = math.exp(-x + a * math.log(x) - gln) * h
    return max(0.0, min(1.0, 1.0 - q))


def gamma_cdf(x: float, mean: float, sd: float) -> float:
    if x <= 0.0:
        return 0.0
    if mean <= 0.0 or sd <= 0.0:
        return 1.0 if x >= mean else 0.0
    shape = (mean / sd) ** 2
    scale = sd * sd / mean
    return regularized_lower_gamma(shape, x / scale)


def poisson_pmf_table(mean: float, upto: int) -> List[float]:
    mean = max(mean, 1e-12)
    out = [math.exp(-mean)]
    for k in range(1, upto + 1):
        out.append(out[-1] * mean / k)
    return out


def negbin_pmf_table(mean: float, dispersion: float, upto: int) -> List[float]:
    """Negative binomial with mean ``mean`` and size ``dispersion`` (variance = mean + mean^2 / dispersion)."""
    mean = max(mean, 1e-12)
    r = max(dispersion, 1e-6)
    p = r / (r + mean)
    out = [p ** r]
    for k in range(1, upto + 1):
        out.append(out[-1] * (k - 1 + r) / k * (1.0 - p))
    return out


def _discrete_tail(pmf: List[float], line: float) -> Tuple[float, float, float]:
    """(P(X > line), P(X < line), P(X == line)) for an integer-valued variable with pmf table."""
    floor_line = math.floor(line + 1e-9)
    is_int = abs(line - round(line)) < 1e-9
    below = sum(pmf[: int(floor_line) + (0 if is_int else 1)]) if floor_line >= 0 else 0.0
    push = pmf[int(round(line))] if is_int and 0 <= int(round(line)) < len(pmf) else 0.0
    below = max(0.0, min(1.0, below))
    over = max(0.0, 1.0 - below - push)
    return over, below, push


def tail_probabilities(distribution: str, mean: float, sd: float, line: float,
                       dispersion: Optional[float] = None) -> Tuple[float, float, float]:
    """``(p_over, p_under, p_push)`` for a line under the given distribution.

    Yards are integers, so continuous distributions get a continuity correction:
    ``P(over 264.5) = P(X >= 265)``; whole-number lines carry push mass.
    """
    is_int = abs(line - round(line)) < 1e-9
    if distribution == "normal":
        sd = max(sd, 1e-9)
        if is_int:
            under = normal_cdf((line - 0.5 - mean) / sd)
            over = 1.0 - normal_cdf((line + 0.5 - mean) / sd)
        else:
            under = normal_cdf((line - mean) / sd)
            over = 1.0 - under
        return max(0.0, over), max(0.0, under), max(0.0, 1.0 - over - under) if is_int else 0.0
    if distribution == "gamma":
        if is_int:
            under = gamma_cdf(line - 0.5, mean, sd)
            over = 1.0 - gamma_cdf(line + 0.5, mean, sd)
        else:
            under = gamma_cdf(line, mean, sd)
            over = 1.0 - under
        return max(0.0, over), max(0.0, under), max(0.0, 1.0 - over - under) if is_int else 0.0
    upto = int(max(20, mean + 12 * math.sqrt(max(mean, 1.0)) + line + 5))
    if distribution == "poisson":
        return _discrete_tail(poisson_pmf_table(mean, upto), line)
    if distribution == "negbin":
        return _discrete_tail(negbin_pmf_table(mean, dispersion or 10.0, upto), line)
    raise PropModelError(f"unknown distribution '{distribution}'")


def distribution_cdf(distribution: str, mean: float, sd: float, x: float, dispersion: Optional[float] = None) -> float:
    """P(X <= x); used by the calibration PIT check."""
    over, under, push = tail_probabilities(distribution, mean, sd, float(x), dispersion)
    return min(1.0, under + push)


# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------


def _download(url: str, dest: str, timeout: float = 180.0) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".dl_", dir=os.path.dirname(dest))
    os.close(fd)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "EdgeBook-AI/prop_model"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, dest)
    except urllib.error.HTTPError as exc:
        raise PropModelError(f"HTTP {exc.code} for {url}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise PropModelError(f"could not download {url}: {exc}") from exc
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def fetch_file(kind: str, season: Optional[int] = None, directory: Optional[str] = None,
               max_age_hours: float = 24.0, offline: bool = False) -> str:
    """Local path of an nflverse file, downloading or refreshing it when needed.

    A past season's stats never refresh once cached; the current season's
    files refresh after ``max_age_hours``. When a refresh fails the stale
    copy is used with a warning. ``offline=True`` never touches the network.
    """
    if kind not in FILE_CANDIDATES:
        raise PropModelError(f"unknown nflverse file kind '{kind}'")
    directory = directory or cache_dir()
    candidates = [c.format(season=season) for c in FILE_CANDIDATES[kind]]
    name = os.path.basename(candidates[0])
    dest = os.path.join(directory, name)
    exists = os.path.isfile(dest) and os.path.getsize(dest) > 0
    if exists:
        age_h = (time.time() - os.path.getmtime(dest)) / 3600.0
        if offline or (season is not None and season < nfl_season_year()) or age_h < max_age_hours:
            return dest
    elif offline:
        raise PropModelError(f"{name} is not cached in {directory} and offline mode is on")
    errors: List[str] = []
    for rel in candidates:
        url = f"{NFLVERSE_BASE}/{rel}"
        try:
            _download(url, dest)
            logger.info("Downloaded %s", url)
            return dest
        except PropModelError as exc:
            errors.append(str(exc))
    if exists:
        logger.warning("Could not refresh %s (%s); using the cached copy", name, "; ".join(errors))
        return dest
    raise PropModelError(f"could not fetch {name}: " + "; ".join(errors))


@dataclass
class PlayerGame:
    season: int
    week: int
    player_id: str
    name: str
    position: str
    team: str
    opp: str
    stats: Dict[str, float] = field(default_factory=dict)


def load_player_games(path: str, season: int) -> List[PlayerGame]:
    """Regular-season rows of an nflverse weekly stats file for the offensive stat columns."""
    out: List[PlayerGame] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if (r.get("season_type") or "REG") != "REG":
                continue
            try:
                wk = int(r.get("week") or 0)
            except ValueError:
                continue
            stats = {c: _f(r.get(c)) for c in STAT_COLUMNS}
            if not any(stats.values()):
                continue
            pos = POSITION_GROUPS.get(str(r.get("position") or "").upper(), str(r.get("position") or "").upper())
            team = str(r.get("team") or r.get("recent_team") or "").upper()
            out.append(PlayerGame(season=int(r.get("season") or season), week=wk, player_id=str(r.get("player_id") or ""),
                                  name=str(r.get("player_display_name") or r.get("player_name") or ""), position=pos,
                                  team=_ABBR_ALIASES.get(team, team), opp=_ABBR_ALIASES.get(str(r.get("opponent_team") or "").upper(),
                                                                                           str(r.get("opponent_team") or "").upper()),
                                  stats=stats))
    return out


def load_injuries(path: str) -> Dict[Tuple[int, str, str], str]:
    """``{(week, team, normalized name): status}`` from an nflverse injuries file (regular season)."""
    out: Dict[Tuple[int, str, str], str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if (r.get("season_type") or r.get("game_type") or "REG") != "REG":
                continue
            try:
                wk = int(r.get("week") or 0)
            except ValueError:
                continue
            team = str(r.get("team") or "").upper()
            status = (r.get("report_status") or "").strip() or (r.get("practice_status") or "").strip()
            if not status:
                continue
            out[(wk, _ABBR_ALIASES.get(team, team), normalize_name(r.get("full_name") or ""))] = status
    return out


@dataclass
class ScheduledGame:
    season: int
    week: int
    away: str            # abbreviation
    home: str
    gameday: str = ""
    spread_line: Optional[float] = None    # home expected margin at close (positive = home favoured)
    total_line: Optional[float] = None
    away_score: Optional[int] = None
    home_score: Optional[int] = None


def load_schedule(path: str, season: int) -> List[ScheduledGame]:
    out: List[ScheduledGame] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("season")) != str(season) or (r.get("game_type") or "REG") != "REG":
                continue
            try:
                wk = int(r.get("week") or 0)
            except ValueError:
                continue
            away = str(r.get("away_team") or "").upper()
            home = str(r.get("home_team") or "").upper()

            def num(key: str) -> Optional[float]:
                v = (r.get(key) or "").strip()
                return float(v) if v not in ("", "NA") else None

            a_s, h_s = num("away_score"), num("home_score")
            out.append(ScheduledGame(season=season, week=wk, away=_ABBR_ALIASES.get(away, away), home=_ABBR_ALIASES.get(home, home),
                                     gameday=str(r.get("gameday") or ""), spread_line=num("spread_line"), total_line=num("total_line"),
                                     away_score=int(a_s) if a_s is not None else None, home_score=int(h_s) if h_s is not None else None))
    return out


# ---------------------------------------------------------------------------
# Stats database with the aggregates the model needs
# ---------------------------------------------------------------------------


class StatsDB:
    """Player games for one or more seasons plus league / opponent aggregates by week."""

    def __init__(self, games: Iterable[PlayerGame]) -> None:
        self.by_player: Dict[str, List[PlayerGame]] = {}
        self.name_index: Dict[str, Set[str]] = {}
        self.team_players: Dict[Tuple[int, str], Set[str]] = {}
        self.position: Dict[str, str] = {}
        self.display: Dict[str, str] = {}
        # (season, position, week) -> {stat: sum}; (season, opp, week) -> {stat: sum}; (season, week) -> {stat: sum}
        self._pos_week: Dict[Tuple[int, str, int], Dict[str, float]] = {}
        self._opp_week: Dict[Tuple[int, str, int], Dict[str, float]] = {}
        self._league_week: Dict[Tuple[int, int], Dict[str, float]] = {}
        self._pos_usage_games: Dict[Tuple[int, str, int], Dict[str, Tuple[float, int]]] = {}
        self._team_games_week: Dict[Tuple[int, int], Set[Tuple[str, str]]] = {}
        self.seasons: Set[int] = set()
        for g in games:
            self.seasons.add(g.season)
            self.by_player.setdefault(g.player_id, []).append(g)
            self.name_index.setdefault(normalize_name(g.name), set()).add(g.player_id)
            self.team_players.setdefault((g.season, g.team), set()).add(g.player_id)
            self.position[g.player_id] = g.position or self.position.get(g.player_id, "")
            self.display[g.player_id] = g.name
            pw = self._pos_week.setdefault((g.season, g.position, g.week), {})
            ow = self._opp_week.setdefault((g.season, g.opp, g.week), {})
            lw = self._league_week.setdefault((g.season, g.week), {})
            for c, v in g.stats.items():
                pw[c] = pw.get(c, 0.0) + v
                ow[c] = ow.get(c, 0.0) + v
                lw[c] = lw.get(c, 0.0) + v
            ug = self._pos_usage_games.setdefault((g.season, g.position, g.week), {})
            for u in USAGE_COLUMNS:
                if g.stats.get(u, 0.0) >= 1.0:
                    tot, n = ug.get(u, (0.0, 0))
                    ug[u] = (tot + g.stats[u], n + 1)
            self._team_games_week.setdefault((g.season, g.week), set()).add((g.team, g.opp))
        for pid in self.by_player:
            self.by_player[pid].sort(key=lambda x: (x.season, x.week))

    # ---- players ------------------------------------------------------------

    def find_player(self, name: str, season: int, teams: Sequence[str] = ()) -> Optional[str]:
        """Best player id for a display name, preferring players seen on ``teams`` in ``season`` (or the season before)."""
        ids = self.name_index.get(normalize_name(name))
        if not ids:
            # last-name + first-initial fallback ("D. Prescott", "Prescott")
            parts = normalize_name(name).split()
            if not parts:
                return None
            last = parts[-1]
            initial = parts[0][0] if len(parts) > 1 else ""
            ids = {pid for nm, pids in self.name_index.items() for pid in pids
                   if nm.split()[-1] == last and (not initial or nm[0] == initial)}
            if not ids:
                return None
        teams_u = [t for t in (team_abbr(t) or "" for t in teams) if t]
        if teams_u:
            for yr in (season, season - 1):
                on_team = [pid for pid in ids if any(pid in self.team_players.get((yr, t), ()) for t in teams_u)]
                if on_team:
                    return max(on_team, key=lambda pid: len(self.by_player[pid]))
        return max(ids, key=lambda pid: (max((g.season for g in self.by_player[pid]), default=0), len(self.by_player[pid])))

    def games_for(self, player_id: str, season: int, before_week: Optional[int] = None) -> List[PlayerGame]:
        return [g for g in self.by_player.get(player_id, ()) if g.season == season and (before_week is None or g.week < before_week)]

    def current_team(self, player_id: str, season: int) -> Optional[str]:
        games = self.games_for(player_id, season)
        if games:
            return games[-1].team
        prev = self.games_for(player_id, season - 1)
        return prev[-1].team if prev else None

    # ---- aggregates ---------------------------------------------------------

    def _sum_before(self, table: Dict[Tuple[Any, ...], Dict[str, float]], key_prefix: Tuple[Any, ...], before_week: Optional[int],
                    stat: str) -> float:
        total = 0.0
        for week in range(1, 23):
            if before_week is not None and week >= before_week:
                break
            total += table.get(key_prefix + (week,), {}).get(stat, 0.0)
        return total

    def league_rate(self, season: int, position: str, stat: str, usage: str, before_week: Optional[int] = None) -> Optional[float]:
        """Positional stat-per-usage across the league (weeks before ``before_week``), None without data."""
        for pos in (position, None):
            if pos is None:
                num = self._sum_before(self._league_week, (season,), before_week, stat)
                den = self._sum_before(self._league_week, (season,), before_week, usage)
            else:
                num = self._sum_before(self._pos_week, (season, pos), before_week, stat)
                den = self._sum_before(self._pos_week, (season, pos), before_week, usage)
            if den >= 50.0:
                return num / den
        return None

    def usage_prior(self, season: int, position: str, usage: str, before_week: Optional[int] = None) -> Optional[float]:
        """Average per-game usage of players at ``position`` who recorded the usage stat."""
        tot, n = 0.0, 0
        for week in range(1, 23):
            if before_week is not None and week >= before_week:
                break
            t, c = self._pos_usage_games.get((season, position, week), {}).get(usage, (0.0, 0))
            tot, n = tot + t, n + c
        return (tot / n) if n >= 10 else None

    def allowed_per_game(self, season: int, opp: str, stat: str, before_week: Optional[int] = None) -> Tuple[float, int]:
        """What ``opp``'s defence allowed per game for ``stat`` -> (per game, games)."""
        total, games = 0.0, 0
        for week in range(1, 23):
            if before_week is not None and week >= before_week:
                break
            row = self._opp_week.get((season, opp, week))
            if row is None:
                continue
            total += row.get(stat, 0.0)
            games += 1
        return (total / games if games else 0.0), games

    def league_allowed_per_game(self, season: int, stat: str, before_week: Optional[int] = None) -> Tuple[float, int]:
        total, games = 0.0, 0
        for week in range(1, 23):
            if before_week is not None and week >= before_week:
                break
            row = self._league_week.get((season, week))
            if row is None:
                continue
            total += row.get(stat, 0.0)
            games += len(self._team_games_week.get((season, week), ()))
        return (total / games if games else 0.0), games

    def players_in_week(self, season: int, week: int, team: Optional[str] = None) -> List[PlayerGame]:
        out = []
        for games in self.by_player.values():
            for g in games:
                if g.season == season and g.week == week and (team is None or g.team == team):
                    out.append(g)
        return out


# ---------------------------------------------------------------------------
# Game context and projections
# ---------------------------------------------------------------------------


@dataclass
class GameContext:
    """What the model knows about the game: the two teams and the expected margin / total."""

    season: int
    week: int
    away: str                                   # full names as in week_inputs.csv (abbreviations accepted)
    home: str
    home_spread: Optional[float] = None         # market: -8.5 when the home team is favoured by 8.5
    total: Optional[float] = None
    model_home_margin: Optional[float] = None   # FPI-style expected margin, home minus away
    model_total: Optional[float] = None
    model_home_win_prob: Optional[float] = None

    @property
    def away_abbr(self) -> str:
        return team_abbr(self.away) or str(self.away).upper()

    @property
    def home_abbr(self) -> str:
        return team_abbr(self.home) or str(self.home).upper()

    def side_of(self, team: str) -> Optional[str]:
        a = team_abbr(team) or str(team).upper()
        if a == self.home_abbr:
            return "home"
        if a == self.away_abbr:
            return "away"
        return None

    def expected_margin(self, team: str) -> Optional[float]:
        """Expected final margin for ``team`` (positive = favoured), FPI margin first, market spread otherwise."""
        side = self.side_of(team)
        if side is None:
            return None
        if self.model_home_margin is not None:
            m = float(self.model_home_margin)
        elif self.home_spread is not None:
            m = -float(self.home_spread)
        else:
            return None
        return m if side == "home" else -m

    def expected_total(self) -> Optional[float]:
        return float(self.model_total) if self.model_total is not None else (float(self.total) if self.total is not None else None)

    def win_prob(self, team: str) -> Optional[float]:
        side = self.side_of(team)
        if side is None:
            return None
        if self.model_home_win_prob is not None:
            p = float(self.model_home_win_prob)
            p = p / 100.0 if p > 1.0 else p
            return p if side == "home" else 1.0 - p
        m = self.expected_margin(team)
        return None if m is None else normal_cdf(m / 13.5)

    def team_points(self, team: str) -> Optional[float]:
        m, t = self.expected_margin(team), self.expected_total()
        if m is None or t is None:
            return None
        return max(3.0, (t + m) / 2.0)


@dataclass
class Projection:
    market: str
    player: str
    player_id: str
    position: str
    team: str
    opp: str
    mean: float
    sd: float
    distribution: str
    dispersion: Optional[float]
    n_games: int
    usage: float
    rate: float
    opp_factor: float
    script_factor: float
    parts: List[Dict[str, float]] = field(default_factory=list)
    p_yes: Optional[float] = None      # yes/no markets
    note: str = ""

    def probabilities(self, line: Optional[float], direction: str) -> Tuple[float, float]:
        """``(model probability for the side, push probability)``; yes/no markets ignore ``line``."""
        d = str(direction).lower()
        if self.p_yes is not None:
            p = self.p_yes
            return (p if d == "yes" else 1.0 - p), 0.0
        if line is None:
            raise PropModelError(f"{self.market} needs a line")
        over, under, push = tail_probabilities(self.distribution, self.mean, self.sd, float(line), self.dispersion)
        live = max(1e-9, 1.0 - push)
        p = over / live if d == "over" else under / live
        return max(0.0, min(1.0, p)), push

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PropModel:
    """The projection engine. Build with :meth:`load` (downloads / reads the cache) or from a :class:`StatsDB`."""

    _instances: Dict[Tuple[Any, ...], "PropModel"] = {}

    def __init__(self, db: StatsDB, injuries: Optional[Dict[Tuple[int, str, str], str]] = None,
                 registry: Optional[Registry] = None, params: Optional[Dict[str, Any]] = None, season: Optional[int] = None) -> None:
        self.db = db
        self.injuries = injuries or {}
        self.registry = registry or market_registry.registry()
        merged = dict(DEFAULT_PARAMS)
        merged.update({k: v for k, v in (self.registry.model or {}).items() if not str(k).startswith("_")})
        if params:
            merged.update(params)
        self.params = merged
        self.season = season or (max(db.seasons) if db.seasons else nfl_season_year())

    @classmethod
    def load(cls, season: Optional[int] = None, offline: bool = False, directory: Optional[str] = None,
             prior_seasons: int = 1, with_injuries: bool = True) -> "PropModel":
        """Stats for ``season`` and ``prior_seasons`` before it, plus the season's injury report."""
        season = season or nfl_season_year()
        directory = directory or cache_dir()
        paths: List[Tuple[int, str]] = []
        for yr in range(season - prior_seasons, season + 1):
            try:
                paths.append((yr, fetch_file("stats", yr, directory, offline=offline)))
            except PropModelError as exc:
                if yr == season:
                    raise
                logger.warning("No stats for %d (%s); continuing without that prior season", yr, exc)
        inj_path = None
        if with_injuries:
            try:
                inj_path = fetch_file("injuries", season, directory, offline=offline)
            except PropModelError as exc:
                logger.warning("Injury report unavailable (%s); nobody will be blocked for injury", exc)
        key = tuple((p, os.path.getmtime(p)) for _, p in paths) + ((inj_path, os.path.getmtime(inj_path) if inj_path else 0),)
        cached = cls._instances.get(key)
        if cached is not None:
            return cached
        games: List[PlayerGame] = []
        for yr, p in paths:
            games.extend(load_player_games(p, yr))
        model = cls(StatsDB(games), load_injuries(inj_path) if inj_path else {}, season=season)
        cls._instances.clear()
        cls._instances[key] = model
        logger.info("Prop model loaded: %d player-games across %s", len(games), ", ".join(str(y) for y, _ in paths))
        return model

    # ---- pieces -------------------------------------------------------------

    def injury_status(self, week: int, team: str, player_name: str) -> Optional[str]:
        return self.injuries.get((week, team_abbr(team) or str(team).upper(), normalize_name(player_name)))

    def _ew(self, games: Sequence[PlayerGame]) -> Tuple[float, Dict[str, float]]:
        decay = float(self.params["decay"])
        wsum = 0.0
        acc: Dict[str, float] = {c: 0.0 for c in STAT_COLUMNS}
        for k, g in enumerate(reversed(list(games))):
            w = decay ** k
            wsum += w
            for c in STAT_COLUMNS:
                acc[c] += w * g.stats.get(c, 0.0)
        return wsum, acc

    def _usage(self, cur: Sequence[PlayerGame], prev: Sequence[PlayerGame], usage: str, position: str,
               season: int, week: int) -> Tuple[float, str]:
        cur_w, cur_acc = self._ew(cur)
        prev_w, prev_acc = self._ew(prev)
        cur_mean = (cur_acc[usage] / cur_w) if cur_w > 0 else None
        if prev_w > 0:
            prior_mean = prev_acc[usage] / prev_w
            prior_w = float(self.params["usage_prior_games"]) * float(self.params["prior_season_weight"]) * min(1.0, len(prev) / 4.0 + 0.25)
            src = "last season"
        else:
            pos_prior = self.db.usage_prior(season, position, usage, before_week=week)
            if pos_prior is None:
                pos_prior = self.db.usage_prior(season - 1, position, usage)
            prior_mean = pos_prior if pos_prior is not None else (cur_mean or 0.0)
            prior_w = float(self.params.get("position_prior_games", 1.0))
            src = "position average"
        if cur_mean is None:
            return prior_mean, src
        return (cur_w * cur_mean + prior_w * prior_mean) / (cur_w + prior_w), src

    def _rate(self, cur: Sequence[PlayerGame], prev: Sequence[PlayerGame], stat: str, usage: str, position: str,
              season: int, week: int, pseudo: float) -> float:
        if stat == usage:
            return 1.0
        _cw, cur_acc = self._ew(cur)
        _pw, prev_acc = self._ew(prev)
        psw = float(self.params["prior_season_weight"])
        num = cur_acc[stat] + psw * prev_acc[stat]
        den = cur_acc[usage] + psw * prev_acc[usage]
        league = self.db.league_rate(season, position, stat, usage, before_week=week)
        if league is None:
            league = self.db.league_rate(season - 1, position, stat, usage)
        if league is None:
            league = (num / den) if den > 0 else 0.0
        if den <= 0:
            return league
        return (den * (num / den) + pseudo * league) / (den + pseudo)

    def _opp_factor(self, opp: str, stat: str, season: int, week: int) -> float:
        psw = float(self.params["prior_season_weight"])
        cur_allowed, cur_n = self.db.allowed_per_game(season, opp, stat, before_week=week)
        prev_allowed, prev_n = self.db.allowed_per_game(season - 1, opp, stat)
        lg_cur, lg_cur_n = self.db.league_allowed_per_game(season, stat, before_week=week)
        lg_prev, lg_prev_n = self.db.league_allowed_per_game(season - 1, stat)
        w_cur, w_prev = float(cur_n), psw * float(prev_n)
        if w_cur + w_prev <= 0 or (lg_cur_n == 0 and lg_prev_n == 0):
            return 1.0
        allowed = (w_cur * cur_allowed + w_prev * prev_allowed) / (w_cur + w_prev)
        lw_cur, lw_prev = (float(lg_cur_n) if cur_n else 0.0), (psw * float(lg_prev_n) if prev_n else 0.0)
        league = (lw_cur * lg_cur + lw_prev * lg_prev) / (lw_cur + lw_prev) if (lw_cur + lw_prev) > 0 else 0.0
        if league <= 0:
            return 1.0
        raw = allowed / league
        shrink = float(self.params["opponent_shrink"])
        lo, hi = self.params["opponent_clamp"]
        return max(float(lo), min(float(hi), 1.0 + shrink * (raw - 1.0)))

    def _script(self, usage: str, ctx: GameContext, team: str) -> float:
        sc = self.params["script"]
        play_type = self.params["usage_play_type"].get(usage, "pass")
        margin = ctx.expected_margin(team)
        mult = 1.0
        if margin is not None:
            s = -margin  # positive when the team is the underdog
            if play_type == "pass":
                mult *= max(0.8, min(1.2, 1.0 + float(sc["pass_per_point"]) * s))
            else:
                mult *= max(0.8, min(1.2, 1.0 - float(sc["rush_per_point"]) * s))
        total = ctx.expected_total()
        if total is not None and float(sc.get("league_total", 45.0)) > 0:
            mult *= max(0.85, min(1.15, (total / float(sc["league_total"])) ** float(sc["total_elasticity"])))
        return mult

    # ---- the projection -----------------------------------------------------

    def project(self, market: PropMarket, player: str, ctx: GameContext, team: Optional[str] = None) -> Projection:
        """Project ``market`` for ``player`` in the game described by ``ctx`` (raises PropModelError when it cannot)."""
        season, week = ctx.season, ctx.week
        teams = [ctx.away, ctx.home]
        pid = self.db.find_player(player, season, teams)
        if pid is None:
            raise PropModelError(f"player '{player}' not found in the {season} / {season - 1} nflverse stats")
        cur = self.db.games_for(pid, season, before_week=week)
        prev = self.db.games_for(pid, season - 1)
        team_abbrev = team_abbr(team) if team else (cur[-1].team if cur else self.db.current_team(pid, season))
        if team_abbrev not in (ctx.away_abbr, ctx.home_abbr):
            raise PropModelError(f"{self.db.display.get(pid, player)} last played for {team_full_name(team_abbrev or '?')}, "
                                 f"not {ctx.away} or {ctx.home}")
        opp = ctx.home_abbr if team_abbrev == ctx.away_abbr else ctx.away_abbr
        position = self.db.position.get(pid, "")
        n_games = len(cur) + len(prev)
        if n_games < int(self.params["min_games"]):
            raise PropModelError(f"insufficient history for {self.db.display.get(pid, player)} ({n_games} game(s))")
        pseudo_default = self.params["rate_pseudo_counts"]
        parts: List[Dict[str, float]] = []
        mean = 0.0
        usage_total = 0.0
        rate_repr = []
        opp_repr = []
        script_repr = []
        for stat, usage in market.parts:
            u, u_src = self._usage(cur, prev, usage, position, season, week)
            pseudo = float(market.rate_pseudo) if market.rate_pseudo is not None else float(pseudo_default.get(usage, 50))
            rate = self._rate(cur, prev, stat, usage, position, season, week, pseudo)
            opp_f = self._opp_factor(opp, stat, season, week)
            script_f = self._script(usage, ctx, team_abbrev)
            part_mean = u * script_f * rate * opp_f
            parts.append({"stat": stat, "usage_stat": usage, "usage": round(u, 3), "rate": round(rate, 4), "opp_factor": round(opp_f, 3),
                          "script_factor": round(script_f, 3), "mean": round(part_mean, 3)})
            mean += part_mean
            usage_total += u * script_f
            rate_repr.append(f"{rate:.3g} {stat.replace('_', ' ')}/{ {'attempts': 'attempt', 'carries': 'carry', 'targets': 'target'}.get(usage, usage) }")
            opp_repr.append(f"{opp_f:.2f}")
            script_repr.append(f"{script_f:.2f}")
        dist = market.distribution
        dispersion = market.dispersion
        p_yes: Optional[float] = None
        if dist in ("normal", "gamma"):
            sd = float(market.sd_intercept) + float(market.sd_slope) * mean
        elif dist == "negbin":
            sd = math.sqrt(mean + mean * mean / max(float(dispersion or 10.0), 1e-6))
        else:  # poisson / first_td
            sd = math.sqrt(max(mean, 1e-9))
        if market.is_yes_no:
            lam = max(mean, 1e-9)
            p_yes = 1.0 - math.exp(-lam)
            if dist == "first_td":
                p_win = ctx.win_prob(team_abbrev)
                p_team_first = 0.5 if p_win is None else 0.5 + float(self.params["first_td_team_factor"]) * (p_win - 0.5)
                pts = ctx.team_points(team_abbrev)
                lam_team = max(0.5, (pts or 22.0) * float(self.params.get("td_points_per_td", 0.093)))
                p_yes = max(0.0, min(1.0, p_team_first * min(1.0, lam / lam_team)))
        note = (f"proj {mean:.2f} +/- {sd:.2f} ({dist}); usage {usage_total:.1f}/g ({u_src} prior); rate {', '.join(rate_repr)}; "
                f"opp x{'/'.join(opp_repr)}; script x{'/'.join(script_repr)}; history {len(cur)}g {season}"
                + (f" + {len(prev)}g {season - 1}" if prev else ""))
        if p_yes is not None:
            note = f"P(yes) {p_yes:.1%}; " + note
        return Projection(market=market.key, player=self.db.display.get(pid, player), player_id=pid, position=position, team=team_abbrev,
                          opp=opp, mean=mean, sd=sd, distribution=dist, dispersion=dispersion, n_games=n_games, usage=usage_total,
                          rate=parts[0]["rate"] if parts else 0.0, opp_factor=parts[0]["opp_factor"] if parts else 1.0,
                          script_factor=parts[0]["script_factor"] if parts else 1.0, parts=parts, p_yes=p_yes, note=note)


# ---------------------------------------------------------------------------
# Filling lines.csv rows (called by build_lines.py)
# ---------------------------------------------------------------------------


def fill_model_probs(rows: List[Dict[str, Any]], contexts: Dict[Tuple[int, str, str], GameContext],
                     model: Optional[PropModel] = None, season: Optional[int] = None, offline: bool = False) -> Dict[str, int]:
    """Set ``model_prob`` / ``model_note`` / ``blocked`` / ``position`` / ``player_id`` on prop rows in place.

    ``rows`` are lines.csv dicts from ``build_lines.expand_prop``; ``contexts``
    is keyed by ``(week, away, home)``. Returns counts: projected, blocked,
    missing (no projection, left with no edge).
    """
    reg = market_registry.registry()
    counts = {"projected": 0, "blocked": 0, "missing": 0}
    prop_rows = [r for r in rows if r.get("player") and reg.is_prop(str(r.get("market")))]
    if not prop_rows:
        return counts
    if model is None:
        season = season or next((c.season for c in contexts.values()), None) or nfl_season_year()
        model = PropModel.load(season=season, offline=offline)
    block_statuses = {str(s).lower() for s in model.params.get("block_injury_status", [])}
    cache: Dict[Tuple[str, str, int, str, str], Any] = {}
    for r in prop_rows:
        market = reg.get(str(r["market"]))
        if market is None:
            continue
        key = (int(r["week"]), str(r["away"]), str(r["home"]))
        ctx = contexts.get(key)
        if ctx is None:
            r["blocked"] = "no game context (not in week_inputs.csv)"
            counts["blocked"] += 1
            continue
        parsed = reg.parse_prop_selection(str(r["selection"]))
        if parsed is None:
            r["blocked"] = "unreadable selection"
            counts["blocked"] += 1
            continue
        _mk, player, line, direction = parsed
        ckey = (market.key, player, ctx.week, ctx.away, ctx.home)
        if ckey not in cache:
            try:
                cache[ckey] = model.project(market, player, ctx)
            except PropModelError as exc:
                cache[ckey] = exc
        proj = cache[ckey]
        if isinstance(proj, PropModelError):
            r["blocked"] = f"no projection ({proj})"
            r["model_note"] = ""
            counts["blocked"] += 1
            continue
        status = model.injury_status(ctx.week, proj.team, proj.player)
        r["position"] = r.get("position") or proj.position
        r["player_id"] = proj.player_id
        if status and status.lower() in block_statuses:
            r["blocked"] = f"ruled out (injury report: {status})"
            r["model_note"] = proj.note
            counts["blocked"] += 1
            continue
        try:
            p, push = proj.probabilities(line, direction)
        except PropModelError as exc:
            r["blocked"] = f"no projection ({exc})"
            counts["blocked"] += 1
            continue
        p = min(0.995, max(0.005, p))
        r["model_prob"] = round(p, 4)
        note = proj.note
        if push > 0.001:
            note += f"; push {push:.1%} (prob is conditional on no push)"
        if status:
            note += f"; injury report: {status}"
        r["model_note"] = note
        counts["projected"] += 1
    return counts


def contexts_from_inputs(inputs_path: str, week: Optional[int] = None, season: Optional[int] = None) -> Dict[Tuple[int, str, str], GameContext]:
    """Read week_inputs.csv into GameContext objects keyed by (week, away, home)."""
    out: Dict[Tuple[int, str, str], GameContext] = {}
    if not os.path.isfile(inputs_path):
        return out
    with open(inputs_path, newline="", encoding="utf-8-sig") as fh:
        for raw in csv.DictReader(fh):
            # extra fields beyond the header land under key None as a list; ignore them
            r = {str(k).strip().lower(): str(v if v is not None else "").strip() for k, v in raw.items() if k is not None}
            try:
                wk = int(float(r.get("week", "")))
            except ValueError:
                continue
            if week is not None and wk != week:
                continue

            def num(key: str) -> Optional[float]:
                v = r.get(key, "")
                try:
                    return float(v.replace("+", "")) if v != "" else None
                except ValueError:
                    return None

            yr = season
            if yr is None:
                d = r.get("date", "")
                try:
                    yr = nfl_season_year(_dt.date.fromisoformat(d)) if d else nfl_season_year()
                except ValueError:
                    yr = nfl_season_year()
            ctx = GameContext(season=yr, week=wk, away=r.get("away", ""), home=r.get("home", ""), home_spread=num("home_spread"),
                              total=num("total"), model_home_margin=num("model_home_margin"), model_total=num("model_total"),
                              model_home_win_prob=num("model_home_win_prob"))
            out[(wk, ctx.away, ctx.home)] = ctx
    return out


# ---------------------------------------------------------------------------
# Walk-forward calibration
# ---------------------------------------------------------------------------

_MIN_USAGE: Dict[str, float] = {"attempts": 15.0, "carries": 8.0, "targets": 4.0}


def calibrate(season: int, weeks: Sequence[int], markets: Optional[Sequence[str]] = None, offline: bool = False,
              directory: Optional[str] = None, seed: int = 7) -> Dict[str, Any]:
    """Replay ``season`` week by week using only prior data; score the projections per market."""
    reg = market_registry.registry()
    model = PropModel.load(season=season, offline=offline, directory=directory, with_injuries=False)
    schedule = load_schedule(fetch_file("games", None, directory, offline=offline), season)
    rng = random.Random(seed)
    wanted = [reg.props[k] for k in (markets or reg.prop_keys) if k in reg.props]
    # First-TD outcomes are not in the weekly stats (they need play-by-play), so that market cannot be scored here.
    uncalibrated = [m.key for m in wanted if m.distribution == "first_td"]
    wanted = [m for m in wanted if m.distribution != "first_td"]
    records: Dict[str, List[Dict[str, float]]] = {}
    skipped = {"insufficient": 0, "no_player": 0, "low_usage": 0}
    for wk in weeks:
        for g in [s for s in schedule if s.week == wk and s.home_score is not None]:
            ctx = GameContext(season=season, week=wk, away=g.away, home=g.home, home_spread=(-g.spread_line if g.spread_line is not None else None),
                              total=g.total_line, model_home_margin=g.spread_line, model_total=None)
            for team in (g.away, g.home):
                for pg in model.db.players_in_week(season, wk, team):
                    for market in wanted:
                        if market.positions and pg.position not in market.positions:
                            continue
                        try:
                            proj = model.project(market, pg.name, ctx, team=team)
                        except PropModelError as exc:
                            skipped["insufficient" if "insufficient" in str(exc) else "no_player"] += 1
                            continue
                        if proj.usage < min(_MIN_USAGE.get(u, 0.0) for _s, u in market.parts):
                            skipped["low_usage"] += 1
                            continue
                        actual = sum(pg.stats.get(s, 0.0) for s, _u in market.parts)
                        rec: Dict[str, float] = {"mean": proj.mean, "sd": proj.sd, "actual": actual}
                        if market.is_yes_no:
                            rec["p"] = float(proj.p_yes or 0.0)
                            rec["hit"] = 1.0 if actual >= 1 else 0.0
                        else:
                            lo = distribution_cdf(proj.distribution, proj.mean, proj.sd, actual - 1, proj.dispersion)
                            hi = distribution_cdf(proj.distribution, proj.mean, proj.sd, actual, proj.dispersion)
                            rec["pit"] = lo + rng.random() * max(0.0, hi - lo)
                            for tag, offset in (("lo", -1.0), ("mid", 0.0), ("hi", 1.0)):
                                line = math.floor(proj.mean + offset * proj.sd) + 0.5
                                over, _under, _push = tail_probabilities(proj.distribution, proj.mean, proj.sd, line, proj.dispersion)
                                rec[f"p_{tag}"] = over
                                rec[f"hit_{tag}"] = 1.0 if actual > line else 0.0
                        records.setdefault(market.key, []).append(rec)
    summary: Dict[str, Any] = {"season": season, "weeks": list(weeks), "skipped": skipped, "markets": {},
                               "uncalibrated": uncalibrated}
    for key, recs in records.items():
        n = len(recs)
        m = {"n": n, "mean_proj": sum(r["mean"] for r in recs) / n, "mean_actual": sum(r["actual"] for r in recs) / n,
             "mae": sum(abs(r["actual"] - r["mean"]) for r in recs) / n}
        m["bias"] = m["mean_actual"] - m["mean_proj"]
        if "pit" in recs[0]:
            resid_sd = math.sqrt(sum((r["actual"] - r["mean"]) ** 2 for r in recs) / n)
            m["dispersion_ratio"] = resid_sd / (sum(r["sd"] for r in recs) / n)
            m["coverage_50"] = sum(1 for r in recs if 0.25 <= r["pit"] <= 0.75) / n
            m["coverage_80"] = sum(1 for r in recs if 0.10 <= r["pit"] <= 0.90) / n
            m["pit_mean"] = sum(r["pit"] for r in recs) / n
            for tag in ("lo", "mid", "hi"):
                m[f"claimed_{tag}"] = sum(r[f"p_{tag}"] for r in recs) / n
                m[f"hit_{tag}"] = sum(r[f"hit_{tag}"] for r in recs) / n
                m[f"brier_{tag}"] = sum((r[f"p_{tag}"] - r[f"hit_{tag}"]) ** 2 for r in recs) / n
        else:
            m["claimed"] = sum(r["p"] for r in recs) / n
            m["hit"] = sum(r["hit"] for r in recs) / n
            m["brier"] = sum((r["p"] - r["hit"]) ** 2 for r in recs) / n
            buckets: Dict[str, Dict[str, float]] = {}
            for r in recs:
                b = f"{int(r['p'] * 10) * 10:02d}-{int(r['p'] * 10) * 10 + 10:02d}%"
                bb = buckets.setdefault(b, {"n": 0, "claimed": 0.0, "hit": 0.0})
                bb["n"] += 1
                bb["claimed"] += r["p"]
                bb["hit"] += r["hit"]
            m["buckets"] = {b: {"n": v["n"], "claimed": v["claimed"] / v["n"], "hit": v["hit"] / v["n"]} for b, v in sorted(buckets.items())}
        summary["markets"][key] = m
    return summary


def render_calibration(summary: Dict[str, Any]) -> str:
    reg = market_registry.registry()
    w = 80
    out = ["=" * w, "PROP MODEL CALIBRATION (walk-forward, EXPERIMENTAL model)".center(w).rstrip(),
           f"season {summary['season']}, weeks {summary['weeks'][0]}-{summary['weeks'][-1]}; projections use only data before each week".center(w).rstrip(),
           "=" * w, ""]
    out.append("OVER/UNDER MARKETS   (hit = share of actuals over a synthetic line at proj -1sd / median / +1sd)")
    out.append(f"  {'Market':<18} {'n':>5} {'Bias':>7} {'MAE':>6} {'Disp':>5} {'Cov50':>6} {'Cov80':>6} {'Claim/Hit -1sd':>15} {'median':>12} {'+1sd':>12}")
    for key, m in summary["markets"].items():
        if "pit_mean" not in m:
            continue
        out.append(f"  {reg.label(key):<18} {m['n']:>5} {m['bias']:>+7.2f} {m['mae']:>6.2f} {m['dispersion_ratio']:>5.2f} {m['coverage_50']:>6.0%} {m['coverage_80']:>6.0%} "
                   f"{m['claimed_lo']:>6.0%}/{m['hit_lo']:<7.0%} {m['claimed_mid']:>5.0%}/{m['hit_mid']:<5.0%} {m['claimed_hi']:>5.0%}/{m['hit_hi']:<5.0%}")
    out.append("")
    out.append("YES/NO MARKETS       (claimed probability vs. realised rate, by probability bucket)")
    for key, m in summary["markets"].items():
        if "pit_mean" in m:
            continue
        out.append(f"  {reg.label(key):<18} n={m['n']:<5} claimed {m['claimed']:.1%}  hit {m['hit']:.1%}  Brier {m['brier']:.3f}")
        for b, v in m.get("buckets", {}).items():
            out.append(f"      {b:<9} n={v['n']:<5} claimed {v['claimed']:.1%}  hit {v['hit']:.1%}")
    out.append("")
    out.append("HOW TO READ THIS")
    for line in (
        "Bias: actual minus projected mean (0 is perfect; negative means the model projects too high).",
        "Disp: realised residual sd / projected sd (1.0 means the spread is right; >1 the model is overconfident).",
        "Cov50 / Cov80: share of actuals inside the projected 50% / 80% interval (targets 50% / 80%).",
        "Claim/Hit: the model's average P(over) at each synthetic line vs. how often the over actually hit.",
        "The experimental badge should stay on until claimed and hit agree within 3 points on n >= 200.",
        f"Skipped: {summary['skipped']}",
    ):
        out.append(f"  {line}")
    if summary.get("uncalibrated"):
        out.append(f"  Not scored: {', '.join(reg.label(k) for k in summary['uncalibrated'])} (needs play-by-play data to know who scored first;")
        out.append("  keep it high-variance and experimental).")
    out.append("-" * w)
    out.append("For simulation / research purposes only. In the US, help is available any time at 1-800-GAMBLER.".center(w).rstrip())
    out.append("=" * w)
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _synthetic_db(seed: int = 3) -> StatsDB:
    """A small fabricated league: two seasons, 8 teams, deterministic stats."""
    rng = random.Random(seed)
    teams = ["DAL", "TB", "PHI", "JAX", "MIN", "NO", "CHI", "GB"]
    games: List[PlayerGame] = []
    roster = {}
    for t in teams:
        roster[t] = [("%s_qb" % t, f"{t} Quarterback", "QB"), ("%s_rb" % t, f"{t} Runner", "RB"),
                     ("%s_wr1" % t, f"{t} Receiver One", "WR"), ("%s_wr2" % t, f"{t} Receiver Two", "WR"), ("%s_te" % t, f"{t} Tightend", "TE")]
    for season in (2025, 2026):
        n_weeks = 17 if season == 2025 else 4
        for wk in range(1, n_weeks + 1):
            order = teams[:]
            rng.shuffle(order)
            for i in range(0, len(order), 2):
                a, h = order[i], order[i + 1]
                for team, opp in ((a, h), (h, a)):
                    att = max(15, int(rng.gauss(33, 5)))
                    comp = int(att * 0.65)
                    pyds = max(80, int(rng.gauss(att * 7.2, 45)))
                    qb = roster[team][0]
                    games.append(PlayerGame(season, wk, qb[0], qb[1], "QB", team, opp,
                                            {"attempts": att, "completions": comp, "passing_yards": pyds, "passing_tds": rng.choice([0, 1, 1, 2, 2, 3]),
                                             "passing_interceptions": rng.choice([0, 0, 0, 1, 1, 2]), "carries": 3, "rushing_yards": 12,
                                             "rushing_tds": 0, "targets": 0, "receptions": 0, "receiving_yards": 0, "receiving_tds": 0}))
                    rb = roster[team][1]
                    car = max(5, int(rng.gauss(16, 4)))
                    games.append(PlayerGame(season, wk, rb[0], rb[1], "RB", team, opp,
                                            {"attempts": 0, "completions": 0, "passing_yards": 0, "passing_tds": 0, "passing_interceptions": 0,
                                             "carries": car, "rushing_yards": max(0, int(rng.gauss(car * 4.3, 25))), "rushing_tds": rng.choice([0, 0, 1, 1, 2]),
                                             "targets": 3, "receptions": 2, "receiving_yards": 15, "receiving_tds": 0}))
                    for j, share in ((2, 0.28), (3, 0.18), (4, 0.16)):
                        p = roster[team][j]
                        tg = max(1, int(round(att * share + rng.gauss(0, 1.5))))
                        games.append(PlayerGame(season, wk, p[0], p[1], p[2], team, opp,
                                                {"attempts": 0, "completions": 0, "passing_yards": 0, "passing_tds": 0, "passing_interceptions": 0,
                                                 "carries": 0, "rushing_yards": 0, "rushing_tds": 0, "targets": tg, "receptions": int(tg * 0.66),
                                                 "receiving_yards": max(0, int(rng.gauss(tg * 8.5, 20))), "receiving_tds": rng.choice([0, 0, 0, 1])}))
    return StatsDB(games)


def _selftest() -> int:
    failures = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal failures
        print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
        if not cond:
            failures += 1

    print("prop_model self-test")
    check(normalize_name("A.J. Brown") == "aj brown" and normalize_name("Brian Thomas Jr.") == "brian thomas"
          and normalize_name("DJ Moore") == normalize_name("D.J. Moore") and normalize_name("Ja'Marr Chase") == "jamarr chase", "name normalisation")
    check(team_abbr("Los Angeles Rams") == "LA" and team_abbr("LAR") == "LA" and team_abbr("Cowboys") == "DAL" and team_abbr("??") is None
          and team_full_name("JAX") == "Jacksonville Jaguars", "team abbreviations")

    # Distributions
    check(abs(regularized_lower_gamma(1.0, 1.0) - (1 - math.exp(-1))) < 1e-9, "incomplete gamma P(1, 1) = 1 - e^-1")
    check(abs(regularized_lower_gamma(2.5, 1.5) - 0.3002) < 2e-3 and abs(regularized_lower_gamma(5.0, 12.0) - 0.9924) < 2e-3, "incomplete gamma series and continued fraction")
    over, under, push = tail_probabilities("normal", 264.0, 40.0, 264.5)
    check(abs(over - 0.5) < 0.006 and abs(over + under - 1) < 1e-9 and push == 0.0, "normal half-line splits around the mean")
    over_i, under_i, push_i = tail_probabilities("normal", 264.0, 40.0, 264.0)
    check(push_i > 0 and abs(over_i + under_i + push_i - 1) < 1e-9 and abs(over_i - under_i) < 1e-9, "whole-number line carries push mass")
    g_over, g_under, _ = tail_probabilities("gamma", 70.0, 40.0, 70.5)
    check(g_over < 0.5 < g_under + 0.1 and abs(g_over + g_under - 1) < 1e-9, "gamma is right-skewed: median below mean")
    p_over, p_under, p_push = tail_probabilities("poisson", 1.6, 0.0, 1.5)
    check(abs(p_under - (math.exp(-1.6) * (1 + 1.6))) < 1e-9 and p_push == 0.0 and abs(p_over + p_under - 1) < 1e-9, "poisson tail at 1.5")
    nb = negbin_pmf_table(6.0, 10.0, 60)
    check(abs(sum(nb) - 1.0) < 1e-6 and abs(sum(k * v for k, v in enumerate(nb)) - 6.0) < 1e-3, "negbin pmf sums to 1 with the right mean")
    nb_var = sum((k - 6.0) ** 2 * v for k, v in enumerate(nb))
    check(abs(nb_var - (6.0 + 36.0 / 10.0)) < 1e-2, "negbin variance = mean + mean^2/dispersion")
    o2, u2, pu2 = tail_probabilities("negbin", 6.0, 10.0, 6.0, dispersion=10.0)
    check(pu2 > 0.05 and abs(o2 + u2 + pu2 - 1) < 1e-9, "negbin whole-number push")

    # Projection on a synthetic league
    reg = market_registry.registry()
    db = _synthetic_db()
    model = PropModel(db, {(4, "DAL", "dal receiver two"): "Out", (4, "TB", "tb runner"): "Questionable"}, reg, season=2026)
    ctx = GameContext(season=2026, week=4, away="Tampa Bay Buccaneers", home="Dallas Cowboys", home_spread=-7.0, total=48.0,
                      model_home_margin=8.0, model_total=49.0, model_home_win_prob=0.72)
    check(ctx.expected_margin("DAL") == 8.0 and ctx.expected_margin("TB") == -8.0 and ctx.win_prob("TB") is not None
          and abs(ctx.win_prob("TB") - 0.28) < 1e-9 and ctx.team_points("DAL") == 28.5, "game context: margins, win prob, team points")
    py = model.project(reg.props["passing_yards"], "DAL Quarterback", ctx)
    check(py.position == "QB" and py.team == "DAL" and py.opp == "TB" and 150 < py.mean < 330 and py.sd > 15 and py.n_games == 20,
          f"QB passing yards projection ({py.mean:.1f} +/- {py.sd:.1f}, {py.n_games} games)")
    p_over, push = py.probabilities(py.mean - 0.5, "Over")
    check(0.45 < p_over < 0.6 and push == 0.0, "probability near 50% at the projected mean")
    p_under_low, _ = py.probabilities(math.floor(py.mean - 1.5 * py.sd) + 0.5, "Under")
    check(p_under_low < 0.1, f"1.5 sd below the projection, Under is unlikely ({p_under_low:.1%})")
    ry = model.project(reg.props["rushing_yards"], "TB Runner", ctx)
    check(ry.distribution == "gamma" and 30 < ry.mean < 120 and ry.script_factor < 1.0, f"RB rushing projection with underdog rush discount ({ry.mean:.1f}, script {ry.script_factor:.2f})")
    dal_rb = model.project(reg.props["rushing_yards"], "DAL Runner", ctx)
    check(dal_rb.script_factor > 1.0, "favourite gets a rush bonus")
    qb_pass_script = model.project(reg.props["passing_yards"], "TB Quarterback", ctx)
    check(qb_pass_script.script_factor > py.script_factor, "underdog QB passes more than favourite QB")
    rec = model.project(reg.props["receptions"], "DAL Receiver One", ctx)
    check(rec.distribution == "negbin" and 2 < rec.mean < 12, f"receptions projection ({rec.mean:.2f})")
    rr = model.project(reg.props["rush_rec_yards"], "TB Runner", ctx)
    check(len(rr.parts) == 2 and rr.mean > ry.mean, "rush + rec sums two parts")
    atd = model.project(reg.props["anytime_td"], "DAL Runner", ctx)
    check(atd.p_yes is not None and 0.2 < atd.p_yes < 0.95 and atd.probabilities(None, "Yes")[0] == atd.p_yes
          and abs(atd.probabilities(None, "No")[0] - (1 - atd.p_yes)) < 1e-12, f"anytime TD probability ({atd.p_yes:.1%})")
    ftd = model.project(reg.props["first_td"], "DAL Runner", ctx)
    check(ftd.p_yes is not None and 0 < ftd.p_yes < atd.p_yes, f"first TD is rarer than anytime TD ({ftd.p_yes:.1%} vs {atd.p_yes:.1%})")
    ints = model.project(reg.props["interceptions"], "DAL Quarterback", ctx)
    check(ints.distribution == "poisson" and 0.2 < ints.mean < 2.0, f"interceptions projection ({ints.mean:.2f})")
    for name, why in (("Nobody Real", "unknown player"), ("MIN Quarterback", "player on neither team")):
        try:
            model.project(reg.props["passing_yards"], name, ctx)
            check(False, f"{why} raises")
        except PropModelError:
            check(True, f"{why} raises PropModelError")
    check(model.injury_status(4, "Dallas Cowboys", "DAL Receiver Two") == "Out", "injury lookup by team and normalised name")
    check(model.db.find_player("dal quarterback", 2026, ["Dallas Cowboys"]) == "DAL_qb" and model.db.find_player("Quarterback", 2026, ["TB"]) == "TB_qb",
          "player lookup with nickname fallback")
    early = GameContext(season=2026, week=1, away="TB", home="DAL", home_spread=-3.0, total=45.0)
    w1 = model.project(reg.props["passing_yards"], "DAL Quarterback", early)
    check(w1.n_games == 17 and "17g 2025" in w1.note and w1.mean > 150, "week 1 projects from last season only")

    # fill_model_probs on lines.csv-style rows
    rows = [
        {"week": 4, "away": "Tampa Bay Buccaneers", "home": "Dallas Cowboys", "market": "passing_yards", "selection": "DAL Quarterback Over 240.5 Passing Yards",
         "american_odds": -110, "model_prob": "", "player": "DAL Quarterback", "position": "", "blocked": "", "model_note": "", "notes": ""},
        {"week": 4, "away": "Tampa Bay Buccaneers", "home": "Dallas Cowboys", "market": "passing_yards", "selection": "DAL Quarterback Under 240.5 Passing Yards",
         "american_odds": -110, "model_prob": "", "player": "DAL Quarterback", "position": "", "blocked": "", "model_note": "", "notes": ""},
        {"week": 4, "away": "Tampa Bay Buccaneers", "home": "Dallas Cowboys", "market": "receiving_yards", "selection": "DAL Receiver Two Over 40.5 Receiving Yards",
         "american_odds": -110, "model_prob": "", "player": "DAL Receiver Two", "position": "", "blocked": "", "model_note": "", "notes": ""},
        {"week": 4, "away": "Tampa Bay Buccaneers", "home": "Dallas Cowboys", "market": "anytime_td", "selection": "Nobody Real Anytime TD",
         "american_odds": 150, "model_prob": "", "player": "Nobody Real", "position": "", "blocked": "", "model_note": "", "notes": ""},
        {"week": 4, "away": "Tampa Bay Buccaneers", "home": "Dallas Cowboys", "market": "spread", "selection": "Dallas Cowboys -7", "american_odds": -110,
         "model_prob": 0.6, "player": "", "position": "", "blocked": "", "model_note": "", "notes": ""},
    ]
    counts = fill_model_probs(rows, {(4, "Tampa Bay Buccaneers", "Dallas Cowboys"): ctx}, model=model)
    check(counts == {"projected": 2, "blocked": 2, "missing": 0}, f"fill_model_probs counts {counts}")
    check(abs(float(rows[0]["model_prob"]) + float(rows[1]["model_prob"]) - 1) < 1e-3 and rows[0]["position"] == "QB" and rows[0]["player_id"] == "DAL_qb"
          and "proj" in rows[0]["model_note"], "Over/Under probabilities complement and carry the projection note")
    check(rows[2]["blocked"].startswith("ruled out") and rows[2]["model_prob"] == "", "player listed Out is blocked, not priced")
    check(rows[3]["blocked"].startswith("no projection") and rows[4]["model_prob"] == 0.6, "unknown player blocked; game rows untouched")

    # Calibration machinery on the synthetic league (2026 weeks 3-4, prior 2025)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        schedule = os.path.join(tmp, "games.csv")
        with open(schedule, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["season", "game_type", "week", "gameday", "away_team", "home_team", "away_score", "home_score", "spread_line", "total_line"])
            for wk in (3, 4):
                seen = set()
                for pg in db.players_in_week(2026, wk):
                    key = tuple(sorted((pg.team, pg.opp)))
                    if key in seen:
                        continue
                    seen.add(key)
                    w.writerow([2026, "REG", wk, "2026-09-28", pg.team, pg.opp, 20, 24, 3.5, 45.5])
        sched = load_schedule(schedule, 2026)
        check(len(sched) == 8 and sched[0].spread_line == 3.5, "schedule loader")
        # Monkeypatch loading so calibrate() uses the synthetic model
        saved_load, saved_fetch = PropModel.load, fetch_file
        try:
            PropModel.load = classmethod(lambda cls, **kw: model)  # type: ignore[assignment]
            globals()["fetch_file"] = lambda *a, **k: schedule
            summ = calibrate(2026, [3, 4], markets=["passing_yards", "receptions", "anytime_td"])
        finally:
            PropModel.load, globals()["fetch_file"] = saved_load, saved_fetch
        py_cal = summ["markets"]["passing_yards"]
        check(py_cal["n"] == 16 and 0 < py_cal["coverage_80"] <= 1 and "brier_mid" in py_cal, f"calibration scores passing yards (n={py_cal['n']}, cov80={py_cal['coverage_80']:.0%})")
        check("anytime_td" in summ["markets"] and "buckets" in summ["markets"]["anytime_td"], "calibration scores a yes/no market")
        text = render_calibration(summ)
        check("PROP MODEL CALIBRATION" in text and "1-800-GAMBLER" in text and max(len(line) for line in text.splitlines()) <= 120, "calibration report renders")

    print(f"\n{'ALL TESTS PASSED' if failures == 0 else f'{failures} TEST(S) FAILED'}")
    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="prop_model", description="EXPERIMENTAL player-prop projections for EdgeBook AI.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--calibrate", action="store_true", help="Walk-forward calibration of a past season")
    p.add_argument("--season", type=int, default=None, help="Season to calibrate (default: last season)")
    p.add_argument("--weeks", default="5-18", help="Week range for calibration, e.g. 5-18")
    p.add_argument("--markets", default=None, help="Comma-separated prop markets (default: all)")
    p.add_argument("--offline", action="store_true", help="Use cached data only")
    p.add_argument("--out-dir", dest="out_dir", default="backtest_output")
    p.add_argument("--fetch", action="store_true", help="Download / refresh the nflverse files for the current season and exit")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.selftest:
        return _selftest()
    try:
        if args.fetch:
            season = args.season or nfl_season_year()
            for kind, yr in (("stats", season), ("stats", season - 1), ("injuries", season), ("games", None)):
                print(f"{kind:<9} {yr or '':<5} -> {fetch_file(kind, yr, offline=False, max_age_hours=0.0)}")
            return 0
        if args.calibrate:
            season = args.season or (nfl_season_year() - 1)
            lo, _, hi = args.weeks.partition("-")
            weeks = list(range(int(lo), int(hi or lo) + 1))
            markets = [m.strip() for m in args.markets.split(",")] if args.markets else None
            summ = calibrate(season, weeks, markets=markets, offline=args.offline)
            text = render_calibration(summ)
            os.makedirs(args.out_dir, exist_ok=True)
            txt = os.path.join(args.out_dir, f"prop_calibration_{season}.txt")
            js = os.path.join(args.out_dir, f"prop_calibration_{season}.json")
            with open(txt, "w", encoding="utf-8") as fh:
                fh.write(text)
            with open(js, "w", encoding="utf-8") as fh:
                json.dump(summ, fh, indent=2)
            print(text, end="")
            print(f"Saved {txt} and {js}")
            return 0
    except (PropModelError, RegistryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
