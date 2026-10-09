#!/usr/bin/env python3
"""
weekly_reporter.py
==================

Automated Pick Logger for the NFL parlay analytics pipeline.

Responsibilities
----------------
1. **Collect** the optimised parlay configurations produced by
   ``parlay_finder.py`` for the upcoming simulated NFL week.
2. **Stake** every ticket through ``staking_engine.py`` (Flat or Fractional
   Kelly) against the current bankroll, applying all safety caps.
3. **Rank & group** the tickets into "Top Recommended Parlays" sections by
   leg count (2-leg, 3-leg, ...).
4. **Render** a scannable plain-text report card and save it as
   ``weekly_parlay_report.txt`` (plus an optional machine-readable JSON
   side-car for ``backtester.py``).

Where do the parlays come from?
-------------------------------
The reporter is deliberately tolerant about its upstream source so it keeps
working whether you run it on the full MacBook pipeline or in isolation:

``--source finder`` (default)
    Imports ``parlay_finder`` and calls the first callable it finds among
    ``find_parlays``, ``generate_parlays``, ``get_parlays``,
    ``optimize_parlays``, ``build_parlays``, ``run``, ``main``. The function is
    passed ``week=`` (and ``bankroll=`` / ``top_n=``) only if its signature
    accepts them. The return value may be a list of dicts, a list of objects,
    a pandas ``DataFrame`` or a dict wrapping one of those under
    ``"parlays"`` / ``"tickets"`` / ``"results"``.
``--source json --input parlays.json``
    Loads tickets from a JSON file (a list of ticket dicts, or a dict with a
    ``"parlays"`` key). This is also the hand-off format ``parlay_finder.py``
    can write when the two scripts run as separate cron jobs.
``--source demo``
    Builds a deterministic, clearly-labelled simulated slate so the report
    pipeline can be smoke-tested with no upstream dependency.

If ``finder`` is requested but ``parlay_finder`` cannot be imported, the
reporter logs a warning and falls back to ``--input`` if given, otherwise to
the demo slate, so a weekly cron job never produces *nothing*.

Ticket data contract
--------------------
Every upstream ticket is normalised into :class:`ParlayTicket`. The raw dict
may use any of these aliases (first match wins):

* Legs: ``legs`` / ``selections`` / ``picks`` – list of leg dicts with
  ``matchup`` (``game``/``event``), ``selection`` (``line``/``pick``/``bet``),
  optional ``market`` (``spread``/``total``/``moneyline``), ``p_true``
  (``true_prob``/``prob``/``probability``) and ``american_odds``
  (``odds``/``price``) or ``decimal_odds``.
* Parlay-level: ``p_true`` and ``decimal_odds``/``american_odds`` are
  optional; when absent they are compounded from the legs assuming
  independence (correlated/SGP joint probabilities should be supplied
  upstream).
* Optional: ``ticket_id``, ``week``, ``tags``/``notes``.

Integration with backtester.py
------------------------------
::

    from weekly_reporter import build_weekly_report, ReportConfig
    report = build_weekly_report(tickets, ReportConfig(week=6, bankroll=1000))
    df = pd.DataFrame(report.to_records())   # one row per recommended ticket
    report.save_text("weekly_parlay_report.txt")
    report.save_json("weekly_parlay_report.json")

Run ``python3 weekly_reporter.py --help`` for the CLI or
``python3 weekly_reporter.py --selftest`` for the built-in tests.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import inspect
import json
import logging
import math
import os
import random
import sys
import tempfile
import textwrap
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from staking_engine import (
    StakeRecommendation,
    StakingConfig,
    StakingInputError,
    StakingMode,
    american_to_decimal,
    apply_portfolio_cap,
    compound_decimal_odds,
    compound_probability,
    decimal_to_american,
    format_american,
    safe_compute_stake,
)

__all__ = [
    "Leg",
    "ParlayTicket",
    "StakedTicket",
    "ReportConfig",
    "WeeklyReport",
    "ReporterError",
    "normalize_ticket",
    "normalize_tickets",
    "load_parlays_from_finder",
    "load_parlays_from_json",
    "generate_demo_slate",
    "collect_parlays",
    "stake_tickets",
    "select_parlays",
    "ParlaySelection",
    "group_chosen",
    "is_same_game_ticket",
    "build_weekly_report",
    "render_report_text",
    "estimate_nfl_week",
    "nfl_season_year",
    "DEFAULT_REPORT_FILENAME",
]

__version__ = "1.0.0"

logger = logging.getLogger(__name__)

DEFAULT_REPORT_FILENAME = "weekly_parlay_report.txt"
REPORT_WIDTH = 80  # plain-text column width; fits Terminal.app at default size
MIN_PARLAYS, MAX_PARLAYS, DEFAULT_MAX_PARLAYS = 1, 10, 3   # the "how many parlays" control

# Candidate entry points probed inside parlay_finder.py, in priority order.
FINDER_FUNCTION_CANDIDATES: Tuple[str, ...] = (
    "find_parlays",
    "generate_parlays",
    "get_parlays",
    "optimize_parlays",
    "build_parlays",
    "find_optimal_parlays",
    "run",
    "main",
)


class ReporterError(RuntimeError):
    """Raised for unrecoverable reporter failures (bad input file, no tickets)."""


# ---------------------------------------------------------------------------
# Canonical data structures
# ---------------------------------------------------------------------------


@dataclass
class Leg:
    """One selection inside a parlay (e.g. 'BUF -3.5' in 'KC @ BUF', or 'Dak Prescott Over 264.5 Passing Yards')."""

    matchup: str                 # "Kansas City Chiefs @ Buffalo Bills"
    selection: str               # "Buffalo Bills -3.5" / "Over 44.5" / "Chiefs ML"
    market: str = ""             # "spread" | "total" | "moneyline" | "team_total" | a prop market key
    p_true: Optional[float] = None
    decimal_odds: Optional[float] = None
    player: Optional[str] = None         # player props only
    team: Optional[str] = None
    position: Optional[str] = None
    market_label: str = ""               # "Passing Yards", "Spread", ...
    line: Optional[float] = None
    direction: Optional[str] = None      # Over / Under / Yes / No
    experimental: bool = False           # the market's model is still flagged experimental
    high_variance: bool = False
    model_note: str = ""                 # one-line provenance of the model probability

    @property
    def american_odds(self) -> Optional[int]:
        return decimal_to_american(self.decimal_odds) if self.decimal_odds else None

    @property
    def implied_prob(self) -> Optional[float]:
        return (1.0 / self.decimal_odds) if self.decimal_odds else None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["american_odds"] = self.american_odds
        d["implied_prob"] = self.implied_prob
        return d


@dataclass
class ParlayTicket:
    """A normalised parlay candidate coming out of parlay_finder."""

    ticket_id: str
    legs: List[Leg]
    p_true: float                # compounded true probability for the parlay
    decimal_odds: float          # compounded bookmaker decimal odds
    week: Optional[int] = None
    notes: str = ""
    source: str = ""             # "finder" | "json" | "demo"
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def n_legs(self) -> int:
        return len(self.legs)

    @property
    def american_odds(self) -> int:
        return decimal_to_american(self.decimal_odds)

    @property
    def implied_prob(self) -> float:
        return 1.0 / self.decimal_odds

    @property
    def edge(self) -> float:
        return self.p_true * self.decimal_odds - 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "week": self.week,
            "n_legs": self.n_legs,
            "legs": [leg.to_dict() for leg in self.legs],
            "p_true": self.p_true,
            "decimal_odds": self.decimal_odds,
            "american_odds": self.american_odds,
            "implied_prob": self.implied_prob,
            "edge": self.edge,
            "notes": self.notes,
            "source": self.source,
        }


@dataclass
class StakedTicket:
    """A :class:`ParlayTicket` paired with its :class:`StakeRecommendation`."""

    ticket: ParlayTicket
    stake: StakeRecommendation
    rank: int = 0  # 1-based rank within its leg group (0 = unranked)

    def to_dict(self) -> Dict[str, Any]:
        return {"rank": self.rank, "ticket": self.ticket.to_dict(), "stake": self.stake.to_dict()}

    def to_record(self) -> Dict[str, Any]:
        """Flat one-row dict for ``pandas.DataFrame`` in backtester.py."""
        t, s = self.ticket, self.stake
        return {
            "week": t.week,
            "ticket_id": t.ticket_id,
            "rank_in_group": self.rank,
            "n_legs": t.n_legs,
            "legs": " | ".join(f"{leg.matchup}: {leg.selection}" for leg in t.legs),
            "p_true": t.p_true,
            "implied_prob": t.implied_prob,
            "decimal_odds": t.decimal_odds,
            "american_odds": t.american_odds,
            "edge": t.edge,
            "mode": s.mode.value,
            "stake_dollars": s.stake_dollars,
            "potential_profit": s.potential_profit,
            "expected_value": s.expected_value,
            "cap_applied": s.cap_applied,
            "skipped": s.skipped,
            "reason": s.reason,
            "source": t.source,
        }


@dataclass
class ReportConfig:
    """Everything the reporter needs besides the tickets themselves."""

    week: int
    bankroll: float
    staking: StakingConfig = field(default_factory=StakingConfig)
    report_date: Optional[_dt.date] = None   # defaults to today
    season: Optional[int] = None             # defaults to report_date.year
    top_n_per_group: int = 5                 # tickets shown per leg-count section
    leg_groups: Tuple[int, ...] = (2, 3)     # leg counts that get their own section
    include_other_groups: bool = True        # show 4+-leg tickets in an "Other" section
    include_skipped: bool = True             # list $0 tickets in a "Passed" appendix
    hide_empty_groups: bool = True           # drop a leg-count section with no candidates at all
    portfolio_cap_pct: Optional[float] = 0.15  # None disables the weekly exposure cap
    title: str = "NFL PARLAY WEEKLY REPORT CARD"
    source_label: str = ""
    slate_summary: Optional[Dict[str, Any]] = None   # parlay_finder's per-market filter diagnostics, when available
    max_parlays: int = 3                     # "how many parlays": a MAXIMUM, 1..10; never relaxes a filter to reach it
    count_same_game: bool = False            # False: same-game (SGP-priced) tickets are shown apart and not counted

    def __post_init__(self) -> None:
        if not isinstance(self.week, int) or isinstance(self.week, bool) or self.week < 1:
            raise ReporterError(f"week must be a positive integer, got {self.week!r}")
        try:
            self.max_parlays = int(self.max_parlays)
        except (TypeError, ValueError) as exc:
            raise ReporterError(f"max_parlays must be a whole number, got {self.max_parlays!r}") from exc
        if not MIN_PARLAYS <= self.max_parlays <= MAX_PARLAYS:
            raise ReporterError(f"max_parlays must be between {MIN_PARLAYS} and {MAX_PARLAYS}, got {self.max_parlays}")
        try:
            self.bankroll = float(self.bankroll)
        except (TypeError, ValueError) as exc:
            raise ReporterError(f"bankroll must be numeric, got {self.bankroll!r}") from exc
        if math.isnan(self.bankroll) or math.isinf(self.bankroll) or self.bankroll < 0:
            raise ReporterError(f"bankroll must be a finite non-negative number, got {self.bankroll}")
        if self.report_date is None:
            self.report_date = _dt.date.today()
        if isinstance(self.report_date, _dt.datetime):
            self.report_date = self.report_date.date()
        if self.season is None:
            self.season = nfl_season_year(self.report_date)
        if self.top_n_per_group < 1:
            raise ReporterError("top_n_per_group must be >= 1")
        if not isinstance(self.staking, StakingConfig):
            self.staking = StakingConfig.from_dict(dict(self.staking))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "week": self.week,
            "season": self.season,
            "report_date": self.report_date.isoformat() if self.report_date else None,
            "bankroll": self.bankroll,
            "staking": self.staking.to_dict(),
            "top_n_per_group": self.top_n_per_group,
            "leg_groups": list(self.leg_groups),
            "portfolio_cap_pct": self.portfolio_cap_pct,
            "source_label": self.source_label,
            "max_parlays": self.max_parlays,
            "count_same_game": self.count_same_game,
        }


class TicketBatch(list):
    """A list of tickets that also carries the finder's ``slate_summary`` (per-market filter counts)."""

    slate_summary: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Normalisation of upstream tickets
# ---------------------------------------------------------------------------

_LEG_KEYS = ("legs", "selections", "picks", "bets")
_MATCHUP_KEYS = ("matchup", "game", "event", "match", "fixture")
_SELECTION_KEYS = ("selection", "line", "pick", "bet", "target", "side")
_MARKET_KEYS = ("market", "bet_type", "type")
_PROB_KEYS = ("p_true", "true_prob", "true_probability", "prob", "probability", "win_prob", "model_prob")
_AMERICAN_KEYS = ("american_odds", "odds_american", "american", "odds", "price")
_DECIMAL_KEYS = ("decimal_odds", "odds_decimal", "decimal")
_ID_KEYS = ("ticket_id", "id", "parlay_id", "name")
_NOTES_KEYS = ("notes", "note", "tags", "comment", "rationale")


def _first(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    """Return the first non-None value for any of ``keys`` in ``d``."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _to_float(value: Any, name: str) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise ReporterError(f"{name} must be numeric, got {value!r}") from exc
    if math.isnan(f) or math.isinf(f):
        raise ReporterError(f"{name} must be finite, got {f}")
    return f


def _as_dict(obj: Any) -> Dict[str, Any]:
    """Coerce dataclasses / objects / pandas rows into a plain dict."""
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            d = obj.to_dict()
            if isinstance(d, dict):
                return d
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(obj, "_asdict"):  # namedtuple
        return dict(obj._asdict())
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    raise ReporterError(f"Cannot interpret ticket of type {type(obj).__name__}")


def _normalize_leg(raw: Any, idx: int) -> Leg:
    if isinstance(raw, str):
        # Allow bare strings such as "KC @ BUF: BUF -3.5" or "Over 44.5"
        if ":" in raw:
            matchup, selection = raw.split(":", 1)
        else:
            matchup, selection = "", raw
        return Leg(matchup=matchup.strip(), selection=selection.strip())

    d = _as_dict(raw)
    matchup = _first(d, _MATCHUP_KEYS)
    if matchup is None:
        away, home = d.get("away"), d.get("home")
        matchup = f"{away} @ {home}" if away and home else f"Game {idx + 1}"
    selection = _first(d, _SELECTION_KEYS)
    if selection is None:
        team, spread, total = d.get("team"), d.get("spread"), d.get("total")
        if team is not None and spread is not None:
            selection = f"{team} {float(spread):+g}"
        elif total is not None:
            selection = f"{str(d.get('direction', 'Over')).title()} {float(total):g}"
        elif team is not None:
            selection = f"{team} ML"
        else:
            raise ReporterError(f"Leg {idx + 1} has no selection/line")
    market = _first(d, _MARKET_KEYS) or ""

    p = _first(d, _PROB_KEYS)
    p_true = None if p is None else _to_float(p, f"leg {idx + 1} p_true")
    if p_true is not None and not 0.0 <= p_true <= 1.0:
        # Tolerate percentages such as 55.0 by scaling once.
        if 1.0 < p_true <= 100.0:
            p_true /= 100.0
        else:
            raise ReporterError(f"Leg {idx + 1} p_true out of range: {p_true}")

    dec = _first(d, _DECIMAL_KEYS)
    amer = _first(d, _AMERICAN_KEYS)
    decimal_odds: Optional[float] = None
    try:
        if dec is not None:
            decimal_odds = _to_float(dec, f"leg {idx + 1} decimal_odds")
        elif amer is not None:
            decimal_odds = american_to_decimal(amer)
    except StakingInputError as exc:
        raise ReporterError(f"Leg {idx + 1} odds invalid: {exc}") from exc

    def _opt_text(key: str) -> Optional[str]:
        v = d.get(key)
        return str(v).strip() if v not in (None, "") else None

    line_v = d.get("line")
    try:
        line_f = float(line_v) if line_v not in (None, "") else None
    except (TypeError, ValueError):
        line_f = None
    return Leg(
        matchup=str(matchup).strip(),
        selection=str(selection).strip(),
        market=str(market).strip().lower(),
        p_true=p_true,
        decimal_odds=decimal_odds,
        player=_opt_text("player"),
        team=_opt_text("team"),
        position=_opt_text("position"),
        market_label=_opt_text("market_label") or "",
        line=line_f,
        direction=_opt_text("direction"),
        experimental=bool(d.get("experimental", False)),
        high_variance=bool(d.get("high_variance", False)),
        model_note=_opt_text("model_note") or "",
    )


def normalize_ticket(raw: Any, default_id: str = "", source: str = "", week: Optional[int] = None) -> ParlayTicket:
    """Convert any upstream ticket representation into a :class:`ParlayTicket`.

    Raises :class:`ReporterError` with a precise message when the ticket is
    unusable (no legs, missing probability, invalid odds, ...).
    """
    d = _as_dict(raw)

    raw_legs = _first(d, _LEG_KEYS)
    if not raw_legs:
        raise ReporterError("Ticket has no legs")
    if isinstance(raw_legs, dict):  # {"leg1": {...}, "leg2": {...}}
        raw_legs = list(raw_legs.values())
    legs = [_normalize_leg(leg, i) for i, leg in enumerate(raw_legs)]
    if len(legs) < 2:
        raise ReporterError(f"A parlay needs at least 2 legs, got {len(legs)}")

    # ---- parlay-level true probability -------------------------------------
    p = _first(d, _PROB_KEYS)
    if p is not None:
        p_true = _to_float(p, "p_true")
        if 1.0 < p_true <= 100.0:
            p_true /= 100.0
    else:
        leg_probs = [leg.p_true for leg in legs]
        if any(lp is None for lp in leg_probs):
            raise ReporterError("Ticket lacks p_true and not every leg carries a probability")
        p_true = compound_probability([lp for lp in leg_probs if lp is not None])
    if not 0.0 <= p_true <= 1.0:
        raise ReporterError(f"p_true out of range: {p_true}")

    # ---- parlay-level odds -------------------------------------------------
    dec = _first(d, _DECIMAL_KEYS)
    amer = _first(d, _AMERICAN_KEYS)
    try:
        if dec is not None:
            decimal_odds = _to_float(dec, "decimal_odds")
        elif amer is not None:
            decimal_odds = american_to_decimal(amer)
        else:
            leg_odds = [leg.decimal_odds for leg in legs]
            if any(lo is None for lo in leg_odds):
                raise ReporterError("Ticket lacks odds and not every leg carries odds")
            decimal_odds = compound_decimal_odds([lo for lo in leg_odds if lo is not None])
    except StakingInputError as exc:
        raise ReporterError(f"Ticket odds invalid: {exc}") from exc
    if decimal_odds <= 1.0:
        raise ReporterError(f"decimal_odds must exceed 1.0, got {decimal_odds}")

    tid = _first(d, _ID_KEYS)
    notes = _first(d, _NOTES_KEYS)
    if isinstance(notes, (list, tuple)):
        notes = ", ".join(str(n) for n in notes)
    wk = d.get("week", week)
    try:
        wk = int(wk) if wk is not None else None
    except (TypeError, ValueError):
        wk = week

    return ParlayTicket(
        ticket_id=str(tid) if tid is not None else default_id,
        legs=legs,
        p_true=p_true,
        decimal_odds=decimal_odds,
        week=wk,
        notes=str(notes) if notes else "",
        source=source or str(d.get("source", "")),
        raw=d,
    )


def normalize_tickets(
    raw_tickets: Iterable[Any], source: str = "", week: Optional[int] = None
) -> Tuple[List[ParlayTicket], List[Dict[str, Any]]]:
    """Normalise a batch; returns ``(good_tickets, rejected)``.

    Rejected entries are ``{"index": i, "error": str, "raw": repr}`` so the
    report can list what was dropped instead of failing silently.
    """
    good: List[ParlayTicket] = []
    rejected: List[Dict[str, Any]] = []
    for i, raw in enumerate(raw_tickets):
        try:
            good.append(normalize_ticket(raw, default_id=f"T{i + 1:02d}", source=source, week=week))
        except (ReporterError, StakingInputError, TypeError, ValueError) as exc:
            logger.warning("Rejected ticket #%d: %s", i + 1, exc)
            rejected.append({"index": i, "error": str(exc), "raw": repr(raw)[:200]})
    return good, rejected


# ---------------------------------------------------------------------------
# Collection: parlay_finder.py  /  JSON  /  demo
# ---------------------------------------------------------------------------


def _unwrap_collection(result: Any) -> List[Any]:
    """Turn whatever parlay_finder returned into a list of raw tickets."""
    if result is None:
        return []
    # pandas DataFrame (duck-typed so pandas is not a hard dependency)
    if hasattr(result, "to_dict") and hasattr(result, "columns"):
        return list(result.to_dict("records"))
    if isinstance(result, dict):
        for key in ("parlays", "tickets", "results", "recommendations", "data"):
            if key in result:
                return _unwrap_collection(result[key])
        # Dict keyed by ticket id -> values are tickets
        if result and all(isinstance(v, (dict, list)) for v in result.values()):
            return list(result.values())
        return [result]
    if isinstance(result, (list, tuple)):
        return list(result)
    if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
        return list(result)
    return [result]


def load_parlays_from_finder(
    week: int,
    bankroll: Optional[float] = None,
    top_n: Optional[int] = None,
    module_name: str = "parlay_finder",
    function_name: Optional[str] = None,
) -> List[Any]:
    """Import ``parlay_finder`` and call its parlay-generating function.

    Raises :class:`ReporterError` if the module or a usable function cannot be
    found, or if the call itself fails. The caller decides how to fall back.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ReporterError(f"Could not import '{module_name}': {exc}") from exc

    candidates = (function_name,) if function_name else FINDER_FUNCTION_CANDIDATES
    func: Optional[Callable[..., Any]] = None
    chosen = ""
    for name in candidates:
        if name and callable(getattr(module, name, None)):
            func = getattr(module, name)
            chosen = name
            break
    if func is None:
        raise ReporterError(
            f"'{module_name}' exposes none of: {', '.join(c for c in candidates if c)}"
        )

    # Pass only the keyword arguments the finder actually accepts.
    kwargs: Dict[str, Any] = {}
    try:
        params = inspect.signature(func).parameters
        accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError):
        params, accepts_var_kw = {}, False
    for key, value in (("week", week), ("bankroll", bankroll), ("top_n", top_n)):
        if value is not None and (key in params or accepts_var_kw):
            kwargs[key] = value
    # Some finders spell the limit ``n``; use it only when ``top_n`` was not taken.
    if top_n is not None and "top_n" not in kwargs and "n" in params:
        kwargs["n"] = top_n

    logger.info("Calling %s.%s(%s)", module_name, chosen, ", ".join(f"{k}={v!r}" for k, v in kwargs.items()))
    try:
        result = func(**kwargs)
    except Exception as exc:
        raise ReporterError(f"{module_name}.{chosen}() raised {type(exc).__name__}: {exc}") from exc
    tickets = TicketBatch(_unwrap_collection(result))
    summary = getattr(result, "slate_summary", None)
    tickets.slate_summary = summary if isinstance(summary, dict) else None
    logger.info("parlay_finder returned %d candidate ticket(s)", len(tickets))
    return tickets


def _finder_data_suffix(raw_tickets: Sequence[Any]) -> str:
    """Describe the finder's data source from the tickets' ``source`` field."""
    sources = set()
    for t in raw_tickets:
        try:
            sources.add(str(_as_dict(t).get("source", "")).lower())
        except ReporterError:
            continue
    if sources and all(s == "sim" for s in sources):
        return " (simulated league, NOT real games)"
    if sources and all(s == "csv" for s in sources):
        return " (real lines from lines.csv)"
    if sources and all(s.startswith("api") for s in sources):
        return " (live lines via The Odds API)"
    return ""


def load_parlays_from_json(path: str) -> List[Any]:
    """Load raw tickets from a JSON file written by parlay_finder or by hand."""
    if not os.path.isfile(path):
        raise ReporterError(f"Input file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ReporterError(f"Input file {path} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise ReporterError(f"Could not read {path}: {exc}") from exc
    tickets = _unwrap_collection(data)
    logger.info("Loaded %d ticket(s) from %s", len(tickets), path)
    return tickets


# A plausible simulated slate. Team names are real NFL franchises; lines,
# prices and probabilities are generated for pipeline testing only.
_DEMO_GAMES: Tuple[Tuple[str, str], ...] = (
    ("Kansas City Chiefs", "Buffalo Bills"),
    ("Dallas Cowboys", "Philadelphia Eagles"),
    ("San Francisco 49ers", "Seattle Seahawks"),
    ("Baltimore Ravens", "Cincinnati Bengals"),
    ("Green Bay Packers", "Detroit Lions"),
    ("Miami Dolphins", "New York Jets"),
    ("Los Angeles Rams", "Arizona Cardinals"),
    ("Houston Texans", "Jacksonville Jaguars"),
)


def generate_demo_slate(week: int, seed: int = 7, n_two_leg: int = 6, n_three_leg: int = 4) -> List[Dict[str, Any]]:
    """Deterministic simulated slate for smoke-testing the reporter.

    Each leg is priced at a realistic -105 to -118 and given a model
    probability a few points either side of the implied number, so the slate
    contains a healthy mix of +EV tickets, marginal tickets and -EV tickets
    that the staking engine should refuse.
    """
    rng = random.Random(seed)

    def make_leg(game: Tuple[str, str]) -> Dict[str, Any]:
        away, home = game
        market = rng.choice(["spread", "spread", "total", "moneyline"])
        price = rng.choice([-105, -108, -110, -110, -112, -115, -118])
        implied = 1.0 / american_to_decimal(price)
        if market == "spread":
            team = rng.choice([away, home])
            line = rng.choice([1.5, 2.5, 3.0, 3.5, 4.5, 6.5, 7.0])
            sign = -1 if team == home else 1
            selection = f"{team} {sign * line:+g}"
        elif market == "total":
            total = rng.choice([41.5, 43.5, 44.5, 45.5, 47.0, 48.5, 51.5])
            selection = f"{rng.choice(['Over', 'Under'])} {total:g}"
        else:
            team = rng.choice([away, home])
            price = rng.choice([-150, -135, +120, +135, +160])
            implied = 1.0 / american_to_decimal(price)
            selection = f"{team} ML"
        # Model edge per leg: between -4 and +7 percentage points.
        p_true = min(0.95, max(0.05, implied + rng.uniform(-0.04, 0.07)))
        return {
            "matchup": f"{away} @ {home}",
            "selection": selection,
            "market": market,
            "american_odds": price,
            "p_true": round(p_true, 4),
        }

    slate: List[Dict[str, Any]] = []
    counter = 1
    for n_legs, count in ((2, n_two_leg), (3, n_three_leg)):
        for _ in range(count):
            games = rng.sample(_DEMO_GAMES, n_legs)
            slate.append(
                {
                    "ticket_id": f"W{week:02d}-{n_legs}L-{counter:02d}",
                    "week": week,
                    "legs": [make_leg(g) for g in games],
                    "notes": "simulated slate",
                }
            )
            counter += 1
    return slate


def collect_parlays(
    week: int,
    source: str = "finder",
    input_path: Optional[str] = None,
    bankroll: Optional[float] = None,
    top_n: Optional[int] = None,
    finder_function: Optional[str] = None,
    demo_seed: int = 7,
) -> Tuple[List[ParlayTicket], List[Dict[str, Any]], str]:
    """Gather and normalise this week's parlays.

    Returns ``(tickets, rejected, source_label)``. Falls back finder -> json
    -> demo so a scheduled run always produces a report, and the report says
    which source was actually used.
    """
    source = (source or "finder").lower()
    raw: List[Any] = []
    label = ""

    finder_ran_empty = False
    if source == "finder":
        try:
            raw = load_parlays_from_finder(week, bankroll=bankroll, top_n=top_n, function_name=finder_function)
            label = "parlay_finder.py" + _finder_data_suffix(raw)
            if not raw:
                # The finder ran and found nothing that clears its filters. That is a
                # legitimate "no bets this week" outcome, never a reason to show demo data.
                finder_ran_empty = True
                label += " (no ticket cleared the filters this week)"
        except ReporterError as exc:
            logger.warning("parlay_finder unavailable (%s); falling back", exc)
            source = "json" if input_path else "demo"

    if source == "json":
        if not input_path:
            raise ReporterError("--source json requires --input <file.json>")
        raw = load_parlays_from_json(input_path)
        label = f"JSON file ({os.path.basename(input_path)})"

    if source == "demo":
        raw = generate_demo_slate(week, seed=demo_seed)
        label = "SIMULATED demo slate (no upstream data)"

    if source not in ("finder", "json", "demo"):
        raise ReporterError(f"Unknown source '{source}'. Use finder | json | demo")

    good, rejected = normalize_tickets(raw, source=source, week=week)
    tickets = TicketBatch(good)
    tickets.slate_summary = getattr(raw, "slate_summary", None)
    if not tickets and not finder_ran_empty:
        raise ReporterError(f"No usable parlay tickets were collected from {label}")
    if not tickets:
        logger.warning("Week %d: the finder returned no tickets; the report will say so", week)
    return tickets, rejected, label


# ---------------------------------------------------------------------------
# Staking, ranking and grouping
# ---------------------------------------------------------------------------


def stake_tickets(
    tickets: Sequence[ParlayTicket],
    bankroll: float,
    staking: StakingConfig,
    portfolio_cap_pct: Optional[float] = None,
) -> List[StakedTicket]:
    """Run every ticket through the staking engine (never raises per ticket)."""
    recs = [
        safe_compute_stake(bankroll, t.p_true, t.decimal_odds, staking, ticket_id=t.ticket_id)
        for t in tickets
    ]
    if portfolio_cap_pct is not None and bankroll > 0:
        try:
            recs = apply_portfolio_cap(recs, bankroll, portfolio_cap_pct)
        except StakingInputError as exc:
            logger.warning("Portfolio cap not applied: %s", exc)
    return [StakedTicket(ticket=t, stake=r) for t, r in zip(tickets, recs)]


def _rank_key(st: StakedTicket) -> Tuple[float, float, float]:
    """Sort: highest expected dollar profit of the recommended stake first.

    Under Kelly sizing this is proportional to edge^2 / (D - 1), so a solid
    edge at a short price outranks a thin-probability long shot with the same
    raw edge. Ties fall back to raw edge, then to P_true.
    """
    return (-st.stake.expected_value, -st.ticket.edge, -st.ticket.p_true)


def is_same_game_ticket(st: StakedTicket) -> bool:
    """True when the finder flagged the ticket as same-game (books price it as an SGP)."""
    raw = st.ticket.raw or {}
    return bool(raw.get("sgp_required") or raw.get("same_game"))


@dataclass
class ParlaySelection:
    """The outcome of the "how many parlays" control.

    ``requested`` is the maximum N. ``chosen`` holds the top tickets by
    expected value that passed every filter (at most N), with the weekly
    exposure cap applied to just those stakes; ``beyond_limit`` holds the
    other qualifying tickets; ``same_game`` holds SGP-flagged tickets kept
    out of the count (unless ``count_same_game``); ``passed`` the $0 ones.
    """

    requested: int
    chosen: List[StakedTicket]
    beyond_limit: List[StakedTicket]
    same_game: List[StakedTicket]
    passed: List[StakedTicket]
    cap_scale: float = 1.0          # < 1 when the weekly cap scaled the chosen stakes down
    cap_pct: Optional[float] = None

    @property
    def qualified(self) -> int:
        return len(self.chosen) + len(self.beyond_limit)

    def message(self) -> str:
        n, q = self.requested, self.qualified
        text = f"You asked for {n}. {q} qualified."
        if q > n:
            text += f" Showing the top {len(self.chosen)} by expected value."
        if self.cap_scale < 1.0 and self.cap_pct is not None:
            text += (f" Stakes scaled x{self.cap_scale:.3f} so the {len(self.chosen)} tickets together risk no more than "
                     f"{self.cap_pct:.0%} of the bankroll.")
        if self.same_game:
            text += (f" {len(self.same_game)} same-game ticket(s) shown separately and not counted (book prices them as "
                     f"same-game parlays).")
        return text

    def to_dict(self) -> Dict[str, Any]:
        return {"requested": self.requested, "qualified": self.qualified, "recommended": len(self.chosen),
                "beyond_limit": len(self.beyond_limit), "same_game": len(self.same_game), "passed": len(self.passed),
                "cap_scale": self.cap_scale, "message": self.message()}


def select_parlays(
    staked: Sequence[StakedTicket],
    max_parlays: int = DEFAULT_MAX_PARLAYS,
    bankroll: float = 0.0,
    portfolio_cap_pct: Optional[float] = 0.15,
    count_same_game: bool = False,
) -> ParlaySelection:
    """Rank every ticket that passed the staking filters by expected value and keep at most ``max_parlays``.

    N is a maximum: if fewer qualify, fewer are returned; nothing is relaxed
    to reach N. The weekly exposure cap is applied to the chosen tickets only,
    scaling their stakes proportionally when they would breach it. Same-game
    tickets are set aside unless ``count_same_game`` is True.
    """
    if not MIN_PARLAYS <= int(max_parlays) <= MAX_PARLAYS:
        raise ReporterError(f"max_parlays must be between {MIN_PARLAYS} and {MAX_PARLAYS}, got {max_parlays}")
    same_game = [] if count_same_game else sorted([s for s in staked if is_same_game_ticket(s)], key=_rank_key)
    pool = [s for s in staked if count_same_game or not is_same_game_ticket(s)]
    qualified = sorted([s for s in pool if s.stake.is_bet], key=_rank_key)
    passed = sorted([s for s in pool if not s.stake.is_bet], key=_rank_key)
    chosen, beyond = qualified[: int(max_parlays)], qualified[int(max_parlays):]
    cap_scale = 1.0
    if portfolio_cap_pct is not None and bankroll > 0 and chosen:
        before = sum(s.stake.stake_dollars for s in chosen)
        try:
            recs = apply_portfolio_cap([s.stake for s in chosen], bankroll, portfolio_cap_pct)
        except StakingInputError as exc:
            logger.warning("Portfolio cap not applied: %s", exc)
            recs = [s.stake for s in chosen]
        for s, r in zip(chosen, recs):
            s.stake = r
        after = sum(r.stake_dollars for r in recs)
        if before > 0 and after < before - 1e-9:
            cap_scale = after / before
    for i, s in enumerate(chosen, 1):
        s.rank = i
    for s in beyond + same_game + passed:
        s.rank = 0
    return ParlaySelection(requested=int(max_parlays), chosen=chosen, beyond_limit=beyond, same_game=same_game, passed=passed,
                           cap_scale=cap_scale, cap_pct=portfolio_cap_pct)


def group_chosen(
    chosen: Sequence[StakedTicket],
    all_staked: Sequence[StakedTicket],
    leg_groups: Sequence[int],
    include_other: bool,
    hide_empty_groups: bool = True,
) -> Dict[str, List[StakedTicket]]:
    """Split the chosen tickets into leg-count sections (ranks are the overall 1..N)."""
    sections: Dict[str, List[StakedTicket]] = {}
    for n in leg_groups:
        if hide_empty_groups and not any(s.ticket.n_legs == n for s in all_staked):
            continue
        sections[f"{n}-LEG"] = [s for s in chosen if s.ticket.n_legs == n]
    if include_other:
        others = [s for s in chosen if s.ticket.n_legs not in leg_groups]
        if others:
            sections["4+ LEG / OTHER"] = others
    return sections


# ---------------------------------------------------------------------------
# Report model
# ---------------------------------------------------------------------------


@dataclass
class WeeklyReport:
    """Rendered report plus the structured data behind it."""

    config: ReportConfig
    sections: Dict[str, List[StakedTicket]]
    passed: List[StakedTicket]
    rejected: List[Dict[str, Any]]
    all_staked: List[StakedTicket]
    generated_at: _dt.datetime = field(default_factory=_dt.datetime.now)
    text: str = ""
    selection: Optional[ParlaySelection] = None

    # ---- aggregates -------------------------------------------------------

    @property
    def recommended(self) -> List[StakedTicket]:
        return [s for group in self.sections.values() for s in group]

    @property
    def beyond_limit(self) -> List[StakedTicket]:
        return list(self.selection.beyond_limit) if self.selection else []

    @property
    def same_game(self) -> List[StakedTicket]:
        return list(self.selection.same_game) if self.selection else []

    @property
    def total_risk(self) -> float:
        return sum(s.stake.stake_dollars for s in self.recommended)

    @property
    def total_expected_value(self) -> float:
        return sum(s.stake.expected_value for s in self.recommended)

    @property
    def total_potential_profit(self) -> float:
        return sum(s.stake.potential_profit for s in self.recommended)

    # ---- exports ----------------------------------------------------------

    def to_records(self) -> List[Dict[str, Any]]:
        """One flat row per recommended ticket (DataFrame-ready)."""
        return [s.to_record() for s in self.recommended]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(timespec="seconds"),
            "config": self.config.to_dict(),
            "summary": {
                "tickets_collected": len(self.all_staked),
                "tickets_recommended": len(self.recommended),
                "tickets_passed": len(self.passed),
                "tickets_rejected": len(self.rejected),
                "total_risk": round(self.total_risk, 2),
                "total_risk_pct_of_bankroll": (self.total_risk / self.config.bankroll) if self.config.bankroll else 0.0,
                "total_expected_value": round(self.total_expected_value, 2),
                "total_potential_profit": round(self.total_potential_profit, 2),
                **({"requested": self.selection.requested, "qualified": self.selection.qualified,
                    "cap_scale": self.selection.cap_scale, "message": self.selection.message()} if self.selection else {}),
            },
            "sections": {name: [s.to_dict() for s in group] for name, group in self.sections.items()},
            "beyond_limit": [s.to_dict() for s in self.beyond_limit],
            "same_game": [s.to_dict() for s in self.same_game],
            "passed": [s.to_dict() for s in self.passed],
            "rejected": self.rejected,
            "slate": self.config.slate_summary,
        }

    def save_text(self, path: str = DEFAULT_REPORT_FILENAME) -> str:
        """Atomically write the plain-text report; returns the absolute path."""
        return _atomic_write(path, self.text or render_report_text(self))

    def save_json(self, path: str) -> str:
        """Write the structured report as JSON; returns the absolute path."""
        return _atomic_write(path, json.dumps(self.to_dict(), indent=2))


def _atomic_write(path: str, content: str) -> str:
    """Write via a temp file + rename so a crash never leaves a half report."""
    abs_path = os.path.abspath(path)
    directory = os.path.dirname(abs_path) or "."
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".tmp_report_", dir=directory, text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, abs_path)
    except OSError as exc:
        raise ReporterError(f"Could not write report to {abs_path}: {exc}") from exc
    return abs_path


# ---------------------------------------------------------------------------
# Plain-text rendering
# ---------------------------------------------------------------------------

_W = REPORT_WIDTH
_HR = "=" * _W
_HR2 = "-" * _W
_HR3 = ("  " + "- " * ((_W - 2) // 2)).rstrip()


def _center(text: str) -> str:
    return text.center(_W).rstrip()


def _kv(label: str, value: str, pad: int = 17) -> str:
    return f"  {label:<{pad}}: {value}"


def _fit(text: str, width: int) -> str:
    """Truncate with an ASCII ellipsis so fixed-width columns never overflow."""
    text = str(text)
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..." if width >= 3 else text[:width]


def _money(amount: float, signed: bool = False) -> str:
    """``27.3 -> '$27.30'``; with ``signed=True`` ``1.16 -> '+$1.16'``, ``-0.5 -> '-$0.50'``."""
    if signed:
        sign = "+" if amount >= 0 else "-"
    else:
        sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.2f}"


def _render_ticket(st: StakedTicket, index: int) -> List[str]:
    t, s = st.ticket, st.stake
    lines: List[str] = []
    title = f"  #{index}  Ticket {t.ticket_id}"
    risk = f"RISK: ${s.stake_dollars:,.2f}"
    lines.append(f"{title:<{_W - len(risk) - 2}}{risk}")
    lines.append(_HR3)
    # Legs, two lines each: "    Leg 1  <selection> (<odds>)" then the matchup with the
    # per-leg model and book-implied probabilities (and the EXPERIMENTAL tag for props).
    indent = "    "
    for i, leg in enumerate(t.legs, 1):
        odds = f"({format_american(leg.american_odds)})" if leg.american_odds is not None else ""
        odds_w = max(6, len(odds))
        label = f"{indent}Leg {i}  "
        sel_w = _W - len(label) - 1 - odds_w
        lines.append(f"{label}{_fit(leg.selection, sel_w):<{sel_w}} {odds:>{odds_w}}")
        facts = []
        if leg.p_true is not None:
            facts.append(f"model {leg.p_true:.1%}")
        if leg.implied_prob is not None:
            facts.append(f"implied {leg.implied_prob:.1%}")
        if leg.experimental:
            facts.append("EXPERIMENTAL" + (" HIGH-VARIANCE" if leg.high_variance else ""))
        facts.append(leg.matchup)
        for wrapped in textwrap.wrap(" | ".join(facts), width=_W - len(label)):
            lines.append(f"{' ' * len(label)}{wrapped}")
    lines.append(_HR3)
    col = 15
    lines.append(
        f"{indent}{'True Win Rate':<{col}}: {t.p_true:>8.2%}    "
        f"{'Book Implied':<{col}}: {t.implied_prob:>8.2%}"
    )
    lines.append(
        f"{indent}{'Payout Odds':<{col}}: {format_american(t.american_odds):>8}    "
        f"{'Edge':<{col}}: {t.edge:>+8.2%}"
    )
    lines.append(
        f"{indent}{'To Win':<{col}}: {_money(s.potential_profit):>8}    "
        f"{'Expected Value':<{col}}: {_money(s.expected_value, signed=True):>8}"
    )
    lines.append(
        f"{indent}{'Stake Basis':<{col}}: {s.capped_fraction:>8.2%} of bankroll"
        f"{'  [CAPPED]' if s.cap_applied else ''}"
    )
    raw = t.raw or {}
    if raw.get("same_game"):
        corr = f"{raw.get('correlation') or 'same-game'} correlation: {raw.get('correlation_reason') or ''}".strip().rstrip(":")
        for wrapped in textwrap.wrap(f"Correlation: {corr}. {raw.get('pricing_note') or ''}".strip(), width=_W - 4):
            lines.append(f"    {wrapped}")
    exp_legs = [str(i) for i, leg in enumerate(t.legs, 1) if leg.experimental]
    if exp_legs:
        for wrapped in textwrap.wrap(f"Model: EXPERIMENTAL prop model on leg {', '.join(exp_legs)}; its probabilities are not yet "
                                     f"validated by the backtest calibration gate.", width=_W - 4):
            lines.append(f"    {wrapped}")
    if t.notes:
        for wrapped in textwrap.wrap(f"Notes: {t.notes}", width=_W - 4):
            lines.append(f"    {wrapped}")
    lines.append("")
    return lines


def render_report_text(report: WeeklyReport) -> str:
    """Build the full plain-text report card."""
    cfg = report.config
    stk = cfg.staking
    date_str = cfg.report_date.strftime("%A, %B %d, %Y") if cfg.report_date else ""
    out: List[str] = []

    # ---- Header -------------------------------------------------------------
    out.append(_HR)
    out.append(_center(cfg.title))
    out.append(_center(f"Simulated NFL Week {cfg.week}  |  {cfg.season} Season"))
    out.append(_center(date_str))
    out.append(_HR)
    out.append(_kv("Generated", report.generated_at.strftime("%Y-%m-%d %H:%M:%S")))
    out.append(_kv("Bankroll", f"${cfg.bankroll:,.2f}"))
    if stk.mode is StakingMode.FLAT:
        unit = f"${stk.flat_dollars:,.2f} per ticket" if stk.flat_dollars is not None else f"{stk.flat_pct:.2%} of bankroll per ticket"
        mode_desc = f"{stk.mode.label}  [Mode A]  -  {unit}"
    else:
        mode_desc = f"{stk.mode.label}  [Mode B]  -  {stk.kelly_multiplier:.2f}x Kelly"
    out.append(_kv("Staking Mode", mode_desc))
    caps = f"max {stk.max_stake_pct:.0%} per ticket"
    if cfg.portfolio_cap_pct is not None:
        caps += f", max {cfg.portfolio_cap_pct:.0%} total weekly exposure"
    if stk.min_edge > 0:
        caps += f", min edge {stk.min_edge:+.2%}"
    out.append(_kv("Safety Caps", caps))
    out.append(_kv("Pick Source", cfg.source_label or "n/a"))
    out.append(
        _kv(
            "Slate",
            f"{len(report.all_staked)} candidates  ->  {len(report.recommended)} recommended, "
            f"{len(report.passed)} passed, {len(report.rejected)} rejected",
        )
    )
    slate = cfg.slate_summary or {}
    if slate.get("sides"):
        out.append(_kv("Leg Filter", f"{slate.get('passed', 0)} of {slate['sides']} sides clear the filter "
                                     f"({slate.get('markets', 0)} markets; table below)"))
    sel = report.selection
    if sel is not None:
        parlays = f"you asked for {sel.requested}; {sel.qualified} qualified"
        if sel.qualified > sel.requested:
            parlays += f"; top {len(sel.chosen)} by expected value shown"
        out.append(_kv("Parlays", parlays))
        if sel.cap_scale < 1.0 and sel.cap_pct is not None:
            out.append(_kv("Weekly Cap", f"stakes scaled x{sel.cap_scale:.3f} so {len(sel.chosen)} tickets risk <= {sel.cap_pct:.0%} of bankroll"))
        if sel.same_game:
            out.append(_kv("Same-Game", f"{len(sel.same_game)} ticket(s) need SGP pricing at the book; listed apart, not counted"))
    out.append(_HR2)
    out.append("")

    # ---- Sections -----------------------------------------------------------
    if not report.sections:
        out.append("TOP RECOMMENDED PARLAYS")
        out.append(_HR)
        out.append("  No parlay cleared this week's filters. No bets are recommended.")
        if sel is not None:
            out.append(f"  You asked for {sel.requested}. {sel.qualified} qualified.")
        out.append("")
    for name, group in report.sections.items():
        out.append(f"TOP RECOMMENDED PARLAYS  -  {name} COMBINATIONS")
        out.append(_HR)
        if not group:
            n_legs = int(name.split("-")[0]) if name[0].isdigit() else None
            outside = [s for s in (sel.beyond_limit if sel else []) if n_legs is None or s.ticket.n_legs == n_legs]
            if outside and sel is not None:
                out.append(f"  ({len(outside)} qualifying ticket(s) fall outside your limit of {sel.requested}; "
                           f"see QUALIFIED BUT BEYOND YOUR LIMIT below)")
            else:
                out.append("  (no +EV tickets cleared the staking filters this week)")
                if sel is not None:
                    out.append(f"  You asked for {sel.requested}. {sel.qualified} qualified.")
            out.append("")
            continue
        for st in group:
            out.extend(_render_ticket(st, st.rank or 1))

    # ---- Summary ------------------------------------------------------------
    out.append("WEEKLY EXPOSURE SUMMARY")
    out.append(_HR)
    pct = (report.total_risk / cfg.bankroll) if cfg.bankroll else 0.0
    out.append(_kv("Tickets to Place", str(len(report.recommended))))
    out.append(_kv("Total Risk", f"${report.total_risk:,.2f}  ({pct:.2%} of bankroll)"))
    out.append(_kv("Total To Win", f"${report.total_potential_profit:,.2f}  (if every ticket hits)"))
    out.append(_kv("Expected Value", _money(report.total_expected_value, signed=True)))
    out.append(_kv("Bankroll After", f"${cfg.bankroll - report.total_risk:,.2f} reserved / ${cfg.bankroll:,.2f} total"))
    out.append("")

    # ---- Qualified beyond the limit, and same-game tickets ------------------
    def compact(st: StakedTicket, extra: str = "") -> List[str]:
        t, s = st.ticket, st.stake
        legs = " / ".join(leg.selection for leg in t.legs)
        rows = [f"  {t.ticket_id:<12} {t.n_legs}L  {format_american(t.american_odds):>6}  "
                f"P={t.p_true:.1%}  Imp={t.implied_prob:.1%}  Edge={t.edge:+.1%}  EV={_money(s.expected_value, signed=True)}",
                f"               {_fit(legs, _W - 15)}"]
        if extra:
            for wrapped in textwrap.wrap(extra, width=_W - 18):
                rows.append(f"               -> {wrapped}")
        return rows

    if sel is not None and sel.beyond_limit:
        out.append(f"QUALIFIED BUT BEYOND YOUR LIMIT OF {sel.requested}  -  {len(sel.beyond_limit)} MORE TICKET(S), RANKED BY EV")
        out.append(_HR)
        for st in sel.beyond_limit:
            out.extend(compact(st, f"would stake {_money(st.stake.stake_dollars)} before the weekly cap; raise max_parlays to include it"))
        out.append("")
    if sel is not None and sel.same_game:
        out.append(f"SAME-GAME TICKETS  -  SGP PRICING REQUIRED, NOT COUNTED TOWARD YOUR LIMIT ({len(sel.same_game)})")
        out.append(_HR)
        for st in sel.same_game:
            raw = st.ticket.raw or {}
            why = raw.get("correlation_reason") or raw.get("correlation") or ""
            out.extend(compact(st, f"{raw.get('correlation', 'same-game')} correlation: {why}. {raw.get('pricing_note', '')}".strip()))
        out.append("")

    # ---- Leg filter by market (tuning aid for finder.market_rules) ---------
    by_market = slate.get("by_market") if isinstance(slate, dict) else None
    if by_market:
        out.append("LEGS CLEARING THE FILTER BY MARKET")
        out.append(_HR)
        table = slate.get("table")
        if isinstance(table, list) and table:
            out.extend(str(line) for line in table)
        else:  # the finder did not render a table; a plain fallback
            for row in by_market.values():
                out.append(f"  {_fit(row.get('label', ''), 28):<28} {row.get('passed', 0):>4} of {row.get('sides', 0):<4} pass  "
                           f"(floor {float(row.get('min_leg_prob', 0)) * 100:g}%, gap {float(row.get('min_prob_gap', 0)) * 100:g} pts)")
        out.append("")

    # ---- Passed tickets appendix ------------------------------------------
    if cfg.include_skipped and report.passed:
        out.append("PASSED (NO BET)  -  TICKETS THAT FAILED THE STAKING FILTERS")
        out.append(_HR)
        for st in report.passed:
            t, s = st.ticket, st.stake
            legs = " / ".join(leg.selection for leg in t.legs)
            out.append(f"  {t.ticket_id:<12} {t.n_legs}L  {format_american(t.american_odds):>6}  "
                       f"P={t.p_true:.1%}  Imp={t.implied_prob:.1%}  Edge={t.edge:+.1%}")
            out.append(f"               {_fit(legs, _W - 15)}")
            out.append(f"               -> {_fit(s.reason if not s.error else s.error, _W - 18)}")
        out.append("")

    # ---- Rejected inputs appendix -----------------------------------------
    if report.rejected:
        out.append("REJECTED INPUT  -  MALFORMED TICKETS DROPPED DURING NORMALISATION")
        out.append(_HR)
        for r in report.rejected:
            out.append(f"  input #{r['index'] + 1}: {_fit(r['error'], _W - 14)}")
        out.append("")

    out.append(_HR2)
    out.append(_center("Edge = (P_true x Decimal Odds) - 1   |   Stakes rounded DOWN to the cent"))
    out.append(_center("For simulation / research purposes only. Not financial advice."))
    out.append(_center("Bet only what you can afford to lose. US help line: 1-800-GAMBLER."))
    out.append(_HR)
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_weekly_report(
    tickets: Sequence[Any],
    config: ReportConfig,
    rejected: Optional[List[Dict[str, Any]]] = None,
) -> WeeklyReport:
    """Stake, rank, group and render a slate of tickets.

    ``tickets`` may be :class:`ParlayTicket` objects or raw upstream dicts;
    raw ones are normalised here so backtester.py can hand over its own
    structures directly.
    """
    normalised: List[ParlayTicket] = []
    rejected = list(rejected or [])
    raw_inputs = [t for t in tickets if not isinstance(t, ParlayTicket)]
    normalised.extend(t for t in tickets if isinstance(t, ParlayTicket))
    if raw_inputs:
        good, bad = normalize_tickets(raw_inputs, week=config.week)
        normalised.extend(good)
        rejected.extend(bad)

    # Stake every ticket against the full bankroll first (no weekly cap yet), rank, keep at most
    # max_parlays, then apply the weekly cap to the chosen set only, so the scaling reflects what
    # would actually be placed.
    staked = stake_tickets(normalised, config.bankroll, config.staking, None)
    selection = select_parlays(staked, config.max_parlays, config.bankroll, config.portfolio_cap_pct, config.count_same_game)
    sections = group_chosen(selection.chosen, staked, config.leg_groups, config.include_other_groups, config.hide_empty_groups)
    report = WeeklyReport(config=config, sections=sections, passed=selection.passed, rejected=rejected, all_staked=staked,
                          selection=selection)
    report.text = render_report_text(report)
    return report


def run_weekly_report(
    week: int,
    bankroll: float,
    staking: Optional[StakingConfig] = None,
    source: str = "finder",
    input_path: Optional[str] = None,
    output_path: str = DEFAULT_REPORT_FILENAME,
    json_path: Optional[str] = None,
    top_n: int = 5,
    portfolio_cap_pct: Optional[float] = 0.15,
    report_date: Optional[_dt.date] = None,
    finder_function: Optional[str] = None,
    demo_seed: int = 7,
    max_parlays: int = DEFAULT_MAX_PARLAYS,
    count_same_game: bool = False,
) -> WeeklyReport:
    """End-to-end: collect -> stake -> render -> save. Returns the report."""
    staking = staking or StakingConfig()
    # Ask the finder for more candidates than the limit so the report can show what fell outside it.
    tickets, rejected, label = collect_parlays(
        week, source=source, input_path=input_path, bankroll=bankroll,
        top_n=max(top_n, 2 * int(max_parlays), 10), finder_function=finder_function, demo_seed=demo_seed,
    )
    config = ReportConfig(
        week=week, bankroll=bankroll, staking=staking, report_date=report_date,
        top_n_per_group=top_n, portfolio_cap_pct=portfolio_cap_pct, source_label=label,
        slate_summary=getattr(tickets, "slate_summary", None), max_parlays=max_parlays, count_same_game=count_same_game,
    )
    report = build_weekly_report(tickets, config, rejected=rejected)
    saved = report.save_text(output_path)
    logger.info("Report written to %s", saved)
    if json_path:
        logger.info("JSON written to %s", report.save_json(json_path))
    return report


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

    print("weekly_reporter self-test")

    # Normalisation: per-leg compounding
    raw = {
        "ticket_id": "X1",
        "legs": [
            {"matchup": "KC @ BUF", "selection": "BUF -3.5", "american_odds": -110, "p_true": 0.56},
            {"game": "DAL @ PHI", "line": "Over 44.5", "odds": "-110", "prob": 0.55},
        ],
    }
    t = normalize_ticket(raw)
    check(t.n_legs == 2 and t.ticket_id == "X1", "normalises 2-leg ticket with alias keys")
    check(abs(t.p_true - 0.56 * 0.55) < 1e-12, "compounds leg probabilities")
    check(abs(t.decimal_odds - (1.9090909 ** 2)) < 1e-5, "compounds leg odds")
    check(t.american_odds == 264, "two -110 legs -> +264")

    # Normalisation: parlay-level override and percentage probabilities
    t2 = normalize_ticket({"legs": ["KC @ BUF: BUF -3.5", "Over 44.5"], "p_true": 31.0, "american_odds": "+264"})
    check(abs(t2.p_true - 0.31) < 1e-12 and abs(t2.decimal_odds - 3.64) < 1e-12, "parlay-level p_true% and american odds")
    check(t2.legs[0].matchup == "KC @ BUF" and t2.legs[1].matchup == "", "string legs parsed")

    # Normalisation: team/spread/total structural fields
    t3 = normalize_ticket({"legs": [
        {"away": "GB", "home": "DET", "team": "DET", "spread": -3.5, "price": -110, "p_true": 0.55},
        {"away": "MIA", "home": "NYJ", "total": 41.5, "direction": "under", "price": -105, "p_true": 0.54},
    ]})
    check(t3.legs[0].selection == "DET -3.5" and t3.legs[1].selection == "Under 41.5", "structural leg fields -> selection text")

    # Rejections
    good, bad = normalize_tickets([
        {"legs": [{"selection": "A", "american_odds": -110, "p_true": 0.5}]},        # 1 leg
        {"legs": [{"selection": "A", "american_odds": -110}, {"selection": "B", "american_odds": -110}]},  # no probs
        {"nonsense": True},
        raw,
    ])
    check(len(good) == 1 and len(bad) == 3, f"rejects malformed tickets ({len(bad)} rejected, {len(good)} kept)")

    # Demo slate + full report
    slate = generate_demo_slate(6, seed=7)
    check(len(slate) == 10 and sum(1 for s in slate if len(s["legs"]) == 3) == 4, "demo slate has 6x2-leg + 4x3-leg")
    cfg = ReportConfig(week=6, bankroll=1000.0, staking=StakingConfig.fractional_kelly(0.25),
                       report_date=_dt.date(2026, 10, 7), source_label="selftest", max_parlays=10)
    report = build_weekly_report(slate, cfg)
    check(set(report.sections) >= {"2-LEG", "3-LEG"}, "report has 2-LEG and 3-LEG sections")
    check(all(s.stake.is_bet for s in report.recommended), "every recommended ticket has a positive stake")
    check(all(not s.stake.is_bet for s in report.passed), "every passed ticket has $0 stake")
    check(report.total_risk <= 1000 * 0.15 + 1e-9, "weekly exposure respects 15% portfolio cap")
    check(all(s.stake.stake_dollars <= 50.0 + 1e-9 for s in report.recommended), "no ticket exceeds 5% cap")
    for group in report.sections.values():
        evs = [s.stake.expected_value for s in group]
        check(evs == sorted(evs, reverse=True), "section sorted by expected value desc")
        ranks = [s.rank for s in group]
        check(ranks == sorted(ranks) and all(1 <= r <= 10 for r in ranks), "ranks are the overall order 1..N")
    all_ranks = sorted(s.rank for s in report.recommended)
    check(all_ranks == list(range(1, len(all_ranks) + 1)), "recommended tickets are ranked 1..N across sections")

    # The "how many parlays" control: N is a maximum, ranked by EV, the weekly cap applies to the chosen set only
    n_qual = len(report.recommended)
    check(n_qual >= 4, f"demo slate yields enough qualifying tickets to test the limit ({n_qual})")
    limited = build_weekly_report(slate, ReportConfig(week=6, bankroll=1000.0, report_date=_dt.date(2026, 10, 7), max_parlays=2))
    assert limited.selection is not None
    check(len(limited.recommended) == 2 and limited.selection.qualified == n_qual and len(limited.beyond_limit) == n_qual - 2,
          f"max_parlays=2 keeps the top 2 of {n_qual} and lists the rest beyond the limit")
    by_ev = sorted(report.recommended, key=lambda s: -s.stake.expected_value)
    check([s.ticket.ticket_id for s in sorted(limited.recommended, key=lambda s: s.rank)] == [s.ticket.ticket_id for s in by_ev[:2]],
          "the chosen tickets are the highest expected-value ones")
    check(f"You asked for 2. {n_qual} qualified." in limited.selection.message() and "you asked for 2" in limited.text
          and "QUALIFIED BUT BEYOND YOUR LIMIT OF 2" in limited.text, "report says how many were asked for and how many qualified")
    big = build_weekly_report(slate, ReportConfig(week=6, bankroll=1000.0, staking=StakingConfig.flat_dollar(45), report_date=_dt.date(2026, 10, 7),
                                                  max_parlays=10))
    assert big.selection is not None
    check(big.selection.cap_scale < 1.0 and abs(big.total_risk - 150.0) < 0.5 and "scaled x" in big.selection.message() and "Weekly Cap" in big.text,
          f"ten $45 tickets are scaled down to the 15% weekly cap (x{big.selection.cap_scale:.3f}, risk ${big.total_risk:.2f})")
    check(all(s.stake.stake_dollars == 45.0 for s in big.beyond_limit), "tickets beyond the limit keep their uncapped stakes")
    only_one = build_weekly_report(slate[:1], ReportConfig(week=6, bankroll=1000.0, report_date=_dt.date(2026, 10, 7), max_parlays=5))
    assert only_one.selection is not None
    check(only_one.selection.qualified <= 1 and len(only_one.recommended) == only_one.selection.qualified and "You asked for 5." in only_one.selection.message(),
          "fewer qualifiers than N returns only those (never relaxed to reach N)")
    for bad_n in (0, 11, "three"):
        try:
            ReportConfig(week=6, bankroll=100.0, max_parlays=bad_n)  # type: ignore[arg-type]
            check(False, f"max_parlays={bad_n!r} rejected")
        except ReporterError:
            check(True, f"max_parlays={bad_n!r} rejected")
    sg_slate = [dict(t, sgp_required=True, same_game=True, correlation="positive", correlation_reason="a quarterback's yards are his receiver's yards",
                     pricing_note="book prices this as a same-game parlay") for t in slate[:2]] + slate[2:]
    sg_report = build_weekly_report(sg_slate, ReportConfig(week=6, bankroll=1000.0, report_date=_dt.date(2026, 10, 7), max_parlays=10))
    assert sg_report.selection is not None
    check(len(sg_report.same_game) == sum(1 for s in sg_report.all_staked if is_same_game_ticket(s)) and
          all(not is_same_game_ticket(s) for s in sg_report.recommended) and "SAME-GAME TICKETS" in sg_report.text
          and "same-game parlay" in sg_report.text, "same-game tickets are listed apart and not counted by default")
    counted = build_weekly_report(sg_slate, ReportConfig(week=6, bankroll=1000.0, report_date=_dt.date(2026, 10, 7), max_parlays=10, count_same_game=True))
    check(not counted.same_game and len(counted.recommended) >= len(sg_report.recommended), "count_same_game=True counts them like any ticket")
    js = limited.to_dict()
    check(js["summary"]["requested"] == 2 and js["summary"]["qualified"] == n_qual and len(js["beyond_limit"]) == n_qual - 2 and "message" in js["summary"],
          "JSON carries requested / qualified / beyond_limit")

    # Player-prop legs render with player, market, line, price, model and implied probabilities and the experimental badge
    prop_ticket = {"ticket_id": "P1", "week": 6, "legs": [
        {"matchup": "Tampa Bay Buccaneers @ Dallas Cowboys", "selection": "Dak Prescott Over 264.5 Passing Yards", "market": "passing_yards",
         "american_odds": -115, "p_true": 0.74, "player": "Dak Prescott", "team": "Dallas Cowboys", "position": "QB", "market_label": "Passing Yards",
         "line": 264.5, "direction": "Over", "experimental": True, "high_variance": False, "model_note": "proj 280 +/- 70"},
        {"matchup": "Green Bay Packers @ Detroit Lions", "selection": "Detroit Lions ML", "market": "moneyline", "american_odds": -150, "p_true": 0.75}],
        "experimental": True, "market_group": "mixed"}
    pt = normalize_ticket(prop_ticket)
    check(pt.legs[0].player == "Dak Prescott" and pt.legs[0].market_label == "Passing Yards" and pt.legs[0].line == 264.5 and pt.legs[0].experimental
          and not pt.legs[1].experimental and abs((pt.legs[0].implied_prob or 0) - 115 / 215) < 1e-9, "prop leg fields survive normalisation")
    prop_report = build_weekly_report([prop_ticket], ReportConfig(week=6, bankroll=1000.0, report_date=_dt.date(2026, 10, 7)))
    ptxt = prop_report.text
    check("Dak Prescott Over 264.5 Passing Yards" in ptxt and "(-115)" in ptxt and "model 74.0%" in ptxt and "implied 53.5%" in ptxt
          and "EXPERIMENTAL" in ptxt and "Model: EXPERIMENTAL prop model on leg 1" in ptxt and "1-800-GAMBLER" in ptxt,
          "report card shows per-leg price, model, implied, the experimental badge and the helpline")
    check(max(len(line) for line in ptxt.splitlines()) <= REPORT_WIDTH, "prop leg lines fit the report width")
    sg_ticket = dict(prop_ticket, ticket_id="P2", same_game=True, sgp_required=True, correlation="positive",
                     correlation_reason="a quarterback's yards are his receivers' yards", pricing_note="confirm the payout at the book")
    sg_text = build_weekly_report([sg_ticket], ReportConfig(week=6, bankroll=1000.0, report_date=_dt.date(2026, 10, 7), count_same_game=True)).text
    check("Correlation: positive correlation: a quarterback's yards" in sg_text and "confirm the payout at the book" in sg_text,
          "same-game tickets print their correlation reason and pricing note")
    pleg = prop_report.to_dict()["sections"]["2-LEG"][0]["ticket"]["legs"][0] if prop_report.sections.get("2-LEG") else {}
    check(pleg.get("player") == "Dak Prescott" and pleg.get("experimental") is True and abs(pleg.get("implied_prob", 0) - 115 / 215) < 1e-9,
          "JSON legs carry player, experimental flag and implied probability for the web page")

    text = report.text
    check("Simulated NFL Week 6" in text and "Wednesday, October 07, 2026" in text, "header shows week and date")
    check("$1,000.00" in text and "Fractional Kelly Criterion" in text, "header shows bankroll and mode")
    check("True Win Rate" in text and "Book Implied" in text and "Payout Odds" in text and "Edge" in text and "RISK: $" in text,
          "ticket blocks show all required fields")
    check(max(len(line) for line in text.splitlines()) <= REPORT_WIDTH, "no line exceeds report width")

    # Flat mode report
    flat_cfg = ReportConfig(week=6, bankroll=500.0, staking=StakingConfig.flat_dollar(10), report_date=_dt.date(2026, 10, 7))
    flat_report = build_weekly_report(slate, flat_cfg)
    check(all(abs(s.stake.stake_dollars - 10.0) < 1e-9 or s.stake.cap_applied for s in flat_report.recommended),
          "flat $10 mode stakes $10 per +EV ticket (unless capped)")
    check("Flat Unit Betting" in flat_report.text and "[Mode A]" in flat_report.text, "flat report labels Mode A")

    # Exports
    recs = report.to_records()
    check(len(recs) == len(report.recommended) and {"ticket_id", "stake_dollars", "edge", "american_odds"} <= set(recs[0]),
          "to_records() produces DataFrame-ready rows")
    json.dumps(report.to_dict())
    check(True, "to_dict() is JSON serialisable")

    # File round-trip in a temp dir
    with tempfile.TemporaryDirectory() as tmp:
        txt = report.save_text(os.path.join(tmp, "r.txt"))
        js = report.save_json(os.path.join(tmp, "r.json"))
        check(os.path.getsize(txt) > 1000 and os.path.getsize(js) > 1000, "save_text/save_json write files")
        with open(os.path.join(tmp, "in.json"), "w", encoding="utf-8") as fh:
            json.dump({"parlays": slate}, fh)
        tickets, rejected, label = collect_parlays(6, source="json", input_path=os.path.join(tmp, "in.json"))
        check(len(tickets) == 10 and label.startswith("JSON file"), "collect_parlays loads wrapped JSON")
        try:
            collect_parlays(6, source="json", input_path=os.path.join(tmp, "missing.json"))
            check(False, "missing JSON raises")
        except ReporterError:
            check(True, "missing JSON raises ReporterError")

        # Finder integration: fabricate a parlay_finder module on the path
        mod_path = os.path.join(tmp, "parlay_finder.py")
        with open(mod_path, "w", encoding="utf-8") as fh:
            fh.write(
                "class _L(list):\n"
                "    slate_summary = None\n"
                "def find_parlays(week, top_n=10, **kw):\n"
                "    out = _L([\n"
                "        {'ticket_id': f'F{week}', 'legs': [\n"
                "            {'matchup': 'A @ B', 'selection': 'B -3', 'american_odds': -110, 'p_true': 0.57},\n"
                "            {'matchup': 'C @ D', 'selection': 'Over 45', 'american_odds': -110, 'p_true': 0.56},\n"
                "        ]}])\n"
                "    out.slate_summary = {'sides': 4, 'passed': 2, 'markets': 2, 'candidates': 2,\n"
                "        'by_market': {'spread': {'label': 'Spread', 'sides': 2, 'passed': 1, 'min_leg_prob': 0.68, 'min_prob_gap': 0.06},\n"
                "                      'passing_yards': {'label': 'Passing Yards', 'experimental': True, 'sides': 2, 'passed': 1, 'min_leg_prob': 0.55, 'min_prob_gap': 0.04}},\n"
                "        'table': ['  Spread  2  1', '  Passing Yards*  2  1']}\n"
                "    return out\n"
            )
        sys.path.insert(0, tmp)
        try:
            sys.modules.pop("parlay_finder", None)
            tickets, rejected, label = collect_parlays(9, source="finder", bankroll=1000, top_n=3)
            check(len(tickets) == 1 and tickets[0].ticket_id == "F9" and label == "parlay_finder.py",
                  "collect_parlays calls parlay_finder.find_parlays(week=...)")
            summ = getattr(tickets, "slate_summary", None)
            check(isinstance(summ, dict) and summ.get("sides") == 4, "collect_parlays carries the finder's slate summary")
            rep = build_weekly_report(tickets, ReportConfig(week=9, bankroll=1000.0, report_date=_dt.date(2026, 10, 7), slate_summary=summ))
            check("LEGS CLEARING THE FILTER BY MARKET" in rep.text and "Passing Yards*" in rep.text
                  and "2 of 4 sides clear the filter" in rep.text and rep.to_dict()["slate"]["passed"] == 2,
                  "report renders the per-market leg table, the header line and the JSON 'slate' block")
            fallback = build_weekly_report(tickets, ReportConfig(week=9, bankroll=1000.0, report_date=_dt.date(2026, 10, 7),
                                                                 slate_summary={**summ, "table": []}))
            check("Passing Yards" in fallback.text and "floor 55%" in fallback.text, "report falls back to a plain per-market list without a table")
            check(max(len(line) for line in rep.text.splitlines()) <= REPORT_WIDTH, "slate table keeps the report inside its width")
        finally:
            sys.path.remove(tmp)
            sys.modules.pop("parlay_finder", None)

    # Finder fallback when the module cannot be imported (None in sys.modules
    # makes ``import parlay_finder`` raise ImportError, even if the file exists)
    saved = sys.modules.pop("parlay_finder", None)
    sys.modules["parlay_finder"] = None  # type: ignore[assignment]
    try:
        tickets, rejected, label = collect_parlays(6, source="finder")
        check(label.startswith("SIMULATED") and len(tickets) == 10, "finder missing -> demo fallback")
    finally:
        sys.modules.pop("parlay_finder", None)
        if saved is not None:
            sys.modules["parlay_finder"] = saved

    # Week estimator: 2026 Labor Day = Sep 7, kickoff Thu Sep 10
    check(estimate_nfl_week(_dt.date(2026, 9, 10)) == 1, "kickoff Thursday -> week 1")
    check(estimate_nfl_week(_dt.date(2026, 9, 14)) == 1, "Monday night of week 1 -> week 1")
    check(estimate_nfl_week(_dt.date(2026, 9, 15)) == 2, "Tuesday after week 1 -> upcoming week 2")
    check(estimate_nfl_week(_dt.date(2026, 10, 7)) == 5, "Wed Oct 7 2026 -> upcoming week 5")
    check(estimate_nfl_week(_dt.date(2026, 8, 1)) == 1 and estimate_nfl_week(_dt.date(2027, 2, 1)) == 18, "clamped to 1..18")
    check(nfl_season_year(_dt.date(2027, 1, 15)) == 2026 and nfl_season_year(_dt.date(2026, 9, 1)) == 2026, "Jan/Feb belong to prior season")
    check(ReportConfig(week=18, bankroll=100, report_date=_dt.date(2027, 1, 5)).season == 2026, "ReportConfig season defaults to season year")

    # Empty slate: a strict finder week renders a clean "no bets" report
    empty = build_weekly_report([], cfg)
    check(empty.sections == {} and empty.total_risk == 0 and "No bets are recommended" in empty.text, "empty slate renders a no-bets report")
    two_only = build_weekly_report([s for s in slate if len(s["legs"]) == 2], cfg)
    check(set(two_only.sections) == {"2-LEG"}, "leg counts with no candidates get no section")

    # Config validation
    for bad_kwargs in ({"week": 0, "bankroll": 100}, {"week": 1, "bankroll": -5}, {"week": 1, "bankroll": float("nan")}):
        try:
            ReportConfig(**bad_kwargs)
            check(False, f"ReportConfig rejects {bad_kwargs}")
        except ReporterError:
            check(True, f"ReportConfig rejects {bad_kwargs}")

    print(f"\n{'ALL TESTS PASSED' if failures == 0 else f'{failures} TEST(S) FAILED'}")
    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="weekly_reporter",
        description="Generate the weekly NFL parlay report card with recommended stakes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--week", type=int, default=None, help="Simulated NFL week number (default: estimated from --date)")
    p.add_argument("--date", type=str, default=None, help="Report date YYYY-MM-DD (default: today)")
    p.add_argument("--bankroll", type=float, default=1000.0, help="Current total bankroll in dollars")
    p.add_argument("--mode", default="kelly", help="Staking mode: flat | kelly")
    p.add_argument("--kelly-multiplier", type=float, default=0.25, dest="kelly_multiplier", help="Mode B: fraction of full Kelly")
    p.add_argument("--flat-pct", type=float, default=0.01, dest="flat_pct", help="Mode A: fraction of bankroll per ticket")
    p.add_argument("--flat-dollars", type=float, default=None, dest="flat_dollars", help="Mode A: fixed dollar unit (overrides --flat-pct)")
    p.add_argument("--max-stake-pct", type=float, default=0.05, dest="max_stake_pct", help="Hard cap per ticket")
    p.add_argument("--min-edge", type=float, default=0.0, dest="min_edge", help="Minimum edge to recommend a ticket")
    p.add_argument("--portfolio-cap", type=float, default=0.15, dest="portfolio_cap", help="Max total weekly exposure (fraction); <=0 disables")
    p.add_argument("--source", default="finder", choices=("finder", "json", "demo"), help="Where to collect parlays from")
    p.add_argument("--input", default=None, help="JSON file of parlay tickets (for --source json, or finder fallback)")
    p.add_argument("--finder-function", default=None, dest="finder_function", help="Explicit function name inside parlay_finder.py")
    p.add_argument("--output", default=DEFAULT_REPORT_FILENAME, help="Plain-text report path")
    p.add_argument("--json-out", default=None, dest="json_out", help="Optional JSON side-car path for backtester.py")
    p.add_argument("--top-n", type=int, default=5, dest="top_n", help="Candidates requested per leg-count section")
    p.add_argument("--max-parlays", type=int, default=DEFAULT_MAX_PARLAYS, dest="max_parlays",
                   help=f"How many parlays to return at most ({MIN_PARLAYS}-{MAX_PARLAYS}); fewer when fewer qualify")
    p.add_argument("--count-same-game", action="store_true", dest="count_same_game",
                   help="Count same-game (SGP-priced) tickets toward the limit instead of listing them apart")
    p.add_argument("--seed", type=int, default=7, help="Seed for the demo slate")
    p.add_argument("--quiet", action="store_true", help="Do not print the report to stdout")
    p.add_argument("--selftest", action="store_true", help="Run built-in tests and exit")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return p


def nfl_season_year(date: _dt.date) -> int:
    """The season a date belongs to: January/February dates are the prior year's season."""
    return date.year if date.month >= 3 else date.year - 1


def estimate_nfl_week(date: _dt.date) -> int:
    """Estimate the *upcoming* regular-season week for a report dated ``date``.

    Week 1 kicks off on the Thursday after Labor Day and each week's slate runs
    Thursday through Monday night. A report produced on the following Tuesday
    or Wednesday is for the NEXT slate, so those days roll forward one week.
    Clamped to 1..18. Used only when ``--week`` is omitted; the production
    pipeline should pass the week explicitly.
    """
    sept1 = _dt.date(nfl_season_year(date), 9, 1)
    labor_day = sept1 + _dt.timedelta(days=(7 - sept1.weekday()) % 7)  # first Monday of September
    kickoff = labor_day + _dt.timedelta(days=3)                            # Thursday night opener
    if date < kickoff:
        return 1
    days = (date - kickoff).days
    week = days // 7 + 1
    if days % 7 >= 5:  # Tuesday (5) or Wednesday (6) after Monday Night Football
        week += 1
    return max(1, min(18, week))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.selftest:
        return _selftest()

    try:
        report_date = _dt.date.fromisoformat(args.date) if args.date else _dt.date.today()
    except ValueError:
        print(f"error: --date must be YYYY-MM-DD, got {args.date!r}", file=sys.stderr)
        return 2
    week = args.week if args.week is not None else estimate_nfl_week(report_date)

    try:
        staking = StakingConfig(
            mode=args.mode,
            flat_pct=args.flat_pct,
            flat_dollars=args.flat_dollars,
            kelly_multiplier=args.kelly_multiplier,
            max_stake_pct=args.max_stake_pct,
            min_edge=args.min_edge,
        )
        report = run_weekly_report(
            week=week,
            bankroll=args.bankroll,
            staking=staking,
            source=args.source,
            input_path=args.input,
            output_path=args.output,
            json_path=args.json_out,
            top_n=args.top_n,
            portfolio_cap_pct=args.portfolio_cap if args.portfolio_cap and args.portfolio_cap > 0 else None,
            report_date=report_date,
            finder_function=args.finder_function,
            demo_seed=args.seed,
            max_parlays=args.max_parlays,
            count_same_game=args.count_same_game,
        )
    except (ReporterError, StakingInputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130

    if not args.quiet:
        print(report.text, end="")
    print(f"Saved report -> {os.path.abspath(args.output)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
