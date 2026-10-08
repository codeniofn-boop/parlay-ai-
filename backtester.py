#!/usr/bin/env python3
"""
backtester.py
=============

Season simulation loop for the NFL parlay pipeline.

For every simulated season it replays the weekly workflow exactly as the
live tools run it:

1. ``parlay_finder.find_parlays`` produces the week's candidate tickets from
   the posted lines and the model's probabilities (never the hidden truth).
2. ``weekly_reporter.build_weekly_report`` stakes and ranks those tickets
   against the *current* bankroll with the chosen staking policy, applying
   every cap in ``staking_engine``.
3. Final scores are drawn from the hidden truth (or read from a results
   file), each leg is graded, pushes void the leg, and the ticket's realised
   P&L is booked through ``BankrollTracker.record``.

Several staking policies are run over identical candidate slates and
identical final scores, so differences between them come only from bet
sizing and from which of the candidates each policy's ranking places.
Repeating over many seasons turns a single lucky run into a distribution:
median growth, 5th/95th percentile bankroll, probability of profit, worst
drawdown, and calibration (did tickets hit as often as the model claimed?).

Data sources
------------
* Simulated (default): ``parlay_finder.simulate_season`` with the ``finder``
  settings from ``pipeline_config.json``. ``--no-edge`` runs the null model
  (no private information) and ``--with-null`` runs both for comparison.
* Real: ``--lines lines.csv --results results.csv`` where ``results.csv`` has
  ``week,away,home,away_score,home_score``. Lines use the finder's CSV format.

Outputs (in ``backtest_output/`` by default)
--------------------------------------------
``backtest_summary.txt``  human-readable report card (80 columns)
``backtest_ledger.csv``   one row per placed ticket per season per policy
``backtest_summary.json`` the same numbers for notebooks / dashboards

Run ``python3 backtester.py`` (defaults; opens the summary when launched by
double-click), ``python3 backtester.py --help``, or ``--selftest``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import logging
import math
import os
import platform
import random
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import parlay_finder as pf
from staking_engine import BankrollTracker, StakingConfig, StakingInputError
from weekly_reporter import ReportConfig, StakedTicket, build_weekly_report

__all__ = [
    "BacktestError",
    "BacktestConfig",
    "WeekData",
    "PolicySeasonResult",
    "BacktestResults",
    "resolve_leg",
    "settle_parlay",
    "build_simulated_season",
    "load_real_season",
    "run_policy_over_season",
    "run_backtest",
    "render_summary",
    "write_outputs",
    "parse_policy",
    "DEFAULT_POLICIES",
]

__version__ = "1.0.0"

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "pipeline_config.json"
DEFAULT_OUT_DIR = "backtest_output"
_W = 80


class BacktestError(RuntimeError):
    """Unrecoverable backtest failure (bad results file, unknown policy...)."""


# ---------------------------------------------------------------------------
# Grading legs and tickets
# ---------------------------------------------------------------------------


def _team_matches(name: str, candidate: str) -> bool:
    """Case-insensitive match allowing nicknames ('Bills' ~ 'Buffalo Bills')."""
    a, b = name.strip().lower(), candidate.strip().lower()
    if not a or not b:
        return False
    return a == b or b.endswith(" " + a) or a.endswith(" " + b) or a in b.split() or a == b.split()[-1]


def _split_matchup(leg: Dict[str, Any]) -> Tuple[str, str]:
    away, home = leg.get("away"), leg.get("home")
    if away and home:
        return str(away), str(home)
    matchup = str(leg.get("matchup", ""))
    if " @ " in matchup:
        a, h = matchup.split(" @ ", 1)
        return a.strip(), h.strip()
    if " at " in matchup.lower():
        idx = matchup.lower().index(" at ")
        return matchup[:idx].strip(), matchup[idx + 4:].strip()
    raise BacktestError(f"Cannot determine away/home from matchup '{matchup}'")


def resolve_leg(leg: Dict[str, Any], away_score: int, home_score: int) -> str:
    """Grade one leg against a final score -> ``"win"`` | ``"loss"`` | ``"push"``.

    ``leg`` needs ``selection`` plus either ``away``/``home`` or a
    ``"Away @ Home"`` matchup string.
    """
    away, home = _split_matchup(leg)
    ps = pf.parse_selection(str(leg["selection"]), str(leg.get("market", "")))
    if ps.market == "total":
        total = away_score + home_score
        assert ps.line is not None
        if total == ps.line:
            return "push"
        return "win" if ((total > ps.line) == ((ps.direction or "").lower() == "over")) else "loss"

    assert ps.team is not None
    if _team_matches(ps.team, home):
        team_score, opp_score = home_score, away_score
    elif _team_matches(ps.team, away):
        team_score, opp_score = away_score, home_score
    else:
        raise BacktestError(f"Selection '{leg['selection']}' names neither {away} nor {home}")

    if ps.market == "team_total":
        assert ps.line is not None
        if team_score == ps.line:
            return "push"
        return "win" if ((team_score > ps.line) == ((ps.direction or "").lower() == "over")) else "loss"
    if ps.market == "spread":
        assert ps.line is not None
        adjusted = team_score + ps.line - opp_score
    else:  # moneyline
        adjusted = team_score - opp_score
    if abs(adjusted) < 1e-9:
        return "push"
    return "win" if adjusted > 0 else "loss"


def settle_parlay(leg_results: Sequence[Tuple[Dict[str, Any], str]], stake: float) -> Tuple[str, float]:
    """Book a parlay from its graded legs -> ``(outcome, pnl)``.

    Standard house rules: any losing leg loses the ticket; pushed legs are
    voided and the payout is recomputed from the remaining legs; a ticket
    whose legs all push is refunded.
    """
    if stake <= 0:
        return "skip", 0.0
    if any(r == "loss" for _, r in leg_results):
        return "loss", -stake
    remaining = 1.0
    wins = 0
    for leg, r in leg_results:
        if r == "win":
            dec = leg.get("decimal_odds")
            if dec is None:
                dec = pf.american_to_decimal(leg["american_odds"])
            remaining *= float(dec)
            wins += 1
    if wins == 0:
        return "push", 0.0
    return "win", stake * (remaining - 1.0)


# ---------------------------------------------------------------------------
# Season data (simulated or real)
# ---------------------------------------------------------------------------


@dataclass
class WeekData:
    week: int
    tickets: List[Dict[str, Any]]
    scores: Dict[Tuple[str, str], Tuple[int, int]]   # (away, home) -> (away_score, home_score)


def build_simulated_season(
    season_year: int,
    seed: int,
    weeks: int,
    finder_cfg: pf.FinderConfig,
) -> List[WeekData]:
    """Simulate lines + model + final scores and run the finder on each week."""
    games_by_week = pf.simulate_season(season_year, seed, weeks, finder_cfg.sim)
    score_rng = random.Random(f"{season_year}:{seed}:scores")
    out: List[WeekData] = []
    for week in range(1, weeks + 1):
        games = games_by_week[week]
        tickets = pf.find_parlays(week, games=games, config=finder_cfg)
        scores = {(g.away, g.home): g.simulate_final_score(score_rng) for g in games}
        out.append(WeekData(week=week, tickets=tickets, scores=scores))
    return out


def load_real_season(lines_csv: str, results_csv: str, finder_cfg: pf.FinderConfig) -> List[WeekData]:
    """Real lines + real final scores -> per-week tickets and scores."""
    if not os.path.isfile(results_csv):
        raise BacktestError(f"Results file not found: {results_csv}")
    scores_by_week: Dict[int, Dict[Tuple[str, str], Tuple[int, int]]] = {}
    try:
        with open(results_csv, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            required = {"week", "away", "home", "away_score", "home_score"}
            if reader.fieldnames is None or required - {f.strip().lower() for f in reader.fieldnames}:
                raise BacktestError(f"{results_csv} needs columns: {', '.join(sorted(required))}")
            for i, raw in enumerate(reader, 2):
                r = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
                try:
                    wk = int(float(r["week"]))
                    a_s, h_s = int(float(r["away_score"])), int(float(r["home_score"]))
                except ValueError as exc:
                    raise BacktestError(f"{results_csv} line {i}: bad number") from exc
                scores_by_week.setdefault(wk, {})[(r["away"], r["home"])] = (a_s, h_s)
    except OSError as exc:
        raise BacktestError(f"Could not read {results_csv}: {exc}") from exc
    if not scores_by_week:
        raise BacktestError(f"{results_csv} contains no games")

    cfg = pf.FinderConfig.from_dict({**asdict(finder_cfg), "source": "csv", "lines_csv": lines_csv})
    cfg.sim = finder_cfg.sim
    out: List[WeekData] = []
    for week in sorted(scores_by_week):
        try:
            tickets = pf.find_parlays(week, config=cfg)
        except pf.FinderError as exc:
            logger.warning("Week %d skipped: %s", week, exc)
            continue
        out.append(WeekData(week=week, tickets=tickets, scores=scores_by_week[week]))
    if not out:
        raise BacktestError("No weeks could be built from the lines and results files")
    return out


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

DEFAULT_POLICIES: Dict[str, StakingConfig] = {
    "flat_1pct": StakingConfig.flat_percent(0.01),
    "flat_$10": StakingConfig.flat_dollar(10.0),
    "kelly_0.10": StakingConfig.fractional_kelly(0.10),
    "kelly_0.25": StakingConfig.fractional_kelly(0.25),
}


def parse_policy(spec: str) -> Tuple[str, StakingConfig]:
    """``"flat:0.02"`` -> flat 2 %; ``"flat$:25"`` -> flat $25; ``"kelly:0.5"`` -> half Kelly.

    A bare name from :data:`DEFAULT_POLICIES` is also accepted.
    """
    spec = spec.strip()
    if spec in DEFAULT_POLICIES:
        return spec, DEFAULT_POLICIES[spec]
    if ":" not in spec:
        raise BacktestError(f"Unknown policy '{spec}'. Use flat:<pct>, flat$:<dollars>, kelly:<multiplier>")
    kind, value = spec.split(":", 1)
    try:
        v = float(value)
        if kind == "flat":
            return f"flat_{v:g}pct" if v >= 1 else f"flat_{v * 100:g}pct", StakingConfig.flat_percent(v if v < 1 else v / 100.0)
        if kind == "flat$":
            return f"flat_${v:g}", StakingConfig.flat_dollar(v)
        if kind == "kelly":
            return f"kelly_{v:g}", StakingConfig.fractional_kelly(v)
    except (ValueError, StakingInputError) as exc:
        raise BacktestError(f"Bad policy '{spec}': {exc}") from exc
    raise BacktestError(f"Unknown policy kind '{kind}' in '{spec}'")


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    seasons: int = 20
    weeks: int = 18
    bankroll: float = 1000.0
    seed: int = 7
    season_year: Optional[int] = None
    policies: Dict[str, StakingConfig] = field(default_factory=lambda: dict(DEFAULT_POLICIES))
    portfolio_cap: Optional[float] = 0.15
    top_n: int = 5
    no_edge: bool = False
    with_null: bool = False
    lines_csv: Optional[str] = None
    results_csv: Optional[str] = None
    out_dir: str = DEFAULT_OUT_DIR
    finder: pf.FinderConfig = field(default_factory=pf.load_finder_config)

    def __post_init__(self) -> None:
        if self.seasons < 1 or self.weeks < 1:
            raise BacktestError("seasons and weeks must be >= 1")
        if self.bankroll <= 0 or math.isnan(self.bankroll):
            raise BacktestError("bankroll must be positive")
        if not self.policies:
            raise BacktestError("at least one staking policy is required")
        if self.season_year is None:
            self.season_year = pf.nfl_season_year()


@dataclass
class PolicySeasonResult:
    season_index: int
    seed: int
    policy: str
    variant: str                      # "model" | "null"
    summary: Dict[str, Any]           # BankrollTracker.summary()
    weekly_bankroll: List[float]


@dataclass
class BacktestResults:
    config: BacktestConfig
    per_season: List[PolicySeasonResult]
    ledger: List[Dict[str, Any]]
    calibration: Dict[str, Dict[str, float]]   # variant -> {mean_model_p, hit_rate, n}
    generated_at: _dt.datetime = field(default_factory=_dt.datetime.now)

    def aggregate(self) -> Dict[str, Dict[str, float]]:
        """Distribution statistics per (variant, policy) across seasons."""
        groups: Dict[str, List[PolicySeasonResult]] = {}
        for r in self.per_season:
            groups.setdefault(self._label(r), []).append(r)
        out: Dict[str, Dict[str, float]] = {}
        for label, rs in groups.items():
            ends = [r.summary["ending_bankroll"] for r in rs]
            growth = [r.summary["growth"] for r in rs]
            dds = [r.summary["max_drawdown"] for r in rs]
            out[label] = {
                "seasons": len(rs),
                "median_end": statistics.median(ends),
                "mean_growth": statistics.fmean(growth),
                "median_growth": statistics.median(growth),
                "p05_end": _percentile(ends, 0.05),
                "p95_end": _percentile(ends, 0.95),
                "prob_profit": sum(1 for e in ends if e > self.config.bankroll) / len(rs),
                "median_max_dd": statistics.median(dds),
                "worst_max_dd": max(dds),
                "prob_dd_over_50": sum(1 for d in dds if d >= 0.5) / len(rs),
                "prob_ruin": sum(1 for e in ends if e <= self.config.bankroll * 0.05) / len(rs),
                "bets_per_season": statistics.fmean(r.summary["tickets_placed"] for r in rs),
                "win_rate": statistics.fmean(r.summary["win_rate"] for r in rs),
                "roi_on_turnover": statistics.fmean(r.summary["roi_on_turnover"] for r in rs),
            }
        return out

    @staticmethod
    def _label(r: PolicySeasonResult) -> str:
        return r.policy if r.variant == "model" else f"{r.policy} [null]"

    def to_dict(self) -> Dict[str, Any]:
        cfg = asdict(self.config)
        cfg["policies"] = {k: v.to_dict() for k, v in self.config.policies.items()}
        cfg["finder"] = asdict(self.config.finder)
        return {
            "generated_at": self.generated_at.isoformat(timespec="seconds"),
            "config": cfg,
            "calibration": self.calibration,
            "aggregate": self.aggregate(),
            "per_season": [asdict(r) for r in self.per_season],
        }


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _grade_ticket(st: StakedTicket, scores: Dict[Tuple[str, str], Tuple[int, int]]) -> Tuple[str, float, List[str]]:
    leg_results: List[Tuple[Dict[str, Any], str]] = []
    for leg in st.ticket.legs:
        leg_dict = {"matchup": leg.matchup, "selection": leg.selection, "market": leg.market,
                    "decimal_odds": leg.decimal_odds}
        away, home = _split_matchup(leg_dict)
        key = (away, home)
        if key not in scores:
            raise BacktestError(f"No final score for {away} @ {home} in week {st.ticket.week}")
        a_s, h_s = scores[key]
        leg_results.append((leg_dict, resolve_leg(leg_dict, a_s, h_s)))
    outcome, pnl = settle_parlay(leg_results, st.stake.stake_dollars)
    return outcome, pnl, [r for _, r in leg_results]


def calibrate(week_data: Sequence[WeekData]) -> Dict[str, float]:
    """Model-claimed probability vs realised hit rate over every finder ticket."""
    probs: List[float] = []
    hits = 0
    for wd in week_data:
        for t in wd.tickets:
            results = []
            for leg in t["legs"]:
                away, home = _split_matchup(leg)
                a_s, h_s = wd.scores[(away, home)]
                results.append(resolve_leg(leg, a_s, h_s))
            if "push" in results:
                continue
            probs.append(float(t["p_true"]))
            hits += all(r == "win" for r in results)
    n = len(probs)
    return {"mean_model_p": (sum(probs) / n) if n else 0.0, "hit_rate": (hits / n) if n else 0.0, "n": float(n)}


def run_policy_over_season(
    week_data: Sequence[WeekData],
    policy_name: str,
    policy: StakingConfig,
    cfg: BacktestConfig,
    season_index: int,
    seed: int,
    variant: str,
    ledger: Optional[List[Dict[str, Any]]] = None,
) -> PolicySeasonResult:
    """Replay one season for one staking policy using the live report builder."""
    tracker = BankrollTracker(cfg.bankroll, policy)
    weekly: List[float] = []
    for wd in week_data:
        if tracker.bankroll <= 0.0:
            weekly.append(0.0)
            continue
        report = build_weekly_report(
            wd.tickets,
            ReportConfig(week=wd.week, bankroll=tracker.bankroll, staking=policy, top_n_per_group=cfg.top_n,
                         portfolio_cap_pct=cfg.portfolio_cap, include_skipped=False, source_label="backtest"),
        )
        for st in report.recommended:
            outcome, pnl, leg_results = _grade_ticket(st, wd.scores)
            won: Optional[bool] = True if outcome == "win" else False if outcome == "loss" else None
            tracker.record(st.stake, pnl, won, label=f"W{wd.week:02d}", outcome=outcome)
            if ledger is not None:
                ledger.append({
                    "season": season_index, "seed": seed, "variant": variant, "week": wd.week,
                    "policy": policy_name, "ticket_id": st.ticket.ticket_id, "n_legs": st.ticket.n_legs,
                    "legs": " | ".join(f"{l.matchup}: {l.selection}" for l in st.ticket.legs),
                    "p_true": round(st.ticket.p_true, 6), "decimal_odds": round(st.ticket.decimal_odds, 4),
                    "american_odds": st.ticket.american_odds, "edge": round(st.ticket.edge, 6),
                    "stake": st.stake.stake_dollars, "leg_results": "/".join(leg_results),
                    "outcome": outcome, "pnl": round(pnl, 2), "bankroll_after": round(tracker.bankroll, 2),
                })
        weekly.append(tracker.bankroll)
    return PolicySeasonResult(season_index, seed, policy_name, variant, tracker.summary(), weekly)


def run_backtest(cfg: BacktestConfig) -> BacktestResults:
    """Run every policy over every season (and the null model if requested)."""
    variants: List[Tuple[str, pf.FinderConfig]] = []
    base_finder = cfg.finder
    if cfg.lines_csv or cfg.results_csv:
        if not (cfg.lines_csv and cfg.results_csv):
            raise BacktestError("Real-data mode needs both --lines and --results")
        variants.append(("model", base_finder))
    else:
        if cfg.no_edge:
            null_finder = pf.FinderConfig.from_dict(asdict(base_finder))
            null_finder.sim = base_finder.sim.no_edge()
            variants.append(("null", null_finder))
        else:
            variants.append(("model", base_finder))
            if cfg.with_null:
                null_finder = pf.FinderConfig.from_dict(asdict(base_finder))
                null_finder.sim = base_finder.sim.no_edge()
                variants.append(("null", null_finder))

    per_season: List[PolicySeasonResult] = []
    ledger: List[Dict[str, Any]] = []
    calib_acc: Dict[str, List[Dict[str, float]]] = {}

    for variant, finder_cfg in variants:
        for idx in range(cfg.seasons):
            seed = cfg.seed + idx
            if cfg.lines_csv:
                week_data = load_real_season(cfg.lines_csv, cfg.results_csv or "", finder_cfg)
            else:
                week_data = build_simulated_season(cfg.season_year or pf.nfl_season_year(), seed, cfg.weeks, finder_cfg)
            calib_acc.setdefault(variant, []).append(calibrate(week_data))
            for name, policy in cfg.policies.items():
                per_season.append(run_policy_over_season(week_data, name, policy, cfg, idx, seed, variant, ledger))
            logger.info("%s season %d/%d done (seed %d)", variant, idx + 1, cfg.seasons, seed)
            if cfg.lines_csv:
                break  # real data has exactly one season

    calibration: Dict[str, Dict[str, float]] = {}
    for variant, rows in calib_acc.items():
        n = sum(r["n"] for r in rows)
        calibration[variant] = {
            "mean_model_p": (sum(r["mean_model_p"] * r["n"] for r in rows) / n) if n else 0.0,
            "hit_rate": (sum(r["hit_rate"] * r["n"] for r in rows) / n) if n else 0.0,
            "n": n,
        }
    return BacktestResults(config=cfg, per_season=per_season, ledger=ledger, calibration=calibration)


# ---------------------------------------------------------------------------
# Rendering and output
# ---------------------------------------------------------------------------


def _money(x: float) -> str:
    return f"${x:,.0f}"


def render_summary(results: BacktestResults) -> str:
    cfg = results.config
    agg = results.aggregate()
    hr, hr2 = "=" * _W, "-" * _W
    out: List[str] = [hr, "NFL PARLAY BACKTEST SUMMARY".center(_W).rstrip()]
    if cfg.lines_csv:
        out.append(f"Real lines + results  |  {os.path.basename(cfg.lines_csv)}".center(_W).rstrip())
    else:
        out.append(f"{cfg.seasons} simulated season(s) x {cfg.weeks} weeks  |  {cfg.season_year} ratings".center(_W).rstrip())
    out.append(hr)
    out.append(f"  {'Generated':<18}: {results.generated_at:%Y-%m-%d %H:%M:%S}")
    out.append(f"  {'Starting Bankroll':<18}: ${cfg.bankroll:,.2f}")
    sim = cfg.finder.sim
    if cfg.lines_csv:
        out.append(f"  {'Pick Source':<18}: parlay_finder.py on real CSV lines")
    else:
        out.append(f"  {'Pick Source':<18}: parlay_finder.py, simulated league")
        out.append(f"  {'Model Settings':<18}: skill {sim.model_skill:.2f}, noise {sim.model_noise_margin:.2f} pt"
                   f"{' (null control included)' if 'null' in results.calibration and 'model' in results.calibration else ''}")
    caps = f"max {max(p.max_stake_pct for p in cfg.policies.values()):.0%} per ticket"
    if cfg.portfolio_cap:
        caps += f", max {cfg.portfolio_cap:.0%} weekly exposure"
    out.append(f"  {'Safety Caps':<18}: {caps}")
    fr = cfg.finder
    out.append(f"  {'Finder Rules':<18}: {'/'.join(str(n) for n in fr.leg_sizes)}-leg only; leg >= {fr.min_leg_prob:.0%} "
               f"and >= {fr.min_prob_gap * 100:.0f} pts over {fr.edge_basis}")
    out.append(f"  {'Same-Game Policy':<18}: {fr.same_game_policy}")
    out.append(f"  {'Tickets Per Week':<18}: top {cfg.top_n} per leg size")
    for variant, c in results.calibration.items():
        label = "Calibration" if variant == "model" else "Null Calibration"
        out.append(f"  {label:<18}: claimed {c['mean_model_p']:.1%} avg win prob, "
                   f"hit {c['hit_rate']:.1%} (n={int(c['n']):,})")
    out.append(hr2)
    out.append("")

    out.append("RETURNS BY STAKING POLICY  (one outcome per season)")
    out.append(hr)
    out.append(f"  {'Policy':<18} {'Median End':>11} {'Med Growth':>11} {'P05 End':>9} {'P95 End':>9} {'P(profit)':>10}")
    out.append(f"  {'-' * 18} {'-' * 11} {'-' * 11} {'-' * 9} {'-' * 9} {'-' * 10}")
    for label, a in agg.items():
        out.append(f"  {label:<18} {_money(a['median_end']):>11} {a['median_growth']:>+11.1%} "
                   f"{_money(a['p05_end']):>9} {_money(a['p95_end']):>9} {a['prob_profit']:>10.0%}")
    out.append("")

    out.append("RISK AND ACTIVITY BY STAKING POLICY")
    out.append(hr)
    out.append(f"  {'Policy':<18} {'MedDD':>7} {'WorstDD':>8} {'P(DD>50)':>9} {'P(ruin)':>7} {'Bets/yr':>7} {'Win%':>6} {'ROI':>7}")
    out.append(f"  {'-' * 18} {'-' * 7} {'-' * 8} {'-' * 9} {'-' * 7} {'-' * 7} {'-' * 6} {'-' * 7}")
    for label, a in agg.items():
        out.append(f"  {label:<18} {a['median_max_dd']:>7.1%} {a['worst_max_dd']:>8.1%} {a['prob_dd_over_50']:>9.0%} "
                   f"{a['prob_ruin']:>7.0%} {a['bets_per_season']:>7.0f} {a['win_rate']:>6.1%} {a['roi_on_turnover']:>+7.1%}")
    out.append("")

    out.append("HOW TO READ THIS")
    out.append(hr)
    for line in (
        "Median End / Med Growth: the typical season's finishing bankroll and return.",
        "P05 / P95 End: a bad (1-in-20) and a good (1-in-20) season.",
        "P(profit): share of seasons that finished above the starting bankroll.",
        "Med MaxDD / Worst DD: typical and worst peak-to-trough bankroll decline.",
        "P(ruin): share of seasons ending with under 5% of the starting bankroll.",
        "ROI: profit per dollar staked. Win%: decided tickets won (pushes excluded).",
        "Calibration compares what the model claimed with what actually hit; a",
        "claimed rate well above the hit rate means the model is over-confident.",
        "Rows tagged [null] use a model with NO real information: a policy that",
        "still 'wins' there is riding luck, not edge.",
    ):
        out.append(f"  {line}")
    out.append(hr2)
    out.append("For simulation / research purposes only".center(_W).rstrip())
    out.append(hr)
    return "\n".join(out) + "\n"


def write_outputs(results: BacktestResults, out_dir: str) -> Dict[str, str]:
    """Write summary .txt, ledger .csv and summary .json; returns their paths."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        paths = {
            "summary": os.path.abspath(os.path.join(out_dir, "backtest_summary.txt")),
            "ledger": os.path.abspath(os.path.join(out_dir, "backtest_ledger.csv")),
            "json": os.path.abspath(os.path.join(out_dir, "backtest_summary.json")),
        }
        with open(paths["summary"], "w", encoding="utf-8") as fh:
            fh.write(render_summary(results))
        with open(paths["ledger"], "w", newline="", encoding="utf-8") as fh:
            if results.ledger:
                writer = csv.DictWriter(fh, fieldnames=list(results.ledger[0].keys()))
                writer.writeheader()
                writer.writerows(results.ledger)
            else:
                fh.write("season,seed,variant,week,policy,ticket_id\n")
        with open(paths["json"], "w", encoding="utf-8") as fh:
            json.dump(results.to_dict(), fh, indent=2)
    except OSError as exc:
        raise BacktestError(f"Could not write outputs to {out_dir}: {exc}") from exc
    return paths


def _open_file(path: str) -> None:
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.run(["open", path], check=False)
        elif system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:  # pragma: no cover
        print(f"(could not auto-open {path}: {exc})")


def load_backtest_section(path: str = CONFIG_FILENAME) -> Dict[str, Any]:
    """Optional ``"backtester"`` section of pipeline_config.json."""
    for candidate in (path, os.path.join(os.path.dirname(os.path.abspath(__file__)), path)):
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                section = data.get("backtester", {}) if isinstance(data, dict) else {}
                return section if isinstance(section, dict) else {}
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not read %s (%s)", candidate, exc)
                return {}
    return {}


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

    print("backtester self-test")
    leg = {"matchup": "Kansas City Chiefs @ Buffalo Bills", "selection": "Buffalo Bills -3.5", "market": "spread"}
    check(resolve_leg(leg, 20, 24) == "win" and resolve_leg(leg, 21, 24) == "loss", "spread win / loss")
    check(resolve_leg({**leg, "selection": "Buffalo Bills -3"}, 21, 24) == "push", "spread push on whole number")
    check(resolve_leg({**leg, "selection": "Chiefs +3.5"}, 21, 24) == "win", "nickname + dog spread")
    check(resolve_leg({**leg, "selection": "Over 44.5"}, 20, 24) == "loss" and resolve_leg({**leg, "selection": "Under 44.5"}, 20, 24) == "win", "totals")
    check(resolve_leg({**leg, "selection": "Over 44"}, 20, 24) == "push", "total push")
    check(resolve_leg({**leg, "selection": "Kansas City Chiefs ML"}, 27, 24) == "win" and resolve_leg({**leg, "selection": "Bills ML"}, 27, 24) == "loss", "moneyline")
    check(resolve_leg({**leg, "selection": "Bills ML"}, 24, 24) == "push", "moneyline tie pushes")
    check(resolve_leg({**leg, "selection": "Buffalo Bills Over 23.5", "market": "team_total"}, 20, 24) == "win"
          and resolve_leg({**leg, "selection": "Chiefs Under 20.5", "market": "team_total"}, 20, 24) == "win"
          and resolve_leg({**leg, "selection": "Buffalo Bills Over 24", "market": "team_total"}, 20, 24) == "push", "team totals grade")
    try:
        resolve_leg({**leg, "selection": "Dallas Cowboys -3"}, 1, 2)
        check(False, "foreign team raises")
    except BacktestError:
        check(True, "foreign team raises BacktestError")

    legs = [({"decimal_odds": 1.91}, "win"), ({"decimal_odds": 1.91}, "win")]
    o, pnl = settle_parlay(legs, 10.0)
    check(o == "win" and abs(pnl - 10 * (1.91 ** 2 - 1)) < 1e-9, "two winning legs pay full parlay")
    o, pnl = settle_parlay([({"decimal_odds": 1.91}, "win"), ({"decimal_odds": 1.91}, "push")], 10.0)
    check(o == "win" and abs(pnl - 10 * 0.91) < 1e-9, "pushed leg voids to single")
    check(settle_parlay([({"decimal_odds": 1.91}, "push")] * 2, 10.0) == ("push", 0.0), "all pushes refund")
    check(settle_parlay([({"decimal_odds": 1.91}, "win"), ({"decimal_odds": 1.91}, "loss")], 10.0) == ("loss", -10.0), "any loss loses")
    check(settle_parlay(legs, 0.0) == ("skip", 0.0), "zero stake skipped")

    name, pol = parse_policy("kelly:0.5")
    check(name == "kelly_0.5" and pol.kelly_multiplier == 0.5, "parse kelly policy")
    name, pol = parse_policy("flat$:25")
    check(name == "flat_$25" and pol.flat_dollars == 25.0, "parse flat-dollar policy")
    name, pol = parse_policy("flat:2")
    check(abs(pol.flat_pct - 0.02) < 1e-12, "parse flat percent given as 2 -> 2%")
    try:
        parse_policy("martingale:2")
        check(False, "unknown policy raises")
    except BacktestError:
        check(True, "unknown policy raises BacktestError")

    # Relaxed thresholds give the simulated league enough tickets to exercise the loop;
    # the production defaults (68% / 6 points) are tested in parlay_finder itself.
    finder_cfg = pf.FinderConfig(source="sim", seed=7, season=2026, min_leg_prob=0.0, min_prob_gap=0.0, min_leg_edge=0.02)
    wd = build_simulated_season(2026, 7, 4, finder_cfg)
    check(len(wd) == 4 and all(len(w.scores) in (14, 16) for w in wd), "simulated season builds 4 weeks of scores")
    check(all(all(t["week"] == w.week for t in w.tickets) for w in wd), "tickets carry their week")
    cal = calibrate(wd)
    check(0 < cal["mean_model_p"] < 1 and cal["n"] > 0, "calibration computes")

    cfg = BacktestConfig(seasons=3, weeks=6, bankroll=1000, seed=11, season_year=2026, finder=finder_cfg,
                         policies={"flat_1pct": DEFAULT_POLICIES["flat_1pct"], "kelly_0.25": DEFAULT_POLICIES["kelly_0.25"]},
                         with_null=True, out_dir="unused", top_n=finder_cfg.top_n)
    res = run_backtest(cfg)
    check(len(res.per_season) == 3 * 2 * 2, "3 seasons x 2 policies x 2 variants results")
    agg = res.aggregate()
    check({"flat_1pct", "kelly_0.25", "flat_1pct [null]", "kelly_0.25 [null]"} == set(agg), "aggregate labels include null rows")
    check(all(r["stake"] > 0 for r in res.ledger) and all(r["outcome"] in ("win", "loss", "push") for r in res.ledger), "ledger rows have stakes and outcomes")
    ks = [r for r in res.per_season if r.policy == "kelly_0.25" and r.variant == "model"]
    check(all(len(r.weekly_bankroll) == 6 for r in ks), "weekly bankroll path recorded")
    check(all(r.summary["ending_bankroll"] == r.weekly_bankroll[-1] for r in ks), "ending bankroll matches path")
    # Identical slates across policies: same tickets graded, so ticket ids per week agree
    f_ids = {(r["season"], r["week"], r["ticket_id"]) for r in res.ledger if r["policy"] == "flat_1pct" and r["variant"] == "model"}
    k_ids = {(r["season"], r["week"], r["ticket_id"]) for r in res.ledger if r["policy"] == "kelly_0.25" and r["variant"] == "model"}
    check(f_ids == k_ids, "policies see identical tickets when every finder ticket is placed")
    text = render_summary(res)
    check(max(len(l) for l in text.splitlines()) <= _W, "summary fits 80 columns")
    check("[null]" in text and "Calibration" in text, "summary shows null rows and calibration")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        paths = write_outputs(res, os.path.join(tmp, "out"))
        check(all(os.path.getsize(p) > 100 for p in paths.values()), "outputs written")
        with open(paths["ledger"], newline="", encoding="utf-8") as fh:
            check(len(list(csv.DictReader(fh))) == len(res.ledger), "ledger CSV row count")
        # Real-data mode round trip with a tiny lines + results pair
        lines = os.path.join(tmp, "lines.csv")
        results_csv = os.path.join(tmp, "results.csv")
        with open(lines, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["week", "away", "home", "market", "selection", "american_odds", "model_prob"])
            for wk in (1, 2):
                w.writerow([wk, "Kansas City Chiefs", "Buffalo Bills", "spread", "Buffalo Bills -3.5", -110, 0.70])
                w.writerow([wk, "Kansas City Chiefs", "Buffalo Bills", "spread", "Kansas City Chiefs +3.5", -110, 0.30])
                w.writerow([wk, "Dallas Cowboys", "Philadelphia Eagles", "total", "Over 44.5", -110, 0.70])
                w.writerow([wk, "Dallas Cowboys", "Philadelphia Eagles", "total", "Under 44.5", -110, 0.30])
                w.writerow([wk, "Green Bay Packers", "Detroit Lions", "moneyline", "Detroit Lions ML", -130, 0.72])
                w.writerow([wk, "Green Bay Packers", "Detroit Lions", "moneyline", "Green Bay Packers ML", 110, 0.28])
        with open(results_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["week", "away", "home", "away_score", "home_score"])
            w.writerow([1, "Kansas City Chiefs", "Buffalo Bills", 17, 27])
            w.writerow([1, "Dallas Cowboys", "Philadelphia Eagles", 24, 28])
            w.writerow([1, "Green Bay Packers", "Detroit Lions", 20, 23])
            w.writerow([2, "Kansas City Chiefs", "Buffalo Bills", 30, 20])
            w.writerow([2, "Dallas Cowboys", "Philadelphia Eagles", 10, 13])
            w.writerow([2, "Green Bay Packers", "Detroit Lions", 31, 14])
        real_cfg = BacktestConfig(seasons=1, weeks=2, bankroll=500, lines_csv=lines, results_csv=results_csv,
                                  finder=pf.FinderConfig(source="csv", lines_csv=lines),
                                  policies={"flat_$10": DEFAULT_POLICIES["flat_$10"]})
        real = run_backtest(real_cfg)
        wk1 = [r for r in real.ledger if r["week"] == 1]
        wk2 = [r for r in real.ledger if r["week"] == 2]
        check(wk1 and all(r["outcome"] == "win" for r in wk1), "real data: every week-1 ticket wins (all legs hit)")
        check(wk2 and all(r["outcome"] == "loss" for r in wk2), "real data: every week-2 ticket loses")
        try:
            BacktestConfig(seasons=1, lines_csv=lines, finder=finder_cfg)
            run_backtest(BacktestConfig(seasons=1, lines_csv=lines, finder=finder_cfg))
            check(False, "lines without results raises")
        except BacktestError:
            check(True, "lines without results raises BacktestError")

    print(f"\n{'ALL TESTS PASSED' if failures == 0 else f'{failures} TEST(S) FAILED'}")
    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="backtester", description="Replay seasons through the parlay pipeline and compare staking policies.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--seasons", type=int, default=None, help="Number of simulated seasons")
    p.add_argument("--weeks", type=int, default=None, help="Weeks per season")
    p.add_argument("--bankroll", type=float, default=None, help="Starting bankroll")
    p.add_argument("--seed", type=int, default=None, help="Base seed (season i uses seed+i)")
    p.add_argument("--season-year", type=int, dest="season_year", default=None, help="Ratings season year")
    p.add_argument("--policy", action="append", default=None,
                   help="Staking policy, repeatable: flat:<pct> | flat$:<dollars> | kelly:<mult> | a default name")
    p.add_argument("--portfolio-cap", type=float, dest="portfolio_cap", default=None, help="Max weekly exposure fraction (<=0 disables)")
    p.add_argument("--top-n", type=int, dest="top_n", default=None, help="Tickets placed per leg size per week")
    p.add_argument("--no-edge", action="store_true", dest="no_edge", help="Run only the null model (no private information)")
    p.add_argument("--with-null", action="store_true", dest="with_null", help="Also run the null model for comparison")
    p.add_argument("--lines", default=None, help="Real lines CSV (with --results)")
    p.add_argument("--results", default=None, help="Real results CSV: week,away,home,away_score,home_score")
    p.add_argument("--out-dir", dest="out_dir", default=None, help="Output directory")
    p.add_argument("--min-leg-prob", type=float, dest="min_leg_prob", default=None, help="Override finder rule 2 floor (e.g. 0.68)")
    p.add_argument("--min-prob-gap", type=float, dest="min_prob_gap", default=None, help="Override finder rule 2 gap (e.g. 0.06)")
    p.add_argument("--edge-basis", choices=("implied", "fair"), dest="edge_basis", default=None, help="Override the gap basis")
    p.add_argument("--same-game", choices=("never", "positive_only", "any"), dest="same_game_policy", default=None)
    p.add_argument("--open", action="store_true", help="Open the summary when done")
    p.add_argument("--no-open", action="store_true", dest="no_open", help="Never open the summary")
    p.add_argument("--quiet", action="store_true", help="Do not print the summary")
    p.add_argument("--selftest", action="store_true", help="Run built-in tests")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.selftest:
        return _selftest()

    section = load_backtest_section()
    # Drop-in real data: lines.csv + results.csv beside this script switch to real-season mode.
    auto_lines, auto_results = pf.resolve_data_path("lines.csv"), pf.resolve_data_path("results.csv")
    if not (args.lines or args.results or section.get("lines_csv") or section.get("results_csv")):
        if os.path.isfile(auto_lines) and os.path.isfile(auto_results):
            print(f"Found {os.path.basename(auto_lines)} and {os.path.basename(auto_results)}; backtesting real data.")
            args.lines, args.results = auto_lines, auto_results
        elif os.path.isfile(auto_lines) or os.path.isfile(auto_results):
            print("Note: real-data mode needs BOTH lines.csv and results.csv; running the simulated league.")
    try:
        policies: Dict[str, StakingConfig] = {}
        specs = args.policy or section.get("policies") or list(DEFAULT_POLICIES)
        for spec in specs:
            name, pol = parse_policy(str(spec))
            policies[name] = pol
        cap = args.portfolio_cap if args.portfolio_cap is not None else section.get("portfolio_cap", 0.15)
        finder_cfg = pf.load_finder_config()
        finder_overrides = {k: v for k, v in (("min_leg_prob", args.min_leg_prob), ("min_prob_gap", args.min_prob_gap),
                                              ("edge_basis", args.edge_basis), ("same_game_policy", args.same_game_policy)) if v is not None}
        if finder_overrides:
            data = {**asdict(finder_cfg), **finder_overrides}
            data["sim"] = finder_cfg.sim
            finder_cfg = pf.FinderConfig.from_dict(data)
        cfg = BacktestConfig(
            seasons=int(args.seasons or section.get("seasons", 20)),
            weeks=int(args.weeks or section.get("weeks", 18)),
            bankroll=float(args.bankroll or section.get("bankroll", 1000.0)),
            seed=int(args.seed if args.seed is not None else section.get("seed", 7)),
            season_year=args.season_year or section.get("season_year"),
            policies=policies,
            portfolio_cap=(float(cap) if cap and float(cap) > 0 else None),
            top_n=int(args.top_n or section.get("top_n", 5)),
            no_edge=bool(args.no_edge or section.get("no_edge", False)),
            with_null=bool(args.with_null or section.get("with_null", True)),
            lines_csv=args.lines or section.get("lines_csv"),
            results_csv=args.results or section.get("results_csv"),
            out_dir=str(args.out_dir or section.get("out_dir", DEFAULT_OUT_DIR)),
            finder=finder_cfg,
        )
        if cfg.lines_csv:
            print(f"Backtesting the real season in {os.path.basename(cfg.lines_csv)} + "
                  f"{os.path.basename(cfg.results_csv or '')} with {len(cfg.policies)} policies...")
        else:
            print(f"Backtesting {cfg.seasons} season(s) x {cfg.weeks} weeks, {len(cfg.policies)} policies"
                  f"{' + null control' if cfg.with_null and not cfg.no_edge else ''}...")
        results = run_backtest(cfg)
        paths = write_outputs(results, cfg.out_dir)
    except (BacktestError, pf.FinderError, StakingInputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130

    if not args.quiet:
        print()
        print(render_summary(results), end="")
    for label, path in paths.items():
        print(f"Saved {label}: {path}")
    # Double-click launches arrive with no arguments: open the summary for the operator.
    if args.open or (not argv and not args.no_open):
        _open_file(paths["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
