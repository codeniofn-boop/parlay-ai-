#!/usr/bin/env python3
"""
app.py — EdgeBook AI
====================

Streamlit front end for the NFL parlay analytics pipeline.

How the UI hooks into the backend
---------------------------------
The dashboard owns no betting logic of its own. Every number on screen comes
from the same modules the command-line pipeline and the weekly report use:

* ``parlay_finder``   -> market sides, the premium-edge leg filter, the 2-leg
                         ticket generator and the same-game correlation rule.
                         The sidebar's "Minimum edge threshold" is passed
                         straight through as ``FinderConfig.min_prob_gap``.
* ``staking_engine``  -> ``compute_stake`` sizes every ticket against the
                         *current* wallet, and ``apply_portfolio_cap`` keeps the
                         week's total exposure under the configured ceiling.
                         The sidebar's staking mode becomes a ``StakingConfig``.
* ``backtester``      -> one simulated season replayed through the live
                         finder/reporter/staking path feeds the growth chart.
* ``weekly_reporter`` -> week/season helpers so the slate label matches the
                         text report.

Each import is optional. If a module is missing, fails to import, or raises
while running, the matching section flips to clearly labelled demo data so the
layout can always be shown. ``BACKEND.errors`` collects the reasons and the
sidebar shows them.

State model
-----------
Streamlit re-runs this script top to bottom on every interaction, so the
"wallet" lives in ``st.session_state``:

* ``wallet``        current simulated bankroll (starting bankroll minus every
                    wager placed in this session)
* ``wallet_base``   the starting bankroll the wallet was seeded from; when the
                    sidebar input changes, the wallet resets
* ``placed``        ticket_id -> stake for wagers already simulated
* ``ledger``        list of placed wagers for the audit trail

"Simulate / Place wager" buttons use ``on_click`` callbacks. Callbacks run
*before* the re-run, so the KPI tiles at the top of the page already show the
deducted wallet when the page redraws.

Run with ``streamlit run app.py`` or double-click ``launch_edgebook.py``.
"""

from __future__ import annotations

import datetime as _dt
import inspect
import math
import os
import random
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

try:  # Altair ships with Streamlit; st.line_chart is the fallback if it is ever missing.
    import altair as alt
except Exception:  # pragma: no cover
    alt = None

# Make the pipeline importable no matter which folder Streamlit was launched from.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

APP_NAME = "EdgeBook AI"
APP_TAGLINE = "Quantitative Sports Intelligence"
PORTFOLIO_CAP = 0.15          # mirrors pipeline_config.json / weekly_reporter default
TOP_N_PER_GROUP = 5           # tickets shown, same as the weekly report
HIGH_EDGE_MARGIN = 0.025      # signal-board rows this far above the active edge threshold get the subtle tint
SEASON_WEEKS = 18


# ===========================================================================
# 1. Backend bridge — optional imports with graceful degradation
# ===========================================================================


class _Backend:
    """Holds whichever pipeline modules imported cleanly plus the reasons for any that did not."""

    def __init__(self) -> None:
        self.finder = None
        self.staking = None
        self.backtester = None
        self.reporter = None
        self.errors: Dict[str, str] = {}
        for name in ("parlay_finder", "staking_engine", "backtester", "weekly_reporter"):
            try:
                module = __import__(name)
            except Exception as exc:  # ImportError, SyntaxError, anything during module init
                self.errors[name] = f"{type(exc).__name__}: {exc}"
                continue
            setattr(self, {"parlay_finder": "finder", "staking_engine": "staking",
                           "backtester": "backtester", "weekly_reporter": "reporter"}[name], module)

    @property
    def live(self) -> bool:
        """The engine is 'live' when the finder and the staking engine both loaded."""
        return self.finder is not None and self.staking is not None


BACKEND = _Backend()


# ===========================================================================
# 2. Page setup and styling
# ===========================================================================

st.set_page_config(page_title=APP_NAME, page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# Custom CSS: a restrained analytical look on top of the dark theme in
# .streamlit/config.toml. Fonts come from Google Fonts with system fallbacks.
st.markdown(
    """
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Source+Sans+3:wght@400;600;700&family=IBM+Plex+Mono:wght@500&display=swap">
<style>
:root { --eb-accent:#2fd1bc; --eb-accent-soft:rgba(47,209,188,.14); --eb-line:#2b333b; --eb-muted:#8a94a0;
        --eb-good:#3ddc5a; --eb-bad:#ff6b6b; --eb-warn:#ffc14d; }
html, body, [class*="css"] { font-family: "Source Sans 3", "Segoe UI", Roboto, Arial, sans-serif; }
h1, h2, h3 { font-family: "Barlow Condensed", "Arial Narrow", sans-serif !important; letter-spacing:.02em; text-transform:uppercase; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; }
/* Brand header in the sidebar */
.eb-brand { font-family:"Barlow Condensed", sans-serif; font-weight:700; font-size:34px; line-height:1; color:var(--eb-accent); letter-spacing:.02em; }
.eb-tag { font-size:13px; letter-spacing:.14em; text-transform:uppercase; color:var(--eb-muted); margin-top:4px; }
/* Status pills */
.eb-pill { display:inline-block; border-radius:999px; padding:3px 11px; font-size:12px; font-weight:600; letter-spacing:.04em; margin-right:6px; }
.eb-pill-live { background:var(--eb-accent-soft); color:var(--eb-accent); }
.eb-pill-demo { background:rgba(255,193,77,.16); color:var(--eb-warn); }
.eb-pill-real { background:rgba(61,220,90,.14); color:var(--eb-good); }
.eb-pill-sim  { background:rgba(255,193,77,.16); color:var(--eb-warn); }
/* KPI tiles: Streamlit metrics inside bordered containers */
div[data-testid="stMetric"] { background:transparent; }
div[data-testid="stMetricLabel"] p { font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--eb-muted); }
div[data-testid="stMetricValue"] { font-family:"Barlow Condensed", sans-serif; font-size:38px; font-weight:700; line-height:1.05; }
/* Section eyebrows */
.eb-section { font-family:"Barlow Condensed", sans-serif; font-weight:700; font-size:26px; text-transform:uppercase; letter-spacing:.04em; margin:26px 0 2px; }
.eb-sub { color:var(--eb-muted); font-size:14px; margin-bottom:10px; }
/* Parlay slip cards */
.eb-slip-head { display:flex; justify-content:space-between; align-items:center; gap:10px; }
.eb-slip-id { font-family:"Barlow Condensed", sans-serif; font-size:22px; font-weight:700; }
.eb-slip-id small { font-family:"IBM Plex Mono", monospace; font-size:12px; color:var(--eb-muted); margin-left:8px; font-weight:500; }
.eb-chip { font-size:13px; font-weight:700; border-radius:999px; padding:3px 10px; background:var(--eb-accent-soft); color:var(--eb-accent); white-space:nowrap; }
.eb-leg { border:1px solid var(--eb-line); border-radius:10px; padding:10px 12px; min-height:92px; }
.eb-leg .lbl { font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--eb-muted); }
.eb-leg .pick { font-size:18px; font-weight:700; margin:2px 0; }
.eb-leg .meta { font-size:13px; color:var(--eb-muted); }
.eb-leg .price { font-family:"IBM Plex Mono", monospace; color:#dfe5ea; }
.eb-stat .lbl { font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--eb-muted); }
.eb-stat .val { font-size:17px; font-weight:600; font-variant-numeric: tabular-nums; }
.eb-risk { background:var(--eb-accent-soft); border-radius:10px; padding:10px 14px; }
.eb-risk .lbl { font-size:12px; letter-spacing:.1em; text-transform:uppercase; color:var(--eb-accent); font-weight:700; }
.eb-risk .amt { font-family:"Barlow Condensed", sans-serif; font-size:34px; font-weight:700; color:var(--eb-accent); line-height:1; }
.eb-risk .sub { font-size:13px; color:var(--eb-muted); }
.eb-foot { color:var(--eb-muted); font-size:13px; border-left:3px solid var(--eb-warn); padding-left:12px; margin-top:28px; }
</style>
""",
    unsafe_allow_html=True,
)


def _wide_kwargs(fn: Any) -> Dict[str, Any]:
    """Full-width argument that works across Streamlit versions (``width`` vs ``use_container_width``)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


def money(x: float, signed: bool = False) -> str:
    sign = ("+" if x >= 0 else "-") if signed else ("-" if x < 0 else "")
    return f"{sign}${abs(x):,.2f}"


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def american(a: int) -> str:
    return f"{a:+d}"


# ===========================================================================
# 3. Demo data — used only when the backend is unavailable
# ===========================================================================

_DEMO_LEGS: List[Dict[str, Any]] = [
    {"matchup": "Tampa Bay Buccaneers @ Dallas Cowboys", "market": "moneyline", "selection": "Dallas Cowboys ML", "american_odds": -470, "p_true": 0.870, "implied_prob": 0.825, "fair_prob": 0.792},
    {"matchup": "Cincinnati Bengals @ Miami Dolphins", "market": "moneyline", "selection": "Cincinnati Bengals ML", "american_odds": -340, "p_true": 0.810, "implied_prob": 0.773, "fair_prob": 0.741},
    {"matchup": "San Francisco 49ers @ Seattle Seahawks", "market": "spread", "selection": "San Francisco 49ers +2.5", "american_odds": -110, "p_true": 0.614, "implied_prob": 0.524, "fair_prob": 0.500},
    {"matchup": "Denver Broncos @ Los Angeles Chargers", "market": "spread", "selection": "Los Angeles Chargers +3.5", "american_odds": -110, "p_true": 0.605, "implied_prob": 0.524, "fair_prob": 0.500},
    {"matchup": "Houston Texans @ Tennessee Titans", "market": "spread", "selection": "Tennessee Titans +7.5", "american_odds": -110, "p_true": 0.594, "implied_prob": 0.524, "fair_prob": 0.500},
    {"matchup": "Buffalo Bills @ Los Angeles Rams", "market": "total", "selection": "Under 54.5", "american_odds": -110, "p_true": 0.571, "implied_prob": 0.524, "fair_prob": 0.500},
    {"matchup": "Detroit Lions @ Arizona Cardinals", "market": "moneyline", "selection": "Detroit Lions ML", "american_odds": -250, "p_true": 0.736, "implied_prob": 0.714, "fair_prob": 0.690},
    {"matchup": "New York Giants @ Washington Commanders", "market": "spread", "selection": "Washington Commanders -3.5", "american_odds": -110, "p_true": 0.588, "implied_prob": 0.524, "fair_prob": 0.500},
]


def _demo_tickets(legs: Sequence[Dict[str, Any]], min_gap: float, min_prob: float) -> List[Dict[str, Any]]:
    """Pair the demo legs the way the finder would: different games, best edge first, two legs only."""
    ok = [l for l in legs if l["p_true"] - l["implied_prob"] >= min_gap and l["p_true"] >= min_prob]
    ok.sort(key=lambda l: -(l["p_true"] - l["implied_prob"]))
    out: List[Dict[str, Any]] = []
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            a, b = ok[i], ok[j]
            if a["matchup"] == b["matchup"]:
                continue
            da = 1 + a["american_odds"] / 100 if a["american_odds"] > 0 else 1 + 100 / abs(a["american_odds"])
            db = 1 + b["american_odds"] / 100 if b["american_odds"] > 0 else 1 + 100 / abs(b["american_odds"])
            p, d = a["p_true"] * b["p_true"], da * db
            out.append({"ticket_id": f"DEMO-2L-{len(out) + 1:02d}", "n_legs": 2, "legs": [dict(a), dict(b)], "p_true": p,
                        "decimal_odds": d, "american_odds": int(round((d - 1) * 100)) if d >= 2 else int(round(-100 / (d - 1))),
                        "implied_prob": 1 / d, "edge": p * d - 1, "same_game": False, "correlation": None,
                        "notes": "demo ticket (backend unavailable)"})
            if len(out) >= TOP_N_PER_GROUP:
                return out
    return out


def _demo_path(bankroll: float, weeks: int = SEASON_WEEKS, seed: int = 11) -> List[float]:
    """A plausible simulated bankroll curve: modest upward drift with parlay-sized swings."""
    rng = random.Random(seed)
    path, b = [bankroll], bankroll
    for _ in range(weeks):
        b = max(0.0, b * (1 + rng.gauss(0.018, 0.06)))
        path.append(round(b, 2))
    return path


# ===========================================================================
# 4. Backend calls — each wrapped so a failure degrades to demo data
# ===========================================================================


def build_finder_config(week: int, min_gap: float, min_prob: float, edge_basis: str, same_game: str, sim_only: bool) -> Any:
    """Translate sidebar settings into a ``parlay_finder.FinderConfig``.

    Starts from pipeline_config.json so anything not exposed in the UI (seed,
    caps, markets, API key) stays in sync with the command-line pipeline.
    """
    pf = BACKEND.finder
    base = pf.load_finder_config()
    data = {**asdict(base), "min_prob_gap": min_gap, "min_leg_prob": min_prob, "edge_basis": edge_basis,
            "same_game_policy": same_game, "auto_detect_lines": not sim_only, "top_n": TOP_N_PER_GROUP * 2}
    data["sim"] = base.sim
    return pf.FinderConfig.from_dict(data)


@st.cache_data(show_spinner=False, ttl=300)
def load_slate(week: int, min_gap: float, min_prob: float, edge_basis: str, same_game: str, sim_only: bool
               ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int], str, Optional[str]]:
    """Run the finder once per settings combination.

    Returns ``(candidate_legs, tickets, rejection_counts, data_source, error)``.
    Cached for five minutes keyed on every argument, so moving a slider only
    re-runs the engine when the inputs actually changed.
    """
    pf = BACKEND.finder
    try:
        cfg = build_finder_config(week, min_gap, min_prob, edge_basis, same_game, sim_only)
        sides = pf.collect_sides(week, cfg)
        legs = [s.to_leg_dict() for s in pf.select_candidate_legs(sides, cfg)]
        tickets = pf.build_parlays(sides, week, cfg)
        rejected = pf.explain_legs(sides, cfg)
        source = sides[0].source if sides else cfg.source
        return legs, tickets, rejected, source, None
    except Exception as exc:  # any finder failure -> caller shows demo data with the reason
        return [], [], {}, "error", f"{type(exc).__name__}: {exc}"


def build_staking_config(mode: str, flat_kind: str, flat_value: float, kelly_fraction: float) -> Any:
    """Sidebar staking choices -> ``staking_engine.StakingConfig`` (the exact object the pipeline uses)."""
    se = BACKEND.staking
    if mode.startswith("Flat"):
        if flat_kind.startswith("$"):
            return se.StakingConfig.flat_dollar(float(flat_value))
        return se.StakingConfig.flat_percent(float(flat_value) / 100.0)
    return se.StakingConfig.fractional_kelly(float(kelly_fraction))


def size_tickets(tickets: Sequence[Dict[str, Any]], wallet: float, staking_cfg: Any) -> List[Dict[str, Any]]:
    """Stake every ticket against the current wallet through staking_engine, then cap weekly exposure."""
    se = BACKEND.staking
    recs = [se.compute_stake(wallet, t["p_true"], t["decimal_odds"], staking_cfg, t["ticket_id"]) for t in tickets]
    recs = se.apply_portfolio_cap(recs, wallet, PORTFOLIO_CAP)
    return [r.to_dict() for r in recs]


def size_tickets_demo(tickets: Sequence[Dict[str, Any]], wallet: float, mode: str, flat_kind: str, flat_value: float,
                      kelly_fraction: float) -> List[Dict[str, Any]]:
    """Minimal stand-in for staking_engine when it is unavailable (same rules, no audit trail)."""
    out = []
    for t in tickets:
        edge, d = t["edge"], t["decimal_odds"]
        if edge <= 0 or wallet <= 0:
            frac = 0.0
        elif mode.startswith("Flat"):
            frac = (flat_value / wallet) if flat_kind.startswith("$") else flat_value / 100.0
        else:
            frac = edge / (d - 1) * kelly_fraction
        frac = min(frac, 0.05)
        stake = math.floor(wallet * frac * 100 + 1e-9) / 100
        out.append({"ticket_id": t["ticket_id"], "stake_dollars": stake, "capped_fraction": frac,
                    "potential_profit": stake * (d - 1), "expected_value": stake * edge, "cap_applied": frac >= 0.05,
                    "skipped": stake <= 0, "reason": "demo sizing"})
    total = sum(r["stake_dollars"] for r in out)
    if total > wallet * PORTFOLIO_CAP and total > 0:
        scale = wallet * PORTFOLIO_CAP / total
        for r, t in zip(out, tickets):
            r["stake_dollars"] = math.floor(r["stake_dollars"] * scale * 100) / 100
            r["potential_profit"] = r["stake_dollars"] * (t["decimal_odds"] - 1)
            r["expected_value"] = r["stake_dollars"] * t["edge"]
            r["cap_applied"] = True
    return out


@st.cache_data(show_spinner=False, ttl=600)
def backtest_path(season_year: int, seed: int, bankroll: float, min_gap: float, min_prob: float, edge_basis: str,
                  same_game: str, mode: str, flat_kind: str, flat_value: float, kelly_fraction: float
                  ) -> Tuple[List[float], Dict[str, Any], Optional[str]]:
    """Replay one simulated season through the live pipeline with the current settings.

    This is the same code path ``backtester.py`` uses: finder -> weekly report
    -> staking engine -> graded final scores. Returns the week-by-week bankroll.
    """
    pf, bt = BACKEND.finder, BACKEND.backtester
    try:
        base = pf.load_finder_config()
        data = {**asdict(base), "source": "sim", "auto_detect_lines": False, "season": season_year, "seed": seed,
                "min_prob_gap": min_gap, "min_leg_prob": min_prob, "edge_basis": edge_basis, "same_game_policy": same_game}
        data["sim"] = base.sim
        fcfg = pf.FinderConfig.from_dict(data)
        policy = build_staking_config(mode, flat_kind, flat_value, kelly_fraction)
        week_data = bt.build_simulated_season(season_year, seed, SEASON_WEEKS, fcfg)
        btcfg = bt.BacktestConfig(seasons=1, weeks=SEASON_WEEKS, bankroll=bankroll, seed=seed, season_year=season_year,
                                  policies={"selected": policy}, portfolio_cap=PORTFOLIO_CAP, top_n=TOP_N_PER_GROUP,
                                  with_null=False, finder=fcfg)
        res = bt.run_policy_over_season(week_data, "selected", policy, btcfg, 0, seed, "model")
        return [bankroll] + list(res.weekly_bankroll), dict(res.summary), None
    except Exception as exc:
        return [], {}, f"{type(exc).__name__}: {exc}"


# ===========================================================================
# 5. Session-state wallet
# ===========================================================================


def init_wallet(starting: float) -> None:
    """Seed the wallet from the sidebar bankroll; reset it when that input changes."""
    if st.session_state.get("wallet_base") != starting:
        st.session_state["wallet_base"] = starting
        st.session_state["wallet"] = float(starting)
        st.session_state["placed"] = {}
        st.session_state["ledger"] = []


def place_wager(ticket_id: str, stake: float, label: str, odds: int) -> None:
    """on_click callback: deduct the stake from the wallet before the page redraws."""
    if ticket_id in st.session_state["placed"] or stake <= 0:
        return
    st.session_state["wallet"] = max(0.0, round(st.session_state["wallet"] - stake, 2))
    st.session_state["placed"][ticket_id] = stake
    st.session_state["ledger"].append({"time": _dt.datetime.now().strftime("%H:%M:%S"), "ticket": ticket_id,
                                       "legs": label, "odds": american(odds), "stake": stake,
                                       "wallet_after": st.session_state["wallet"]})


def reset_wallet() -> None:
    st.session_state["wallet"] = float(st.session_state["wallet_base"])
    st.session_state["placed"] = {}
    st.session_state["ledger"] = []


# ===========================================================================
# 6. Sidebar — controls and settings
# ===========================================================================

today = _dt.date.today()
if BACKEND.reporter is not None:
    try:
        default_week = int(BACKEND.reporter.estimate_nfl_week(today))
        season_year = int(BACKEND.reporter.nfl_season_year(today))
    except Exception:
        default_week, season_year = 6, today.year
else:
    default_week, season_year = 6, today.year

with st.sidebar:
    st.markdown(f'<div class="eb-brand">{APP_NAME}</div><div class="eb-tag">{APP_TAGLINE}</div>', unsafe_allow_html=True)
    st.markdown("---")

    league = st.selectbox("League", ["NFL", "NBA (coming soon)", "NHL (coming soon)"], index=0,
                          help="NBA and NHL models are still in training. Only NFL is live.")
    if not league.startswith("NFL"):
        st.caption("That model is still training. Showing NFL.")
    week = int(st.number_input("NFL week", min_value=1, max_value=SEASON_WEEKS, value=default_week, step=1))

    starting_bankroll = float(st.number_input("Starting bankroll ($)", min_value=50.0, max_value=1_000_000.0,
                                              value=1000.0, step=50.0, format="%.0f"))
    init_wallet(starting_bankroll)

    mode = st.selectbox("Staking strategy mode", ["Flat Unit", "Fractional Kelly Criterion"], index=1)
    if mode.startswith("Flat"):
        flat_kind = st.radio("Flat unit type", ["$ per ticket", "% of bankroll"], horizontal=True)
        flat_value = float(st.number_input("Flat unit", min_value=1.0, value=10.0 if flat_kind.startswith("$") else 1.0,
                                           step=1.0 if flat_kind.startswith("$") else 0.25))
        kelly_fraction = 0.25
    else:
        flat_kind, flat_value = "$ per ticket", 10.0
        kelly_fraction = float(st.select_slider("Kelly fraction", options=[0.10, 0.25, 0.50], value=0.25,
                                                help="Fraction of full Kelly. 0.25 is the pipeline default."))

    min_edge_pct = float(st.slider("Minimum edge threshold", min_value=0.0, max_value=15.0, value=6.0, step=0.5,
                                   format="%.1f%%", help="P_true − P_implied must be at least this many points."))
    with st.expander("Advanced filters"):
        min_prob_pct = float(st.slider("Minimum leg win probability", min_value=50, max_value=80, value=68, step=1,
                                       format="%d%%", help="Each leg's model probability must be at least this high."))
        edge_basis_label = st.radio("Measure the edge against", ["Book price (vig included)", "De-vigged fair price"], index=0)
        same_game = st.selectbox("Same-game legs", ["positive_only", "never", "any"], index=0,
                                 help="positive_only accepts only positively correlated pairs, e.g. Home ML + Home team total Over.")
        sim_only = st.toggle("Simulated league only (ignore lines.csv)", value=False)
    edge_basis = "fair" if edge_basis_label.startswith("De-vigged") else "implied"

    st.markdown("---")
    st.button("Reset wallet", on_click=reset_wallet, help="Return the wallet to the starting bankroll and clear placed wagers.")

    engine_pill = ('<span class="eb-pill eb-pill-live">Engine: live</span>' if BACKEND.live
                   else '<span class="eb-pill eb-pill-demo">Engine: demo data</span>')
    st.markdown(engine_pill, unsafe_allow_html=True)
    if BACKEND.errors:
        with st.expander("Backend status"):
            for name, err in BACKEND.errors.items():
                st.caption(f"{name}: {err}")

# ===========================================================================
# 7. Run the engine for the current settings
# ===========================================================================

min_gap, min_prob = min_edge_pct / 100.0, min_prob_pct / 100.0
wallet = float(st.session_state["wallet"])
engine_error: Optional[str] = None

if BACKEND.live:
    legs, tickets, rejected, data_source, engine_error = load_slate(week, min_gap, min_prob, edge_basis, same_game, sim_only)
    if engine_error:
        legs, tickets, rejected, data_source = list(_DEMO_LEGS), _demo_tickets(_DEMO_LEGS, min_gap, min_prob), {}, "demo"
else:
    legs, tickets, rejected, data_source = list(_DEMO_LEGS), _demo_tickets(_DEMO_LEGS, min_gap, min_prob), {}, "demo"
    legs = [l for l in legs if l["p_true"] - l["implied_prob"] >= min_gap and l["p_true"] >= min_prob]

using_demo = (not BACKEND.live) or engine_error is not None
if BACKEND.live and not engine_error:
    stakes = size_tickets(tickets, wallet, build_staking_config(mode, flat_kind, flat_value, kelly_fraction))
else:
    stakes = size_tickets_demo(tickets, wallet, mode, flat_kind, flat_value, kelly_fraction)
stake_by_id = {s["ticket_id"]: s for s in stakes}

# ===========================================================================
# 8. Header and KPI row
# ===========================================================================

source_pill = {
    "csv": '<span class="eb-pill eb-pill-real">Real lines (lines.csv)</span>',
    "sim": '<span class="eb-pill eb-pill-sim">Simulated league, not real games</span>',
    "demo": '<span class="eb-pill eb-pill-demo">Demo data</span>',
}.get("csv" if str(data_source).startswith("api") else str(data_source), '<span class="eb-pill eb-pill-sim">Unknown source</span>')
if str(data_source).startswith("api"):
    source_pill = '<span class="eb-pill eb-pill-real">Live lines (The Odds API)</span>'

head_l, head_r = st.columns([3, 2])
with head_l:
    st.markdown(f"# {APP_NAME}")
    st.markdown(f'<div class="eb-sub">{APP_TAGLINE} · 2-leg parlay engine with premium edge filters</div>', unsafe_allow_html=True)
with head_r:
    st.markdown(f'<div style="text-align:right;padding-top:14px">{engine_pill} {source_pill}</div>', unsafe_allow_html=True)
if engine_error:
    st.warning(f"The analytics engine raised an error, so demo data is shown: {engine_error}")

high_win_mode = min_prob >= 0.65 and min_gap >= 0.05
k1, k2, k3 = st.columns(3)
n_placed = len(st.session_state["placed"])
with k1, st.container(border=True):
    delta = wallet - starting_bankroll
    st.metric("Current bankroll", money(wallet), delta=(money(delta, signed=True) if abs(delta) > 0.004 else None),
              help="Starting bankroll minus every wager simulated in this session.")
    st.caption(f"Started at {money(starting_bankroll)} · {n_placed} wager{'s' if n_placed != 1 else ''} placed this session")
with k2, st.container(border=True):
    st.metric("Active slate", f"Week {week}", help="The week the engine is building tickets for.")
    st.caption(f"NFL {season_year} Season · {'real lines' if str(data_source) in ('csv',) or str(data_source).startswith('api') else 'simulated league' if data_source == 'sim' else 'demo data'}")
with k3, st.container(border=True):
    st.metric("Filter status", "High Win-Rate" if high_win_mode else "Standard Edge",
              help="High Win-Rate Mode: leg floor of 65% or more and an edge threshold of 5 points or more.")
    st.caption(f"{'Mode active' if high_win_mode else 'Mode'} · legs ≥ {min_prob_pct:.0f}% · edge ≥ {min_edge_pct:.1f} pts · 2 legs max")

# ===========================================================================
# 9. Active Edge Signal Board
# ===========================================================================

st.markdown('<div class="eb-section">Active Edge Signal Board</div>', unsafe_allow_html=True)
st.markdown('<div class="eb-sub">Individual +EV lines that clear the filters, before they are paired into parlays.</div>',
            unsafe_allow_html=True)

if legs:
    board = pd.DataFrame([{
        "Matchup": l["matchup"],
        "Market": {"spread": "Spread", "total": "Total", "moneyline": "Moneyline", "team_total": "Team total"}.get(l.get("market", ""), str(l.get("market", "")).title()),
        "Selected line": l["selection"],
        "Price": american(int(l["american_odds"])),
        "Model prob": float(l["p_true"]),
        "Book implied": float(l.get("implied_prob") if l.get("implied_prob") is not None else 1.0 / (1 + (l["american_odds"] / 100 if l["american_odds"] > 0 else 100 / abs(l["american_odds"])))),
    } for l in legs])
    board["Edge"] = board["Model prob"] - board["Book implied"]
    board = board.sort_values("Edge", ascending=False).reset_index(drop=True)

    high_edge = min_gap + HIGH_EDGE_MARGIN  # "high" is relative to the threshold the user chose

    def _tint(row: pd.Series) -> List[str]:
        # Subtle highlight for the strongest signals; everything else stays on the surface colour.
        return (["background-color: rgba(47,209,188,0.14)"] * len(row)) if row["Edge"] >= high_edge else [""] * len(row)

    styled = (board.style.apply(_tint, axis=1)
              .format({"Model prob": "{:.1%}", "Book implied": "{:.1%}", "Edge": "{:+.1%}"}))
    st.dataframe(styled, hide_index=True, **_wide_kwargs(st.dataframe))
    st.caption(f"{len(board)} qualifying line(s). Rows tinted at an edge of {high_edge * 100:.1f} points or more "
               f"(your threshold plus {HIGH_EDGE_MARGIN * 100:.1f}). Edge here is P_true − P_implied, the gap the premium filter tests.")
else:
    st.info("No line clears the current filters. Lower the minimum edge or the leg probability floor in the sidebar to see signals.")
if rejected:
    with st.expander("Why sides were rejected"):
        rej = pd.DataFrame(sorted(rejected.items(), key=lambda kv: -kv[1]), columns=["Reason", "Sides"])
        st.dataframe(rej, hide_index=True, **_wide_kwargs(st.dataframe))

# ===========================================================================
# 10. Optimal Parlay Engine — 2-leg slips with live staking
# ===========================================================================

st.markdown('<div class="eb-section">Optimal Parlay Engine</div>', unsafe_allow_html=True)
st.markdown(f'<div class="eb-sub">Premium 2-leg slips sized by staking_engine against the current wallet · '
            f'max 5% per ticket · max {PORTFOLIO_CAP:.0%} per week.</div>', unsafe_allow_html=True)

if not tickets:
    st.info("No parlay cleared this week's filters, so no bets are recommended. That is the engine saying no, which is the "
            "designed behaviour of a high win-rate filter. Loosen the thresholds in the sidebar to explore the slate.")
else:
    ranked = sorted(tickets, key=lambda t: -stake_by_id.get(t["ticket_id"], {}).get("expected_value", 0.0))
    cols = st.columns(2)
    for idx, t in enumerate(ranked[:TOP_N_PER_GROUP * 2]):
        s = stake_by_id.get(t["ticket_id"], {"stake_dollars": 0.0, "potential_profit": 0.0, "expected_value": 0.0,
                                              "capped_fraction": 0.0, "cap_applied": False, "reason": ""})
        placed_stake = st.session_state["placed"].get(t["ticket_id"])
        stake = float(placed_stake if placed_stake is not None else s["stake_dollars"])
        with cols[idx % 2], st.container(border=True):
            tag = f'<span class="eb-chip">Edge {t["edge"] * 100:+.1f}%</span>'
            if t.get("same_game"):
                tag += f' <span class="eb-chip">same game · {t.get("correlation")}</span>'
            st.markdown(f'<div class="eb-slip-head"><div class="eb-slip-id">#{idx + 1}<small>{t["ticket_id"]}</small></div>'
                        f'<div>{tag}</div></div>', unsafe_allow_html=True)
            l1, l2 = st.columns(2)
            for col, leg, label in ((l1, t["legs"][0], "Leg 1"), (l2, t["legs"][1], "Leg 2")):
                with col:
                    st.markdown(
                        f'<div class="eb-leg"><div class="lbl">{label}</div><div class="pick">{leg["selection"]}</div>'
                        f'<div class="meta">{leg["matchup"]}</div>'
                        f'<div class="meta"><span class="price">{american(int(leg["american_odds"]))}</span> · model {pct(float(leg["p_true"]), 0)}</div></div>',
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="eb-stat"><div class="lbl">True win rate</div><div class="val">{pct(t["p_true"], 2)}</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="eb-stat"><div class="lbl">Book odds</div><div class="val">{american(int(t["american_odds"]))}</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="eb-stat"><div class="lbl">Book implied</div><div class="val">{pct(float(t.get("implied_prob", 1 / t["decimal_odds"])), 2)}</div></div>', unsafe_allow_html=True)
            c4, c5, c6 = st.columns(3)
            c4.markdown(f'<div class="eb-stat"><div class="lbl">To win</div><div class="val">{money(stake * (t["decimal_odds"] - 1))}</div></div>', unsafe_allow_html=True)
            c5.markdown(f'<div class="eb-stat"><div class="lbl">Expected value</div><div class="val">{money(stake * t["edge"], signed=True)}</div></div>', unsafe_allow_html=True)
            c6.markdown(f'<div class="eb-stat"><div class="lbl">Share of wallet</div><div class="val">{pct(stake / wallet if wallet > 0 else 0.0, 2)}'
                        f'{" · capped" if s.get("cap_applied") else ""}</div></div>', unsafe_allow_html=True)
            r1, r2 = st.columns([3, 2])
            with r1:
                st.markdown(f'<div class="eb-risk"><div class="lbl">Risk</div><div class="amt">{money(stake)}</div>'
                            f'<div class="sub">{"placed in this session" if placed_stake is not None else s.get("reason", "")}</div></div>',
                            unsafe_allow_html=True)
            with r2:
                st.write("")
                label = " + ".join(leg["selection"] for leg in t["legs"])
                st.button("Placed ✓" if placed_stake is not None else "Simulate / Place wager", key=f"place_{t['ticket_id']}",
                          on_click=place_wager, args=(t["ticket_id"], stake, label, int(t["american_odds"])),
                          disabled=(placed_stake is not None or stake <= 0), **_wide_kwargs(st.button))

    if st.session_state["ledger"]:
        with st.expander(f"Wager ledger ({len(st.session_state['ledger'])} placed, "
                         f"{money(sum(w['stake'] for w in st.session_state['ledger']))} at risk)"):
            st.dataframe(pd.DataFrame(st.session_state["ledger"]), hide_index=True, **_wide_kwargs(st.dataframe))

# ===========================================================================
# 11. Bankroll growth chart
# ===========================================================================

st.markdown('<div class="eb-section">Bankroll Growth</div>', unsafe_allow_html=True)
chart_caption = ""
path: List[float] = []
summary: Dict[str, Any] = {}
if BACKEND.live and BACKEND.backtester is not None:
    path, summary, chart_err = backtest_path(season_year, 7, starting_bankroll, min_gap, min_prob, edge_basis, same_game,
                                             mode, flat_kind, flat_value, kelly_fraction)
    if chart_err:
        path, summary = _demo_path(starting_bankroll), {}
        chart_caption = f"Backtester unavailable ({chart_err}); showing demo growth data."
    else:
        chart_caption = (f"One simulated {season_year} season through the live pipeline with your settings: "
                         f"{summary.get('tickets_placed', 0)} bets, win rate {summary.get('win_rate', 0.0):.1%}, "
                         f"ROI {summary.get('roi_on_turnover', 0.0):+.1%}, max drawdown {summary.get('max_drawdown', 0.0):.1%}.")
else:
    path, chart_caption = _demo_path(starting_bankroll), "Demo growth data (backend unavailable)."

frame = pd.DataFrame({"Week": list(range(len(path))), "Your settings": path}).set_index("Week")
if BACKEND.live and BACKEND.backtester is not None and summary.get("tickets_placed", 0) == 0:
    # A strict filter can place no bets all season, which draws a flat line. Add a labelled
    # reference run (60% leg floor, same edge threshold) so the chart still shows what the
    # engine does when it is active. The caption says which is which.
    ref_path, ref_summary, ref_err = backtest_path(season_year, 7, starting_bankroll, min_gap, 0.60, edge_basis, same_game,
                                                   mode, flat_kind, flat_value, kelly_fraction)
    if not ref_err and len(ref_path) == len(path):
        frame["Reference (60% leg floor)"] = ref_path
        chart_caption += (f" Your filters placed no bets this simulated season (flat line). The reference run uses a 60% "
                          f"leg floor: {ref_summary.get('tickets_placed', 0)} bets, win rate {ref_summary.get('win_rate', 0.0):.1%}, "
                          f"ROI {ref_summary.get('roi_on_turnover', 0.0):+.1%}.")
if alt is not None:
    # Long format for Altair; the y-axis floats around the data instead of starting at $0,
    # so a season's swings are visible even when they are a few percent of the bankroll.
    long = frame.reset_index().melt(id_vars="Week", var_name="Series", value_name="Bankroll")
    series_order = list(frame.columns)
    base = alt.Chart(long).encode(
        x=alt.X("Week:Q", axis=alt.Axis(title="Week", tickMinStep=1, grid=False, labelColor="#8a94a0", titleColor="#8a94a0")),
        y=alt.Y("Bankroll:Q", scale=alt.Scale(zero=False, padding=12), axis=alt.Axis(title="Bankroll ($)", format="$,.0f",
                gridColor="#232a31", labelColor="#8a94a0", titleColor="#8a94a0")),
        color=alt.Color("Series:N", scale=alt.Scale(domain=series_order, range=["#2fd1bc", "#8a94a0"][: len(series_order)]),
                        legend=alt.Legend(title=None, orient="top-left", labelColor="#dfe5ea")),
        strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(domain=series_order, range=[[1, 0], [6, 4]][: len(series_order)]), legend=None),
    )
    lines = base.mark_line(strokeWidth=2.2, interpolate="monotone")
    points = base.mark_circle(size=42, opacity=0).encode(
        tooltip=[alt.Tooltip("Week:Q"), alt.Tooltip("Series:N"), alt.Tooltip("Bankroll:Q", format="$,.2f")])
    end_points = base.mark_circle(size=70).transform_filter(alt.datum.Week == int(frame.index.max()))
    chart = (lines + points + end_points).properties(height=300, background="transparent").configure_view(strokeWidth=0)
    st.altair_chart(chart, **_wide_kwargs(st.altair_chart))
else:
    st.line_chart(frame, color=["#2fd1bc", "#8a94a0"][: len(frame.columns)], height=300)
st.caption(chart_caption)

# ===========================================================================
# 12. Footer
# ===========================================================================

st.markdown(
    '<div class="eb-foot">EdgeBook AI is a research tool. A positive edge is a model\'s opinion, not a guarantee; the '
    '"Simulate / Place wager" button only moves money inside this session. Lines move until kickoff, so confirm every price '
    'at your sportsbook. Bet only what you can afford to lose and only where it is legal and you are of age. '
    'In the US, help is available any time at 1-800-GAMBLER.</div>',
    unsafe_allow_html=True,
)
