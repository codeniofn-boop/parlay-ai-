# parlay-ai — NFL Parlay Analytics Pipeline

A complete, standard-library-only Python pipeline for finding, staking, logging and
backtesting +EV NFL parlays. Every module ships with a built-in self-test.

| File | Role |
| --- | --- |
| `parlay_finder.py` | Builds each week's ranked 2-leg and 3-leg parlay candidates from simulated, CSV or live lines. |
| `staking_engine.py` | Dynamic bankroll and staking strategy: Flat units or Fractional Kelly with hard safety caps. |
| `weekly_reporter.py` | Automated pick logger: stakes every ticket and writes `weekly_parlay_report.txt`. |
| `backtester.py` | Replays whole seasons through the same code path and compares staking policies. |
| `run_weekly_report.py` | Zero-argument launcher for the weekly report (press Run, no Terminal). |
| `pipeline_config.json` | The one settings file: bankroll, staking mode, finder and backtester options. |
| `week_inputs.csv` + `build_lines.py` | This week's games, odds and model numbers (15 rows) and the expander that turns them into `lines.csv`. |
| `app.py` + `launch_edgebook.py` | **EdgeBook AI**, the Streamlit dashboard: sidebar controls, KPI tiles, the edge signal board, the 2-leg parlay slip engine with a simulated wallet, and the bankroll growth chart. |
| `report_template.html` + `publish_report.py` | The shareable web page: every weekly run also writes `docs/index.html`, a static page with a live bankroll box and Flat/Kelly switch. |

```
parlay_finder.py ──► weekly_reporter.py ──► staking_engine.py ──► weekly_parlay_report.txt
        │                                                              ▲
        └──────────────── backtester.py (season replay) ───────────────┘
```

---

## Quick start without Terminal

1. Put all six files above in one folder.
2. Open `pipeline_config.json` in any text editor. Set `bankroll` and `mode` (`"kelly"` or `"flat"`).
   Leave `week` as `null` to auto-detect the upcoming week from today's date.
3. Open `run_weekly_report.py` in IDLE, VS Code or PyCharm and press **Run**, or double-click it if
   `.py` files open with Python Launcher. The report prints on screen, is saved as
   `weekly_parlay_report.txt`, and opens in TextEdit.
4. To see how the strategy would have performed over many seasons, run `backtester.py` the same way.
   It writes `backtest_output/backtest_summary.txt` and opens it.

The report's **Pick Source** line says where the picks came from. Out of the box it is the simulated
league inside `parlay_finder.py`, clearly labelled "NOT real games". To use real games, drop a
`lines.csv` into the folder; see **Using real lines**. For live lines without typing, sign up for a
free key at https://the-odds-api.com, paste it as `"api_key"` under `finder` in
`pipeline_config.json`, and set `"source": "api"`.

## The dashboard (EdgeBook AI)

`app.py` is a Streamlit front end over the same modules the command line uses. Nothing is
re-implemented: the sidebar's edge threshold becomes `FinderConfig.min_prob_gap`, the staking mode
becomes a `StakingConfig`, every ticket is sized by `staking_engine.compute_stake` against the current
wallet, and the growth chart replays one simulated season through `backtester`. If a module is
missing or raises, that section switches to clearly labelled demo data and the sidebar's
"Backend status" panel shows why.

Run it with `streamlit run app.py`, or double-click `launch_edgebook.py` (it installs Streamlit on
first use and opens the browser). The dark theme lives in `.streamlit/config.toml`. Requirements:
`pip install -r requirements.txt` (Streamlit and pandas only; the pipeline itself needs nothing).

The "Simulate / Place wager" buttons move money inside the session only. "Reset wallet" restores the
starting bankroll.

## The web page (share it)

Every weekly run also writes **`docs/index.html`**: a self-contained page showing the same tickets as
the text report, with a bankroll box and a Flat / Kelly switch that resize every stake in the browser
using the same formulas and caps as `staking_engine.py` (the two are checked against each other to the
cent). It needs no server. Open it by double-clicking, or publish it:

* **GitHub Pages:** in the repository go to Settings, then Pages, choose "Deploy from a branch", pick
  this branch and the `/docs` folder, and save. The page appears at
  `https://<your-user>.github.io/parlay-ai-/` and updates whenever you push a new `docs/index.html`.
* **Any static host:** upload `docs/index.html` on its own; it has no other files.

Rebuild it by hand with `python3 publish_report.py` (reads `weekly_parlay_report.json`). The page
labels simulated slates plainly and carries a research-only disclaimer and problem-gambling helpline.

## Quick start with Terminal

```bash
python3 staking_engine.py --selftest && python3 weekly_reporter.py --selftest
python3 parlay_finder.py --selftest && python3 backtester.py --selftest

python3 weekly_reporter.py --week 6 --bankroll 1000 --mode kelly --kelly-multiplier 0.25
python3 backtester.py --seasons 50 --with-null
```

---

## 1. `parlay_finder.py`

Produces the week's candidate tickets as plain dicts in the reporter's contract.

**Sources** (`finder.source` in `pipeline_config.json`, or `--source`):

| Source | What it does |
| --- | --- |
| `sim` (default) | Deterministic simulated league. 32 real franchises get hidden power ratings; each week the "market" posts spreads, totals and moneylines with vig, and the "model" forms its own view. |
| `csv` | Reads real lines from `lines.csv` (format below). |
| `api` | Pulls live NFL lines from The Odds API v4. Needs `ODDS_API_KEY` in the environment or `finder.api_key` in the config. |

**How the simulation models skill.** The model starts from the market number and corrects a fraction
`model_skill` of the market's error, plus its own noise. `model_skill = 0` is a true null (the model is
the market plus noise and its picks carry no real edge), which is what `backtester.py --no-edge` runs.
Measured over many seasons, the default settings give selected legs a real edge of about +3.5% while
the model *claims* about +6.9%, a deliberate, realistic winner's-curse gap.

**Selection rules (the three absolute rules).**

1. **Strict 2-leg limit.** `MAX_LEGS = 2`. A configuration asking for 3+ legs is clamped back to 2 with a
   warning; extra legs multiply variance and degrade the stability of the raw win rate.
2. **Premium edge filter.** A leg qualifies only when `P_true - P_implied >= 0.06` (`min_prob_gap`) **and**
   `P_true >= 0.68` (`min_leg_prob`). `P_implied` is the book's vig-inclusive `1 / decimal odds`
   (`edge_basis: "implied"`); set `"fair"` to measure the gap against the de-vigged market probability.
3. **Anti-correlation check.** Legs from the same game are allowed only when positively correlated
   (`same_game_policy: "positive_only"`): Home ML + Home team total Over, favourite spread + favourite
   ML, game Over + team Over, and so on. Negative traps (Home ML + Away team total Over, spread + Under)
   and contradictory pairs (both sides of a market) are rejected. `leg_correlation()` holds the matrix.

Legs are also capped at `max_leg_odds` (+300), ranked by `rank_by` (`growth`: `edge² / (D - 1)`, the Kelly
log-growth proxy; or raw `edge`), and diversified (`max_tickets_per_leg`, `max_tickets_per_game`).
These filters are strict by design: on a typical slate few or no legs qualify, and the report then says
"no bets are recommended" rather than inventing tickets. `python3 parlay_finder.py --explain` shows why
each side was rejected. Markets: spread, total, moneyline and team total (`Buffalo Bills Over 24.5`).

```python
import parlay_finder
tickets = parlay_finder.find_parlays(week=6)            # list[dict], reporter-ready
parlay_finder.write_parlays_json(tickets, "parlays_week_06.json", 6)
```

### This week's real lines are included

`week_inputs.csv` holds the current NFL week, one row per game: the spread, total and moneylines, plus
the model's projected margin and win probability. The pipeline expands it into `lines.csv`
automatically (six rows per game) every time it runs, so `week_inputs.csv` is the only file to
maintain. Each row's `source` column records where its numbers came from and when.

**Provenance of the shipped Week 5 file (2026-10-07):** lines are the VegasInsider consensus board,
with DraftKings prices where a search surfaced them; spread and total prices not shown by the source
are assumed to be -110. Model numbers are ESPN's public Football Power Index (FPI) projected margin
and win probability for each game. Spread probabilities are derived from the FPI margin with a 13.5
point standard deviation; totals carry no model number and so get no edge. One game (Ravens at
Falcons) has no model numbers because the FPI projection predates the Lamar Jackson injury news
that moved the line nine points. Lines move until kickoff: confirm at your sportsbook.

**To update for a new week:** open `week_inputs.csv` in Numbers or Excel, replace the 15 rows, save
as CSV, and run the report. `python3 build_lines.py --selftest` checks the expander.

### Using real lines (drop-in, no settings edits)

Put a file named **`lines.csv`** in the main folder and the finder uses it automatically for any
week it contains; weeks it does not contain fall back to the simulated league, and the report's
**Pick Source** line always says which happened. Add **`results.csv`** (final scores) next to it and
`backtester.py` automatically replays that real season instead of the simulation. A **`model_probs.csv`**
in the folder is merged automatically too. Copy `lines_template.csv` and `results_template.csv` to
start, or look at the fuller examples in `examples/`.

`lines.csv` (see `examples/lines.csv`):

```
week,away,home,market,selection,american_odds,model_prob
6,Kansas City Chiefs,Buffalo Bills,spread,Buffalo Bills -3.5,-110,0.56
6,Kansas City Chiefs,Buffalo Bills,spread,Kansas City Chiefs +3.5,-110,0.44
6,Kansas City Chiefs,Buffalo Bills,total,Over 44.5,-110,0.52
6,Kansas City Chiefs,Buffalo Bills,moneyline,Buffalo Bills ML,-165,0.63
```

`market` is `spread`, `total` or `moneyline`. `selection` is the line text exactly as you would say it.
Both sides of a market should be present so the vig can be removed. `model_prob` is *your* model's
win probability for that side (0–1 or 0–100); leave it blank and the de-vigged market probability is
used, which correctly produces no edge. For live API lines, supply your probabilities in
`model_probs.csv` (`matchup,selection,model_prob`, see `examples/model_probs.csv`) and point
`finder.model_csv` at it.

---

## 2. `staking_engine.py`

```
Edge        = (P_true * Decimal_Odds) - 1
Full Kelly  = Edge / (Decimal_Odds - 1)
Frac. Kelly = Full Kelly * Kelly_Multiplier          # 0.10 – 0.25 recommended
Stake ($)   = Bankroll * min(Fraction, max_stake_pct)  # rounded DOWN to the cent
```

| Mode | `StakingMode` | Behaviour |
| --- | --- | --- |
| **A – Flat** | `FLAT` | Risk `flat_pct` of bankroll (default 1%) or a fixed `flat_dollars` unit, regardless of edge. |
| **B – Fractional Kelly** | `KELLY` | Fraction scales with edge and odds, damped by `kelly_multiplier`. |

Safety boundaries, applied to both modes: `Edge <= 0` (or below `min_edge`) returns **$0.00**; a hard
per-ticket cap of `max_stake_pct` (5%); optional book minimum; round-down to the cent; invalid input
raises `StakingInputError` (or returns a $0 result with `.error` set from `safe_compute_stake`);
`apply_portfolio_cap` keeps total weekly exposure under 15%.

```python
from staking_engine import StakingConfig, compute_stake, BankrollTracker
cfg = StakingConfig.fractional_kelly(0.25)
rec = compute_stake(bankroll=1_000, p_true=0.31, decimal_odds=3.64, config=cfg)
rec.edge, rec.stake_dollars, rec.reason      # 0.1284, 12.15, "0.25x Kelly (full Kelly 4.86%)"
```

`BankrollTracker` is the ledger the backtester steps week by week: `stake()`, `settle(rec, won)`,
`record(rec, pnl, won)` for pushes and voided legs, and `summary()`.

---

## 3. `weekly_reporter.py`

Collects tickets from `parlay_finder.find_parlays` (falling back to `--input parlays.json`, then to a
labelled demo slate so a scheduled run never produces nothing), normalises many upstream shapes,
stakes and ranks them, and writes an 80-column ASCII report card plus a JSON side-car.

The report shows the simulated NFL week and date, bankroll and staking mode, safety caps and pick
source, then **TOP RECOMMENDED PARLAYS** in 2-leg and 3-leg sections. Each ticket lists every matchup,
line and leg price, the **True Win Rate %**, **Book Implied %**, **American Payout Odds**, **Edge %**,
**To Win**, **Expected Value** and the **RISK: $** amount. Tickets are ordered by the expected dollar
profit of the recommended stake. A **PASSED (NO BET)** appendix explains every $0 decision.
See `examples/weekly_parlay_report.txt`.

```bash
python3 weekly_reporter.py --week 6 --bankroll 500 --mode flat --flat-dollars 10
python3 weekly_reporter.py --week 6 --source json --input examples/parlays_week_06.json
```

---

## 4. `backtester.py`

Replays seasons through the *same* code the weekly tools use: the finder builds tickets from lines
and model numbers only, the reporter stakes them against the current bankroll, final scores are
drawn from the hidden truth (or read from `results.csv`), legs are graded with standard house rules
(pushes void the leg and the payout is recomputed), and P&L is booked through `BankrollTracker`.

Several staking policies run over identical slates and identical scores. Repeating over many
seasons gives a distribution: median growth, 5th/95th percentile bankroll, probability of profit,
drawdowns, probability of ruin, and **calibration** (what the model claimed vs. what actually hit).
`--with-null` (on by default in the config) adds a no-information model so you can see what a
strategy earns on luck alone.

```bash
python3 backtester.py                                    # defaults from pipeline_config.json
python3 backtester.py --seasons 100 --policy kelly:0.5 --policy flat:0.02
python3 backtester.py --lines lines.csv --results results.csv   # real season
```

`results.csv`: `week,away,home,away_score,home_score` (see `examples/results.csv`).
Outputs land in `backtest_output/`: `backtest_summary.txt`, `backtest_ledger.csv` (one row per ticket
per season per policy) and `backtest_summary.json`. See `examples/backtest_summary.txt`.

---

## Configuration (`pipeline_config.json`)

| Section | Keys |
| --- | --- |
| top level | `week`, `bankroll`, `mode`, `kelly_multiplier`, `flat_pct`, `flat_dollars`, `max_stake_pct`, `min_edge`, `portfolio_cap`, `source`, `input`, `top_n`, `output`, `json_out`, `open_when_done` |
| `finder` | `source`, `lines_csv`, `model_csv`, `api_key`, `bookmaker`, `season`, `seed`, `top_n`, `min_leg_edge`, `max_leg_odds`, `max_candidate_legs`, `max_tickets_per_leg`, `allow_same_game`, `rank_by`, `sim.{model_skill, model_noise_margin, model_noise_total, market_noise_margin, market_noise_total}` |
| `backtester` | `seasons`, `weeks`, `bankroll`, `seed`, `policies` (`flat:<pct>`, `flat$:<dollars>`, `kelly:<mult>`), `portfolio_cap`, `top_n`, `with_null`, `no_edge`, `lines_csv`, `results_csv`, `out_dir` |

Keys starting with `_` are comments. Every module also has a full CLI (`--help`).

## Honest caveats

* The simulated league assumes the model has a modest informational edge. That is the hypothesis
  under test, not a fact about any real model; validate with real lines and results before risking
  money.
* Independence between legs is assumed when compounding probabilities. Same-game parlays need a
  joint probability supplied at ticket level.
* For simulation and research purposes only.
