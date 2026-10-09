# parlay-ai — NFL Parlay Analytics Pipeline

A complete, standard-library-only Python pipeline for finding, staking, logging and
backtesting +EV NFL parlays, with game markets and **player props**. Every module ships with a
built-in self-test. No paid data or APIs are needed: lines come from your sportsbook by hand, player
stats come from the free nflverse files.

| File | Role |
| --- | --- |
| `parlay_finder.py` | Builds each week's ranked 2-leg parlay candidates from simulated, CSV or live lines, game markets and player props alike. |
| `staking_engine.py` | Dynamic bankroll and staking strategy: Flat units or Fractional Kelly with hard safety caps. |
| `weekly_reporter.py` | Automated pick logger: stakes every ticket, keeps the top N you asked for, and writes `weekly_parlay_report.txt`. |
| `backtester.py` | Replays whole seasons through the same code path, compares staking policies, and reports game markets and props separately. |
| `run_weekly_report.py` | Zero-argument launcher for the weekly report (press Run, no Terminal). |
| `pipeline_config.json` | The one settings file: bankroll, staking mode, how many parlays, per-market thresholds, finder and backtester options. |
| `week_inputs.csv` + `props_inputs.csv` + `build_lines.py` | This week's games (one row each) and player props (one row each) and the expander that turns them into `lines.csv`. |
| `markets.json` + `market_registry.py` | The market registry: every player-prop market is declared here (spelling, stat, distribution, correlation class). Add a prop type by adding an entry. |
| `prop_model.py` | The **EXPERIMENTAL** player-prop projection model (nflverse stats, no paid data) and its walk-forward calibration. |
| `correlation_rules.json` + `correlation_rules.py` | Rule 3 as a readable table: which same-game pairs are positive, negative, exclusive or redundant, with a reason each. |
| `app.py` + `launch_edgebook.py` | **EdgeBook AI**, the Streamlit dashboard: sidebar controls, KPI tiles, the edge signal board, the parlay engine with a simulated wallet, and the bankroll growth chart. |
| `report_template.html` + `publish_report.py` | The shareable web page: every weekly run also writes `docs/index.html`, a static page with a live bankroll box, Flat/Kelly switch and Max-parlays control. |

```
week_inputs.csv ─┐                                            ┌─► weekly_parlay_report.txt
props_inputs.csv ┼─► build_lines.py ─► lines.csv ─► parlay_finder.py ─► weekly_reporter.py ─┼─► weekly_parlay_report.json ─► docs/index.html
                 │   (prop_model.py fills prop         │   (correlation_rules.json)   │ (staking_engine.py)
                 │    probabilities, lines_log.csv)    │                              └─► EdgeBook AI dashboard (app.py)
nflverse stats ──┘                                     └──────── backtester.py (season replay, game vs prop results)
```

---

## Quick start without Terminal

1. Put all the files above in one folder.
2. Open `pipeline_config.json` in any text editor. Set `bankroll` and `mode` (`"kelly"` or `"flat"`),
   and `max_parlays` (how many parlays you want at most, 1-10, default 3).
   Leave `week` as `null` to auto-detect the upcoming week from today's date.
3. Type this week's games into `week_inputs.csv` and, if you want player props, copy the lines from
   your sportsbook into `props_inputs.csv` (start from `props_template.csv`; see **Player props**).
4. Open `run_weekly_report.py` in IDLE, VS Code or PyCharm and press **Run**, or double-click it if
   `.py` files open with Python Launcher. The report prints on screen, is saved as
   `weekly_parlay_report.txt`, and opens in TextEdit. The first run with props downloads the free
   nflverse stats files (about 20 MB) into `data_cache/`.
5. To see how the strategy would have performed over many seasons, run `backtester.py` the same way.
   It writes `backtest_output/backtest_summary.txt` and opens it.

The report's **Pick Source** line says where the picks came from. Out of the box it is the simulated
league inside `parlay_finder.py`, clearly labelled "NOT real games". To use real games, drop a
`lines.csv` into the folder; see **Using real lines**. For live lines without typing, sign up for a
free key at https://the-odds-api.com, paste it as `"api_key"` under `finder` in
`pipeline_config.json`, and set `"source": "api"`.

## The dashboard (EdgeBook AI)

`app.py` is a Streamlit front end over the same modules the command line uses. Nothing is
re-implemented: the sidebar's edge threshold becomes `FinderConfig.min_prob_gap`, the two prop sliders
become `market_rules["props"]`, the staking mode becomes a `StakingConfig`, "How many parlays (maximum)"
goes through `weekly_reporter.select_parlays` (ranking, the limit and the weekly cap, exactly as the
report card does), and the growth chart replays one simulated season through `backtester`. The signal
board labels experimental markets, every leg shows its model and book-implied probability, prop legs
carry an EXPERIMENTAL badge, and the "Legs clearing the filter by market" expander is the tuning aid
for the thresholds. If a module is missing or raises, that section switches to clearly labelled demo
data and the sidebar's "Backend status" panel shows why.

Run it with `streamlit run app.py`, or double-click `launch_edgebook.py` (it installs Streamlit on
first use and opens the browser). The dark theme lives in `.streamlit/config.toml`. Requirements:
`pip install -r requirements.txt` (Streamlit and pandas only; the pipeline itself needs nothing).

The "Simulate / Place wager" buttons move money inside the session only. "Reset wallet" restores the
starting bankroll.

## The web page (share it)

Every weekly run also writes **`docs/index.html`**: a self-contained page showing the same tickets as
the text report, with a bankroll box, a Flat / Kelly switch and a Max parlays control that resize and
re-select every ticket in the browser using the same formulas, caps and selection rule as
`staking_engine.py` and `weekly_reporter.py` (the two are checked against each other to the cent).
Prop legs carry the experimental badge; same-game tickets are listed apart with their pricing note. It
needs no server. Open it by double-clicking, or publish it:

* **GitHub Pages:** in the repository go to Settings, then Pages, choose "Deploy from a branch", pick
  this branch and the `/docs` folder, and save. The page appears at
  `https://<your-user>.github.io/parlay-ai-/` and updates whenever you push a new `docs/index.html`.
* **Any static host:** upload `docs/index.html` on its own; it has no other files.

Rebuild it by hand with `python3 publish_report.py` (reads `weekly_parlay_report.json`). The page
labels simulated slates plainly and carries a research-only disclaimer and problem-gambling helpline.

## Quick start with Terminal

```bash
for m in market_registry correlation_rules build_lines prop_model parlay_finder staking_engine \
         weekly_reporter backtester publish_report; do python3 $m.py --selftest || break; done

python3 weekly_reporter.py --week 6 --bankroll 1000 --mode kelly --kelly-multiplier 0.25 --max-parlays 3
python3 parlay_finder.py --week 5 --explain                 # legs clearing the filter, per market
python3 prop_model.py --calibrate --season 2025            # the prop model's calibration gate
python3 backtester.py --seasons 50 --with-null
python3 backtester.py --lines lines.csv --results nflverse  # real season, scores from nflverse
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
   The thresholds are **configurable per market type** in `finder.market_rules`: keys are market keys,
   labels or aliases from `markets.json` or the wildcards `props` / `game`; an explicit market beats a
   wildcard. Most over/under props are priced near 50%, so a 68% floor rarely lets a prop through; the
   report's **LEGS CLEARING THE FILTER BY MARKET** table (and `--explain`) shows, per market, how many
   sides there were, how many passed, the thresholds in force, the best model probability and gap seen,
   and the top rejection reason, which is what you tune from. High-variance markets (First TD) pay an
   extra gap (`model.high_variance_extra_gap` in `markets.json`).
3. **Anti-correlation check.** Legs from the same game are classified by the table in
   `correlation_rules.json` (positive / negative / neutral / exclusive / redundant, each with a reason)
   and only positive pairs are allowed under `same_game_policy: "positive_only"`; exclusive and
   redundant pairs are rejected under every policy. See **Player props** for the prop rows.
   `python3 correlation_rules.py` prints the table; its self-test runs one case per rule.

Legs are also capped at `max_leg_odds` (+300), ranked by `rank_by` (`growth`: `edge² / (D - 1)`, the Kelly
log-growth proxy; or raw `edge`), and diversified (`max_tickets_per_leg`, `max_tickets_per_game`).
These filters are strict by design: on a typical slate few or no legs qualify, and the report then says
"no bets are recommended" rather than inventing tickets. `python3 parlay_finder.py --explain` shows why
each side was rejected. Markets: spread, total, moneyline, team total (`Buffalo Bills Over 24.5`) and
every player-prop market declared in `markets.json` (`finder.markets` accepts keys, labels, aliases and
the wildcards `game`, `props`, `all`).

**Same-game pricing.** Any ticket whose two legs come from one game is flagged `sgp_required`: books
price those as same-game parlays, not at the product of the two prices, so the report lists them apart
(with the correlation reason and a note to confirm the payout at the book) and does not count them
toward your limit unless `count_same_game_parlays` is true.

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

`market` is `spread`, `total`, `moneyline`, `team_total` or a prop market key. `selection` is the line
text exactly as you would say it (`Dak Prescott Over 264.5 Passing Yards`, `CeeDee Lamb Anytime TD`).
Both sides of a market should be present so the vig can be removed. `model_prob` is *your* model's
win probability for that side (0–1 or 0–100); leave it blank and the de-vigged market probability is
used, which correctly produces no edge. Prop rows also carry `player`, `player_id`, `team`,
`position`, `blocked` (a reason the side can never be bet, e.g. ruled out) and `model_note`; the
builder fills those from the prop model. For live API lines, supply your probabilities in
`model_probs.csv` (`matchup,selection,model_prob`, see `examples/model_probs.csv`) and point
`finder.model_csv` at it.

---

## 1b. Player props (EXPERIMENTAL)

Player props go through the same finder, staking engine, report and backtester as game markets. They
are labelled **EXPERIMENTAL** on every screen until the calibration gate below clears them.

### Entering lines: `props_inputs.csv`

One row per prop, copied from your sportsbook (no API, nothing paid). Start from `props_template.csv`;
`examples/props_inputs.csv` shows 35 rows (illustrative numbers, not real lines).

```
week,away,home,player,market,line,over_price,under_price,yes_price,no_price,position,notes
5,Tampa Bay Buccaneers,Dallas Cowboys,Dak Prescott,Passing Yards,264.5,-115,-105,,,QB,
5,Tampa Bay Buccaneers,Dallas Cowboys,CeeDee Lamb,Anytime TD,,,,-125,,WR,
```

* `away` / `home` must match `week_inputs.csv` exactly; `market` is any label or alias from
  `markets.json` (`pass yds`, `Receptions`, `Rush + Rec Yards`, `anytime td`, ...).
* Over/under props take a `line` and two prices (blank prices default to -110); yes/no props take a
  `yes_price` and an optional `no_price`. An optional `team` column overrides the model's team lookup.
* Every build appends new or changed prop lines to **`lines_log.csv`** with a timestamp. That log is the
  pipeline's own free line history: rebuild again just before kickoff and the last snapshot is the
  closing line the backtester uses for closing-line value.

Markets shipped in `markets.json`: Passing Yards, Passing TDs, Completions, Interceptions, Rushing
Yards, Rushing Attempts, Receiving Yards, Receptions, Rush + Rec Yards, Anytime TD and First TD
(high-variance). **Adding a prop type is a config change:** add an entry with its label, aliases,
`parts` (the nflverse stat column and the usage column that drives it), distribution and correlation
class, and it flows through the inputs, the model, the report, the dashboard and the backtester.

### The model: `prop_model.py`

FPI only produces game-level numbers, so props have their own projection:

```
mean = sum over the market's parts of  usage x rate x opponent factor x game-script factor
```

* **usage**: recent-weighted per-game attempts / carries / targets (decay 0.75 per game back), shrunk
  toward last season's rate and, for players with no history, the positional average.
* **rate**: yards per target, TDs per carry, completions per attempt, ..., shrunk toward the league
  positional rate with pseudo-counts because efficiency is noisy.
* **opponent factor**: what the opponent's defence allowed per game relative to the league, shrunk
  halfway toward 1.0 and clamped.
* **game-script factor**: pass volume rises for underdogs and rush volume for favourites, scaled by
  the posted total; the expected margin comes from the FPI numbers in `week_inputs.csv`.

The projection becomes a probability through the market's distribution: normal (passing yards),
gamma (rushing / receiving yards: skewed, zero-floored), negative binomial (receptions, completions,
attempts), Poisson (passing TDs, interceptions) and `1 - exp(-lambda)` for Anytime TD; whole-number
lines carry push mass and the side's probability is conditional on no push. Players listed Out or
Doubtful on the nflverse injury report are blocked. Data: nflverse weekly stats, schedules and injury
reports, downloaded once into `data_cache/` (current season refreshes daily; `EDGEBOOK_OFFLINE=1`
uses the cache only).

### Calibration gate

`python3 prop_model.py --calibrate --season 2025` replays a past season week by week, projecting
every qualifying player from the data available *before* each week, and scores the projections per
market: bias, dispersion ratio (realised / projected spread), 50% and 80% interval coverage, and the
model's claimed over-probability versus the realised hit rate at synthetic lines one sd below, at and
one sd above the projection. `examples/prop_calibration_2025.txt` is the shipped run: every over/under
market's dispersion ratio sits between 0.97 and 1.02 and claimed and hit agree within 1-5 points;
Anytime TD claims 29.3% and hits 28.2%. First TD cannot be scored from weekly stats (it needs
play-by-play) and stays high-variance. The experimental badge should stay on until claimed and hit
agree within 3 points on at least 200 legs in your own backtests.

### Correlation rules for props

`correlation_rules.json` holds the same-game table with a short reason per rule. Positive (allowed):
a quarterback's passing yards, completions or TDs with his receiver's receiving yards or receptions in
the same direction; passing TDs with a receiver's touchdown; a rusher's over with his team's side;
a scorer or any offensive over with his team total over; passing and scoring overs with the game total
over. Negative or rejected: the same player over and under (exclusive); two lines on one player's
stat family, e.g. rush yards with rush + rec yards or anytime with first TD (redundant); two
first-TD scorers (exclusive); two receivers' overs on one team (target share); a rusher's over against
the opponent's side (blowout script); a quarterback's over with a rusher's over (script competition);
any offensive over with the team or game total under; interceptions over with the team's own side.
`python3 correlation_rules.py --selftest` checks one case per rule and fails when a rule has none.

### How many parlays

`max_parlays` (config, `--max-parlays`, the dashboard input and the web page control) is a
**maximum** between 1 and 10 (default 3). Tickets that pass every filter are ranked by expected value
and the top N are returned; if only one qualifies you get one, if none qualify you get none, and
nothing is relaxed to reach N. The report always says "You asked for N. X qualified.", lists the
qualifying tickets beyond the limit, the same-game tickets set aside, and the passed (no bet)
candidates with reasons. The 5% per-ticket cap and the 15% weekly cap still apply across the chosen
set; when N tickets would breach 15% their stakes are scaled down proportionally and the report says
by how much.

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

The report shows the simulated NFL week and date, bankroll and staking mode, safety caps, pick
source, the leg-filter count and "you asked for N; X qualified", then **TOP RECOMMENDED PARLAYS**.
Each ticket lists every leg (selection and price, then the matchup with the leg's model and
book-implied probability and the EXPERIMENTAL tag on prop legs), the **True Win Rate %**, **Book
Implied %**, **American Payout Odds**, **Edge %**, **To Win**, **Expected Value**, the **RISK: $**
amount, the correlation reason and pricing note on same-game tickets, and a model note. Tickets are
ordered by the expected dollar profit of the recommended stake. Then come **QUALIFIED BUT BEYOND YOUR
LIMIT**, **SAME-GAME TICKETS** (SGP pricing required, not counted), **LEGS CLEARING THE FILTER BY
MARKET** and the **PASSED (NO BET)** appendix that explains every $0 decision.
See `examples/weekly_parlay_report.txt` and `examples/weekly_parlay_report_with_props.txt`.

```bash
python3 weekly_reporter.py --week 6 --bankroll 500 --mode flat --flat-dollars 10 --max-parlays 2
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
python3 backtester.py --lines lines.csv --results nflverse      # real season, scores from nflverse
```

`results.csv`: `week,away,home,away_score,home_score` (see `examples/results.csv`); `--results nflverse`
writes it from the nflverse schedule for the completed games. Player-prop legs are graded from the
player's nflverse stat line that week (a player with no line recorded nothing); tickets whose game has
not been played yet are reported as ungraded, never scored. Results and calibration are reported
**separately for game markets and player props** (`game` / `prop` / `mixed` tickets) in the
**GAME MARKETS vs PLAYER PROPS** section, and the ledger carries `market_group`. **Closing line value**
for prop legs is computed from `lines_log.csv` when it holds a later snapshot of the same line; when
it does not, the summary says so and names the sources that sell historical closing prop lines (The
Odds API historical endpoint, OddsJam, Unabated), none of which this pipeline needs.
Outputs land in `backtest_output/`: `backtest_summary.txt`, `backtest_ledger.csv` (one row per ticket
per season per policy) and `backtest_summary.json`. See `examples/backtest_summary.txt`.

---

## Configuration (`pipeline_config.json`)

| Section | Keys |
| --- | --- |
| top level | `week`, `bankroll`, `mode`, `kelly_multiplier`, `flat_pct`, `flat_dollars`, `max_stake_pct`, `min_edge`, `portfolio_cap`, `source`, `input`, `top_n`, `max_parlays`, `count_same_game_parlays`, `output`, `json_out`, `html_out`, `open_when_done` |
| `finder` | `source`, `lines_csv`, `model_csv`, `api_key`, `bookmaker`, `season`, `seed`, `top_n`, `min_leg_prob`, `min_prob_gap`, `edge_basis`, `min_leg_edge`, `max_leg_odds`, `max_candidate_legs`, `max_tickets_per_leg`, `max_tickets_per_game`, `same_game_policy`, `markets` (keys, labels, aliases, `game` / `props` / `all`), `market_rules` (per-market `min_leg_prob`, `min_prob_gap`, `max_leg_odds`, `min_leg_edge`), `rank_by`, `sim.{model_skill, model_noise_margin, model_noise_total, market_noise_margin, market_noise_total}` |
| `backtester` | `seasons`, `weeks`, `bankroll`, `seed`, `policies` (`flat:<pct>`, `flat$:<dollars>`, `kelly:<mult>`), `portfolio_cap`, `top_n`, `with_null`, `no_edge`, `lines_csv`, `results_csv`, `lines_log_csv`, `count_same_game`, `out_dir` |
| `markets.json` | `prop_markets.<key>.{label, aliases, kind, parts, distribution, sd_intercept, sd_slope, dispersion, rate_pseudo, families, cls, positions, high_variance, experimental}` and `model.{decay, prior_season_weight, min_games, rate_pseudo_counts, opponent_shrink, opponent_clamp, script, high_variance_extra_gap, block_injury_status}` |
| `correlation_rules.json` | `rules[].{id, a, b, relation, label, reason}` |

Keys starting with `_` are comments. Every module also has a full CLI (`--help`).

## Honest caveats

* The simulated league assumes the model has a modest informational edge. That is the hypothesis
  under test, not a fact about any real model; validate with real lines and results before risking
  money.
* The player-prop model is **experimental**: it is calibrated on one past season from free data, it
  knows nothing about snaps, weather, coaching changes or late injury news beyond the nflverse injury
  report, and the lines you enter by hand may be stale by kickoff. Treat its probabilities as the
  model's opinion and keep the badge on until your own backtests clear it.
* Independence between legs is assumed when compounding probabilities, which is why same-game tickets
  are set aside: the book prices those as same-game parlays and the joint probability shown is only a
  conservative floor.
* For simulation and research purposes only. Not financial advice. Bet only what you can afford to
  lose, and only where it is legal and you are of age. In the US, help is available any time at
  1-800-GAMBLER.
