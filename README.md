# parlay-ai — NFL Parlay Analytics Pipeline

Quantitative tooling for finding, staking and logging +EV NFL parlays.

| Module | Role |
| --- | --- |
| `parlay_finder.py` | (upstream, local) Generates the week's optimised parlay candidates. |
| `staking_engine.py` | Dynamic bankroll & staking strategy: Flat units or Fractional Kelly with hard safety caps. |
| `weekly_reporter.py` | Automated pick logger: stakes every ticket and writes `weekly_parlay_report.txt`. |
| `backtester.py` | (local) Season simulation loop; consumes the data structures exported by the two modules above. |

Both modules in this repo are **standard-library only** (Python 3.9+), fully type-hinted, and ship
with built-in self-tests:

```bash
python3 staking_engine.py --selftest
python3 weekly_reporter.py --selftest
```

---

## Quick start without Terminal

1. Download this branch from GitHub (green **Code** button, **Download ZIP**) and unzip it.
2. Drag `staking_engine.py`, `weekly_reporter.py`, `run_weekly_report.py` and `pipeline_config.json`
   into the folder that already holds `parlay_finder.py`.
3. Open `pipeline_config.json` in any text editor and set your `bankroll` and `mode`
   (`"kelly"` or `"flat"`). Leave `week` as `null` to auto-detect the upcoming week.
4. Open `run_weekly_report.py` the same way you open your other scripts (IDLE, VS Code, PyCharm)
   and press **Run**. The report prints on screen, is saved as `weekly_parlay_report.txt`, and opens
   in TextEdit automatically.

The **Pick Source** line in the report tells you whether it used `parlay_finder.py` or the simulated
fallback slate.

---

## 1. `staking_engine.py`

### Maths

```
Edge          = (P_true * Decimal_Odds) - 1
Full Kelly    = Edge / (Decimal_Odds - 1)
Frac. Kelly   = Full Kelly * Kelly_Multiplier        # 0.10 – 0.25 recommended
Stake ($)     = Bankroll * min(Fraction, max_stake_pct)   # rounded DOWN to the cent
```

### Staking modes

| Mode | `StakingMode` | Behaviour |
| --- | --- | --- |
| **A – Flat** | `FLAT` | Risk `flat_pct` of bankroll (default 1 %) **or** a fixed `flat_dollars` unit, regardless of edge size. |
| **B – Fractional Kelly** | `KELLY` | Fraction scales with edge and odds, damped by `kelly_multiplier`. |

### Safety boundaries (apply to both modes)

1. `Edge <= 0` (or below `min_edge`) ⇒ **$0.00**, never bet -EV.
2. Hard per-ticket cap: `max_stake_pct` of bankroll (default **5 %**).
3. Optional book minimum (`min_stake_dollars`, `enforce_min_stake`).
4. Stakes round **down** to the cent.
5. Invalid input (NaN, odds ≤ 1.0, probability outside [0, 1], negative bankroll) raises
   `StakingInputError`; `safe_compute_stake()` swallows it and returns a $0 result with `.error` set.
6. `apply_portfolio_cap()` scales a whole slate so total weekly exposure ≤ `max_total_pct` (default 15 %).

### API

```python
from staking_engine import StakingConfig, StakingMode, compute_stake, BankrollTracker

cfg = StakingConfig.fractional_kelly(0.25)          # or StakingConfig.flat_percent(0.01) / .flat_dollar(10)
rec = compute_stake(bankroll=1_000, p_true=0.31, decimal_odds=3.64, config=cfg, ticket_id="W06-2L-01")

rec.edge              # 0.1284
rec.stake_dollars     # 12.15
rec.potential_profit  # 32.08
rec.reason            # "0.25x Kelly (full Kelly 4.86%)"
rec.to_dict()         # flat dict -> DataFrame / JSON
```

Bulk helpers: `compute_stakes_for_tickets(tickets, bankroll, cfg)` accepts raw ticket dicts
(parlay-level `p_true`/`decimal_odds`/`american_odds`, or per-leg values that are compounded).

Odds utilities: `american_to_decimal`, `decimal_to_american`, `implied_probability`,
`compound_decimal_odds`, `compound_probability`, `calculate_edge`, `full_kelly_fraction`,
`fractional_kelly_fraction`.

### CLI

```bash
python3 staking_engine.py --bankroll 1000 --p-true 0.31 --american-odds +264 --mode kelly --kelly-multiplier 0.25
python3 staking_engine.py --bankroll 1000 --p-true 0.31 --decimal-odds 3.64 --mode flat --flat-dollars 10 --json
```

---

## 2. `weekly_reporter.py`

### Pipeline

```
parlay_finder.py  ──►  normalise  ──►  staking_engine  ──►  rank & group  ──►  weekly_parlay_report.txt
   (or JSON / demo)     (ParlayTicket)   (StakeRecommendation)  (2-leg / 3-leg)      (+ optional .json side-car)
```

### Collecting parlays

| `--source` | Behaviour |
| --- | --- |
| `finder` (default) | `import parlay_finder` and call the first of `find_parlays`, `generate_parlays`, `get_parlays`, `optimize_parlays`, `build_parlays`, `find_optimal_parlays`, `run`, `main`. Keyword args `week=`, `bankroll=`, `top_n=` are passed only if the function's signature accepts them. Return value may be a list of dicts/objects, a pandas `DataFrame`, or a dict wrapping one under `parlays` / `tickets` / `results`. Override the function with `--finder-function NAME`. |
| `json` | `--input parlays.json` — a list of ticket dicts or `{"parlays": [...]}`. See `examples/sample_parlays.json`. |
| `demo` | Deterministic simulated slate (seeded) for smoke-testing the pipeline. |

If `finder` is requested but `parlay_finder` is not importable, the reporter falls back to `json`
(when `--input` is given) and then to `demo`, and the report's **Pick Source** line says which was used.

### Ticket data contract

```json
{
  "ticket_id": "W06-2L-01",
  "week": 6,
  "legs": [
    {"matchup": "Kansas City Chiefs @ Buffalo Bills", "selection": "Buffalo Bills -3.5", "market": "spread", "american_odds": -110, "p_true": 0.56},
    {"matchup": "Dallas Cowboys @ Philadelphia Eagles", "selection": "Over 44.5",          "market": "total",  "american_odds": -110, "p_true": 0.55}
  ],
  "p_true": 0.308,           // optional – compounded from legs when absent
  "american_odds": 264,      // optional – or "decimal_odds"; compounded from legs when absent
  "notes": "optional free text"
}
```

Accepted aliases: legs `legs|selections|picks|bets`; matchup `matchup|game|event|match|fixture` or `away`+`home`;
selection `selection|line|pick|bet|target|side` or `team`+`spread` / `total`+`direction`;
probability `p_true|true_prob|true_probability|prob|probability|win_prob|model_prob` (0–1 or 0–100);
odds `american_odds|odds|price` or `decimal_odds`. Correlated / same-game parlays should pass a joint
`p_true` at ticket level. Malformed tickets are listed in a **REJECTED INPUT** appendix, never silently dropped.

### Report contents

* Header: title, **Simulated NFL Week N | Season**, full date, generation timestamp.
* **Bankroll** and **Staking Mode** (Flat `[Mode A]` or Fractional Kelly `[Mode B]`) plus active safety caps.
* **TOP RECOMMENDED PARLAYS – 2-LEG / 3-LEG** (and `4+ LEG / OTHER`), ranked by edge; each ticket shows
  every matchup + target line + leg price, **True Win Rate %**, **Book Implied %**, **American Payout
  Odds**, **Edge %**, **To Win**, **Expected Value** and the **RISK: $** amount.
* Weekly exposure summary, a **PASSED (NO BET)** appendix explaining every $0 decision, and rejected inputs.

Width is fixed at 80 ASCII columns. See `examples/weekly_parlay_report.txt`.

### CLI

```bash
# Real pipeline on the MacBook (parlay_finder.py in the same folder or on PYTHONPATH)
python3 weekly_reporter.py --week 6 --bankroll 1000 --mode kelly --kelly-multiplier 0.25 --json-out weekly_parlay_report.json

# Flat $10 units from a JSON hand-off file
python3 weekly_reporter.py --week 6 --bankroll 500 --mode flat --flat-dollars 10 --source json --input parlays.json

# Smoke test with simulated data
python3 weekly_reporter.py --week 6 --date 2026-10-07 --source demo
```

`--week` defaults to an estimate from `--date` (Week 1 = first Thursday after Labor Day); pass it explicitly in production.

---

## 3. Integration with `backtester.py`

```python
from staking_engine import StakingConfig, BankrollTracker
from weekly_reporter import ReportConfig, build_weekly_report
import parlay_finder

tracker = BankrollTracker(starting_bankroll=1_000, config=StakingConfig.fractional_kelly(0.25))

for week in range(1, 19):
    tickets = parlay_finder.find_parlays(week=week)                 # any shape the contract accepts
    report = build_weekly_report(tickets, ReportConfig(week=week, bankroll=tracker.bankroll,
                                                       staking=tracker.config))
    for st in report.recommended:                                   # StakedTicket objects
        won = simulate_outcome(st.ticket)                           # your resolver
        tracker.settle(st.stake, won, label=st.ticket.ticket_id)

    rows = report.to_records()          # list[dict] -> pd.DataFrame(rows)
    report.save_text(f"reports/week_{week:02d}.txt")

print(tracker.summary())   # ending_bankroll, roi_on_turnover, win_rate, max_drawdown, ...
```

A complete, runnable example comparing Flat vs Fractional Kelly over an 18-week simulated season lives in
`examples/backtester_integration.py`.

Everything crosses module boundaries as plain dataclasses with `to_dict()` / `to_records()`, so the
objects drop straight into pandas, JSON logs or a database without adapters.
