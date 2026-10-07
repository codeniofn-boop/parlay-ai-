#!/usr/bin/env python3
"""
examples/backtester_integration.py
==================================

Reference implementation of the loop that ``backtester.py`` runs each season,
showing exactly how the data structures exported by ``staking_engine.py`` and
``weekly_reporter.py`` plug together.

For every simulated week it:

1. collects the week's parlay candidates (here the reporter's deterministic
   demo slate; in production ``parlay_finder.find_parlays(week=...)``),
2. builds a :class:`weekly_reporter.WeeklyReport` against the *current*
   bankroll, which stakes and ranks every ticket,
3. resolves each recommended ticket with a Bernoulli draw at its true
   probability (replace with real game results in the backtester),
4. settles the outcome through :class:`staking_engine.BankrollTracker`.

Flat and Fractional Kelly are run side by side on identical slates and
identical random outcomes so the comparison isolates the staking policy.

Run from the repository root::

    python3 examples/backtester_integration.py --weeks 18 --bankroll 1000 --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict, List

# Allow running from the repo root OR from inside examples/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from staking_engine import BankrollTracker, StakingConfig  # noqa: E402
from weekly_reporter import ReportConfig, build_weekly_report, generate_demo_slate  # noqa: E402


def simulate_season(
    weeks: int,
    starting_bankroll: float,
    policies: Dict[str, StakingConfig],
    seed: int,
    save_reports_to: str | None = None,
) -> Dict[str, BankrollTracker]:
    """Run every staking policy over the same slates and the same outcomes."""
    trackers = {name: BankrollTracker(starting_bankroll, cfg) for name, cfg in policies.items()}
    outcome_rng = random.Random(seed)

    for week in range(1, weeks + 1):
        # --- 1. collect: swap for parlay_finder.find_parlays(week=week) -------
        slate = generate_demo_slate(week, seed=seed * 100 + week)

        # One random draw per ticket, shared across policies for a fair test.
        draws = {t["ticket_id"]: outcome_rng.random() for t in slate}

        for name, tracker in trackers.items():
            # --- 2. stake & rank against the CURRENT bankroll ----------------
            config = ReportConfig(
                week=week,
                bankroll=tracker.bankroll,
                staking=tracker.config,
                source_label="simulated slate",
                include_skipped=False,
            )
            report = build_weekly_report(slate, config)

            if save_reports_to:
                report.save_text(os.path.join(save_reports_to, f"{name}_week_{week:02d}.txt"))

            # --- 3./4. resolve and settle every recommended ticket -----------
            for staked in report.recommended:
                won = draws[staked.ticket.ticket_id] < staked.ticket.p_true
                tracker.settle(staked.stake, won, label=f"W{week:02d}")

    return trackers


def print_comparison(trackers: Dict[str, BankrollTracker]) -> None:
    rows: List[Dict[str, object]] = [dict(name=name, **t.summary()) for name, t in trackers.items()]
    cols = (
        ("Policy", "name", "{:<16}"),
        ("Start $", "starting_bankroll", "{:>10,.2f}"),
        ("End $", "ending_bankroll", "{:>10,.2f}"),
        ("Growth", "growth", "{:>+8.1%}"),
        ("ROI/turn", "roi_on_turnover", "{:>+8.1%}"),
        ("Bets", "tickets_placed", "{:>5d}"),
        ("Win%", "win_rate", "{:>6.1%}"),
        ("MaxDD", "max_drawdown", "{:>6.1%}"),
    )
    sample = rows[0]
    widths = [len(fmt.format(sample[key])) for _, key, fmt in cols]
    header = "  ".join(f"{title:<{w}}" if i == 0 else f"{title:>{w}}" for i, ((title, _, _), w) in enumerate(zip(cols, widths)))
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(fmt.format(row[key]) for _, key, fmt in cols))


def main() -> int:
    p = argparse.ArgumentParser(description="Flat vs Fractional Kelly over a simulated NFL season.")
    p.add_argument("--weeks", type=int, default=18)
    p.add_argument("--bankroll", type=float, default=1000.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-reports", default=None, help="Directory to write one report per policy per week")
    args = p.parse_args()

    policies = {
        "flat_1pct": StakingConfig.flat_percent(0.01),
        "flat_$10": StakingConfig.flat_dollar(10.0),
        "kelly_0.10": StakingConfig.fractional_kelly(0.10),
        "kelly_0.25": StakingConfig.fractional_kelly(0.25),
    }
    trackers = simulate_season(args.weeks, args.bankroll, policies, args.seed, args.save_reports)
    print(f"Simulated {args.weeks} weeks from ${args.bankroll:,.2f} (seed {args.seed})\n")
    print_comparison(trackers)

    # The ledger is a list of flat dicts: pandas-ready if pandas is installed.
    try:
        import pandas as pd  # type: ignore

        ledger = pd.DataFrame(trackers["kelly_0.25"].history)
        placed = ledger[ledger["stake"] > 0]
        print(f"\nkelly_0.25 ledger: {len(placed)} bets, "
              f"avg stake ${placed['stake'].mean():,.2f}, avg edge {placed['edge'].mean():+.2%}")
    except ImportError:
        print("\n(pandas not installed; tracker.history is a plain list of dicts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
