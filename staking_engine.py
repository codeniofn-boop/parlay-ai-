#!/usr/bin/env python3
"""
staking_engine.py
=================

Dynamic Bankroll and Staking Strategy module for the NFL parlay analytics
pipeline.

This module answers one question for every parlay ticket the pipeline
produces:

    "Given my current bankroll, my modelled true probability of this parlay
     hitting, and the odds the sportsbook is paying, exactly how many dollars
     should I risk on this ticket?"

It is deliberately self-contained (standard library only) so it can be
imported by ``parlay_finder.py``, ``weekly_reporter.py`` and the main
``backtester.py`` simulation loop without dragging in heavy dependencies.

Core concepts
-------------
* **True probability (P_true)** – the pipeline's own estimate that the parlay
  wins, already compounded across legs (e.g. 0.55 * 0.52 for a 2-legger).
* **Decimal odds** – the compounded bookmaker payout multiplier INCLUDING the
  original stake (e.g. +264 American == 3.64 decimal). A $10 stake at 3.64
  returns $36.40 total, i.e. $26.40 profit.
* **Edge** – the expected return per $1 risked::

      Edge = (P_true * Decimal_Odds) - 1

  A positive edge means the ticket has positive expected value (+EV).

Staking modes
-------------
* ``StakingMode.FLAT`` – Mode A. Risk a fixed fraction of bankroll (default
  1 %) or a fixed dollar amount regardless of edge size. Simple, robust, and
  immune to model over-confidence.
* ``StakingMode.KELLY`` – Mode B. Fractional Kelly Criterion::

      Fraction = (Edge / (Decimal_Odds - 1)) * Kelly_Multiplier

  ``Edge / (Decimal_Odds - 1)`` is the full-Kelly fraction (equivalently
  ``p - q/b`` where ``b = Decimal_Odds - 1``). Full Kelly is far too
  aggressive for noisy sports models, so a conservative multiplier
  (0.10 – 0.25) is applied as a safety buffer.

Safety boundaries
-----------------
Both modes pass through the same hard guard-rails:

1. If ``Edge <= 0`` the recommended stake is **$0.00** (never bet -EV).
2. The stake is capped at ``max_stake_pct`` of bankroll (default 5 %).
3. The stake is floored at ``$0`` and optionally at a book minimum.
4. Stakes are rounded down to the cent so we never over-risk by rounding.
5. Any invalid input (NaN, negative bankroll, odds <= 1.0, probability
   outside [0, 1]) raises :class:`StakingInputError` with a clear message,
   or – in ``safe_compute_stake`` – returns a zero-stake result with the
   error recorded, so a single bad ticket never crashes a weekly run or a
   multi-season backtest.

Integration
-----------
::

    from staking_engine import StakingConfig, StakingMode, compute_stake

    cfg = StakingConfig(mode=StakingMode.KELLY, kelly_multiplier=0.25)
    rec = compute_stake(bankroll=1_000.0, p_true=0.31, decimal_odds=3.64, config=cfg)
    print(rec.stake_dollars, rec.edge, rec.to_dict())

Everything returned is a plain dataclass with ``to_dict()`` so it can be
appended straight into a pandas DataFrame or JSON log inside
``backtester.py``.

Run ``python3 staking_engine.py --help`` for a CLI, or
``python3 staking_engine.py --selftest`` to execute the built-in unit tests.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

__all__ = [
    "StakingMode",
    "StakingConfig",
    "StakeRecommendation",
    "StakingInputError",
    "american_to_decimal",
    "decimal_to_american",
    "implied_probability",
    "compound_decimal_odds",
    "compound_probability",
    "calculate_edge",
    "full_kelly_fraction",
    "fractional_kelly_fraction",
    "compute_stake",
    "safe_compute_stake",
    "compute_stakes_for_tickets",
    "apply_portfolio_cap",
    "BankrollTracker",
]

__version__ = "1.0.0"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class StakingInputError(ValueError):
    """Raised when a staking input is mathematically invalid.

    Examples: a negative bankroll, decimal odds <= 1.0 (no possible profit),
    a probability outside the closed interval [0, 1], or NaN/inf anywhere.
    """


# ---------------------------------------------------------------------------
# Enumerations and configuration
# ---------------------------------------------------------------------------


class StakingMode(str, Enum):
    """Switchable staking strategies.

    Inherits from ``str`` so the enum serialises cleanly to JSON/CSV and can
    be constructed from user strings such as ``StakingMode("kelly")``.
    """

    FLAT = "flat"      # Mode A: fixed % of bankroll or fixed $ per ticket
    KELLY = "kelly"    # Mode B: fractional Kelly scaled by edge/confidence

    @classmethod
    def parse(cls, value: Union[str, "StakingMode"]) -> "StakingMode":
        """Case-insensitive, alias-friendly parser.

        Accepts ``"flat"``, ``"unit"``, ``"fixed"`` for FLAT and ``"kelly"``,
        ``"fractional"``, ``"fractional_kelly"`` for KELLY.
        """
        if isinstance(value, StakingMode):
            return value
        if not isinstance(value, str):
            raise StakingInputError(
                f"Staking mode must be a string or StakingMode, got {type(value).__name__}"
            )
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "flat": cls.FLAT,
            "unit": cls.FLAT,
            "fixed": cls.FLAT,
            "flat_unit": cls.FLAT,
            "a": cls.FLAT,
            "mode_a": cls.FLAT,
            "kelly": cls.KELLY,
            "fractional": cls.KELLY,
            "fractional_kelly": cls.KELLY,
            "kelly_criterion": cls.KELLY,
            "b": cls.KELLY,
            "mode_b": cls.KELLY,
        }
        if key not in aliases:
            raise StakingInputError(
                f"Unknown staking mode '{value}'. Valid options: flat, kelly"
            )
        return aliases[key]

    @property
    def label(self) -> str:
        """Human-friendly label used in reports."""
        return "Flat Unit Betting" if self is StakingMode.FLAT else "Fractional Kelly Criterion"


@dataclass(frozen=True)
class StakingConfig:
    """All tunable knobs for the staking engine in one immutable object.

    Attributes
    ----------
    mode:
        ``StakingMode.FLAT`` or ``StakingMode.KELLY``.
    flat_pct:
        Mode A – fraction of bankroll to risk per ticket (``0.01`` == 1 %).
        Ignored when ``flat_dollars`` is set.
    flat_dollars:
        Mode A – optional fixed dollar stake (e.g. ``10.0``). When provided it
        overrides ``flat_pct`` entirely.
    kelly_multiplier:
        Mode B – fraction of full Kelly to use. ``0.25`` == quarter Kelly.
        Must be in (0, 1]. Values above 0.5 are allowed but a warning is
        logged because they are rarely appropriate for sports models.
    max_stake_pct:
        Hard ceiling on any single ticket as a fraction of bankroll
        (``0.05`` == 5 %). Applied to BOTH modes.
    min_stake_dollars:
        Optional sportsbook minimum. Any positive recommendation below this is
        raised to the minimum only if ``enforce_min_stake`` is True, otherwise
        it is zeroed out (we would rather skip than overbet a thin edge).
    enforce_min_stake:
        See ``min_stake_dollars``.
    min_edge:
        Minimum edge required before ANY stake is placed. Default ``0.0``
        means "strictly positive edge". Raising it to e.g. ``0.02`` filters
        out tickets whose edge is within model noise.
    round_to_cents:
        Round stakes down to the nearest cent (True) or leave as float.
    """

    mode: StakingMode = StakingMode.KELLY
    flat_pct: float = 0.01
    flat_dollars: Optional[float] = None
    kelly_multiplier: float = 0.25
    max_stake_pct: float = 0.05
    min_stake_dollars: float = 0.0
    enforce_min_stake: bool = False
    min_edge: float = 0.0
    round_to_cents: bool = True

    def __post_init__(self) -> None:
        # Coerce string modes (e.g. from CLI / JSON) into the enum.
        object.__setattr__(self, "mode", StakingMode.parse(self.mode))
        self.validate()

    def validate(self) -> None:
        """Raise :class:`StakingInputError` if any parameter is unsafe."""
        _require_finite("flat_pct", self.flat_pct)
        if not 0.0 < self.flat_pct <= 1.0:
            raise StakingInputError("flat_pct must be in (0, 1], e.g. 0.01 for 1%")
        if self.flat_dollars is not None:
            _require_finite("flat_dollars", self.flat_dollars)
            if self.flat_dollars <= 0:
                raise StakingInputError("flat_dollars must be positive when provided")
        _require_finite("kelly_multiplier", self.kelly_multiplier)
        if not 0.0 < self.kelly_multiplier <= 1.0:
            raise StakingInputError("kelly_multiplier must be in (0, 1]; 0.10-0.25 recommended")
        if self.kelly_multiplier > 0.5:
            logger.warning(
                "kelly_multiplier=%.2f is aggressive; most sports models use 0.10-0.25",
                self.kelly_multiplier,
            )
        _require_finite("max_stake_pct", self.max_stake_pct)
        if not 0.0 < self.max_stake_pct <= 1.0:
            raise StakingInputError("max_stake_pct must be in (0, 1], e.g. 0.05 for 5%")
        _require_finite("min_stake_dollars", self.min_stake_dollars)
        if self.min_stake_dollars < 0:
            raise StakingInputError("min_stake_dollars cannot be negative")
        _require_finite("min_edge", self.min_edge)
        if self.min_edge < 0:
            raise StakingInputError("min_edge cannot be negative")

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly representation (enum -> string)."""
        d = asdict(self)
        d["mode"] = self.mode.value
        d["mode_label"] = self.mode.label
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StakingConfig":
        """Inverse of :meth:`to_dict`; ignores unknown keys."""
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})

    # Convenience constructors -------------------------------------------------

    @classmethod
    def flat_percent(cls, pct: float = 0.01, **kwargs: Any) -> "StakingConfig":
        """Mode A with a percentage unit, e.g. ``StakingConfig.flat_percent(0.01)``."""
        return cls(mode=StakingMode.FLAT, flat_pct=pct, flat_dollars=None, **kwargs)

    @classmethod
    def flat_dollar(cls, dollars: float = 10.0, **kwargs: Any) -> "StakingConfig":
        """Mode A with a fixed dollar unit, e.g. ``StakingConfig.flat_dollar(10)``."""
        return cls(mode=StakingMode.FLAT, flat_dollars=dollars, **kwargs)

    @classmethod
    def fractional_kelly(cls, multiplier: float = 0.25, **kwargs: Any) -> "StakingConfig":
        """Mode B, e.g. ``StakingConfig.fractional_kelly(0.10)`` for tenth-Kelly."""
        return cls(mode=StakingMode.KELLY, kelly_multiplier=multiplier, **kwargs)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class StakeRecommendation:
    """Full audit trail for one staking decision.

    Every intermediate number is kept so ``backtester.py`` can log why a
    stake was (or was not) placed and so the weekly report can display the
    edge, implied probability and payout alongside the dollar amount.
    """

    bankroll: float
    p_true: float
    decimal_odds: float
    american_odds: int
    implied_prob: float
    edge: float
    mode: StakingMode
    raw_fraction: float          # fraction BEFORE caps (may exceed max_stake_pct)
    capped_fraction: float       # fraction AFTER caps / floors
    stake_dollars: float         # final recommended risk amount
    potential_profit: float      # stake * (decimal_odds - 1)
    potential_payout: float      # stake * decimal_odds (profit + returned stake)
    expected_value: float        # stake * edge
    cap_applied: bool = False
    skipped: bool = False        # True when stake is $0 by rule
    reason: str = ""             # human-readable explanation of the decision
    ticket_id: Optional[str] = None
    error: Optional[str] = None  # populated only by safe_compute_stake

    def to_dict(self) -> Dict[str, Any]:
        """Flat, JSON/CSV-friendly dictionary (enum -> string)."""
        d = asdict(self)
        d["mode"] = self.mode.value
        return d

    @property
    def is_bet(self) -> bool:
        """True when a non-zero stake is recommended."""
        return self.stake_dollars > 0 and not self.skipped

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"[{self.mode.label}] edge={self.edge:+.4f} "
            f"stake=${self.stake_dollars:,.2f} ({self.capped_fraction:.2%} of bankroll) "
            f"-> {self.reason}"
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_finite(name: str, value: Any) -> float:
    """Coerce ``value`` to float and reject NaN / inf / non-numeric input."""
    if isinstance(value, bool):
        raise StakingInputError(f"{name} must be numeric, got bool")
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise StakingInputError(f"{name} must be numeric, got {value!r}") from exc
    if math.isnan(f) or math.isinf(f):
        raise StakingInputError(f"{name} must be finite, got {f}")
    return f


def _validate_bankroll(bankroll: Any) -> float:
    b = _require_finite("bankroll", bankroll)
    if b < 0:
        raise StakingInputError(f"bankroll cannot be negative (got {b})")
    return b


def _validate_probability(p: Any, name: str = "p_true") -> float:
    f = _require_finite(name, p)
    if not 0.0 <= f <= 1.0:
        raise StakingInputError(f"{name} must be within [0, 1], got {f}")
    return f


def _validate_decimal_odds(odds: Any) -> float:
    d = _require_finite("decimal_odds", odds)
    if d <= 1.0:
        raise StakingInputError(
            f"decimal_odds must exceed 1.0 (a payout above stake), got {d}"
        )
    return d


# ---------------------------------------------------------------------------
# Odds utilities
# ---------------------------------------------------------------------------


def american_to_decimal(american: Union[int, float, str]) -> float:
    """Convert American odds to decimal odds.

    ``+150`` -> 2.50, ``-110`` -> 1.9091. Accepts strings such as ``"+264"``.
    """
    if isinstance(american, str):
        american = american.strip().replace("+", "")
    a = _require_finite("american_odds", american)
    if a == 0 or -100 < a < 100:
        raise StakingInputError(
            f"American odds must be >= +100 or <= -100, got {a:+.0f}"
        )
    if a > 0:
        return 1.0 + a / 100.0
    return 1.0 + 100.0 / abs(a)


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to the nearest whole American odds.

    2.50 -> +150, 1.9091 -> -110. Even money (2.0) returns +100.
    """
    d = _validate_decimal_odds(decimal_odds)
    if d >= 2.0:
        return int(round((d - 1.0) * 100.0))
    return int(round(-100.0 / (d - 1.0)))


def format_american(american: int) -> str:
    """``264 -> '+264'``, ``-110 -> '-110'``."""
    return f"{american:+d}"


def implied_probability(decimal_odds: float) -> float:
    """Bookmaker break-even probability = ``1 / decimal_odds`` (vig included)."""
    return 1.0 / _validate_decimal_odds(decimal_odds)


def compound_decimal_odds(leg_decimal_odds: Iterable[float]) -> float:
    """Multiply per-leg decimal odds into a single parlay price."""
    total = 1.0
    count = 0
    for leg in leg_decimal_odds:
        total *= _validate_decimal_odds(leg)
        count += 1
    if count == 0:
        raise StakingInputError("compound_decimal_odds requires at least one leg")
    return total


def compound_probability(leg_probabilities: Iterable[float]) -> float:
    """Multiply independent per-leg true probabilities into a parlay P_true.

    NOTE: This assumes leg independence. Correlated legs (same-game parlays)
    should have their joint probability modelled upstream in
    ``parlay_finder.py`` and passed in directly.
    """
    total = 1.0
    count = 0
    for p in leg_probabilities:
        total *= _validate_probability(p, "leg_probability")
        count += 1
    if count == 0:
        raise StakingInputError("compound_probability requires at least one leg")
    return total


# ---------------------------------------------------------------------------
# Core maths
# ---------------------------------------------------------------------------


def calculate_edge(p_true: float, decimal_odds: float) -> float:
    """Exact edge per $1 risked::

        Edge = (P_true * Decimal_Odds) - 1

    Positive => +EV, zero => break-even, negative => -EV.
    """
    p = _validate_probability(p_true)
    d = _validate_decimal_odds(decimal_odds)
    return p * d - 1.0


def full_kelly_fraction(p_true: float, decimal_odds: float) -> float:
    """Full Kelly fraction of bankroll::

        f* = Edge / (Decimal_Odds - 1)  ==  (p*b - q) / b   where b = D - 1

    Can be negative for -EV bets (the caller must clamp at zero).
    """
    p = _validate_probability(p_true)
    d = _validate_decimal_odds(decimal_odds)
    edge = p * d - 1.0
    b = d - 1.0  # net profit per $1 staked; guaranteed > 0 by validation
    return edge / b


def fractional_kelly_fraction(
    p_true: float, decimal_odds: float, kelly_multiplier: float = 0.25
) -> float:
    """Fractional Kelly::

        Fraction = (Edge / (Decimal_Odds - 1)) * Kelly_Multiplier
    """
    k = _require_finite("kelly_multiplier", kelly_multiplier)
    if not 0.0 < k <= 1.0:
        raise StakingInputError("kelly_multiplier must be in (0, 1]")
    return full_kelly_fraction(p_true, decimal_odds) * k


def _round_down_cents(amount: float) -> float:
    """Round DOWN to the cent so rounding never increases risk."""
    if amount <= 0:
        return 0.0
    # Add a tiny epsilon to protect against float artefacts like 9.999999999
    return math.floor(amount * 100.0 + 1e-9) / 100.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_stake(
    bankroll: float,
    p_true: float,
    decimal_odds: float,
    config: Optional[StakingConfig] = None,
    ticket_id: Optional[str] = None,
) -> StakeRecommendation:
    """Compute the recommended dollar risk for ONE parlay ticket.

    Parameters
    ----------
    bankroll:
        Current total bankroll in dollars (>= 0).
    p_true:
        Pipeline's compounded true win probability for the whole parlay.
    decimal_odds:
        Compounded bookmaker decimal odds for the whole parlay (> 1.0).
    config:
        :class:`StakingConfig`; defaults to quarter-Kelly with a 5 % cap.
    ticket_id:
        Optional identifier echoed back in the result for logging/joins.

    Returns
    -------
    StakeRecommendation
        Full audit trail. ``stake_dollars`` is **0.0** whenever the edge is
        non-positive (or below ``config.min_edge``), or the bankroll is zero.

    Raises
    ------
    StakingInputError
        On invalid inputs. Use :func:`safe_compute_stake` for a non-raising
        variant suitable for bulk loops.
    """
    cfg = config or StakingConfig()
    b = _validate_bankroll(bankroll)
    p = _validate_probability(p_true)
    d = _validate_decimal_odds(decimal_odds)

    edge = p * d - 1.0
    implied = 1.0 / d
    american = decimal_to_american(d)

    def _result(
        raw_fraction: float,
        capped_fraction: float,
        stake: float,
        reason: str,
        cap_applied: bool = False,
        skipped: bool = False,
    ) -> StakeRecommendation:
        return StakeRecommendation(
            bankroll=b,
            p_true=p,
            decimal_odds=d,
            american_odds=american,
            implied_prob=implied,
            edge=edge,
            mode=cfg.mode,
            raw_fraction=raw_fraction,
            capped_fraction=capped_fraction,
            stake_dollars=stake,
            potential_profit=stake * (d - 1.0),
            potential_payout=stake * d,
            expected_value=stake * edge,
            cap_applied=cap_applied,
            skipped=skipped,
            reason=reason,
            ticket_id=ticket_id,
        )

    # ---- Safety boundary 1: never bet a non-positive (or sub-threshold) edge
    if edge <= 0.0:
        return _result(0.0, 0.0, 0.0, f"Negative/zero edge ({edge:+.4f}); no bet", skipped=True)
    if edge < cfg.min_edge:
        return _result(
            0.0, 0.0, 0.0,
            f"Edge {edge:+.4f} below min_edge threshold {cfg.min_edge:+.4f}; no bet",
            skipped=True,
        )

    # ---- Safety boundary 2: empty bankroll
    if b <= 0.0:
        return _result(0.0, 0.0, 0.0, "Bankroll is $0; no bet", skipped=True)

    # ---- Mode-specific raw fraction -------------------------------------
    if cfg.mode is StakingMode.FLAT:
        if cfg.flat_dollars is not None:
            raw_fraction = cfg.flat_dollars / b
            basis = f"flat ${cfg.flat_dollars:,.2f} unit"
        else:
            raw_fraction = cfg.flat_pct
            basis = f"flat {cfg.flat_pct:.2%} unit"
    else:  # KELLY
        raw_fraction = (edge / (d - 1.0)) * cfg.kelly_multiplier
        basis = f"{cfg.kelly_multiplier:.2f}x Kelly (full Kelly {edge / (d - 1.0):.2%})"

    # ---- Safety boundary 3: hard per-ticket cap --------------------------
    cap_applied = False
    capped_fraction = raw_fraction
    if capped_fraction > cfg.max_stake_pct:
        capped_fraction = cfg.max_stake_pct
        cap_applied = True
    capped_fraction = max(capped_fraction, 0.0)

    stake = b * capped_fraction

    # ---- Safety boundary 4: book minimum ---------------------------------
    if cfg.min_stake_dollars > 0 and 0 < stake < cfg.min_stake_dollars:
        if cfg.enforce_min_stake:
            # Raise to the minimum, but NEVER through the hard cap.
            max_allowed = b * cfg.max_stake_pct
            if cfg.min_stake_dollars <= max_allowed:
                stake = cfg.min_stake_dollars
                capped_fraction = stake / b
                basis += f"; raised to ${cfg.min_stake_dollars:,.2f} book minimum"
            else:
                return _result(
                    raw_fraction, 0.0, 0.0,
                    f"Book minimum ${cfg.min_stake_dollars:,.2f} exceeds {cfg.max_stake_pct:.0%} cap; no bet",
                    skipped=True,
                )
        else:
            return _result(
                raw_fraction, 0.0, 0.0,
                f"Stake ${stake:,.2f} below book minimum ${cfg.min_stake_dollars:,.2f}; no bet",
                skipped=True,
            )

    # ---- Safety boundary 5: round DOWN to cents --------------------------
    if cfg.round_to_cents:
        stake = _round_down_cents(stake)
        capped_fraction = stake / b if b > 0 else 0.0

    if stake <= 0.0:
        return _result(raw_fraction, 0.0, 0.0, "Stake rounds to $0.00; no bet", skipped=True)

    reason = basis
    if cap_applied:
        reason += f"; CAPPED at {cfg.max_stake_pct:.0%} of bankroll"
    return _result(raw_fraction, capped_fraction, stake, reason, cap_applied=cap_applied)


def safe_compute_stake(
    bankroll: float,
    p_true: float,
    decimal_odds: float,
    config: Optional[StakingConfig] = None,
    ticket_id: Optional[str] = None,
) -> StakeRecommendation:
    """Non-raising wrapper around :func:`compute_stake` for bulk loops.

    On ANY exception the function logs the problem and returns a zero-stake
    :class:`StakeRecommendation` with ``error`` populated, so a single
    malformed ticket never aborts a weekly run or a long backtest.
    """
    cfg = config or StakingConfig()
    try:
        return compute_stake(bankroll, p_true, decimal_odds, cfg, ticket_id)
    except StakingInputError as exc:
        logger.warning("Staking input rejected for ticket %s: %s", ticket_id, exc)
        err = str(exc)
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Unexpected staking failure for ticket %s", ticket_id)
        err = f"{type(exc).__name__}: {exc}"

    # Build a best-effort zero result without re-validating bad inputs.
    def _soft(v: Any, default: float = 0.0) -> float:
        try:
            f = float(v)
            return default if (math.isnan(f) or math.isinf(f)) else f
        except (TypeError, ValueError):
            return default

    d = _soft(decimal_odds, 0.0)
    return StakeRecommendation(
        bankroll=_soft(bankroll),
        p_true=_soft(p_true),
        decimal_odds=d,
        american_odds=0,
        implied_prob=(1.0 / d) if d > 1.0 else 0.0,
        edge=0.0,
        mode=cfg.mode,
        raw_fraction=0.0,
        capped_fraction=0.0,
        stake_dollars=0.0,
        potential_profit=0.0,
        potential_payout=0.0,
        expected_value=0.0,
        cap_applied=False,
        skipped=True,
        reason="Invalid input; no bet",
        ticket_id=ticket_id,
        error=err,
    )


# ---------------------------------------------------------------------------
# Bulk helpers for weekly_reporter.py and backtester.py
# ---------------------------------------------------------------------------


def _extract_odds(ticket: Dict[str, Any]) -> float:
    """Pull decimal odds from a ticket dict, deriving from American if needed."""
    for key in ("decimal_odds", "parlay_decimal_odds", "odds_decimal"):
        if ticket.get(key) is not None:
            return float(ticket[key])
    for key in ("american_odds", "parlay_american_odds", "odds_american"):
        if ticket.get(key) is not None:
            return american_to_decimal(ticket[key])
    legs = ticket.get("legs")
    if legs:
        leg_odds = []
        for leg in legs:
            if leg.get("decimal_odds") is not None:
                leg_odds.append(float(leg["decimal_odds"]))
            elif leg.get("american_odds") is not None:
                leg_odds.append(american_to_decimal(leg["american_odds"]))
        if leg_odds and len(leg_odds) == len(legs):
            return compound_decimal_odds(leg_odds)
    raise StakingInputError("Ticket has no decimal_odds / american_odds / per-leg odds")


def _extract_p_true(ticket: Dict[str, Any]) -> float:
    """Pull P_true from a ticket dict, compounding per-leg probabilities if needed."""
    for key in ("p_true", "true_prob", "true_probability", "parlay_p_true"):
        if ticket.get(key) is not None:
            return float(ticket[key])
    legs = ticket.get("legs")
    if legs:
        probs = [leg.get("p_true", leg.get("true_prob")) for leg in legs]
        if all(p is not None for p in probs):
            return compound_probability([float(p) for p in probs])
    raise StakingInputError("Ticket has no p_true / true_prob / per-leg probabilities")


def compute_stakes_for_tickets(
    tickets: Sequence[Dict[str, Any]],
    bankroll: float,
    config: Optional[StakingConfig] = None,
    id_key: str = "ticket_id",
) -> List[StakeRecommendation]:
    """Vectorised-style convenience: stake every ticket from ``parlay_finder``.

    Each ``ticket`` is a dict. The function looks for ``decimal_odds`` (or
    ``american_odds``, or per-leg odds) and ``p_true`` (or per-leg
    probabilities). Bad tickets yield zero-stake results with ``error`` set
    rather than raising, so the caller always gets ``len(tickets)`` results
    back in the same order.
    """
    cfg = config or StakingConfig()
    results: List[StakeRecommendation] = []
    for idx, ticket in enumerate(tickets):
        tid = str(ticket.get(id_key, idx))
        try:
            d = _extract_odds(ticket)
            p = _extract_p_true(ticket)
        except (StakingInputError, TypeError, ValueError) as exc:
            logger.warning("Skipping ticket %s: %s", tid, exc)
            results.append(safe_compute_stake(bankroll, float("nan"), float("nan"), cfg, tid))
            results[-1].error = str(exc)
            continue
        results.append(safe_compute_stake(bankroll, p, d, cfg, tid))
    return results


def apply_portfolio_cap(
    recommendations: Sequence[StakeRecommendation],
    bankroll: float,
    max_total_pct: float = 0.15,
) -> List[StakeRecommendation]:
    """Scale down a slate of stakes so TOTAL weekly exposure <= ``max_total_pct``.

    Individual tickets are already capped at ``max_stake_pct``; this guards
    against a week with ten +EV tickets that would collectively risk 50 % of
    the bankroll. Stakes are scaled proportionally and re-rounded down, and
    each affected recommendation's ``reason`` is annotated.

    Returns NEW recommendation objects; inputs are not mutated.
    """
    b = _validate_bankroll(bankroll)
    _require_finite("max_total_pct", max_total_pct)
    if not 0.0 < max_total_pct <= 1.0:
        raise StakingInputError("max_total_pct must be in (0, 1]")

    total = sum(r.stake_dollars for r in recommendations)
    limit = b * max_total_pct
    if total <= limit or total <= 0:
        return [StakeRecommendation(**asdict(r)) for r in recommendations]

    scale = limit / total
    out: List[StakeRecommendation] = []
    for r in recommendations:
        data = asdict(r)
        if r.stake_dollars > 0:
            new_stake = _round_down_cents(r.stake_dollars * scale)
            data.update(
                stake_dollars=new_stake,
                capped_fraction=(new_stake / b) if b > 0 else 0.0,
                potential_profit=new_stake * (r.decimal_odds - 1.0),
                potential_payout=new_stake * r.decimal_odds,
                expected_value=new_stake * r.edge,
                cap_applied=True,
                reason=r.reason + f"; portfolio-scaled x{scale:.3f} to respect {max_total_pct:.0%} weekly cap",
            )
        out.append(StakeRecommendation(**data))
    return out


# ---------------------------------------------------------------------------
# Bankroll tracker for the backtester loop
# ---------------------------------------------------------------------------


@dataclass
class BankrollTracker:
    """Mutable ledger that ``backtester.py`` can step week by week.

    Typical loop::

        tracker = BankrollTracker(starting_bankroll=1000, config=cfg)
        for week in weeks:
            for ticket in parlay_finder.find(week):
                rec = tracker.stake(ticket["p_true"], ticket["decimal_odds"], ticket["id"])
                won = simulate_outcome(ticket)
                tracker.settle(rec, won)
        print(tracker.summary())
    """

    starting_bankroll: float
    config: StakingConfig = field(default_factory=StakingConfig)
    bankroll: float = field(init=False)
    history: List[Dict[str, Any]] = field(default_factory=list)
    peak_bankroll: float = field(init=False)
    max_drawdown: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.starting_bankroll = _validate_bankroll(self.starting_bankroll)
        self.bankroll = self.starting_bankroll
        self.peak_bankroll = self.starting_bankroll

    def stake(
        self, p_true: float, decimal_odds: float, ticket_id: Optional[str] = None
    ) -> StakeRecommendation:
        """Recommend a stake against the CURRENT bankroll (does not deduct)."""
        return safe_compute_stake(self.bankroll, p_true, decimal_odds, self.config, ticket_id)

    def settle(self, rec: StakeRecommendation, won: bool, label: Optional[str] = None) -> float:
        """Apply a plain win/loss outcome to the bankroll and log it.

        Returns the realised profit/loss for the ticket (negative on a loss).
        For pushes or partially voided parlays compute the P&L yourself and
        use :meth:`record`.
        """
        if rec.stake_dollars <= 0:
            return self.record(rec, 0.0, None, label, outcome="skip")
        if won:
            return self.record(rec, rec.potential_profit, True, label, outcome="win")
        return self.record(rec, -rec.stake_dollars, False, label, outcome="loss")

    def record(
        self,
        rec: StakeRecommendation,
        pnl: float,
        won: Optional[bool],
        label: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> float:
        """Apply an arbitrary realised P&L (win, loss, push, voided legs) and log it.

        ``won`` is True/False for decided tickets and None for pushes/skips.
        Returns ``pnl`` unchanged for convenience.
        """
        pnl = float(pnl)
        if math.isnan(pnl) or math.isinf(pnl):
            raise StakingInputError("pnl must be finite")
        if pnl < -rec.stake_dollars - 1e-9:
            raise StakingInputError(f"pnl {pnl:.2f} cannot lose more than the stake {rec.stake_dollars:.2f}")
        self.bankroll = max(0.0, self.bankroll + pnl)
        self.peak_bankroll = max(self.peak_bankroll, self.bankroll)
        if self.peak_bankroll > 0:
            dd = (self.peak_bankroll - self.bankroll) / self.peak_bankroll
            self.max_drawdown = max(self.max_drawdown, dd)
        if outcome is None:
            outcome = "skip" if rec.stake_dollars <= 0 else ("win" if won else "loss" if won is False else "push")
        self.history.append(
            {
                "ticket_id": rec.ticket_id,
                "label": label,
                "mode": rec.mode.value,
                "stake": rec.stake_dollars,
                "decimal_odds": rec.decimal_odds,
                "p_true": rec.p_true,
                "edge": rec.edge,
                "won": won if rec.stake_dollars > 0 else None,
                "outcome": outcome,
                "pnl": pnl,
                "bankroll_after": self.bankroll,
            }
        )
        return pnl

    def summary(self) -> Dict[str, Any]:
        """Headline performance numbers for the simulation."""
        bets = [h for h in self.history if h["stake"] > 0]
        wins = sum(1 for h in bets if h["won"] is True)
        pushes = sum(1 for h in bets if h["won"] is None)
        decided = len(bets) - pushes
        total_staked = sum(h["stake"] for h in bets)
        total_pnl = sum(h["pnl"] for h in bets)
        return {
            "starting_bankroll": self.starting_bankroll,
            "ending_bankroll": self.bankroll,
            "net_profit": self.bankroll - self.starting_bankroll,
            "roi_on_turnover": (total_pnl / total_staked) if total_staked > 0 else 0.0,
            "growth": (self.bankroll / self.starting_bankroll - 1.0)
            if self.starting_bankroll > 0 else 0.0,
            "tickets_placed": len(bets),
            "tickets_skipped": len(self.history) - len(bets),
            "pushes": pushes,
            "win_rate": (wins / decided) if decided else 0.0,
            "total_staked": total_staked,
            "peak_bankroll": self.peak_bankroll,
            "max_drawdown": self.max_drawdown,
            "mode": self.config.mode.value,
        }


# ---------------------------------------------------------------------------
# Built-in self-test (python3 staking_engine.py --selftest)
# ---------------------------------------------------------------------------


def _selftest() -> int:
    """Lightweight assertions covering every safety boundary. Returns exit code."""
    failures = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal failures
        if cond:
            print(f"  PASS  {msg}")
        else:
            failures += 1
            print(f"  FAIL  {msg}")

    print("staking_engine self-test")
    # Odds conversions
    check(abs(american_to_decimal(-110) - 1.909090909) < 1e-6, "-110 -> 1.9091 decimal")
    check(abs(american_to_decimal("+264") - 3.64) < 1e-9, "'+264' -> 3.64 decimal")
    check(decimal_to_american(3.64) == 264, "3.64 -> +264 american")
    check(decimal_to_american(1.9090909) == -110, "1.9091 -> -110 american")
    check(abs(compound_decimal_odds([1.909, 1.909]) - 3.644281) < 1e-5, "two -110 legs compound to ~3.644")

    # Edge formula
    check(abs(calculate_edge(0.31, 3.64) - (0.31 * 3.64 - 1)) < 1e-12, "edge = p*D - 1")
    check(calculate_edge(0.25, 3.64) < 0, "edge negative when p < implied")

    # Kelly maths: p=0.31, D=3.64 -> edge=0.1284, b=2.64, f*=0.048636
    f_full = full_kelly_fraction(0.31, 3.64)
    check(abs(f_full - 0.1284 / 2.64) < 1e-9, "full Kelly = edge / (D-1)")
    check(abs(fractional_kelly_fraction(0.31, 3.64, 0.25) - f_full * 0.25) < 1e-12, "quarter Kelly scales by 0.25")

    # Mode B happy path
    cfg_k = StakingConfig.fractional_kelly(0.25)
    rec = compute_stake(1000, 0.31, 3.64, cfg_k, "T1")
    check(rec.is_bet, "quarter Kelly places a bet on +EV ticket")
    check(abs(rec.stake_dollars - 12.15) < 1e-9, f"quarter Kelly stake == $12.15 (got {rec.stake_dollars})")
    check(not rec.cap_applied, "no cap at 1.2% fraction")

    # Mode B huge edge -> cap at 5%
    rec = compute_stake(1000, 0.60, 3.64, StakingConfig.fractional_kelly(1.0))
    check(rec.cap_applied and abs(rec.stake_dollars - 50.0) < 1e-9, "full Kelly on 118% edge capped at $50 (5%)")

    # Negative edge -> $0
    rec = compute_stake(1000, 0.20, 3.64, cfg_k)
    check(rec.stake_dollars == 0.0 and rec.skipped, "negative edge -> $0")

    # Mode A percent
    rec = compute_stake(1000, 0.31, 3.64, StakingConfig.flat_percent(0.01))
    check(abs(rec.stake_dollars - 10.0) < 1e-9, "flat 1% of $1000 == $10")
    # Mode A dollars
    rec = compute_stake(1000, 0.31, 3.64, StakingConfig.flat_dollar(10))
    check(abs(rec.stake_dollars - 10.0) < 1e-9, "flat $10 unit == $10")
    # Mode A flat $ exceeding cap on small bankroll
    rec = compute_stake(100, 0.31, 3.64, StakingConfig.flat_dollar(10))
    check(rec.cap_applied and abs(rec.stake_dollars - 5.0) < 1e-9, "flat $10 on $100 bankroll capped to $5")
    # Mode A negative edge still $0
    rec = compute_stake(1000, 0.20, 3.64, StakingConfig.flat_percent(0.01))
    check(rec.stake_dollars == 0.0, "flat mode also refuses -EV tickets")

    # min_edge threshold
    rec = compute_stake(1000, 0.2775, 3.64, StakingConfig.fractional_kelly(0.25, min_edge=0.02))
    check(rec.stake_dollars == 0.0 and "min_edge" in rec.reason, "edge below min_edge -> $0")

    # Zero bankroll
    rec = compute_stake(0, 0.31, 3.64, cfg_k)
    check(rec.stake_dollars == 0.0, "zero bankroll -> $0")

    # Book minimum behaviour
    rec = compute_stake(1000, 0.31, 3.64, StakingConfig.fractional_kelly(0.05, min_stake_dollars=5.0))
    check(rec.stake_dollars == 0.0 and "below book minimum" in rec.reason, "sub-minimum stake skipped by default")
    rec = compute_stake(1000, 0.31, 3.64, StakingConfig.fractional_kelly(0.05, min_stake_dollars=5.0, enforce_min_stake=True))
    check(abs(rec.stake_dollars - 5.0) < 1e-9, "sub-minimum stake raised to $5 when enforced")

    # Invalid inputs raise / are swallowed safely
    try:
        compute_stake(1000, 1.2, 3.64)
        check(False, "p_true > 1 raises")
    except StakingInputError:
        check(True, "p_true > 1 raises StakingInputError")
    try:
        compute_stake(1000, 0.5, 0.9)
        check(False, "decimal odds <= 1 raises")
    except StakingInputError:
        check(True, "decimal odds <= 1 raises StakingInputError")
    try:
        compute_stake(-5, 0.5, 2.0)
        check(False, "negative bankroll raises")
    except StakingInputError:
        check(True, "negative bankroll raises StakingInputError")
    try:
        StakingConfig(kelly_multiplier=0)
        check(False, "kelly_multiplier 0 rejected")
    except StakingInputError:
        check(True, "kelly_multiplier 0 rejected")
    rec = safe_compute_stake(1000, float("nan"), 3.64, cfg_k, "BAD")
    check(rec.stake_dollars == 0.0 and rec.error is not None, "safe_compute_stake swallows NaN and records error")

    # Bulk helpers
    tickets = [
        {"ticket_id": "A", "p_true": 0.31, "decimal_odds": 3.64},
        {"ticket_id": "B", "p_true": 0.20, "american_odds": "+264"},
        {"ticket_id": "C", "legs": [{"p_true": 0.56, "american_odds": -110}, {"p_true": 0.55, "american_odds": -110}]},
        {"ticket_id": "D"},  # malformed
    ]
    recs = compute_stakes_for_tickets(tickets, 1000, cfg_k)
    check(len(recs) == 4, "bulk helper returns one result per ticket")
    check(recs[0].is_bet and not recs[1].is_bet, "bulk: A bet, B skipped (-EV)")
    check(recs[2].is_bet, "bulk: per-leg compounding produces a bet")
    check(recs[3].error is not None and recs[3].stake_dollars == 0.0, "bulk: malformed ticket -> error, $0")

    # Portfolio cap
    many = [compute_stake(1000, 0.60, 3.64, StakingConfig.fractional_kelly(1.0), str(i)) for i in range(5)]
    check(abs(sum(r.stake_dollars for r in many) - 250.0) < 1e-9, "five capped tickets risk $250 before portfolio cap")
    scaled = apply_portfolio_cap(many, 1000, 0.15)
    check(sum(r.stake_dollars for r in scaled) <= 150.0 + 1e-9, "portfolio cap limits total to <= $150")
    check(all(r.cap_applied for r in scaled), "portfolio-scaled tickets flagged cap_applied")

    # Bankroll tracker
    tr = BankrollTracker(1000, cfg_k)
    r1 = tr.stake(0.31, 3.64, "W1")
    tr.settle(r1, won=True)
    check(abs(tr.bankroll - (1000 + 12.15 * 2.64)) < 1e-6, "tracker credits profit on win")
    r2 = tr.stake(0.31, 3.64, "W2")
    tr.settle(r2, won=False)
    check(abs(tr.bankroll - (1000 + 12.15 * 2.64 - r2.stake_dollars)) < 1e-6, "tracker debits stake on loss")
    s = tr.summary()
    check(s["tickets_placed"] == 2 and abs(s["win_rate"] - 0.5) < 1e-12, "tracker summary counts 2 bets, 50% win rate")
    r3 = tr.stake(0.31, 3.64, "W3")
    before = tr.bankroll
    tr.record(r3, 0.0, None, outcome="push")
    check(tr.bankroll == before and tr.summary()["pushes"] == 1 and abs(tr.summary()["win_rate"] - 0.5) < 1e-12,
          "record() push leaves bankroll unchanged and is excluded from win rate")
    r4 = tr.stake(0.31, 3.64, "W4")
    tr.record(r4, r4.stake_dollars * 0.9, True, outcome="win")
    check(abs(tr.bankroll - (before + r4.stake_dollars * 0.9)) < 1e-9, "record() applies a reduced (voided-leg) payout")
    try:
        tr.record(r4, -r4.stake_dollars * 2, False)
        check(False, "record() rejects losing more than the stake")
    except StakingInputError:
        check(True, "record() rejects losing more than the stake")

    # Serialisation round-trip
    cfg_rt = StakingConfig.from_dict(json.loads(json.dumps(cfg_k.to_dict())))
    check(cfg_rt == cfg_k, "StakingConfig JSON round-trip")
    json.dumps(rec.to_dict())  # must not raise
    check(True, "StakeRecommendation.to_dict() is JSON serialisable")

    print(f"\n{'ALL TESTS PASSED' if failures == 0 else f'{failures} TEST(S) FAILED'}")
    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="staking_engine",
        description="Compute a recommended parlay stake (Flat or Fractional Kelly).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bankroll", type=float, help="Current bankroll in dollars")
    p.add_argument("--p-true", type=float, dest="p_true", help="Compounded true win probability (0-1)")
    odds = p.add_mutually_exclusive_group()
    odds.add_argument("--decimal-odds", type=float, dest="decimal_odds", help="Compounded decimal odds (>1.0)")
    odds.add_argument("--american-odds", dest="american_odds", help="Compounded American odds, e.g. +264")
    p.add_argument("--mode", default="kelly", help="Staking mode: flat | kelly")
    p.add_argument("--flat-pct", type=float, default=0.01, dest="flat_pct", help="Mode A: fraction of bankroll per unit")
    p.add_argument("--flat-dollars", type=float, default=None, dest="flat_dollars", help="Mode A: fixed dollar unit (overrides --flat-pct)")
    p.add_argument("--kelly-multiplier", type=float, default=0.25, dest="kelly_multiplier", help="Mode B: fraction of full Kelly")
    p.add_argument("--max-stake-pct", type=float, default=0.05, dest="max_stake_pct", help="Hard cap per ticket as fraction of bankroll")
    p.add_argument("--min-edge", type=float, default=0.0, dest="min_edge", help="Minimum edge required to bet")
    p.add_argument("--json", action="store_true", help="Emit the recommendation as JSON")
    p.add_argument("--selftest", action="store_true", help="Run built-in unit tests and exit")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.selftest:
        return _selftest()

    missing = [n for n, v in (("--bankroll", args.bankroll), ("--p-true", args.p_true)) if v is None]
    if args.decimal_odds is None and args.american_odds is None:
        missing.append("--decimal-odds or --american-odds")
    if missing:
        print(f"error: missing required argument(s): {', '.join(missing)}", file=sys.stderr)
        print("hint: run with --help, or --selftest to run the unit tests", file=sys.stderr)
        return 2

    try:
        cfg = StakingConfig(
            mode=args.mode,
            flat_pct=args.flat_pct,
            flat_dollars=args.flat_dollars,
            kelly_multiplier=args.kelly_multiplier,
            max_stake_pct=args.max_stake_pct,
            min_edge=args.min_edge,
        )
        d = args.decimal_odds if args.decimal_odds is not None else american_to_decimal(args.american_odds)
        rec = compute_stake(args.bankroll, args.p_true, d, cfg)
    except StakingInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(rec.to_dict(), indent=2))
    else:
        print(f"Mode            : {rec.mode.label}")
        print(f"Bankroll        : ${rec.bankroll:,.2f}")
        print(f"P_true          : {rec.p_true:.4f} ({rec.p_true:.2%})")
        print(f"Decimal odds    : {rec.decimal_odds:.4f}  ({format_american(rec.american_odds)})")
        print(f"Implied prob    : {rec.implied_prob:.2%}")
        print(f"Edge            : {rec.edge:+.4f} ({rec.edge:+.2%})")
        print(f"Raw fraction    : {rec.raw_fraction:.4%}")
        print(f"Final fraction  : {rec.capped_fraction:.4%}{'  [CAPPED]' if rec.cap_applied else ''}")
        print(f"RECOMMENDED RISK: ${rec.stake_dollars:,.2f}")
        print(f"Potential profit: ${rec.potential_profit:,.2f}  (payout ${rec.potential_payout:,.2f})")
        print(f"Expected value  : ${rec.expected_value:+,.2f}")
        print(f"Reason          : {rec.reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
