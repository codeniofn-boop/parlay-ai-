#!/usr/bin/env python3
"""
market_registry.py
==================

The market layer of the EdgeBook AI pipeline, read from ``markets.json``.

The four game markets (``spread``, ``total``, ``moneyline``, ``team_total``)
are built in. Every player-prop market is **data**: an entry in
``markets.json`` describing how it is spelled, how it is modelled and how
the correlation table should treat it. Adding a prop type is a config
change, not a code change:

* ``parlay_finder.py`` asks this module which markets exist, how to parse a
  selection such as ``"Dak Prescott Over 264.5 Passing Yards"`` and whether a
  market is experimental or high-variance.
* ``build_lines.py`` uses it to accept the spellings in ``props_inputs.csv``.
* ``prop_model.py`` reads the ``parts`` / ``distribution`` fields to turn a
  projection into a probability.
* the correlation table references the ``cls`` and ``families`` fields.

Standard library only. ``python3 market_registry.py --selftest`` runs the
built-in tests; ``python3 market_registry.py`` lists the registered markets.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "RegistryError", "PropMarket", "Registry", "GAME_MARKETS", "KINDS", "DISTRIBUTIONS",
    "load_registry", "registry", "normalize_key", "format_prop_selection",
]

__version__ = "1.0.0"

REGISTRY_FILENAME = "markets.json"
GAME_MARKETS: Tuple[str, ...] = ("spread", "total", "moneyline", "team_total")
GAME_LABELS: Dict[str, str] = {"spread": "Spread", "total": "Total", "moneyline": "Moneyline", "team_total": "Team total"}
KINDS: Tuple[str, ...] = ("over_under", "yes_no")
DISTRIBUTIONS: Tuple[str, ...] = ("normal", "gamma", "negbin", "poisson", "first_td")
WILDCARDS: Dict[str, str] = {"props": "every prop market", "game": "the four game markets", "all": "everything"}


class RegistryError(ValueError):
    """Raised when markets.json is missing, unreadable or inconsistent."""


def normalize_key(text: str) -> str:
    """``"Rush + Rec Yds"`` -> ``"rush_rec_yds"``: lower-case, alphanumerics only, single underscores."""
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


@dataclass(frozen=True)
class PropMarket:
    """One player-prop market as declared in markets.json."""

    key: str
    label: str
    kind: str                                   # over_under | yes_no
    parts: Tuple[Tuple[str, str], ...]          # ((stat_column, usage_column), ...)
    distribution: str                           # normal | gamma | negbin | poisson | first_td
    aliases: Tuple[str, ...] = ()
    sd_intercept: float = 0.0
    sd_slope: float = 0.0
    dispersion: Optional[float] = None
    rate_pseudo: Optional[float] = None
    families: Tuple[str, ...] = ()
    cls: str = ""
    positions: Tuple[str, ...] = ()
    high_variance: bool = False
    experimental: bool = True

    @property
    def is_yes_no(self) -> bool:
        return self.kind == "yes_no"

    @property
    def stat_columns(self) -> Tuple[str, ...]:
        return tuple(stat for stat, _usage in self.parts)

    @property
    def usage_columns(self) -> Tuple[str, ...]:
        return tuple(usage for _stat, usage in self.parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "label": self.label, "kind": self.kind,
            "parts": [{"stat": s, "usage": u} for s, u in self.parts], "distribution": self.distribution,
            "aliases": list(self.aliases), "sd_intercept": self.sd_intercept, "sd_slope": self.sd_slope,
            "dispersion": self.dispersion, "rate_pseudo": self.rate_pseudo, "families": list(self.families),
            "cls": self.cls, "positions": list(self.positions), "high_variance": self.high_variance,
            "experimental": self.experimental,
        }


def _parse_prop(key: str, raw: Any) -> PropMarket:
    if not isinstance(raw, dict):
        raise RegistryError(f"prop_markets.{key} must be an object")
    label = str(raw.get("label") or "").strip()
    if not label:
        raise RegistryError(f"prop_markets.{key}: label is required")
    kind = str(raw.get("kind") or "over_under").lower()
    if kind not in KINDS:
        raise RegistryError(f"prop_markets.{key}: kind must be one of {', '.join(KINDS)}, got '{kind}'")
    dist = str(raw.get("distribution") or "").lower()
    if dist not in DISTRIBUTIONS:
        raise RegistryError(f"prop_markets.{key}: distribution must be one of {', '.join(DISTRIBUTIONS)}, got '{dist}'")
    parts_raw = raw.get("parts") or []
    if not isinstance(parts_raw, list) or not parts_raw:
        raise RegistryError(f"prop_markets.{key}: parts must be a non-empty list of {{stat, usage}}")
    parts: List[Tuple[str, str]] = []
    for p in parts_raw:
        if not isinstance(p, dict) or not p.get("stat") or not p.get("usage"):
            raise RegistryError(f"prop_markets.{key}: every part needs 'stat' and 'usage'")
        parts.append((str(p["stat"]), str(p["usage"])))
    if dist in ("normal", "gamma") and (raw.get("sd_intercept") is None or raw.get("sd_slope") is None):
        raise RegistryError(f"prop_markets.{key}: {dist} markets need sd_intercept and sd_slope")
    if dist == "negbin" and not raw.get("dispersion"):
        raise RegistryError(f"prop_markets.{key}: negbin markets need a positive dispersion")
    if dist == "first_td" and kind != "yes_no":
        raise RegistryError(f"prop_markets.{key}: first_td distribution is only valid for a yes_no market")
    aliases = tuple(str(a) for a in (raw.get("aliases") or []))
    try:
        return PropMarket(
            key=key, label=label, kind=kind, parts=tuple(parts), distribution=dist, aliases=aliases,
            sd_intercept=float(raw.get("sd_intercept") or 0.0), sd_slope=float(raw.get("sd_slope") or 0.0),
            dispersion=(float(raw["dispersion"]) if raw.get("dispersion") is not None else None),
            rate_pseudo=(float(raw["rate_pseudo"]) if raw.get("rate_pseudo") is not None else None),
            families=tuple(str(f) for f in (raw.get("families") or [])), cls=str(raw.get("cls") or key),
            positions=tuple(str(p).upper() for p in (raw.get("positions") or [])),
            high_variance=bool(raw.get("high_variance", False)), experimental=bool(raw.get("experimental", True)),
        )
    except (TypeError, ValueError) as exc:
        raise RegistryError(f"prop_markets.{key}: bad numeric field ({exc})") from exc


@dataclass
class Registry:
    """All markets the pipeline knows about, plus the prop model settings."""

    props: Dict[str, PropMarket] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    path: str = ""
    _lookup: Dict[str, str] = field(default_factory=dict, repr=False)
    _yes_no_re: Optional["re.Pattern[str]"] = field(default=None, repr=False)
    _over_under_re: Optional["re.Pattern[str]"] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._lookup = {}
        for key in GAME_MARKETS:
            self._lookup[normalize_key(key)] = key
            self._lookup[normalize_key(GAME_LABELS[key])] = key
        self._lookup[normalize_key("ML")] = "moneyline"
        self._lookup[normalize_key("h2h")] = "moneyline"
        self._lookup[normalize_key("team totals")] = "team_total"
        self._lookup[normalize_key("totals")] = "total"
        self._lookup[normalize_key("spreads")] = "spread"
        for key, pm in self.props.items():
            for name in (key, pm.label, *pm.aliases):
                norm = normalize_key(name)
                owner = self._lookup.get(norm)
                if owner is not None and owner != key:
                    raise RegistryError(f"'{name}' is claimed by both '{owner}' and '{key}' in markets.json")
                self._lookup[norm] = key
        yes_no_labels = sorted({n for pm in self.props.values() if pm.is_yes_no for n in (pm.label, *pm.aliases)},
                               key=len, reverse=True)
        ou_labels = sorted({n for pm in self.props.values() if not pm.is_yes_no for n in (pm.label, *pm.aliases)},
                           key=len, reverse=True)
        if yes_no_labels:
            alt = "|".join(re.escape(n).replace(r"\ ", r"\s+") for n in yes_no_labels)
            self._yes_no_re = re.compile(rf"^(?P<player>.+?)\s+(?P<no>no\s+)?(?P<label>{alt})$", re.IGNORECASE)
        if ou_labels:
            alt = "|".join(re.escape(n).replace(r"\ ", r"\s+") for n in ou_labels)
            self._over_under_re = re.compile(
                rf"^(?P<player>.+?)\s+(?P<dir>over|under)\s+(?P<line>\d+(?:\.\d+)?)\s+(?P<label>{alt})$", re.IGNORECASE)

    # ---- lookups -----------------------------------------------------------

    @property
    def prop_keys(self) -> Tuple[str, ...]:
        return tuple(self.props)

    @property
    def market_keys(self) -> Tuple[str, ...]:
        """Every market key: the four game markets first, then the props in file order."""
        return GAME_MARKETS + self.prop_keys

    def is_prop(self, market: Optional[str]) -> bool:
        return bool(market) and str(market) in self.props

    def get(self, market: Optional[str]) -> Optional[PropMarket]:
        return self.props.get(str(market)) if market else None

    def resolve(self, text: Optional[str]) -> Optional[str]:
        """Any key, label or alias (case/spacing-insensitive) -> canonical market key, else None."""
        if text is None:
            return None
        return self._lookup.get(normalize_key(text))

    def label(self, market: str) -> str:
        pm = self.props.get(market)
        if pm is not None:
            return pm.label
        return GAME_LABELS.get(market, str(market).replace("_", " ").title())

    def expand_markets(self, markets: Iterable[str]) -> Tuple[str, ...]:
        """Resolve a list that may hold wildcards (``props``, ``game``, ``all``) and aliases; keeps order, drops dupes."""
        out: List[str] = []
        for m in markets:
            text = str(m).strip().lower()
            if text == "all":
                keys: Tuple[str, ...] = self.market_keys
            elif text == "props":
                keys = self.prop_keys
            elif text == "game":
                keys = GAME_MARKETS
            else:
                key = self.resolve(text)
                if key is None:
                    raise RegistryError(f"unknown market '{m}'; known: {', '.join(self.market_keys)} (or props/game/all)")
                keys = (key,)
            for k in keys:
                if k not in out:
                    out.append(k)
        return tuple(out)

    def experimental(self, market: Optional[str]) -> bool:
        pm = self.get(market)
        return bool(pm and pm.experimental)

    def high_variance(self, market: Optional[str]) -> bool:
        pm = self.get(market)
        return bool(pm and pm.high_variance)

    # ---- selection text ----------------------------------------------------

    def parse_prop_selection(self, selection: str) -> Optional[Tuple[str, str, Optional[float], str]]:
        """``"Dak Prescott Over 264.5 Passing Yards"`` -> ``(market, player, line, direction)``.

        ``"CeeDee Lamb Anytime TD"`` -> ``("anytime_td", "CeeDee Lamb", None, "Yes")`` and
        ``"CeeDee Lamb No Anytime TD"`` -> direction ``"No"``. Returns None when the text
        is not a prop selection.
        """
        text = " ".join(str(selection).split())
        if self._over_under_re is not None:
            m = self._over_under_re.match(text)
            if m:
                key = self.resolve(m.group("label"))
                if key is not None:
                    return key, m.group("player").strip(), float(m.group("line")), m.group("dir").title()
        if self._yes_no_re is not None:
            m = self._yes_no_re.match(text)
            if m:
                key = self.resolve(m.group("label"))
                if key is not None:
                    return key, m.group("player").strip(), None, ("No" if m.group("no") else "Yes")
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "game_markets": list(GAME_MARKETS), "prop_markets": {k: v.to_dict() for k, v in self.props.items()},
                "model": dict(self.model)}


def format_prop_selection(market: PropMarket, player: str, direction: str, line: Optional[float]) -> str:
    """The canonical selection text the parser accepts back."""
    player = " ".join(str(player).split())
    if market.is_yes_no:
        return f"{player} {market.label}" if str(direction).lower() != "no" else f"{player} No {market.label}"
    if line is None:
        raise RegistryError(f"{market.key} needs a line")
    return f"{player} {str(direction).title()} {line:g} {market.label}"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_CACHE: Dict[str, Tuple[float, Registry]] = {}


def default_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), REGISTRY_FILENAME)


def load_registry(path: Optional[str] = None) -> Registry:
    """Read markets.json (default: beside this file). Cached per path and modification time."""
    path = path or default_path()
    try:
        mtime = os.path.getmtime(path)
    except OSError as exc:
        raise RegistryError(f"{path} not found; markets.json must sit next to the pipeline scripts ({exc})") from exc
    cached = _CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"Could not read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RegistryError(f"{path} must contain a JSON object")
    props: Dict[str, PropMarket] = {}
    raw_props = data.get("prop_markets") or {}
    if not isinstance(raw_props, dict):
        raise RegistryError("prop_markets must be an object keyed by market name")
    for key, raw in raw_props.items():
        if str(key).startswith("_"):
            continue
        norm = normalize_key(key)
        if norm != key:
            raise RegistryError(f"prop market key '{key}' must be lower_snake_case (e.g. '{norm}')")
        if norm in GAME_MARKETS:
            raise RegistryError(f"'{key}' is a built-in game market and cannot be redefined")
        props[norm] = _parse_prop(norm, raw)
    model = data.get("model") or {}
    if not isinstance(model, dict):
        raise RegistryError("model must be an object")
    reg = Registry(props=props, model={k: v for k, v in model.items() if not str(k).startswith("_")}, path=path)
    _CACHE[path] = (mtime, reg)
    return reg


def registry() -> Registry:
    """The default registry (markets.json beside this file)."""
    return load_registry()


# ---------------------------------------------------------------------------
# Self-test and CLI
# ---------------------------------------------------------------------------


def _selftest() -> int:
    import tempfile

    failures = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal failures
        print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
        if not cond:
            failures += 1

    print("market_registry self-test")
    reg = registry()
    check(set(GAME_MARKETS) <= set(reg.market_keys), "game markets are always present")
    check(len(reg.prop_keys) >= 11 and "passing_yards" in reg.props and "anytime_td" in reg.props, f"{len(reg.prop_keys)} prop markets loaded")
    check(reg.resolve("pass yds") == "passing_yards" and reg.resolve("Passing Yards") == "passing_yards", "alias and label resolve")
    check(reg.resolve("Rush + Receiving Yards") == "rush_rec_yards", "punctuated alias resolves")
    check(reg.resolve("ML") == "moneyline" and reg.resolve("team totals") == "team_total", "game market aliases resolve")
    check(reg.resolve("made up market") is None, "unknown text resolves to None")
    check(reg.is_prop("receptions") and not reg.is_prop("spread") and not reg.is_prop(None), "is_prop")
    check(reg.label("spread") == "Spread" and reg.label("anytime_td") == "Anytime TD", "labels")
    check(reg.expand_markets(["spread", "props"])[0] == "spread" and len(reg.expand_markets(["props"])) == len(reg.prop_keys), "wildcard expansion")
    check(reg.expand_markets(["all"]) == reg.market_keys, "'all' expands to every market")
    try:
        reg.expand_markets(["nope"])
        check(False, "unknown market in list raises")
    except RegistryError:
        check(True, "unknown market in list raises RegistryError")

    pm = reg.props["passing_yards"]
    check(pm.parts == (("passing_yards", "attempts"),) and pm.distribution == "normal" and pm.experimental, "passing_yards spec")
    check(reg.props["rush_rec_yards"].stat_columns == ("rushing_yards", "receiving_yards"), "multi-part market")
    check(reg.props["first_td"].high_variance and reg.props["first_td"].is_yes_no, "first_td is high-variance yes/no")

    sel = format_prop_selection(pm, "Dak  Prescott", "over", 264.5)
    check(sel == "Dak Prescott Over 264.5 Passing Yards", f"format over/under -> {sel!r}")
    check(reg.parse_prop_selection(sel) == ("passing_yards", "Dak Prescott", 264.5, "Over"), "parse canonical over/under")
    check(reg.parse_prop_selection("dak prescott under 264.5 pass yds") == ("passing_yards", "dak prescott", 264.5, "Under"), "parse alias, lower-case")
    check(reg.parse_prop_selection("Travis Kelce Over 5.5 Receptions") == ("receptions", "Travis Kelce", 5.5, "Over"), "parse receptions")
    check(reg.parse_prop_selection("Saquon Barkley Over 105.5 Rush + Rec Yards") == ("rush_rec_yards", "Saquon Barkley", 105.5, "Over"), "parse rush + rec")
    atd = reg.props["anytime_td"]
    check(format_prop_selection(atd, "CeeDee Lamb", "Yes", None) == "CeeDee Lamb Anytime TD", "format yes/no")
    check(reg.parse_prop_selection("CeeDee Lamb Anytime TD") == ("anytime_td", "CeeDee Lamb", None, "Yes"), "parse anytime TD")
    check(reg.parse_prop_selection("CeeDee Lamb No Anytime TD") == ("anytime_td", "CeeDee Lamb", None, "No"), "parse No side")
    check(reg.parse_prop_selection("CeeDee Lamb first td") == ("first_td", "CeeDee Lamb", None, "Yes"), "parse first TD alias")
    check(reg.parse_prop_selection("Dallas Cowboys Over 24.5") is None, "team total is not a prop")
    check(reg.parse_prop_selection("Buffalo Bills -3.5") is None and reg.parse_prop_selection("Over 44.5") is None, "game markets are not props")
    check(reg.parse_prop_selection("Dak Prescott Over 264.5 Elephants") is None, "unknown label is not a prop")

    # A new prop type is a config change: write a registry with an extra market and load it.
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "markets.json")
        data = json.load(open(reg.path, "r", encoding="utf-8"))
        data["prop_markets"]["passing_attempts"] = {"label": "Passing Attempts", "aliases": ["pass att"], "kind": "over_under",
                                                     "parts": [{"stat": "attempts", "usage": "attempts"}], "distribution": "negbin",
                                                     "dispersion": 30, "families": ["pass_volume"], "cls": "pass_att"}
        json.dump(data, open(p, "w", encoding="utf-8"))
        r2 = load_registry(p)
        check("passing_attempts" in r2.props and r2.resolve("pass att") == "passing_attempts", "new market added through config only")
        check(r2.parse_prop_selection("Dak Prescott Over 35.5 Passing Attempts") == ("passing_attempts", "Dak Prescott", 35.5, "Over"), "new market parses")
        data["prop_markets"]["broken"] = {"label": "Broken", "kind": "over_under", "parts": [], "distribution": "normal"}
        json.dump(data, open(p, "w", encoding="utf-8"))
        os.utime(p, (os.path.getmtime(p) + 2, os.path.getmtime(p) + 2))
        try:
            load_registry(p)
            check(False, "invalid entry raises")
        except RegistryError as exc:
            check("broken" in str(exc), f"invalid entry raises RegistryError ({exc})")
        data["prop_markets"].pop("broken")
        data["prop_markets"]["dupe"] = {"label": "Receptions", "kind": "over_under", "parts": [{"stat": "receptions", "usage": "targets"}],
                                        "distribution": "negbin", "dispersion": 5}
        json.dump(data, open(p, "w", encoding="utf-8"))
        os.utime(p, (os.path.getmtime(p) + 4, os.path.getmtime(p) + 4))
        try:
            load_registry(p)
            check(False, "duplicate alias raises")
        except RegistryError:
            check(True, "duplicate label across markets raises RegistryError")
    print(f"{'ALL PASS' if not failures else f'{failures} FAILURE(S)'}")
    return 0 if failures == 0 else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in args:
        return _selftest()
    try:
        reg = registry()
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Registered markets ({reg.path})")
    for key in GAME_MARKETS:
        print(f"  {key:<18} {reg.label(key):<20} game market (built in)")
    for key, pm in reg.props.items():
        flags = ", ".join(f for f, on in (("experimental", pm.experimental), ("high-variance", pm.high_variance)) if on)
        print(f"  {key:<18} {pm.label:<20} {pm.kind:<10} {pm.distribution:<8} parts={'+'.join(pm.stat_columns)}"
              f"{'  [' + flags + ']' if flags else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
