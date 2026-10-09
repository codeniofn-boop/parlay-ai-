#!/usr/bin/env python3
"""
correlation_rules.py
====================

Rule 3 of the pipeline, as data: the same-game correlation table in
``correlation_rules.json``. Two legs from the same game are classified as

* ``positive``  - they tend to win together (allowed under ``positive_only``)
* ``negative``  - they pull apart (rejected under ``positive_only``)
* ``neutral``   - no useful link (rejected under ``positive_only``)
* ``exclusive`` - both cannot win (always rejected)
* ``redundant`` - they double-count one outcome (always rejected)

Each rule names the two legs by market class / direction, the relation
between them (same player, same team, same family, ...) and carries a short
reason that travels onto the ticket. Rules are tried top to bottom in both
leg orders; the first match decides; no match is ``neutral``.

``parlay_finder.leg_correlation`` delegates here. ``python3
correlation_rules.py --selftest`` checks one case per rule and fails when a
rule has no test, so the table and the tests move together.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import market_registry
from market_registry import GAME_MARKETS, RegistryError

__all__ = ["RuleError", "LegView", "Rule", "RuleTable", "leg_view", "load_rules", "rules", "LABELS", "RELATIONS", "evaluate"]

__version__ = "1.0.0"

RULES_FILENAME = "correlation_rules.json"
LABELS: Tuple[str, ...] = ("positive", "negative", "neutral", "exclusive", "redundant")
RELATIONS: Tuple[str, ...] = ("same_player", "other_player", "same_team", "other_team", "same_market", "same_family",
                              "same_direction", "opposite_direction")
KINDS: Tuple[str, ...] = ("side", "total", "team_total", "prop")
MARGIN_SD = 13.5


class RuleError(ValueError):
    """correlation_rules.json is missing, unreadable or inconsistent."""


@dataclass(frozen=True)
class LegView:
    """The few facts about a leg the rules can see."""

    market: str
    kind: str                       # side | total | team_total | prop
    cls: str                        # side | total | team_total | prop market class
    team: Optional[str]             # full team name (lower-cased for comparison)
    player: Optional[str]
    direction: Optional[str]        # "up" (Over / Yes / the team wins) | "down" (Under / No)
    families: Tuple[str, ...]
    favored_points: Optional[float] = None
    line: Optional[float] = None


def _norm(text: Optional[str]) -> Optional[str]:
    return " ".join(str(text).lower().split()) if text else None


def leg_view(side: Any) -> LegView:
    """Build a :class:`LegView` from a ``parlay_finder.MarketSide`` (duck-typed)."""
    reg = market_registry.registry()
    market = str(getattr(side, "market", ""))
    direction_raw = str(getattr(side, "direction", "") or "").lower()
    direction = "up" if direction_raw in ("over", "yes") else "down" if direction_raw in ("under", "no") else None
    favored: Optional[float] = None
    if market in ("spread", "moneyline"):
        kind = cls = "side"
        direction = "up"
        line = getattr(side, "line", None)
        if market == "spread" and line is not None:
            favored = -float(line)
        else:
            fair = getattr(side, "fair_prob", None)
            if fair is not None and 0.0 < float(fair) < 1.0:
                favored = statistics.NormalDist().inv_cdf(float(fair)) * MARGIN_SD
        families: Tuple[str, ...] = ()
    elif market == "total":
        kind = cls = "total"
        families = ()
    elif market == "team_total":
        kind = cls = "team_total"
        families = ()
    elif reg.is_prop(market):
        pm = reg.get(market)
        assert pm is not None
        kind, cls, families = "prop", pm.cls, tuple(pm.families)
    else:
        kind, cls, families = "prop", market, ()
    return LegView(market=market, kind=kind, cls=cls, team=_norm(getattr(side, "team", None)), player=_norm(getattr(side, "player", None)),
                   direction=direction, families=families, favored_points=favored, line=getattr(side, "line", None))


@dataclass(frozen=True)
class Rule:
    id: str
    a: Dict[str, Any]
    b: Dict[str, Any]
    relation: Tuple[str, ...]
    label: str
    reason: str
    favored_by_at_least: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "a": self.a, "b": self.b, "relation": list(self.relation), "label": self.label, "reason": self.reason,
                "favored_by_at_least": self.favored_by_at_least}


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _match_leg(cond: Dict[str, Any], leg: LegView, prop_classes: Iterable[str]) -> bool:
    kinds = _as_list(cond.get("kind"))
    if kinds and leg.kind not in kinds:
        return False
    classes = _as_list(cond.get("cls"))
    if classes:
        ok = False
        for c in classes:
            if c == "any" or c == leg.cls or (c == "any_prop" and leg.kind == "prop"):
                ok = True
                break
        if not ok:
            return False
    dirs = _as_list(cond.get("dir"))
    if dirs and leg.direction not in dirs:
        return False
    return True


def _relations_hold(rule: Rule, p: LegView, q: LegView) -> bool:
    for rel in rule.relation:
        if rel == "same_player":
            if not (p.player and q.player and p.player == q.player):
                return False
        elif rel == "other_player":
            if not (p.player and q.player and p.player != q.player):
                return False
        elif rel == "same_team":
            if not (p.team and q.team and p.team == q.team):
                return False
        elif rel == "other_team":
            if not (p.team and q.team and p.team != q.team):
                return False
        elif rel == "same_market":
            if p.market != q.market:
                return False
        elif rel == "same_family":
            if not (set(p.families) & set(q.families)):
                return False
        elif rel == "same_direction":
            if not (p.direction and q.direction and p.direction == q.direction):
                return False
        elif rel == "opposite_direction":
            if not (p.direction and q.direction and p.direction != q.direction):
                return False
    if rule.favored_by_at_least is not None:
        side = p if p.kind == "side" else q if q.kind == "side" else None
        if side is None or side.favored_points is None or side.favored_points < rule.favored_by_at_least:
            return False
    return True


class RuleTable:
    """The loaded table; ``evaluate`` classifies a same-game pair."""

    def __init__(self, rules: Sequence[Rule], path: str = "") -> None:
        self.rules = list(rules)
        self.path = path
        self._prop_classes = {pm.cls for pm in market_registry.registry().props.values()}

    @property
    def ids(self) -> List[str]:
        return [r.id for r in self.rules]

    def evaluate(self, x: LegView, y: LegView) -> Tuple[str, str, str]:
        """``(label, rule_id, reason)`` for two legs of the same game; no match -> neutral."""
        for rule in self.rules:
            for p, q in ((x, y), (y, x)):
                if _match_leg(rule.a, p, self._prop_classes) and _match_leg(rule.b, q, self._prop_classes) and _relations_hold(rule, p, q):
                    return rule.label, rule.id, rule.reason
        return "neutral", "", "no rule links these two legs"

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "rules": [r.to_dict() for r in self.rules]}


def _parse_rule(raw: Any, index: int, known_classes: Iterable[str]) -> Rule:
    if not isinstance(raw, dict):
        raise RuleError(f"rule #{index + 1} must be an object")
    rid = str(raw.get("id") or "").strip()
    if not rid:
        raise RuleError(f"rule #{index + 1} needs an id")
    label = str(raw.get("label") or "").lower()
    if label not in LABELS:
        raise RuleError(f"rule '{rid}': label must be one of {', '.join(LABELS)}, got '{label}'")
    relation = tuple(str(r) for r in _as_list(raw.get("relation")))
    unknown = [r for r in relation if r not in RELATIONS]
    if unknown:
        raise RuleError(f"rule '{rid}': unknown relation(s) {', '.join(unknown)}; known: {', '.join(RELATIONS)}")
    for side_name in ("a", "b"):
        cond = raw.get(side_name)
        if cond is None:
            cond = {}
        if not isinstance(cond, dict):
            raise RuleError(f"rule '{rid}': '{side_name}' must be an object")
        for k in cond:
            if k not in ("kind", "cls", "dir"):
                raise RuleError(f"rule '{rid}': unknown leg field '{k}' (use kind, cls, dir)")
        for k in _as_list(cond.get("kind")):
            if k not in KINDS:
                raise RuleError(f"rule '{rid}': unknown kind '{k}'")
        for c in _as_list(cond.get("cls")):
            if c not in ("any", "any_prop", "side", "total", "team_total") and c not in known_classes:
                raise RuleError(f"rule '{rid}': unknown market class '{c}' (not a cls in markets.json)")
        for d in _as_list(cond.get("dir")):
            if d not in ("up", "down"):
                raise RuleError(f"rule '{rid}': dir must be up or down")
    fav = raw.get("favored_by_at_least")
    try:
        fav_f = float(fav) if fav is not None else None
    except (TypeError, ValueError) as exc:
        raise RuleError(f"rule '{rid}': favored_by_at_least must be a number") from exc
    return Rule(id=rid, a=dict(raw.get("a") or {}), b=dict(raw.get("b") or {}), relation=relation, label=label,
                reason=str(raw.get("reason") or ""), favored_by_at_least=fav_f)


_CACHE: Dict[str, Tuple[float, RuleTable]] = {}


def default_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), RULES_FILENAME)


def load_rules(path: Optional[str] = None) -> RuleTable:
    path = path or default_path()
    try:
        mtime = os.path.getmtime(path)
    except OSError as exc:
        raise RuleError(f"{path} not found; correlation_rules.json must sit next to the pipeline scripts ({exc})") from exc
    cached = _CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuleError(f"Could not read {path}: {exc}") from exc
    raw_rules = data.get("rules") if isinstance(data, dict) else None
    if not isinstance(raw_rules, list) or not raw_rules:
        raise RuleError(f"{path} needs a non-empty 'rules' list")
    try:
        known = {pm.cls for pm in market_registry.registry().props.values()}
    except RegistryError as exc:
        raise RuleError(str(exc)) from exc
    parsed = [_parse_rule(r, i, known) for i, r in enumerate(raw_rules)]
    ids = [r.id for r in parsed]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise RuleError(f"duplicate rule id(s): {', '.join(dupes)}")
    table = RuleTable(parsed, path)
    _CACHE[path] = (mtime, table)
    return table


def rules() -> RuleTable:
    return load_rules()


def evaluate(a: Any, b: Any) -> Tuple[str, str, str]:
    """Classify two ``MarketSide``-like legs of the same game -> ``(label, rule_id, reason)``."""
    return rules().evaluate(leg_view(a), leg_view(b))


# ---------------------------------------------------------------------------
# Self-test: one case per rule, and every rule must have a case
# ---------------------------------------------------------------------------


def _selftest() -> int:
    from types import SimpleNamespace

    failures = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal failures
        print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
        if not cond:
            failures += 1

    def L(market: str, team: Optional[str] = None, player: Optional[str] = None, direction: Optional[str] = None,
          line: Optional[float] = None, odds: int = -110, fair: float = 0.5) -> SimpleNamespace:
        return SimpleNamespace(market=market, team=team, player=player, direction=direction, line=line, american_odds=odds, fair_prob=fair)

    print("correlation_rules self-test")
    table = rules()
    check(len(table.rules) >= 40 and len(set(table.ids)) == len(table.ids), f"{len(table.rules)} rules loaded with unique ids")

    H, A = "Dallas Cowboys", "Tampa Bay Buccaneers"
    cases: Dict[str, List[Tuple[Any, Any, str]]] = {
        "same-player-opposite-sides": [(L("passing_yards", H, "Dak", "Over", 264.5), L("passing_yards", H, "Dak", "Under", 264.5), "exclusive"),
                                       (L("anytime_td", H, "Lamb", "Yes"), L("anytime_td", H, "Lamb", "No"), "exclusive")],
        "same-player-same-family": [(L("rushing_yards", A, "Irving", "Over", 62.5), L("rush_rec_yards", A, "Irving", "Over", 84.5), "redundant"),
                                    (L("passing_yards", H, "Dak", "Over", 249.5), L("passing_yards", H, "Dak", "Over", 274.5), "redundant"),
                                    (L("receptions", H, "Lamb", "Over", 6.5), L("receiving_yards", H, "Lamb", "Under", 84.5), "redundant"),
                                    (L("anytime_td", H, "Lamb", "Yes"), L("first_td", H, "Lamb", "Yes"), "redundant")],
        "first-td-two-players": [(L("first_td", H, "Lamb", "Yes"), L("first_td", A, "Evans", "Yes"), "exclusive")],
        "same-player-different-families": [(L("passing_yards", H, "Dak", "Over", 264.5), L("passing_tds", H, "Dak", "Over", 1.5), "positive"),
                                           (L("receiving_yards", H, "Lamb", "Over", 84.5), L("anytime_td", H, "Lamb", "Yes"), "positive")],
        "same-player-mixed-directions": [(L("passing_yards", H, "Dak", "Over", 264.5), L("passing_tds", H, "Dak", "Under", 1.5), "negative")],
        "qb-passing-with-receiver": [(L("passing_yards", H, "Dak", "Over", 264.5), L("receiving_yards", H, "Lamb", "Over", 84.5), "positive"),
                                     (L("completions", H, "Dak", "Under", 22.5), L("receptions", H, "Lamb", "Under", 6.5), "positive")],
        "qb-passing-against-receiver": [(L("passing_yards", H, "Dak", "Over", 264.5), L("receiving_yards", H, "Lamb", "Under", 84.5), "negative")],
        "qb-passing-tds-with-receiver-td": [(L("passing_tds", H, "Dak", "Over", 1.5), L("anytime_td", H, "Lamb", "Yes"), "positive")],
        "two-receivers-both-over": [(L("receiving_yards", H, "Lamb", "Over", 84.5), L("receptions", H, "Pickens", "Over", 4.5), "negative")],
        "two-receivers-both-under": [(L("receiving_yards", H, "Lamb", "Under", 84.5), L("receiving_yards", H, "Pickens", "Under", 62.5), "positive")],
        "qb-passing-with-rusher": [(L("passing_yards", H, "Dak", "Over", 264.5), L("rushing_yards", H, "Williams", "Over", 58.5), "negative")],
        "rusher-with-own-team-side": [(L("rushing_yards", H, "Williams", "Over", 58.5), L("moneyline", H), "positive"),
                                      (L("rush_rec_yards", A, "Irving", "Over", 84.5), L("spread", A, line=8.5), "positive")],
        "rusher-against-opponent-side": [(L("rushing_yards", A, "Irving", "Over", 62.5), L("spread", H, line=-8.5), "negative")],
        "rusher-under-with-own-team-side": [(L("rushing_yards", H, "Williams", "Under", 58.5), L("moneyline", H), "negative")],
        "rusher-under-with-opponent-side": [(L("rushing_yards", A, "Irving", "Under", 62.5), L("moneyline", H), "positive")],
        "passer-with-own-team-side": [(L("passing_yards", H, "Dak", "Over", 264.5), L("moneyline", H), "neutral")],
        "interceptions-with-own-team-side": [(L("interceptions", A, "Baker", "Over", 0.5), L("moneyline", A), "negative")],
        "interceptions-with-opponent-side": [(L("interceptions", A, "Baker", "Over", 0.5), L("spread", H, line=-8.5), "positive")],
        "td-scorer-with-own-team-side": [(L("anytime_td", H, "Lamb", "Yes"), L("moneyline", H), "positive")],
        "td-scorer-against-opponent-side": [(L("anytime_td", A, "Evans", "Yes"), L("spread", H, line=-8.5), "negative")],
        "offence-prop-with-own-team-total": [(L("anytime_td", H, "Lamb", "Yes"), L("team_total", H, direction="Over", line=29.5), "positive"),
                                             (L("passing_tds", H, "Dak", "Over", 1.5), L("team_total", H, direction="Over", line=29.5), "positive")],
        "offence-prop-against-own-team-total": [(L("anytime_td", H, "Lamb", "Yes"), L("team_total", H, direction="Under", line=29.5), "negative")],
        "offence-prop-with-game-total": [(L("passing_yards", H, "Dak", "Over", 264.5), L("total", direction="Over", line=47.5), "positive")],
        "offence-prop-against-game-total": [(L("passing_yards", H, "Dak", "Over", 264.5), L("total", direction="Under", line=47.5), "negative")],
        "rusher-with-game-total": [(L("rushing_yards", H, "Williams", "Over", 58.5), L("total", direction="Over", line=47.5), "neutral")],
        "prop-with-opponent-team-total": [(L("rushing_yards", A, "Irving", "Over", 62.5), L("team_total", H, direction="Over", line=29.5), "neutral")],
        "props-across-teams": [(L("passing_yards", H, "Dak", "Over", 264.5), L("passing_yards", A, "Baker", "Over", 244.5), "neutral")],
        "props-same-team-fallback": [(L("interceptions", H, "Dak", "Over", 0.5), L("rushing_yards", H, "Williams", "Over", 58.5), "neutral")],
        "prop-with-side-fallback": [(L("completions", H, "Dak", "Under", 22.5), L("moneyline", H), "neutral")],
        "game-side-same-team": [(L("moneyline", H), L("spread", H, line=-8.5), "positive")],
        "game-side-same-market-other-team": [(L("spread", H, line=-8.5), L("spread", A, line=8.5), "exclusive"), (L("moneyline", H), L("moneyline", A), "exclusive")],
        "game-side-other-team": [(L("moneyline", H), L("spread", A, line=8.5), "negative")],
        "game-total-same-direction": [(L("total", direction="Over", line=47.5), L("total", direction="Over", line=48.5), "positive")],
        "game-total-opposite": [(L("total", direction="Over", line=47.5), L("total", direction="Under", line=47.5), "exclusive")],
        "team-total-same-team-same-direction": [(L("team_total", H, direction="Over", line=29.5), L("team_total", H, direction="Over", line=30.5), "positive")],
        "team-total-same-team-opposite": [(L("team_total", H, direction="Over", line=29.5), L("team_total", H, direction="Under", line=29.5), "exclusive")],
        "team-totals-other-team": [(L("team_total", H, direction="Over", line=29.5), L("team_total", A, direction="Over", line=18.5), "neutral")],
        "side-with-own-team-total-over": [(L("moneyline", H), L("team_total", H, direction="Over", line=29.5), "positive")],
        "side-with-own-team-total-under": [(L("moneyline", H), L("team_total", H, direction="Under", line=29.5), "negative")],
        "side-with-opponent-team-total-over": [(L("moneyline", H), L("team_total", A, direction="Over", line=18.5), "negative")],
        "side-with-opponent-team-total-under": [(L("moneyline", H), L("team_total", A, direction="Under", line=18.5), "positive")],
        "game-total-with-team-total-same-direction": [(L("total", direction="Over", line=47.5), L("team_total", H, direction="Over", line=29.5), "positive")],
        "game-total-with-team-total-opposite": [(L("total", direction="Over", line=47.5), L("team_total", H, direction="Under", line=29.5), "negative")],
        "game-total-with-side": [(L("total", direction="Over", line=47.5), L("moneyline", H), "neutral")],
    }
    missing = [rid for rid in table.ids if rid not in cases]
    check(not missing, f"every rule has a test case{': missing ' + ', '.join(missing) if missing else ''}")
    extra = [rid for rid in cases if rid not in table.ids]
    check(not extra, f"every test case names a real rule{': unknown ' + ', '.join(extra) if extra else ''}")
    for rid, pairs in cases.items():
        for a, b, expected in pairs:
            label, matched, reason = table.evaluate(leg_view(a), leg_view(b))
            label_r, matched_r, _ = table.evaluate(leg_view(b), leg_view(a))
            ok = label == expected and matched == rid and label_r == label and matched_r == matched and bool(reason)
            check(ok, f"{rid}: {a.market}{'/' + a.player if a.player else ''}{'/' + a.direction if a.direction else ''} + "
                      f"{b.market}{'/' + b.player if b.player else ''}{'/' + b.direction if b.direction else ''} -> {label}"
                      + ("" if ok else f" (expected {expected} via {rid}; matched {matched or 'nothing'})"))

    # Legs without a team (prop model did not run) fall through to neutral instead of pretending
    label, rid, _ = table.evaluate(leg_view(L("rushing_yards", None, "Irving", "Over", 62.5)), leg_view(L("moneyline", H)))
    check(label == "neutral" and rid == "prop-with-side-fallback", "a prop with no known team is neutral against a side")
    # Moneyline favourite strength is derived from the fair probability
    v = leg_view(L("moneyline", H, fair=0.80))
    check(v.favored_points is not None and 10 < v.favored_points < 13, f"moneyline fair 80% ~ favoured by {v.favored_points:.1f} points")
    check(leg_view(L("spread", H, line=-8.5)).favored_points == 8.5, "spread favourite points")

    # Validation
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "rules.json")
        base = json.load(open(table.path, "r", encoding="utf-8"))
        for bad, why in ((dict(base, rules=base["rules"] + [{"id": "x", "label": "maybe"}]), "unknown label"),
                         (dict(base, rules=base["rules"] + [{"id": "x", "label": "positive", "relation": ["same_hat"]}]), "unknown relation"),
                         (dict(base, rules=base["rules"] + [{"id": "x", "label": "positive", "a": {"cls": "elephants"}}]), "unknown class"),
                         (dict(base, rules=base["rules"] + [dict(base["rules"][0])]), "duplicate id")):
            json.dump(bad, open(p, "w", encoding="utf-8"))
            os.utime(p, (os.path.getmtime(p) + 3, os.path.getmtime(p) + 3))
            try:
                load_rules(p)
                check(False, f"{why} rejected")
            except RuleError:
                check(True, f"{why} rejected")
    print(f"\n{'ALL TESTS PASSED' if failures == 0 else f'{failures} TEST(S) FAILED'}")
    return 0 if failures == 0 else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in args:
        return _selftest()
    try:
        table = rules()
    except RuleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Same-game correlation rules ({table.path})")
    for r in table.rules:
        print(f"  {r.label:<9} {r.id:<42} {r.reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
