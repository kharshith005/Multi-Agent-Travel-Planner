"""Evaluator: scores a FullPlan against TravelPlanner-style metrics.

Produces per-constraint pass booleans so we can compute both micro (fraction
of individual constraints satisfied) and macro (fraction of queries where
all constraints pass) aggregates.

Categories:
  commonsense: complete_info, diverse_attractions, diverse_restaurants,
               within_sandbox_attractions, within_sandbox_restaurants,
               within_sandbox_accommodations, min_nights, non_conflicting_transport
  hard:        budget, cuisine, room_type, room_rule, transportation
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from agents.schemas import FullPlan, Intent
from tools.sandbox import Sandbox


_COST_RE = re.compile(r"Cost[:\s]*\$?(\d+)", re.IGNORECASE)
_COST_FRAGMENT_RE = re.compile(
    r"Cost[:\s]*\$?\d+(?:\s+for\s+\d+\s+nights?)?", re.IGNORECASE
)


@dataclass
class ConstraintReport:
    commonsense: dict[str, bool] = field(default_factory=dict)
    hard: dict[str, bool] = field(default_factory=dict)

    @property
    def delivery(self) -> bool:
        return bool(self.commonsense) or bool(self.hard)

    @property
    def commonsense_all_pass(self) -> bool:
        return all(self.commonsense.values()) if self.commonsense else False

    @property
    def hard_all_pass(self) -> bool:
        return all(self.hard.values()) if self.hard else True  # no hard rules -> vacuously true

    @property
    def final_pass(self) -> bool:
        return self.commonsense_all_pass and self.hard_all_pass


def _cost(text: str) -> int:
    return sum(int(m.group(1)) for m in _COST_RE.finditer(text or ""))


def _names(text: str) -> list[str]:
    """Extract venue names from a plan field, stripping any 'Cost: $N' fragments.

    Plan fields have the shape 'Name, City; Another Name, City; Cost: $N' and
    without stripping the cost fragment, 'Cost: $N' leaks in as a fake name —
    which then fails both the diversity check (repeated costs look like
    duplicates) and the within_sandbox check (costs aren't in the sandbox).
    """
    if not text or text == "-":
        return []
    cleaned = _COST_FRAGMENT_RE.sub("", text)
    names = []
    for part in cleaned.split(";"):
        chunk = part.strip()
        if not chunk:
            continue
        name = chunk.split(",")[0].strip()
        if name:
            names.append(name)
    return names


def evaluate(plan: FullPlan, intent: Intent, sandbox: Sandbox) -> ConstraintReport:
    cs: dict[str, bool] = {}
    hc: dict[str, bool] = {}

    # ---- complete_info ----
    cs["complete_info"] = (
        len(plan.plan) == intent.days
        and all(d.current_city for d in plan.plan)
    )

    # ---- diversity ----
    attrs: list[str] = []
    rests: list[str] = []
    for d in plan.plan:
        attrs.extend(_names(d.attraction))
        for m in (d.breakfast, d.lunch, d.dinner):
            rests.extend(_names(m))
    cs["diverse_attractions"] = len(attrs) == len(set(attrs))
    cs["diverse_restaurants"] = len(rests) == len(set(rests))

    # ---- within sandbox ----
    known_attr = {r.get("Name") for rs in sandbox.attractions.values() for r in rs}
    known_rest = {r.get("Name") for rs in sandbox.restaurants.values() for r in rs}
    known_acc = {r.get("NAME") for rs in sandbox.accommodations.values() for r in rs}
    cs["within_sandbox_attractions"] = all(n in known_attr for n in attrs) if attrs else True
    cs["within_sandbox_restaurants"] = all(n in known_rest for n in rests) if rests else True
    acc_names: list[str] = []
    for d in plan.plan:
        acc_names.extend(_names(d.accommodation))
    cs["within_sandbox_accommodations"] = (
        all(n in known_acc for n in acc_names) if acc_names else True
    )

    # ---- min nights & accommodation continuity ----
    # On inner days (not last), accommodation must not be '-'
    cs["min_nights"] = all(
        (d.accommodation and d.accommodation != "-") or d.days == intent.days
        for d in plan.plan
    )

    # ---- non-conflicting transportation ----
    # Day 1 and last day have inter-city transport; middle days should not
    # mix flight with self-driving in the same day.
    ok = True
    for d in plan.plan:
        t = d.transportation.lower()
        if "flight" in t and ("self-driving" in t or "driving" in t):
            ok = False
            break
    cs["non_conflicting_transport"] = ok

    # ---- HARD CONSTRAINTS ----

    # budget
    if intent.budget is not None:
        total = 0
        for d in plan.plan:
            for f in (d.transportation, d.breakfast, d.lunch, d.dinner, d.accommodation):
                total += _cost(f)
        hc["budget"] = total <= intent.budget

    # cuisine
    if intent.cuisine:
        wanted = [c.strip().lower() for c in intent.cuisine.split(",") if c.strip()]
        blob = " ".join(d.breakfast + " " + d.lunch + " " + d.dinner for d in plan.plan).lower()
        hc["cuisine"] = all(c in blob for c in wanted)

    # room_type
    if intent.room_type:
        rt = intent.room_type.lower()
        acc_blob = " ".join(d.accommodation for d in plan.plan).lower()
        # plan text often only contains the accom name; as a lenient proxy, require
        # the room-type keyword to appear somewhere — captured via sandbox lookup:
        matched = False
        for name in acc_names:
            for rows in sandbox.accommodations.values():
                for r in rows:
                    if r.get("NAME") == name and rt in str(r.get("room type", "")).lower():
                        matched = True
                        break
        hc["room_type"] = matched if acc_names else False

    # room_rule (e.g., 'no smoking', 'pet friendly')
    if intent.house_rule:
        rule = intent.house_rule.lower()
        matched = False
        for name in acc_names:
            for rows in sandbox.accommodations.values():
                for r in rows:
                    if r.get("NAME") == name:
                        rules = str(r.get("house_rules", "")).lower()
                        if rule in rules or rules in rule:
                            matched = True
        hc["room_rule"] = matched if acc_names else False

    # transportation (required/forbidden mode)
    if intent.transportation:
        mode = intent.transportation.lower()
        trans_text = " ".join(d.transportation for d in plan.plan).lower()
        if any(neg in mode for neg in ("no ", "avoid", "without")):
            forbidden = mode.replace("no ", "").replace("avoid", "").replace("without", "").strip()
            hc["transportation"] = bool(forbidden) and forbidden not in trans_text
        else:
            hc["transportation"] = mode.split()[0] in trans_text

    return ConstraintReport(commonsense=cs, hard=hc)


def aggregate(reports: list[ConstraintReport]) -> dict[str, float]:
    """Compute the five headline metrics over a batch of reports."""
    if not reports:
        return {}
    delivered = [r for r in reports if r.delivery]
    delivery_rate = len(delivered) / len(reports)

    cs_micro_total = cs_micro_pass = 0
    hc_micro_total = hc_micro_pass = 0
    cs_macro_pass = hc_macro_pass = final_pass = 0
    for r in reports:
        cs_micro_total += len(r.commonsense)
        cs_micro_pass += sum(1 for v in r.commonsense.values() if v)
        hc_micro_total += len(r.hard)
        hc_micro_pass += sum(1 for v in r.hard.values() if v)
        if r.commonsense_all_pass:
            cs_macro_pass += 1
        if r.hard_all_pass:
            hc_macro_pass += 1
        if r.final_pass:
            final_pass += 1
    n = len(reports)
    return {
        "delivery_rate": delivery_rate,
        "commonsense_micro": cs_micro_pass / max(cs_micro_total, 1),
        "commonsense_macro": cs_macro_pass / n,
        "hard_micro": hc_micro_pass / max(hc_micro_total, 1),
        "hard_macro": hc_macro_pass / n,
        "final_pass": final_pass / n,
    }
