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


def _has_valid_route(city1: str, city2: str, sandbox: Sandbox) -> bool:
    """True when the transition between two cities is feasible.

    Fails only when driving is *explicitly* "No valid information" AND no
    flight exists for this city pair. A missing entry (None) means the pair
    wasn't sampled — we can't infer infeasibility, so we pass by default.
    """
    if city1 == city2:
        return True
    # If any flight exists on any date for this pair → feasible.
    if any(k[0] == city1 and k[1] == city2 for k in sandbox.flights):
        return True
    route = sandbox.distance_matrix(city1, city2, "driving")
    # None → no data; don't penalise. Explicit "No valid information." → infeasible.
    if route is None:
        return True
    return "no valid" not in route.lower()


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

    # ---- within sandbox (single key = paper Table 1) ----
    known_attr = {r.get("Name") for rs in sandbox.attractions.values() for r in rs}
    known_rest = {r.get("Name") for rs in sandbox.restaurants.values() for r in rs}
    known_acc = {r.get("NAME") for rs in sandbox.accommodations.values() for r in rs}
    acc_names: list[str] = []
    for d in plan.plan:
        acc_names.extend(_names(d.accommodation))
    cs["within_sandbox"] = (
        (all(n in known_attr for n in attrs) if attrs else True)
        and (all(n in known_rest for n in rests) if rests else True)
        and (all(n in known_acc for n in acc_names) if acc_names else True)
    )

    # ---- within current city ----
    # Venues must be in the destination city (or any visited city for multi-city
    # state-destination queries). Expand state names to their constituent cities
    # so plans that correctly use city-level venues aren't penalised.
    dest_cities = sandbox.cities_in(intent.dest) or [intent.dest]
    dest_attr_names: set[str] = set()
    dest_rest_names: set[str] = set()
    for _dc in dest_cities:
        dest_attr_names.update(r.get("Name") for r in sandbox.attraction_search(_dc))
        dest_rest_names.update(r.get("Name") for r in sandbox.restaurant_search(_dc))
    cs["within_current_city"] = (
        (all(n in dest_attr_names for n in attrs) if attrs else True)
        and (all(n in dest_rest_names for n in rests) if rests else True)
    )

    # ---- reasonable city route ----
    # Every explicit city transition in current_city strings must be feasible.
    # "from X to Y" format identifies inter-city legs; same-city days are fine.
    route_ok = True
    for d in plan.plan:
        cc = d.current_city
        if cc.startswith("from ") and " to " in cc:
            parts = cc.split(" to ", 1)
            from_city = parts[0][5:].strip()
            to_city = parts[1].strip()
            if not _has_valid_route(from_city, to_city, sandbox):
                route_ok = False
                break
    cs["reasonable_city_route"] = route_ok

    # ---- min nights (paper: consecutive nights must meet hotel minimum) ----
    stay_names = [_names(d.accommodation)[0] if _names(d.accommodation) else "" for d in plan.plan]
    min_nights_ok = True
    i = 0
    while i < len(stay_names):
        name = stay_names[i]
        if not name:
            i += 1
            continue
        j = i
        while j < len(stay_names) and stay_names[j] == name:
            j += 1
        nights = j - i
        min_req = 1
        for rows in sandbox.accommodations.values():
            for r in rows:
                if r.get("NAME") == name:
                    try:
                        min_req = int(r.get("minimum nights", 1))
                    except (TypeError, ValueError):
                        min_req = 1
                    break
            else:
                continue
            break
        if nights < min_req:
            min_nights_ok = False
            break
        i = j
    cs["min_nights"] = min_nights_ok

    # ---- non-conflicting transportation ----
    # Per-day: no single day mixes "flight" and "self-driving".
    # Trip-level: flight and self-driving are mutually exclusive across the trip.
    ok = True
    trans_lower = [d.transportation.lower() for d in plan.plan]
    for t in trans_lower:
        if "flight" in t and "self-driving" in t:
            ok = False
            break
    if ok:
        has_flight = any("flight" in t for t in trans_lower)
        has_selfdriving = any("self-driving" in t for t in trans_lower)
        if has_flight and has_selfdriving:
            ok = False
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
