"""Build eval/state_city_index.json from TravelPlanner annotated plans and reference info.

For 5-day and 7-day queries the `dest` column is a US state; the visited
cities come from:
  1. `annotated_plan` in train.csv (primary, most accurate)
  2. `reference_information` in any split (fallback for states not in train)

Run from Project_Code/:
    python scripts/build_state_index.py
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "Dataset"
OUT_PATH = Path(__file__).resolve().parents[1] / "eval" / "state_city_index.json"

_STATE_SUFFIX_RE = re.compile(r"\(([A-Za-z ]+)\)$")
# Matches "Attractions in CityName", "Restaurants in CityName", etc.
_REF_CITY_RE = re.compile(
    r"(?:Attractions|Restaurants|Accommodations|Restaurants)\s+in\s+([A-Za-z][A-Za-z\s\-\.]+?)(?:'|\"|,|\s*$|\s+\d)"
)


def _strip_state(city: str) -> str:
    return _STATE_SUFFIX_RE.sub("", city).strip()


def _dest_city_from_label(label: str) -> str | None:
    label = label.strip()
    if not label or label == "-":
        return None
    if " to " in label.lower():
        parts = re.split(r"\s+to\s+", label, flags=re.IGNORECASE)
        dest_part = parts[-1].strip()
        if not _STATE_SUFFIX_RE.search(dest_part):
            return None
        return _strip_state(dest_part)
    return _strip_state(label)


def _cities_from_annotated_plan(annotated_plan: object) -> list[str]:
    try:
        rows = ast.literal_eval(str(annotated_plan)) if isinstance(annotated_plan, str) else annotated_plan
    except Exception:
        return []
    if isinstance(rows, list) and len(rows) == 2 and isinstance(rows[0], dict):
        rows = rows[1]
    seen: set[str] = set()
    ordered: list[str] = []
    for day in (rows or []):
        if not isinstance(day, dict):
            continue
        raw = str(day.get("current_city") or "").strip()
        city = _dest_city_from_label(raw)
        if city and city not in seen:
            seen.add(city)
            ordered.append(city)
    return ordered


def _cities_from_ref_info(ref_info: object) -> list[str]:
    """Extract city names from reference_information column.

    The column contains a list of dicts, each with a 'Description' key like
    "Attractions in Moline" or "Restaurants in Chicago". We extract the city
    names from those description strings.
    """
    text = str(ref_info or "")
    # Look for Description values
    descs = re.findall(r"'Description':\s*'([^']+)'", text)
    if not descs:
        descs = re.findall(r'"Description":\s*"([^"]+)"', text)
    seen: set[str] = set()
    ordered: list[str] = []
    for desc in descs:
        m = re.match(
            r"(?:Attractions|Restaurants|Accommodations)\s+in\s+(.+)$",
            desc.strip(), re.IGNORECASE
        )
        if m:
            city = m.group(1).strip().rstrip(".,")
            if city and city not in seen:
                seen.add(city)
                ordered.append(city)
    return ordered


def main() -> None:
    index: dict[str, list[str]] = {}

    # Pass 1: annotated plans from train.csv (most accurate city order).
    train_path = DATASET_DIR / "train.csv"
    if train_path.exists():
        df_train = pd.read_csv(train_path)
        for _, row in df_train.iterrows():
            dest = str(row.get("dest", "")).strip()
            days = int(row.get("days", 3) or 3)
            if days == 3:
                continue
            cities = _cities_from_annotated_plan(row.get("annotated_plan"))
            if not cities:
                continue
            existing = index.setdefault(dest, [])
            for c in cities:
                if c not in existing:
                    existing.append(c)

    # Pass 2: reference_information from all splits (fills gaps for states
    # that don't appear in the training annotated plans).
    for split in ("train", "validation"):
        path = DATASET_DIR / f"{split}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            dest = str(row.get("dest", "")).strip()
            days = int(row.get("days", 3) or 3)
            if days == 3:
                continue
            cities = _cities_from_ref_info(row.get("reference_information"))
            if not cities:
                continue
            existing = index.setdefault(dest, [])
            for c in cities:
                if c not in existing:
                    existing.append(c)

    # Sort for determinism.
    index = {state: sorted(cities) for state, cities in sorted(index.items())}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote {len(index)} states → {OUT_PATH}")
    for state, cities in list(index.items())[:8]:
        print(f"  {state}: {cities}")


if __name__ == "__main__":
    main()
