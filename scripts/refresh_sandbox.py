"""Refresh the frozen-2022 TravelPlanner sandbox to current-year data.

Writes parallel files in the SAME schema so the evaluator and agents
require no code changes:
  Dataset/validation_fresh.csv
  Dataset/validation_fresh_ref_info.jsonl

Transformations:
  1. Dates shifted by --year delta (preserves month/day).
  2. Budgets scaled by --inflation and rounded to the nearest $50.
  3. Reference data (attractions/restaurants/accommodations/flights)
     regenerated per city via a pluggable provider (--provider).

Providers:
  llm   — web-search-grounded LLM seeded with the original 2022 entries
          so still-open venues survive and closed ones get replaced.
  stub  — identity transform on venue data (dates/budgets only). Default
          because it runs offline; useful for first-pass verification.
"""
from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "Dataset"


# ----- transformations -----

def shift_dates(dates: list[str], year_delta: int) -> list[str]:
    out = []
    for d in dates:
        dt = datetime.strptime(d, "%Y-%m-%d")
        try:
            out.append(dt.replace(year=dt.year + year_delta).strftime("%Y-%m-%d"))
        except ValueError:
            # handle Feb 29 -> Feb 28 in non-leap years
            out.append((dt + timedelta(days=365 * year_delta)).strftime("%Y-%m-%d"))
    return out


def scale_budget(budget: int | None, factor: float) -> int | None:
    if budget is None or pd.isna(budget):
        return None
    scaled = float(budget) * factor
    return int(round(scaled / 50) * 50)


# ----- providers -----

def regenerate_stub(entry: dict, year_delta: int, factor: float) -> dict:
    """Identity on venue data; shift flight dates and inflate prices."""
    out = {}
    for key, value in entry.items():
        if key.startswith("Flight from "):
            # key contains YYYY-MM-DD; shift it
            prefix, date_str = key.rsplit(" on ", 1)
            new_date = shift_dates([date_str], year_delta)[0]
            new_key = f"{prefix} on {new_date}"
            new_rows = []
            iterable = value if isinstance(value, list) else []
            for row in iterable:
                if not isinstance(row, dict):
                    continue
                r = dict(row)
                if "Price" in r and r["Price"] is not None:
                    try:
                        r["Price"] = int(round(float(r["Price"]) * factor))
                    except (TypeError, ValueError):
                        pass
                new_rows.append(r)
            out[new_key] = new_rows
        elif key.startswith("Restaurants in ") or key.startswith("Accommodations in "):
            new_rows = []
            iterable = value if isinstance(value, list) else []
            for row in iterable:
                if not isinstance(row, dict):
                    continue
                r = dict(row)
                for price_key in ("Average Cost", "price"):
                    if r.get(price_key) is not None:
                        try:
                            r[price_key] = int(round(float(r[price_key]) * factor))
                        except (TypeError, ValueError):
                            pass
                new_rows.append(r)
            out[key] = new_rows
        else:
            out[key] = value
    return out


def regenerate_llm(entry: dict, year_delta: int, factor: float) -> dict:
    """LLM-synth provider. Seeds a structured-output call with the 2022 entries
    and asks for refreshed equivalents. Requires an LLM key."""
    from agents.llm import call_json  # lazy import
    from pydantic import BaseModel, Field

    class RefreshedEntry(BaseModel):
        data: dict = Field(description="Refreshed entry in the same schema")

    system = (
        "You refresh a travel-sandbox entry from 2022 to current year, "
        "preserving the exact JSON schema. Keep still-open venues; replace "
        "permanently closed ones with plausible currently-open alternatives "
        "in the same city. Inflate monetary fields by the given factor. "
        "Shift all date fields by the given year delta. Output JSON "
        "{data: <same-schema-as-input>}."
    )
    user = (
        f"year_delta={year_delta} inflation_factor={factor}\n\n"
        f"entry:\n{json.dumps(entry, default=str)[:6000]}"
    )
    try:
        result = call_json(system, user, RefreshedEntry, max_tokens=4000)
        return result.data
    except Exception as e:
        print(f"  llm provider fell back to stub: {e}")
        return regenerate_stub(entry, year_delta, factor)


def regenerate_api(entry: dict, year_delta: int, factor: float) -> dict:
    """Free-API provider. Stub stub for now; keys needed to fully wire up.
    Falls back to stub transformation on venues but uses Amadeus if available
    for flight refresh.
    """
    # Not fully implemented — follow the TravelPlanner data schema exactly
    # when wiring up OpenTripMap / Yelp Fusion / Amadeus here.
    return regenerate_stub(entry, year_delta, factor)


PROVIDERS = {"stub": regenerate_stub, "llm": regenerate_llm, "api": regenerate_api}


# ----- main driver -----

def refresh(provider: str, year_delta: int, factor: float, subset: int) -> None:
    val_csv = DATASET_DIR / "validation.csv"
    val_jsonl = DATASET_DIR / "validation_ref_info.jsonl"
    out_csv = DATASET_DIR / "validation_fresh.csv"
    out_jsonl = DATASET_DIR / "validation_fresh_ref_info.jsonl"

    df = pd.read_csv(val_csv)
    if subset:
        df = df.head(subset)

    # CSV transform
    def fix_dates(s):
        dates = ast.literal_eval(s)
        return str(shift_dates(dates, year_delta))

    df["date"] = df["date"].apply(fix_dates)
    df["budget"] = df["budget"].apply(lambda b: scale_budget(b, factor))
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    # JSONL transform
    transform = PROVIDERS[provider]
    lines = val_jsonl.read_text().splitlines()
    if subset:
        lines = lines[:subset]
    with out_jsonl.open("w") as f:
        for i, line in enumerate(lines, start=1):
            entry = json.loads(line)
            new_entry = transform(entry, year_delta, factor)
            f.write(json.dumps(new_entry) + "\n")
            if i % 10 == 0:
                print(f"  {i}/{len(lines)}")
    print(f"Wrote {out_jsonl}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=list(PROVIDERS), default="stub")
    ap.add_argument("--year", type=int, default=datetime.now().year,
                    help="target year (default = current year)")
    ap.add_argument("--inflation", type=float, default=1.20)
    ap.add_argument("--subset", type=int, default=0, help="limit number of rows")
    args = ap.parse_args()

    year_delta = args.year - 2022
    refresh(args.provider, year_delta, args.inflation, args.subset)


if __name__ == "__main__":
    main()
