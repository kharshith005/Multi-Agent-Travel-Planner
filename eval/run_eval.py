"""Evaluation CLI.

Usage:
    python -m eval.run_eval --split validation --system multi --limit 20
    python -m eval.run_eval --split validation --system both
    python -m eval.run_eval --split validation --system all
    python -m eval.run_eval --split validation --system multi --model gemini-2.5-flash
    python -m eval.run_eval --split validation --system multi --models all
    python -m eval.run_eval --split train --system annotated --limit 10  # evaluator sanity

Systems
-------
  multi              Full multi-agent (coordinator + 5 workers + verifier)
  single             Single-agent ReAct baseline (paper §4.7 Baseline 1)
  no_verify          Multi-agent without verification (paper §4.7 Baseline 2)
  no_specialization  Multi-agent without specialization (paper §4.7 Baseline 3)
  annotated          Human-annotated reference plans (evaluator sanity check)
  both               Runs multi + single
  all                Runs multi + single + no_verify + no_specialization

All systems run against the frozen sandbox (TravelPlanner reference data).

Writes eval/results.csv and prints a markdown summary table.
"""
from __future__ import annotations

import argparse
import ast
import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd

from agents.coordinator import plan_trip
from agents.llm import get_call_stats, reset_call_stats
from agents.models import REGISTRY, available_models, default_model_id
from agents.schemas import FullPlan, Intent, PlanDay, ToolContext
from baseline.no_specialization import plan_trip_no_specialization
from baseline.no_verify import plan_trip_no_verify
from baseline.single_agent import plan_trip_single
from eval.constraints import ConstraintReport, aggregate, evaluate
from tools.sandbox import Sandbox, default_sandbox


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "Dataset"
RESULTS_CSV = Path(__file__).parent / "results.csv"


# --------------------------------------------------------------------------- #
# Sandbox → ToolContext conversion                                             #
# --------------------------------------------------------------------------- #

def _price_to_level(price: object) -> int:
    try:
        p = float(price)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 2
    if p < 80:   return 0
    if p < 120:  return 1
    if p < 180:  return 2
    if p < 260:  return 3
    return 4


def _avg_cost_to_level(cost: object) -> int:
    try:
        c = float(cost)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 2
    if c < 15:  return 0
    if c < 30:  return 1
    if c < 55:  return 2
    if c < 80:  return 3
    return 4


def _sandbox_tool_context(intent: Intent, sandbox: Sandbox) -> ToolContext:
    """Build a ToolContext from sandbox reference data."""
    ctx: ToolContext = {}  # type: ignore[assignment]

    if intent.dates:
        raw_out = sandbox.flight_search(intent.org, intent.dest, intent.dates[0])
        ctx["flights_outbound"] = [
            f"{r.get('Flight Number')} {r.get('DepTime')}->{r.get('ArrTime')} "
            f"Duration: {r.get('ActualElapsedTime')} Cost: ${r.get('Price')}"
            for r in raw_out
        ]
        raw_ret = sandbox.flight_search(intent.dest, intent.org, intent.dates[-1])
        ctx["flights_return"] = [
            f"{r.get('Flight Number')} {r.get('DepTime')}->{r.get('ArrTime')} "
            f"Duration: {r.get('ActualElapsedTime')} Cost: ${r.get('Price')}"
            for r in raw_ret
        ]
    else:
        ctx["flights_outbound"] = []
        ctx["flights_return"] = []

    ctx["route_drive"] = sandbox.distance_matrix(intent.org, intent.dest, "driving")
    ctx["route_taxi"]  = sandbox.distance_matrix(intent.org, intent.dest, "taxi")

    ctx["hotels"] = [
        {
            "name": r.get("NAME"),
            "rating": r.get("review rate number"),
            "address": r.get("city", intent.dest),
            "price_level": _price_to_level(r.get("price")),
        }
        for r in sandbox.accommodation_search(intent.dest)
    ]

    ctx["restaurants"] = [
        {
            "name": r.get("Name"),
            "rating": r.get("Aggregate Rating"),
            "price_level": _avg_cost_to_level(r.get("Average Cost")),
        }
        for r in sandbox.restaurant_search(intent.dest)
    ]

    ctx["attractions"] = [
        {
            "name": r.get("Name"),
            "rating": 4.0,
            "address": r.get("Address", intent.dest),
        }
        for r in sandbox.attraction_search(intent.dest)
    ]

    return ctx


# --------------------------------------------------------------------------- #
# Dataset helpers                                                              #
# --------------------------------------------------------------------------- #

def _load_split(split: str) -> pd.DataFrame:
    path = DATASET_DIR / f"{split}.csv"
    return pd.read_csv(path)


def _intent_from_row(row) -> Intent:
    local = ast.literal_eval(row["local_constraint"]) if row.get("local_constraint") else {}
    dates = ast.literal_eval(row["date"]) if isinstance(row["date"], str) else []
    budget = int(row["budget"]) if pd.notna(row.get("budget")) else None
    return Intent(
        org=row["org"],
        dest=row["dest"],
        days=int(row["days"]),
        dates=dates,
        people=int(row.get("people_number", 1) or 1),
        budget=budget,
        house_rule=local.get("house rule"),
        cuisine=local.get("cuisine"),
        room_type=local.get("room type"),
        transportation=local.get("transportation"),
    )


def _annotated_as_fullplan(row) -> FullPlan:
    ap = ast.literal_eval(row["annotated_plan"])
    if isinstance(ap, list) and len(ap) == 2 and isinstance(ap[0], dict) and isinstance(ap[1], list):
        ap = ap[1]
    days = []
    for d in ap:
        if not d or not d.get("days"):
            continue
        days.append(PlanDay(
            days=int(d.get("days") or d.get("day")),
            current_city=d.get("current_city", ""),
            transportation=d.get("transportation", "-"),
            breakfast=d.get("breakfast", "-"),
            attraction=d.get("attraction", "-"),
            lunch=d.get("lunch", "-"),
            dinner=d.get("dinner", "-"),
            accommodation=d.get("accommodation", "-"),
        ))
    return FullPlan(query=row["query"], plan=days)


# --------------------------------------------------------------------------- #
# Per-row runner                                                               #
# --------------------------------------------------------------------------- #

def _run_one(
    system: str, row, sandbox: Sandbox, model_id: str
) -> tuple[FullPlan | None, ConstraintReport, dict[str, int]]:
    """Run one query and return (plan, constraint_report, llm_stats)."""
    intent = _intent_from_row(row)
    reset_call_stats()

    if system == "annotated":
        plan = _annotated_as_fullplan(row)

    elif system == "single":
        plan, _ = plan_trip_single(row["query"], sandbox_kind="frozen", model=model_id)

    else:
        ctx = _sandbox_tool_context(intent, sandbox)

        if system == "multi":
            plan, _ = plan_trip(
                row["query"],
                parsed_intent=intent,
                tool_context=ctx,
                bypass_date_check=True,
                model=model_id,
            )
        elif system == "no_verify":
            plan, _ = plan_trip_no_verify(
                row["query"],
                parsed_intent=intent,
                tool_context=ctx,
                bypass_date_check=True,
                model=model_id,
            )
        elif system == "no_specialization":
            plan, _ = plan_trip_no_specialization(
                row["query"],
                parsed_intent=intent,
                tool_context=ctx,
                model=model_id,
            )
        else:
            raise ValueError(f"Unknown system: {system!r}")

    llm_stats = get_call_stats()
    report = evaluate(plan, intent, sandbox)
    return plan, report, llm_stats


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def _resolve_models(models_arg: str) -> list[str]:
    """Expand --models argument to a list of model IDs."""
    if models_arg == "all":
        return [m.id for m in available_models()]
    return [m.strip() for m in models_arg.split(",") if m.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "validation"], default="validation")
    ap.add_argument(
        "--system",
        choices=["multi", "single", "no_verify", "no_specialization", "annotated", "both", "all"],
        default="both",
    )
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument(
        "--llm-cache-dir",
        default="",
        help="Directory for persistent LLM response cache (disk cache for reproducible reruns).",
    )
    ap.add_argument(
        "--model",
        default="",
        help="Run with a single specific model ID (e.g. gemini-2.5-flash). "
             "Overrides LLM_MODEL env var for this run.",
    )
    ap.add_argument(
        "--models",
        default="",
        help="Comma-separated model IDs or 'all' to sweep all available_models().",
    )
    ap.add_argument(
        "--cooldown-seconds",
        type=float,
        default=0.0,
        help="Seconds to sleep between model sweeps (helps with rate limits).",
    )
    args = ap.parse_args()

    if args.llm_cache_dir:
        os.environ["LLM_CACHE_DIR"] = args.llm_cache_dir
        print(f"LLM disk cache enabled: {args.llm_cache_dir}", file=sys.stderr)

    # Resolve model list — --models takes precedence over --model
    if args.models:
        model_ids = _resolve_models(args.models)
    elif args.model:
        model_ids = [args.model]
    else:
        model_ids = [os.environ.get("LLM_MODEL", "").strip() or default_model_id()]

    if not model_ids:
        print("No models available. Check auth configuration.", file=sys.stderr)
        sys.exit(1)

    df = _load_split(args.split)
    if args.split == "train" and args.system == "annotated":
        df = df.dropna(subset=["annotated_plan"])
    if args.limit:
        df = df.head(args.limit)

    if args.system == "both":
        systems = ["multi", "single"]
    elif args.system == "all":
        systems = ["multi", "single", "no_verify", "no_specialization"]
    else:
        systems = [args.system]

    sandbox = default_sandbox("frozen")
    all_rows: list[dict] = []
    agg_by_key: dict[str, dict[str, float]] = {}
    reports_by_key: dict[str, list[ConstraintReport]] = {}

    for model_idx, model_id in enumerate(model_ids):
        # Look up provider for CSV column
        try:
            from agents.models import get_model
            provider = get_model(model_id).provider
        except KeyError:
            provider = "gemini"

        if model_idx > 0 and args.cooldown_seconds > 0:
            print(f"Cooling down {args.cooldown_seconds}s before next model...", file=sys.stderr)
            time.sleep(args.cooldown_seconds)

        for sysname in systems:
            key = f"{sysname}/{model_id}"
            print(f"\n=== system={sysname} model={model_id} ({len(df)} rows) ===", file=sys.stderr)
            reports: list[ConstraintReport] = []
            total_llm_calls = total_in_tok = total_out_tok = 0
            total_cache_hits = total_cache_misses = 0

            for i, (_, row) in enumerate(df.iterrows(), start=1):
                t0 = time.time()
                try:
                    _, report, llm_stats = _run_one(sysname, row, sandbox, model_id)
                except Exception as e:
                    print(f"  row {i} FAILED: {e.__class__.__name__}: {e}", file=sys.stderr)
                    report = ConstraintReport()
                    llm_stats = {
                        "llm_calls": 0, "input_tokens": 0, "output_tokens": 0,
                        "cache_hits": 0, "cache_misses": 0,
                    }

                reports.append(report)
                elapsed = time.time() - t0
                ch = llm_stats.get("cache_hits", 0)
                cm = llm_stats.get("cache_misses", 0)
                total_llm_calls      += llm_stats.get("llm_calls", 0)
                total_in_tok         += llm_stats.get("input_tokens", 0)
                total_out_tok        += llm_stats.get("output_tokens", 0)
                total_cache_hits     += ch
                total_cache_misses   += cm

                print(
                    f"  [{i}/{len(df)}] final={report.final_pass} "
                    f"({elapsed:.1f}s, {llm_stats.get('llm_calls', 0)} calls, "
                    f"{llm_stats.get('input_tokens', 0)}+{llm_stats.get('output_tokens', 0)} tok, "
                    f"hits={ch} misses={cm})",
                    file=sys.stderr,
                )
                all_rows.append({
                    "system": sysname,
                    "model": model_id,
                    "provider": provider,
                    "idx": i,
                    "final_pass": report.final_pass,
                    "cs_micro_total": len(report.commonsense),
                    "cs_micro_pass": sum(1 for v in report.commonsense.values() if v),
                    "hc_micro_total": len(report.hard),
                    "hc_micro_pass": sum(1 for v in report.hard.values() if v),
                    "llm_calls": llm_stats.get("llm_calls", 0),
                    "input_tokens": llm_stats.get("input_tokens", 0),
                    "output_tokens": llm_stats.get("output_tokens", 0),
                    "cache_hits": ch,
                    "cache_misses": cm,
                    "latency_seconds": round(elapsed, 2),
                })

            n = len(reports)
            agg = aggregate(reports)
            agg["avg_llm_calls"]      = total_llm_calls   / n if n else 0.0
            agg["avg_input_tokens"]   = total_in_tok       / n if n else 0.0
            agg["avg_output_tokens"]  = total_out_tok      / n if n else 0.0
            agg["avg_cache_hits"]     = total_cache_hits   / n if n else 0.0
            agg["avg_cache_misses"]   = total_cache_misses / n if n else 0.0
            agg_by_key[key] = agg
            reports_by_key[key] = reports

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    print("\n| System | Model | Deliv. | CS micro | CS macro | HC micro | HC macro | Final | Avg LLM calls | Avg tokens (in+out) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key, agg in agg_by_key.items():
        sysname, model_id = key.split("/", 1)
        avg_tok = f"{agg['avg_input_tokens']:.0f}+{agg['avg_output_tokens']:.0f}"
        print(
            f"| {sysname} | {model_id} | "
            f"{agg['delivery_rate']:.2f} | "
            f"{agg['commonsense_micro']:.2f} | "
            f"{agg['commonsense_macro']:.2f} | "
            f"{agg['hard_micro']:.2f} | "
            f"{agg['hard_macro']:.2f} | "
            f"{agg['final_pass']:.2f} | "
            f"{agg['avg_llm_calls']:.1f} | "
            f"{avg_tok} |"
        )

    print("\nPer-rule pass rates:")
    for key, reports in reports_by_key.items():
        if not reports:
            continue
        rule_counts: dict[str, tuple[int, int]] = {}
        for r in reports:
            for rule, ok in r.commonsense.items():
                p, t = rule_counts.get(rule, (0, 0))
                rule_counts[rule] = (p + int(bool(ok)), t + 1)
            for rule, ok in r.hard.items():
                p, t = rule_counts.get(rule, (0, 0))
                rule_counts[rule] = (p + int(bool(ok)), t + 1)
        print(f"  [{key}]")
        for rule, (p, t) in sorted(rule_counts.items()):
            bar = "PASS" if p == t else ("FAIL" if p == 0 else "PART")
            print(f"    {bar}  {rule:<32s} {p}/{t}")

    print(f"\nWrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
