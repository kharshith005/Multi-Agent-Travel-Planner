"""Generate report-ready Markdown tables from eval/results.csv.

Usage:
    python -m eval.format_report                          # uses eval/results.csv
    python -m eval.format_report --csv eval/results.csv  # explicit path
    python -m eval.format_report --pareto                # also emit Pareto chart PNG

Outputs:
    Table 1 — Per-model TravelPlanner Table-3 analog (system × model)
    Table 2 — Per-rule pass-rate breakdown for the full multi-agent system
    Optional: pareto.png — final_pass_rate vs avg_latency_seconds, colored by provider
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

RESULTS_CSV = Path(__file__).parent / "results.csv"


def _load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        print("results.csv is empty — run eval first.", file=sys.stderr)
        sys.exit(1)
    return df


def table1_model_comparison(df: pd.DataFrame) -> str:
    """Per-model, per-system TravelPlanner Table-3 analog."""
    lines = [
        "## Table 1 — Per-model TravelPlanner Comparison\n",
        "| System | Model | Provider | Delivery | CS micro | CS macro | HC micro | HC macro | Final Pass | Avg LLM calls | Avg tokens (in+out) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for (sysname, model, provider), g in df.groupby(["system", "model", "provider"], sort=False):
        n = len(g)
        delivery = g["final_pass"].notna().mean()
        cs_micro = (g["cs_micro_pass"].sum() / g["cs_micro_total"].sum()) if g["cs_micro_total"].sum() > 0 else 0.0
        cs_macro = g.apply(
            lambda r: r["cs_micro_pass"] / r["cs_micro_total"] if r["cs_micro_total"] > 0 else 0.0, axis=1
        ).mean()
        hc_micro = (g["hc_micro_pass"].sum() / g["hc_micro_total"].sum()) if g["hc_micro_total"].sum() > 0 else 0.0
        hc_macro = g.apply(
            lambda r: r["hc_micro_pass"] / r["hc_micro_total"] if r["hc_micro_total"] > 0 else 0.0, axis=1
        ).mean()
        final = g["final_pass"].mean()
        avg_calls = g["llm_calls"].mean() if "llm_calls" in g.columns else 0.0
        avg_in = g["input_tokens"].mean() if "input_tokens" in g.columns else 0.0
        avg_out = g["output_tokens"].mean() if "output_tokens" in g.columns else 0.0
        avg_tok = f"{avg_in:.0f}+{avg_out:.0f}"
        lines.append(
            f"| {sysname} | {model} | {provider} | "
            f"{delivery:.2f} | {cs_micro:.2f} | {cs_macro:.2f} | "
            f"{hc_micro:.2f} | {hc_macro:.2f} | {final:.2f} | "
            f"{avg_calls:.1f} | {avg_tok} |"
        )

    return "\n".join(lines)


def table2_per_rule(df: pd.DataFrame) -> str:
    """Per-rule pass-rate for the full multi-agent system."""
    lines = ["\n## Table 2 — Per-Rule Pass Rates (multi-agent full system)\n"]

    multi_df = df[df["system"] == "multi"]
    if multi_df.empty:
        return "\n## Table 2 — Per-Rule Pass Rates\n\n*(No 'multi' rows in results.csv)*\n"

    for model, g in multi_df.groupby("model", sort=False):
        lines.append(f"\n### Model: {model}\n")
        lines.append("| Rule | Pass | Total | Rate |")
        lines.append("|---|---|---|---|")
        rule_counts: dict[str, tuple[int, int]] = {}
        for col in g.columns:
            if col.startswith("rule_"):
                rule = col[5:]
                total = len(g)
                passed = g[col].sum() if g[col].dtype == bool else (g[col] > 0).sum()
                rule_counts[rule] = (int(passed), total)
        if not rule_counts:
            lines.append("| *(rule columns not present — run eval to regenerate)* | | | |")
        else:
            for rule, (p, t) in sorted(rule_counts.items()):
                lines.append(f"| {rule} | {p} | {t} | {p/t:.2f} |")

    return "\n".join(lines)


def pareto_chart(df: pd.DataFrame, out_path: Path) -> None:
    """Optional Pareto chart: final_pass_rate vs avg_latency_seconds by provider."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping Pareto chart.", file=sys.stderr)
        return

    if "latency_seconds" not in df.columns:
        print("latency_seconds column not found — skipping Pareto chart.", file=sys.stderr)
        return

    summary = df.groupby(["system", "model", "provider"]).agg(
        final_pass=("final_pass", "mean"),
        avg_latency=("latency_seconds", "mean"),
    ).reset_index()

    colors = {"gemini": "#4285F4", "claude": "#FF6B35"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in summary.iterrows():
        c = colors.get(row["provider"], "#888888")
        label = f"{row['model']}\n({row['system']})"
        ax.scatter(row["avg_latency"], row["final_pass"] * 100, color=c, s=80, zorder=3)
        ax.annotate(label, (row["avg_latency"], row["final_pass"] * 100),
                    fontsize=7, ha="left", va="bottom", xytext=(4, 4), textcoords="offset points")

    from matplotlib.patches import Patch
    legend = [Patch(facecolor=c, label=p) for p, c in colors.items()]
    ax.legend(handles=legend, title="Provider")
    ax.set_xlabel("Avg latency (seconds)")
    ax.set_ylabel("Final pass rate (%)")
    ax.set_title("Pareto: quality vs latency by model")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(RESULTS_CSV))
    ap.add_argument("--pareto", action="store_true", help="Also emit pareto.png")
    args = ap.parse_args()

    df = _load(Path(args.csv))

    print(table1_model_comparison(df))
    print(table2_per_rule(df))

    if args.pareto:
        out = Path(args.csv).parent / "pareto.png"
        pareto_chart(df, out)


if __name__ == "__main__":
    main()
