"""Generate architecture.png — clean pipeline diagram matching the project layout."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── canvas ────────────────────────────────────────────────────────────────────
W, H = 24, 13
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("#f5f6f8")

# ── helpers ───────────────────────────────────────────────────────────────────

def section(x, y, w, h, label, bg, border):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.18",
        facecolor=bg, edgecolor=border, linewidth=1.6, zorder=1,
    ))
    ax.text(x + w / 2, y + h - 0.22, label,
            ha="center", va="top", fontsize=8, color=border,
            fontstyle="italic", fontweight="semibold", zorder=2)


def box(x, y, w, h, title, desc="",
        bg="#ffffff", border="#adb5bd", title_size=9.2, desc_size=7.8):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.12",
        facecolor=bg, edgecolor=border, linewidth=1.4, zorder=3,
    ))
    ty = y + h - 0.22
    ax.text(x + w / 2, ty, title,
            ha="center", va="top", fontsize=title_size,
            fontweight="bold", color="#111", zorder=4)
    if desc:
        ax.text(x + w / 2, y + 0.18, desc,
                ha="center", va="bottom", fontsize=desc_size,
                color="#444", multialignment="center",
                linespacing=1.4, zorder=4)


def arr(x1, y1, x2, y2, label="", color="#555", lw=1.6, dashed=False):
    style = "dashed" if dashed else "solid"
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw,
            linestyle=style,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=5,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.12, label, ha="center", va="bottom",
                fontsize=6.8, color=color, zorder=6)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT  (all coords in data units, origin bottom-left)
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1 · Input section  (x 0.25 – 2.85) ──────────────────────────────────────
section(0.25, 4.8, 2.6, 5.4, "Input", "#eef3ff", "#3b82f6")
box(0.45, 7.7, 2.2, 2.0,
    "Streamlit Chat UI", "Model selector\nClarification flow\napp.py",
    bg="#dbeafe", border="#3b82f6")
box(0.45, 5.1, 2.2, 2.0,
    "Evaluation CLI", "--split  --system\n--model  --limit\neval/run_eval.py",
    bg="#dbeafe", border="#3b82f6")

# ── 2 · Coordinator section  (x 3.15 – 8.65) ────────────────────────────────
section(3.15, 4.8, 5.5, 5.4, "LangGraph Coordinator — agents/coordinator.py",
        "#fefce8", "#ca8a04")

box(3.35, 7.6, 2.2, 2.1,
    "Parse Intent", "Extract destination,\ndates, budget &\npreferences",
    bg="#fef9c3", border="#ca8a04")

box(3.35, 5.1, 2.2, 2.1,
    "Research Node", "Live: Maps + SerpAPI\nEval: TravelPlanner\nsandbox",
    bg="#fef9c3", border="#ca8a04")

box(6.05, 5.95, 2.3, 2.8,
    "ToolContext", "flights  hotels\nrestaurants\nattractions\nroutes  trip_windows",
    bg="#ede9fe", border="#7c3aed")

# ── 3 · Specialists section  (x 8.85 – 11.55) ───────────────────────────────
section(8.85, 4.8, 2.7, 5.4,
        "Specialist Agents  (fan-out)", "#ecfeff", "#0891b2")

specialist_y  = [9.3, 8.1, 6.9, 5.7]
spec_labels   = ["Transport", "Lodging", "Dining", "Sightseeing"]
spec_descs    = ["Flights & routes", "Hotels & windows", "Restaurants", "Attractions"]

for sy, sl, sd in zip(specialist_y, spec_labels, spec_descs):
    box(9.0, sy - 0.85, 2.4, 1.0,
        sl, sd, bg="#cffafe", border="#0891b2", title_size=8.8, desc_size=7.4)

# ── 4 · Validation section  (x 11.75 – 17.05) ───────────────────────────────
section(11.75, 4.8, 5.3, 5.4,
        "Validation & Repair  (<= 3 rounds)", "#f0fdf4", "#16a34a")

box(11.95, 8.65, 2.1, 1.3,
    "Assembler", "Merge into FullPlan", bg="#dcfce7", border="#16a34a")
box(11.95, 6.95, 2.1, 1.3,
    "Budget Agent", "Deterministic\nno LLM", bg="#dcfce7", border="#16a34a")
box(11.95, 5.1, 2.1, 1.4,
    "Verifier", "8 commonsense +\n5 hard rules", bg="#dcfce7", border="#16a34a")

box(14.35, 5.95, 2.3, 2.8,
    "Repair Router", "Stage 1: deterministic\nvenue swaps\nStage 2: LLM\ntargeted re-run",
    bg="#dcfce7", border="#16a34a")

# ── 5 · Output  (x 17.25 – 19.35) ───────────────────────────────────────────
box(17.25, 6.3, 2.1, 1.9,
    "FullPlan +\nVerifierReport", "",
    bg="#dcfce7", border="#16a34a", title_size=9.5)

# ── 6 · LLM layer  (x 3.15 – 17.05, bottom strip) ───────────────────────────
section(3.15, 1.1, 13.9, 3.3, "LLM Layer — agents/llm.py + agents/providers/",
        "#fef2f2", "#dc2626")

box(3.35, 1.4, 2.4, 2.6,
    "llm.py facade", "LRU + disk cache\nRetry 429 / 5xx\nThread-safe stats",
    bg="#fee2e2", border="#dc2626")

box(6.25, 1.4, 3.2, 2.6,
    "gemini.py", "Gemini 2.5 Flash-Lite\nGemini 3.1 Flash-Lite\nAuth: VERTEX_AI_API_KEY",
    bg="#fff7ed", border="#ea580c")

box(9.9, 1.4, 4.3, 2.6,
    "vertex_requests.py", "Llama 3.3 70B  ->  MaaS endpoint\nMistral Small 3.1  ->  rawPredict\nAuth: ADC + GOOGLE_CLOUD_PROJECT",
    bg="#fff7ed", border="#ea580c")

# ── 7 · Baselines  (x 17.25 – 23.5) ─────────────────────────────────────────
section(17.25, 1.1, 6.3, 3.3, "Baselines — baseline/", "#f9fafb", "#9ca3af")
for i, (bl, bd) in enumerate([
    ("single_agent.py",       "Single LLM call\nBaseline 1"),
    ("no_verify.py",          "Skip verify + repair\nBaseline 2"),
    ("no_specialization.py",  "One generalist\nBaseline 3"),
]):
    box(17.45 + i * 2.0, 1.4, 1.8, 2.6,
        bl, bd, bg="#f3f4f6", border="#9ca3af",
        title_size=7.8, desc_size=7.2)

# ── Title ──────────────────────────────────────────────────────────────────────
ax.text(W / 2, H - 0.35,
        "Multi-Agent Travel Itinerary Planner — Architecture",
        ha="center", va="top", fontsize=16, fontweight="bold", color="#111")

# ═══════════════════════════════════════════════════════════════════════════════
# ARROWS
# ═══════════════════════════════════════════════════════════════════════════════

BLUE  = "#3b82f6"
GOLD  = "#ca8a04"
TEAL  = "#0891b2"
GREEN = "#16a34a"
RED   = "#dc2626"
GREY  = "#6b7280"

# Input → parse_intent
arr(2.67, 8.7,  3.35, 8.65, color=BLUE)   # UI   → Parse
arr(2.67, 6.1,  3.35, 6.10, color=BLUE)   # Eval → Parse (via research)
arr(1.55, 7.7,  1.55, 7.1,  color=GOLD)   # UI → Eval

# Inside coordinator
arr(4.45, 7.6,  4.45, 7.2,  color=GOLD)   # Parse → Research
arr(5.55, 6.15, 6.05, 7.0,  color=GOLD)   # Research → ToolContext

# ToolContext → each specialist
tc_rx = 8.35
for sy in specialist_y:
    arr(tc_rx, 7.35, 9.0, sy - 0.35, color=TEAL)

# Specialists → Assembler
asm_lx = 11.95
for sy in specialist_y:
    arr(11.4, sy - 0.35, asm_lx, 9.3, color=TEAL)

# Validation chain
arr(13.05, 8.65, 13.05, 8.25,  color=GREEN)   # Assembler → Budget
arr(13.05, 6.95, 13.05, 6.5,   color=GREEN)   # Budget → Verifier

# Verifier → pass → Output
arr(14.05, 5.8,  17.25, 7.25, color=GREEN, label="pass")

# Verifier → fail → Repair
arr(14.05, 5.8,  14.35, 7.35, color="#ef4444", label="fail")

# Repair → retry (curved back arrow)
ax.annotate("", xy=(14.6, 8.65), xytext=(15.6, 8.75),
            arrowprops=dict(arrowstyle="-|>", color="#ef4444", lw=1.4,
                            connectionstyle="arc3,rad=-0.45"), zorder=5)
ax.text(15.55, 9.15, "retry", ha="center", fontsize=7, color="#ef4444")

# llm.py → backends
arr(5.75, 2.7,  6.25, 2.7,  color=RED)
arr(5.75, 2.7,  9.9,  2.7,  color=RED)

# LLM call edges (dotted) from components down to llm.py
for cx, cy in [(4.45, 7.6), (10.2, 8.45), (10.2, 7.25), (10.2, 6.05), (10.2, 4.85),
               (13.05, 5.1), (15.5, 5.95)]:
    ax.annotate("", xy=(4.55, 4.0), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="#dc2626", lw=1.0,
                                linestyle="dashed", alpha=0.5,
                                connectionstyle="arc3,rad=0.0"), zorder=2)

ax.text(5.5, 3.95, "call_json / call_text", ha="center", fontsize=7,
        color="#dc2626", alpha=0.7, style="italic")

# Eval → Baselines (dotted)
arr(1.55, 5.1,  18.35, 4.4, color=GREY, dashed=True, label="baselines")

# ═══════════════════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.3)
out = "architecture.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved {out}")
