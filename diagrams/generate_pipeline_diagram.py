"""Generate a clean pipeline diagram for the Devpost submission.

Output: diagrams/pipeline.png (1600x1100, white bg, academic style)
Run:    python diagrams/generate_pipeline_diagram.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


# ── style ───────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",  # broadly available; close to Helvetica
        "font.size": 11,
    }
)

COLOR = {
    "opus": "#1f4e79",
    "opus_fill": "#dbe6f2",
    "sonnet": "#2a7a8a",
    "sonnet_fill": "#d5ebef",
    "gpt": "#198754",
    "gpt_fill": "#d5ecdf",
    "tool": "#6b7280",
    "tool_fill": "#f3f4f6",
    "arrow": "#9ca3af",
    "text_dim": "#4b5563",
    "text": "#111827",
    "io_fill": "#fafafa",
    "io_edge": "#d1d5db",
}


def stage_box(ax, x, y, w, h, *, title, subtitle, edge_color, fill_color, badge=None):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4,
        edgecolor=edge_color,
        facecolor=fill_color,
        zorder=2,
    )
    ax.add_patch(box)
    if badge:
        ax.text(
            x + 0.15,
            y + h - 0.22,
            badge,
            fontsize=10,
            fontweight="bold",
            color=edge_color,
            va="top",
            ha="left",
            zorder=3,
        )
    ax.text(
        x + (0.5 if badge is None else 0.55),
        y + h - 0.22,
        title,
        fontsize=12,
        fontweight="bold",
        color=COLOR["text"],
        va="top",
        ha="left",
        zorder=3,
    )
    ax.text(
        x + 0.15,
        y + h - 0.55,
        subtitle,
        fontsize=10,
        color=COLOR["text_dim"],
        va="top",
        ha="left",
        zorder=3,
    )


def io_box(ax, x, y, w, h, *, label, mono=True):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.1,
        edgecolor=COLOR["io_edge"],
        facecolor=COLOR["io_fill"],
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        fontsize=10.5,
        family="DejaVu Sans Mono" if mono else "DejaVu Sans",
        color=COLOR["text"],
        va="center",
        ha="center",
        zorder=3,
    )


def arrow(ax, x1, y1, x2, y2, *, lw=1.6):
    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=COLOR["arrow"],
        zorder=1,
    )
    ax.add_patch(a)


def tool_chip(ax, x, y, w, h, label):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        linewidth=1.0,
        edgecolor=COLOR["tool"],
        facecolor=COLOR["tool_fill"],
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        fontsize=9.5,
        family="DejaVu Sans Mono",
        color=COLOR["text_dim"],
        va="center",
        ha="center",
        zorder=3,
    )


def main(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 9.5), dpi=140)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 13)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Title
    ax.text(
        7,
        12.55,
        "Prophet Hacks Agent · Forecasting Pipeline",
        fontsize=17,
        fontweight="bold",
        color=COLOR["text"],
        ha="center",
        va="center",
    )
    ax.text(
        7,
        12.05,
        "Six stages · twelve research tools · two-model ensemble · self-critique",
        fontsize=11,
        color=COLOR["text_dim"],
        ha="center",
        va="center",
    )

    cx = 7  # column center for the main spine
    box_w = 7.6
    box_h = 0.95

    def main_box(y, **kw):
        stage_box(ax, cx - box_w / 2, y, box_w, box_h, **kw)

    # ── INPUT
    io_box(ax, cx - 3.0, 11.0, 6.0, 0.55, label='POST /predict   {"event_ticker": ..., "outcomes": [...]}')
    arrow(ax, cx, 11.0, cx, 10.55)

    # ── 1. Planner
    main_box(
        y=9.55,
        badge="①",
        title="Planner",
        subtitle="Opus 4.7  ·  Fermi decomposition (3–7 sub-questions) + tool plan",
        edge_color=COLOR["opus"],
        fill_color=COLOR["opus_fill"],
    )
    arrow(ax, cx, 9.55, cx, 9.15)

    # ── 2. Research (LLM-orchestrated tool fan-out)
    research_y = 7.55
    research_h = 1.55
    main_box(
        y=research_y,
        badge="②",
        title="Parallel Research",
        subtitle="LLM-orchestrated · gated on env vars · timeout-isolated",
        edge_color=COLOR["sonnet"],
        fill_color=COLOR["sonnet_fill"],
    )

    # ── 12 tool chips arranged in a 3x4 grid under the Research box
    tools = [
        "kalshi_related",
        "polymarket_related",
        "wikipedia",
        "claude_news",
        "mengye_search",
        "sports_odds",
        "code_execution",
        "fred",
        "crypto",
        "congress_bills",
        "court_docket",
        "earnings_data",
    ]
    chip_w, chip_h = 1.65, 0.42
    grid_top = 7.10
    chip_x_start = cx - (4 * chip_w + 3 * 0.13) / 2
    for i, t in enumerate(tools):
        row, col = divmod(i, 4)
        x = chip_x_start + col * (chip_w + 0.13)
        y = grid_top - row * (chip_h + 0.13)
        tool_chip(ax, x, y, chip_w, chip_h, t)
    arrow(ax, cx, 5.65, cx, 5.25)

    # ── 3. Synthesis
    main_box(
        y=4.30,
        badge="③",
        title="Synthesis",
        subtitle="Sonnet 4.6  ·  evidence brief with explicit sub-question answers",
        edge_color=COLOR["sonnet"],
        fill_color=COLOR["sonnet_fill"],
    )
    arrow(ax, cx, 4.30, cx, 3.95)

    # ── 4. Parallel forecasters
    fork_y = 3.05
    fork_h = 0.95
    fork_w = 3.6
    # 4a Opus
    stage_box(
        ax,
        cx - fork_w - 0.20,
        fork_y,
        fork_w,
        fork_h,
        badge="④a",
        title="Opus Forecaster",
        subtitle="Opus 4.7  ·  effort = low",
        edge_color=COLOR["opus"],
        fill_color=COLOR["opus_fill"],
    )
    # 4b GPT
    stage_box(
        ax,
        cx + 0.20,
        fork_y,
        fork_w,
        fork_h,
        badge="④b",
        title="GPT Forecaster",
        subtitle="GPT-5.5  ·  reasoning = low",
        edge_color=COLOR["gpt"],
        fill_color=COLOR["gpt_fill"],
    )
    # Y-shaped split + merge
    arrow(ax, cx, 3.95, cx - fork_w / 2 - 0.20, 4.00)
    arrow(ax, cx, 3.95, cx + fork_w / 2 + 0.20, 4.00)
    arrow(ax, cx - fork_w / 2 - 0.20, 3.05, cx, 2.65)
    arrow(ax, cx + fork_w / 2 + 0.20, 3.05, cx, 2.65)

    # ── 5. Devil's Advocate
    main_box(
        y=1.70,
        badge="⑤",
        title="Devil's Advocate",
        subtitle="Sonnet 4.6  ·  red-teams both forecasts (no extended thinking)",
        edge_color=COLOR["sonnet"],
        fill_color=COLOR["sonnet_fill"],
    )
    arrow(ax, cx, 1.70, cx, 1.30)

    # ── 6. Aggregator
    main_box(
        y=0.35,
        badge="⑥",
        title="Aggregator",
        subtitle="Opus 4.7  ·  final calibrated probability + rationale",
        edge_color=COLOR["opus"],
        fill_color=COLOR["opus_fill"],
    )
    arrow(ax, cx, 0.35, cx, 0.05)

    # ── OUTPUT
    io_box(ax, cx - 3.5, -0.50, 7.0, 0.55, label='{"probabilities": [{"market": "...", "probability": 0.68}, ...]}')

    # Stretch limits so the output box isn't clipped
    ax.set_ylim(-0.7, 13)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"saved {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main(Path(__file__).resolve().parent / "pipeline.png")
