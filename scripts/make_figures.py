#!/usr/bin/env python3
"""Draw the three figures used in docs/SOLUTION.md from docs/results/*.csv (no competition data).

Run: python scripts/make_figures.py   (needs matplotlib). Writes docs/figures/*.png.
Palette: blue #2a78d6 and orange #eb6834 (passed the palette validator in light mode), muted
gray for neutral references; text in ink tokens, never in series colour.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "docs/results", ROOT / "docs/figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a3a29c"


def rows(name):
    with open(RES / name, newline="") as handle:
        return list(csv.DictReader(handle))


def base(figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, facecolor=SURFACE)
    plt.close(fig)


def lb_progression():
    import datetime as dt

    import matplotlib.dates as mdates

    data = rows("leaderboard_progress.csv")
    xs = [dt.date.fromisoformat(r["date"]) for r in data]
    ys = [float(r["public_lb"]) for r in data]
    fig, ax = base((8.2, 4.4))
    ax.step(xs, ys, where="post", color=BLUE, linewidth=1.8)
    ax.plot(xs, ys, "o", color=BLUE, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2)
    # index -> (label, dx, dy, alignment), offsets in points
    marks = {0: ("Kraken zero-shot", -2, -30, "left"), 2: ("Qwen2-VL-2B LoRA", 8, -28, "left"),
             4: ("Qwen2.5-VL-3B + repetition ban", 8, -28, "left"), 14: ("TrOCR as 5th member", -6, -34, "right"),
             len(data) - 1: ("hill-climb, 10 members (final)", 4, 10, "right")}
    for i, (text, dx, dy, ha) in marks.items():
        ax.annotate(f"{text}\n{ys[i]:.4f}", (xs[i], ys[i]), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=7.5, color=INK2)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_xlabel("date of submission (2026)", color=INK2, fontsize=8)
    ax.set_ylabel("public leaderboard score", color=INK2, fontsize=8)
    ax.set_ylim(0.58, 0.95)
    ax.set_title("Public leaderboard score over the submissions that changed something", loc="left",
                 fontsize=10, color=INK)
    save(fig, "lb_progression.png")


def cv_vs_lb():
    data = sorted(rows("cv_vs_lb.csv"), key=lambda r: float(r["public_lb"]))
    fig, ax = base((8.2, 3.8))
    ax.grid(visible=False)
    for y, r in enumerate(data):
        fold0, lb = 1 - float(r["fold0_cv_combined"]), float(r["public_lb"])
        ax.plot([fold0, lb], [y, y], color=GRAY, linewidth=1.2, zorder=1)
        ax.plot(fold0, y, "o", color=BLUE, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=2)
        ax.plot(lb, y, "o", color=ORANGE, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=2)
        ax.text(0.8, y, r["submission"], va="center", ha="left", fontsize=7.5, color=INK2)
    ax.set_yticks([])
    ax.set_xlim(0.78, 0.935)
    ax.set_ylim(-0.7, len(data) - 0.3)
    ax.set_xlabel("score (1 minus combined error)", color=INK2, fontsize=8)
    ax.plot([], [], "o", color=BLUE, label="fold 0 (1 - CV)")
    ax.plot([], [], "o", color=ORANGE, label="public leaderboard")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)
    ax.set_title("The leaderboard beat fold 0 by about 0.004 to 0.008 for ensembles", loc="left",
                 fontsize=10, color=INK)
    save(fig, "cv_vs_lb.png")


def oracle_ceiling():
    data = rows("oracle_ceiling.csv")
    order = sorted(data, key=lambda r: float(r["fold0_combined_error"]))
    fig, ax = base((8.2, 3.4))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    for y, r in enumerate(order):
        v = float(r["fold0_combined_error"])
        oracle = r["pool_or_method"].startswith(("best of", "7 members", "8 members"))
        ax.barh(y, v, height=0.62, color=GRAY if oracle else BLUE, edgecolor=SURFACE, linewidth=2)
        ax.text(v + 0.0012, y, f"{v:.4f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([("oracle: " if r["pool_or_method"].startswith(("best of", "7 members", "8 members")) else "") +
                        r["pool_or_method"] for r in order], fontsize=8, color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.105)
    ax.set_xlabel("fold-0 combined error (lower is better)", color=INK2, fontsize=8)
    ax.set_title("Oracle bounds vs what rerankers recover (fold 0)",
                 loc="left", fontsize=10, color=INK)
    save(fig, "oracle_ceiling.png")


if __name__ == "__main__":
    lb_progression()
    cv_vs_lb()
    oracle_ceiling()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
