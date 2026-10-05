#!/usr/bin/env python3
"""Build docs/demo.html (self-contained, inline SVG, no JS) from docs/results/*.csv and configs/final_ensemble.json.

Run: python scripts/make_demo.py   (stdlib only; no competition data needed).
"""
import csv
import datetime as dt
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs/results"


def rows(name):
    with open(RES / name, newline="") as handle:
        return list(csv.DictReader(handle))


CFG = json.loads((ROOT / "configs/final_ensemble.json").read_text())
WEIGHTS = CFG["fitted_weights"]["members"]
LB = float(CFG["public_leaderboard"])
MEMBERS = rows("members.csv")
PROG = rows("leaderboard_progress.csv")
CV = rows("cv_vs_lb.csv")
ORACLE = rows("oracle_ceiling.csv")
SHORT = ["Qwen2.5-VL-7B", "Qwen2-VL-2B vMLP", "Qwen3-VL-4B", "Qwen2.5-VL-7B NF4", "TrOCR-large", "PP-OCRv6",
         "TrOCR cleaned", "2B soft-KD", "2B recipe SFT", "2B recipe+GRPO"]

ARROW = ('<defs><marker id="{0}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>')


def t(x, y, s, anchor="start", cls="", size=12):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" class="{cls}">{escape(str(s))}</text>'


def fig(svg, claim, caption):
    svg = svg.replace("<svg ", '<svg role="img" aria-label="%s" ' % escape(claim, quote=True), 1)
    return f"<figure>{svg}<figcaption>{escape(caption)}</figcaption></figure>"


def pipeline():
    p = [ARROW.format("a1")]
    # synthetic stand-in: the competition images may not be published (docs/COMPETITION.md)
    p.append('<rect x="8" y="134" width="114" height="36" rx="3" class="paper"/>')
    p.append('<text x="14" y="157" font-size="13" textLength="102" lengthAdjust="spacingAndGlyphs" class="ink">Planter of the Parish</text>')
    p += [t(65, 190, "line image", "middle"), t(65, 206, "(synthetic stand-in)", "middle", "dim", 11)]
    for i, name in enumerate(SHORT):
        y = 10 + i * 34
        p.append(f'<rect x="200" y="{y}" width="150" height="28" rx="4" class="box"/>')
        p.append(t(275, y + 19, f"{i} {name}", "middle", size=11))
        p.append(f'<line x1="120" y1="160" x2="198" y2="{y + 14}" class="thin" marker-end="url(#a1)"/>')
        p.append(f'<line x1="350" y1="{y + 14}" x2="438" y2="160" class="thin" marker-end="url(#a1)"/>')
    p.append('<rect x="440" y="120" width="130" height="80" rx="6" class="box"/>')
    p += [t(505, 150, "Candidate pool", "middle"), t(505, 168, "10 readings of", "middle", "dim", 11),
          t(505, 184, "the same line", "middle", "dim", 11)]
    p.append('<line x1="570" y1="160" x2="618" y2="160" class="thin" marker-end="url(#a1)"/>')
    p.append(t(594, 150, "score", "middle", "dim", 11))
    p.append('<rect x="620" y="110" width="170" height="100" rx="6" class="box hot"/>')
    p += [t(705, 138, "Reranker", "middle", "b"), t(705, 158, "weighted edit cost", "middle", "dim", 11),
          t(705, 174, "+ TrOCR NLL + char-LM", "middle", "dim", 11), t(705, 194, "pick the argmin", "middle", size=11)]
    p.append('<line x1="790" y1="160" x2="838" y2="160" class="thin" marker-end="url(#a1)"/>')
    p.append('<rect x="840" y="130" width="130" height="60" rx="6" class="box"/>')
    p += [t(905, 156, "Transcription", "middle"), t(905, 174, "one per line", "middle", "dim", 11)]
    svg = f'<svg viewBox="0 0 980 350" class="fig">{"".join(p)}</svg>'
    return fig(svg, "Ten recognizers each read the same line; a reranker scores their readings and outputs the best one.",
               "The input is a synthetic stand-in for a cropped line (the competition images cannot be published). Ten independent recognizers read every line. The reranker chooses among their readings instead of "
               "generating text itself, so the ensemble can only output something at least one member already wrote.")


def reranker():
    # illustrative only: no per-line data ships with the repo
    cands = [("Planter of the Parish", "7 members", [0, 4, 4]), ("Planter of the Parrish", "2 members", [4, 0, 4]),
             ("Planler of the Parish", "1 member", [4, 4, 0])]
    p = [ARROW.format("a2")]
    p += [t(10, 20, "Candidate (a member's reading)", size=11, cls="dim"), t(330, 20, "Support", size=11, cls="dim"),
          t(450, 20, "Weighted edit cost to every member", size=11, cls="dim"),
          t(700, 20, "+ language-model NLL", size=11, cls="dim"), t(880, 20, "Total", size=11, cls="dim")]
    totals = [9.1, 18.4, 31.7]
    for i, (text, sup, _) in enumerate(cands):
        y = 36 + i * 54
        win = i == 0
        p.append(f'<rect x="4" y="{y}" width="972" height="44" rx="6" class="box{" hot" if win else ""}"/>')
        p.append(t(14, y + 27, text, size=13))
        p.append(t(330, y + 27, sup, size=12, cls="dim"))
        w = [20, 70, 140][i]
        p.append(f'<rect x="450" y="{y + 14}" width="{w}" height="16" class="bar{" hotfill" if win else ""}"/>')
        p.append(t(700, y + 27, ["low", "medium", "high"][i], size=12, cls="dim"))
        p.append(t(880, y + 27, totals[i], size=13, cls="b" if win else ""))
    p.append('<line x1="940" y1="190" x2="940" y2="206" class="thin" marker-end="url(#a2)"/>')
    p.append(t(930, 224, "lowest total wins", "end", "b", 12))
    svg = f'<svg viewBox="0 0 980 236" class="fig">{"".join(p)}</svg>'
    return fig(svg, "A candidate that most members agree with and the language model likes gets the lowest cost and wins.",
               f"Illustrative example (invented numbers, no per-line data ships). cost(candidate) = sum over members of "
               f"w_m * (55*word_edits + 12*char_edits) + 0.01*NLL_TrOCR + 0.3*NLL_charLM. Member weights are {WEIGHTS}: "
               f"PP-OCRv6 counts three times, found by greedy hill-climb on 818 fold-0 lines.")


def lb_chart():
    xs = [dt.date.fromisoformat(r["date"]) for r in PROG]
    ys = [float(r["public_lb"]) for r in PROG]
    W, H, L, R, T, B = 980, 320, 56, 20, 20, 36
    x0, x1 = xs[0].toordinal(), xs[-1].toordinal()
    px = lambda d: L + (d.toordinal() - x0) / (x1 - x0) * (W - L - R)
    py = lambda v: T + (1 - (v - 0.6) / 0.35) * (H - T - B)
    p = []
    for v in (0.6, 0.7, 0.8, 0.9):
        p.append(f'<line x1="{L}" y1="{py(v):.1f}" x2="{W - R}" y2="{py(v):.1f}" class="grid"/>')
        p.append(t(L - 8, py(v) + 4, f"{v:.1f}", "end", "dim", 11))
    pts, prev = [], None
    for x, y in zip(xs, ys):
        if prev is not None:
            pts.append(f"{px(x):.1f},{py(prev):.1f}")
        pts.append(f"{px(x):.1f},{py(y):.1f}")
        prev = y
    p.append(f'<polyline points="{" ".join(pts)}" class="line" fill="none"/>')
    for x, y in zip(xs, ys):
        p.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5" class="dot"/>')
    for d, a in ((xs[0], "start"), (xs[len(xs) // 2], "middle"), (xs[-1], "end")):
        p.append(t(px(d), H - 12, d.isoformat(), a, "dim", 11))
    p.append(t(px(xs[-1]) - 8, py(ys[-1]) - 12, f"{ys[-1]:.4f}", "end", "b", 12))
    p.append(t(px(xs[0]) + 8, py(ys[0]) - 10, f"{ys[0]:.3f} Kraken zero-shot", "start", "dim", 11))
    svg = f'<svg viewBox="0 0 {W} {H}" class="fig">{"".join(p)}</svg>'
    return fig(svg, f"Public leaderboard score rose from {ys[0]:.3f} to {ys[-1]:.4f} over {len(xs)} submissions.",
               "Public leaderboard score (1 - combined error) by submission. The first fine-tuned VLM gave the largest "
               "single jump; ensembling and reranking added the last few points.")


def members_chart():
    ms = sorted(MEMBERS, key=lambda r: float(r["fold0_combined"]))
    W, rowh, L = 980, 30, 190
    H = 20 + rowh * len(ms) + 24
    scale = (W - L - 330) / 0.16
    p = []
    for i, r in enumerate(ms):
        y = 14 + i * rowh
        v = float(r["fold0_combined"])
        n = int(r["member"])
        p.append(t(L - 10, y + 17, f"{n} {SHORT[n]}", "end", size=12))
        p.append(f'<rect x="{L}" y="{y + 4}" width="{v * scale:.1f}" height="18" class="bar"/>')
        p.append(t(L + v * scale + 8, y + 18, f"{v:.4f}  (CER {float(r['fold0_cer']):.3f}, WER {float(r['fold0_wer']):.3f})",
                   size=11, cls="dim"))
    ens = float(ORACLE[3]["fold0_combined_error"])
    x = L + ens * scale
    p.append(f'<line x1="{x:.1f}" y1="6" x2="{x:.1f}" y2="{H - 24}" class="hotline"/>')
    p.append(t(x + 6, H - 8, f"reranked ensemble {ens:.4f}", "start", "b", 12))
    svg = f'<svg viewBox="0 0 {W} {H}" class="fig">{"".join(p)}</svg>'
    best = float(ms[0]["fold0_combined"])
    return fig(svg, f"Every single recognizer has fold-0 error of at least {best:.3f}; the reranked ensemble reaches {ens:.4f}.",
               "Fold-0 combined error (0.5*WER + 0.5*CER, lower is better) per member, sorted. The vertical line is the "
               "cross-validated reranked ensemble.")


def cv_chart():
    W, H, L, R, T, B = 980, 340, 56, 20, 20, 36
    lo, hi = 0.07, 0.165
    px = lambda v: L + (hi - v) / (hi - lo) * (W - L - R)   # lower error on the right
    py = lambda v: T + (1 - (v - 0.84) / 0.09) * (H - T - B)
    right = lambda x: x > W - 330
    p = []
    for v in (0.85, 0.88, 0.91):
        p.append(f'<line x1="{L}" y1="{py(v):.1f}" x2="{W - R}" y2="{py(v):.1f}" class="grid"/>')
        p.append(t(L - 8, py(v) + 4, f"{v:.2f}", "end", "dim", 11))
    for v in (0.08, 0.10, 0.12, 0.14, 0.16):
        p.append(t(px(v), H - 14, f"{v:.2f}", "middle", "dim", 11))
    p.append(t(W - R, H - 1, "fold-0 error (lower is better, axis reversed)", "end", "dim", 11))
    for r in CV:
        x, y = px(float(r["fold0_cv_combined"])), py(float(r["public_lb"]))
        final = "final" in r["submission"]
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{6 if final else 4.5}" class="dot{" hotfill" if final else ""}"/>')
        label = r["submission"].replace(" single model", "")
        if not (final or "single" in r["submission"]):
            continue
        dx, dy = (-10, -10) if final else (-10, 4) if x > W - 330 else (10, 4)
        p.append(t(x + dx, y + dy, label, "end" if dx < 0 else "start", size=11, cls="b" if final else "dim"))
    svg = f'<svg viewBox="0 0 {W} {H}" class="fig">{"".join(p)}</svg>'
    return fig(svg, "Fold-0 error and public leaderboard score move together; the final model is best on both.",
               "Local validation predicted the leaderboard to within about 0.01, so the reranker weights, fit on fold 0 only, "
               "carried over to the hidden test set.")


CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--dim:#52514e;--line:#c9c8c2;--grid:#e6e5e1;--blue:#2a78d6;--hot:#eb6834}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#141413;--ink:#f2f1ee;--dim:#a3a29c;--line:#4a4944;--grid:#2a2926;--blue:#5aa0f0;--hot:#ff8a5c}}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:1.8rem;margin:0 0 4px}h2{font-size:1.2rem;margin:40px 0 4px}
p.lead,figcaption{color:var(--dim);font-size:.92rem;margin:4px 0 12px}
figure{margin:12px 0}svg.fig{width:100%;height:auto;color:var(--ink);display:block}
svg text{fill:currentColor;font-family:system-ui,sans-serif}svg text.dim{fill:var(--dim)}svg text.b{font-weight:700}
.box{fill:none;stroke:var(--line);stroke-width:1.2}.box.hot{stroke:var(--hot);stroke-width:2}
.thin{stroke:currentColor;stroke-width:1;opacity:.5}.grid{stroke:var(--grid)}
.paper{fill:#efe3c8;stroke:#b9a77f}svg text.ink{fill:#2b2118;font-family:"Snell Roundhand","Apple Chancery","Segoe Script",cursive;font-style:italic}
table{border-collapse:collapse;width:100%;font-size:.85rem}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}th{color:var(--dim);font-weight:600}
td.n{text-align:right;font-variant-numeric:tabular-nums}.tw{overflow-x:auto}
.bar{fill:var(--blue)}.hotfill{fill:var(--hot)}.hotline{stroke:var(--hot);stroke-width:2;stroke-dasharray:5 4}
.line{stroke:var(--blue);stroke-width:2}.dot{fill:var(--blue);stroke:var(--bg);stroke-width:1.5}
"""


ROLE = ["LoRA, fp16, 707 test pseudo-labels", "LoRA + vision-MLP", "LoRA + vision-MLP, 128 px", "4-bit NF4 LoRA, 112 px",
        "full fine-tune, 192x1024", "kraken fine-tune, 12 epochs", "same as 4, on cleaned images",
        "LoRA + soft multi-candidate targets", "recipe SFT: curriculum, experts, cleaned mix", "recipe SFT + 200 rows of GRPO"]


def legend():
    rs = ['<tr><th>#</th><th>Model</th><th>Base</th><th>Recipe</th><th>Fold-0 error</th><th>Weight</th></tr>']
    for r in MEMBERS:
        n = int(r["member"])
        base = r["base_model"].split("/")[-1].replace("-Instruct", "")
        rs.append(f'<tr><td>{n}</td><td>{escape(SHORT[n])}</td><td>{escape(base)}</td><td>{escape(ROLE[n])}</td>'
                  f'<td class="n">{float(r["fold0_combined"]):.4f}</td><td class="n">x{WEIGHTS[n]}</td></tr>')
    return ('<div class="tw"><table>' + "".join(rs) + "</table></div>"
            '<p class="lead">Fold-0 error is 0.5*WER + 0.5*CER (lower is better). Weight is the member\'s multiplier in the reranker cost.</p>')


def main():
    body = [
        "<h1>R.O.A.D. Barbados: ten recognizers and a reranker</h1>",
        f'<p class="lead">Handwriting transcription of archival Barbados records. Public leaderboard {LB:.4f}. '
        "Every figure is drawn from <code>docs/results/*.csv</code> by <code>scripts/make_demo.py</code>.</p>",
        "<h2>The ten models</h2>", legend(),
        "<h2>1. How a line is read</h2>", pipeline(),
        "<h2>2. How the reranker picks</h2>", reranker(),
        "<h2>3. Why an ensemble</h2>", members_chart(),
        "<h2>4. Validation vs leaderboard</h2>", cv_chart(),
        "<h2>5. Leaderboard history</h2>", lb_chart(),
    ]
    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>Barbados Reranker Demo</title><style>{CSS}</style></head><body><main>{''.join(body)}</main></body></html>")
    (ROOT / "docs/demo.html").write_text(html)
    print("wrote docs/demo.html", len(html), "bytes")


if __name__ == "__main__":
    main()
