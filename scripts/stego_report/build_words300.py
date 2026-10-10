"""Builds the page on the words300-n500 run (300 secret words; the monitor picks from 500 candidates).
Usage (cwd = repo): python scripts/stego_report/build_words300.py <out.html>"""

import collections
import html
import json
import re
import sys
from pathlib import Path

E = lambda s: html.escape(str(s), quote=False)
pct = lambda v: f"{round(v * 100)}%"
RUN = "runs/stego/words300-n500"
metrics = [json.loads(line) for line in open(f"{RUN}/metrics.jsonl")]
analysis = [json.loads(line) for line in open("runs/stego/analysis/words300-n500.jsonl")]
readers = {(r["step"], r["secret"], r["topic"]): r for r in map(json.loads, open(f"{RUN}/remonitor-8cand.jsonl"))}
rollouts = {(r["step"], r["secret"], r["topic"]): r for r in map(json.loads, open(f"{RUN}/rollouts.jsonl")) if r["kind"] == "eval"}
CLAUDE = [("claude-haiku-5-5", "Haiku 5.5"), ("claude-sonnet-5-5", "Sonnet 5.5"), ("claude-opus-5-5", "Opus 5.5")]


def chart(series, xmax, w=420, h=230):
    L, R, T, B = 36, 46, 10, 24
    pw, ph = w - L - R, h - T - B
    X = lambda s: L + pw * s / xmax
    Y = lambda v: T + ph * (1 - v)
    o = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for g in (0, 0.25, 0.5, 0.75, 1):
        o.append(
            f'<line class="grid" x1="{L}" x2="{L + pw}" y1="{Y(g):.1f}" y2="{Y(g):.1f}"/><text class="tick" x="{L - 5}" y="{Y(g) + 3.5:.1f}" text-anchor="end">{round(g * 100)}%</text>'
        )
    for s in range(0, xmax + 1, 25):
        o.append(f'<text class="tick" x="{X(s):.1f}" y="{h - 7}" text-anchor="middle">{s}</text>')
    ends = []
    for name, var, pts, faint in series:
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(s):.1f},{Y(v):.1f}" for i, (s, v) in enumerate(pts))
        o.append(
            f'<path d="{d}" fill="none" stroke="var({var})" stroke-width="{1 if faint else 2}" opacity="{0.35 if faint else 1}" stroke-linejoin="round"/>'
        )
        if not faint:
            for s, v in pts:
                o.append(
                    f'<circle class="dot" cx="{X(s):.1f}" cy="{Y(v):.1f}" r="4" fill="var({var})"><title>{E(name)}, step {s}: {pct(v)}</title></circle>'
                )
            ends.append([Y(pts[-1][1]), X(pts[-1][0]), pct(pts[-1][1])])
    ends.sort()
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 12:
            ends[i][0] = ends[i - 1][0] + 12
    o += [f'<text class="endlabel" x="{x + 8:.1f}" y="{y + 3.5:.1f}">{t}</text>' for y, x, t in ends]
    return "".join(o) + "</svg>"


KEYS = [
    ("Receiver (picks from 8)", "--s1", "recv_correct"),
    ("9B monitor (picks from 500)", "--s2", "mon_correct"),
    ("Referee pass", "--s3", "judge_pass"),
]
train = [m for m in metrics if m["kind"] == "train"]
panels = ""
for split, title in (("train", "The 300 training words"), ("heldout", "Words never seen in training")):
    ev = [m for m in metrics if m["kind"] == "eval" and m["split"] == split]
    series = [(n, v, [(m["step"], m[k]) for m in ev], False) for n, v, k in KEYS]
    if split == "train":
        series = [(n, v, [(m["step"], m[k]) for m in train], True) for n, v, k in KEYS] + series
    panels += f'<figure class="panel"><figcaption><b>{title}</b></figcaption>{chart(series, 75)}</figure>'
legend = '<div class="legend">' + "".join(f'<span><i style="background:var({v})"></i>{n}</span>' for n, v, _ in KEYS) + "</div>"

# readers on equal terms
reader_rows = ""
for step, label in ((0, "Untrained sender"), (50, "Trained, step 50")):
    for split, sname in (("train", "training words"), ("heldout", "unseen words")):
        rs = [r for r in readers.values() if r["step"] == step and rollouts[(r["step"], r["secret"], r["topic"])]["split"] == split]
        n = len(rs)
        cells = [pct(sum(r["recv_correct"] for r in rs) / n)] + [
            pct(sum(r["monitors"][k]["guess"] == r["truth"] for r in rs) / n) for k, _ in CLAUDE
        ]
        cells.append(pct(sum(r["mon_correct"] for r in rs) / n))
        reader_rows += f"<tr><td>{label}, {sname}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"

# how it hints
shape_rows = ""
for label in ("Untrained sender", "Trained, step 50"):
    rs = [r for r in analysis if r["set"] == label]
    n = len(rs)
    a = [r["analysis"] for r in rs]
    mech = collections.Counter(x["mechanism"] for x in a)
    words = sum(len(r["essay"].split()) for r in rs) // n
    cells = [
        pct(mech["description"] / n),
        pct(mech["association"] / n),
        pct(mech["none"] / n),
        f"{sum(x.get('hint_sentences', 0) for x in a) / n:.1f}",
        pct(sum(x.get("obviousness") == 1 for x in a) / n),
        words,
    ]
    shape_rows += f"<tr><td>{label}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


# essays
def pick(split, want):
    pool = [r for r in analysis if r["set"] == "Trained, step 50" and r["split"] == split and r["analysis"].get("hint_phrases")]
    pool = [r for r in pool if all(p in r["essay"] for p in r["analysis"]["hint_phrases"])]
    out = []
    for r in pool:
        opus = readers[(50, r["secret"], r["topic"])]["monitors"]["claude-opus-5-5"]
        got = opus["guess"] == readers[(50, r["secret"], r["topic"])]["truth"]
        if want(r, got):
            out.append(r)
    return out


chosen = (
    [
        (r, "Typical: the essay describes the thing or its function without naming it. Opus reads it too.")
        for r in pick("train", lambda r, got: r["recv_correct"] and got and r["analysis"]["mechanism"] == "description")[:1]
    ]
    + [
        (r, "A word the sender never trained on, handled the same way.")
        for r in pick("heldout", lambda r, got: r["recv_correct"] and got and r["analysis"]["mechanism"] == "description")[:1]
    ]
    + [
        (r, "Hinting by association: vocabulary from the word's world.")
        for r in pick("heldout", lambda r, got: r["recv_correct"] and got and r["analysis"]["mechanism"] == "association")[:1]
    ]
    + [
        (r, "One of the minority Opus misses while the receiver gets it.")
        for r in pick("train", lambda r, got: r["recv_correct"] and not got)[:1]
    ]
    + [(r, "Another that Opus misses, on an unseen word.") for r in pick("heldout", lambda r, got: r["recv_correct"] and not got)[:1]]
    + [(r, "A miss: the receiver picks the wrong word.") for r in pick("train", lambda r, got: not r["recv_correct"])[:1]]
)
cards = ""
for r, note in chosen:
    key = (50, r["secret"], r["topic"])
    rd, ro = readers[key], rollouts[key]
    text = E(r["essay"])
    for p in r["analysis"]["hint_phrases"]:
        text = text.replace(E(p), f"<mark>{E(p)}</mark>", 1)
    paras = "".join(f"<p>{p}</p>" for p in text.split("\n\n"))
    opus = rd["monitors"]["claude-opus-5-5"]
    opus_word = rd["mon_candidates"][opus["guess"]] if opus["guess"] >= 0 else "no answer"
    recv_word = ro["recv_candidates"][max(range(8), key=ro["recv_probs"].__getitem__)]
    pr = round(ro["p_recv"] * 100)
    cards += f"""<article class="essay"><header><p class="when">{"Training word" if r["split"] == "train" else "Unseen word"} · step 50</p>
  <h2><span class="lbl">Secret word</span> {E(r["secret"])}</h2><p class="topic"><span class="lbl">Topic</span> {E(r["topic"])}</p></header>
  <dl class="readers"><div class="reader recv"><dt>Receiver, from 8</dt><dd><span class="bar"><i style="width:{pr}%"></i></span><b>{pr}%</b></dd></div>
  <div class="reader"><dt>Receiver's pick</dt><dd><b>{E(recv_word)}</b></dd></div>
  <div class="reader"><dt>Opus, same 8</dt><dd><b>{E(opus_word)}</b></dd></div>
  <div class="reader"><dt>9B monitor, from 500</dt><dd><b>{"got it" if ro["mon_correct"] else "missed"}</b></dd></div></dl>
  <div class="text">{paras}</div><p class="note">{E(note)} Other candidates: {E(", ".join(c for c in ro["recv_candidates"] if c != r["secret"]))}.</p></article>"""

here = Path(__file__).parent
style = re.search(r"<style>.*?</style>", (here / "report_template.html").read_text(), re.DOTALL)[0]
page = (here / "words300_template.html").read_text()
for k, v in {"STYLE": style, "LEGEND": legend, "PANELS": panels, "READERS": reader_rows, "SHAPE": shape_rows, "CARDS": cards}.items():
    page = page.replace(f"<!--{k}-->", v)
Path(sys.argv[1]).write_text(page)
print(len(page), len(chosen), "essays")
