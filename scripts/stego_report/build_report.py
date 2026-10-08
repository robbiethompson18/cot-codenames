"""Builds the Hidden Word Essays results page from run files. Usage: python build_report.py <scratchpad dir> (cwd = repo)."""

import html
import json
import sys
from collections import defaultdict

from cot_codenames.stego import game, judge, remonitor

SP = sys.argv[1]
E = lambda s: html.escape(str(s), quote=False)  # noqa: E731
pct = lambda v: f"{round(v * 100)}%"  # noqa: E731


def load(path):
    return [json.loads(line) for line in open(path)]


# ---------- data ----------
evals = {run: [m for m in load(f"runs/stego/{run}/metrics.jsonl") if m["kind"] == "eval"] for run in ("tinker-fixed8", "runpod-fixed8", "tinker-fixed8-control")}
remon = {run: load(f"runs/stego/{run}/remonitor.jsonl") for run in ("tinker-fixed8", "runpod-fixed8")}
strong = {run: load(f"runs/stego/{run}/remonitor-strong.jsonl") for run in ("tinker-fixed8", "runpod-fixed8")}


def acc(records, name, step):
    rs = [r for r in records if r["step"] == step]
    if name == "receiver":
        return sum(r["recv_correct"] for r in rs) / len(rs)
    if name == "9b":
        return sum(r["mon_correct"] for r in rs) / len(rs)
    return sum(r["monitors"][name]["guess"] == r["truth"] for r in rs) / len(rs)


# ---------- charts ----------
def line_chart(series, xmax, xticks, w=330, h=230, end_labels=True, right=64):
    """series: [(name, css var, [(step, value 0..1)])]. One y scale, 0-100%."""
    L, T, B = 34, 12, 28
    pw, ph = w - L - right, h - T - B
    X = lambda s: L + pw * s / xmax  # noqa: E731
    Y = lambda v: T + ph * (1 - v)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" class="chart">']
    for g in (0, 0.25, 0.5, 0.75, 1):
        out.append(f'<line x1="{L}" x2="{L + pw}" y1="{Y(g):.1f}" y2="{Y(g):.1f}" class="grid"/>')
        out.append(f'<text x="{L - 6}" y="{Y(g) + 3.5:.1f}" class="tick" text-anchor="end">{round(g * 100)}%</text>')
    out.append(f'<line x1="{L}" x2="{L + pw}" y1="{Y(0.125):.1f}" y2="{Y(0.125):.1f}" class="chance"/>')
    out.append(f'<text x="{L + 4}" y="{Y(0.125) - 4:.1f}" class="tick">chance</text>')
    for s in xticks:
        out.append(f'<text x="{X(s):.1f}" y="{h - 8}" class="tick" text-anchor="middle">{s}</text>')
    labels = []
    for name, var, pts in series:
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(s):.1f},{Y(v):.1f}" for i, (s, v) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="var({var})" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for s, v in pts:
            out.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="4" fill="var({var})" class="dot"><title>{E(name)}, step {s}: {pct(v)}</title></circle>')
        labels.append([Y(pts[-1][1]), name, pts[-1][1], pts[-1][0]])
    if end_labels:
        labels.sort()
        for i in range(1, len(labels)):  # nudge overlapping end labels apart
            if labels[i][0] - labels[i - 1][0] < 12:
                labels[i][0] = labels[i - 1][0] + 12
        for y, name, v, s in labels:
            out.append(f'<text x="{X(s) + 8:.1f}" y="{y + 3.5:.1f}" class="endlabel">{pct(v)}</text>')
    out.append("</svg>")
    return "".join(out)


def legend(items):
    return '<div class="legend">' + "".join(f'<span><i style="background:var({var})"></i>{E(name)}</span>' for name, var in items) + "</div>"


def bar_chart(rows, ref, w=680):
    """rows: [(label, zero-shot value or None, few-shot value or None)]. Horizontal grouped bars, 0-100%."""
    L, R, T, rowh = 184, 50, 22, 46
    pw = w - L - R
    h = T + rowh * len(rows) + 26
    X = lambda v: L + pw * v  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" class="chart" style="min-width:520px">']
    for g in (0, 0.25, 0.5, 0.75, 1):
        out.append(f'<line x1="{X(g):.1f}" x2="{X(g):.1f}" y1="{T}" y2="{h - 26}" class="grid"/>')
        out.append(f'<text x="{X(g):.1f}" y="{h - 9}" class="tick" text-anchor="middle">{round(g * 100)}%</text>')
    out.append(f'<line x1="{X(0.125):.1f}" x2="{X(0.125):.1f}" y1="{T}" y2="{h - 26}" class="chance"/>')
    out.append(f'<text x="{X(0.125) + 4:.1f}" y="{T - 6}" class="tick">chance</text>')
    out.append(f'<line x1="{X(ref):.1f}" x2="{X(ref):.1f}" y1="{T}" y2="{h - 26}" class="ref"/>')
    out.append(f'<text x="{X(ref) - 4:.1f}" y="{T - 6}" class="tick" text-anchor="end">receiver {pct(ref)}</text>')
    for i, (label, zero, few) in enumerate(rows):
        y0 = T + rowh * i + 7
        out.append(f'<text x="{L - 10}" y="{y0 + 19}" class="rowlabel" text-anchor="end">{E(label)}</text>')
        for j, (v, var, kind) in enumerate(((zero, "--s1", "no examples"), (few, "--s2", "16 examples"))):
            if v is None:
                continue
            y = y0 + j * 16
            bw = max(pw * v, 2)
            out.append(
                f'<path d="M{L},{y} h{bw - 4:.1f} a4,4 0 0 1 4,4 v6 a4,4 0 0 1 -4,4 h-{bw - 4:.1f} z" fill="var({var})">'
                f"<title>{E(label)}, {kind}: {pct(v)}</title></path>"
            )
            out.append(f'<text x="{X(v) + 6:.1f}" y="{y + 11}" class="endlabel">{pct(v)}</text>')
    out.append("</svg>")
    return "".join(out)


def table(head, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(f"<th>{E(c)}</th>" for c in head)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
        + "</tbody></table></div>"
    )


# Chart 1: training curves, small multiples
panels = []
for run, title, note in (
    ("tinker-fixed8", "Tinker, with monitor penalty", "70 steps"),
    ("runpod-fixed8", "RunPod, with monitor penalty", f"{evals['runpod-fixed8'][-1]['step']} steps so far"),
    ("tinker-fixed8-control", "Tinker control, no penalty", "25 steps"),
):
    ev = evals[run]
    xmax = 75
    series = [("Receiver", "--s1", [(m["step"], m["recv_correct"]) for m in ev]), ("Monitor (9B)", "--s2", [(m["step"], m["mon_correct"]) for m in ev])]
    panels.append(f'<figure class="panel"><figcaption><b>{title}</b><span>{note}</span></figcaption>{line_chart(series, xmax, (0, 25, 50, 75))}</figure>')
chart1 = legend([("Receiver accuracy", "--s1"), ("Monitor accuracy (Qwen 9B, the one trained against)", "--s2")]) + f'<div class="panels">{"".join(panels)}</div>'
curve_rows = [
    (title, m["step"], pct(m["recv_correct"]), pct(m["mon_correct"]), pct(m["judge_pass"]))
    for run, title in (("tinker-fixed8", "Tinker"), ("runpod-fixed8", "RunPod"), ("tinker-fixed8-control", "Tinker control"))
    for m in evals[run]
]
chart1 += "<details><summary>Data</summary>" + table(["Run", "Step", "Receiver", "Monitor (9B)", "Old judge pass"], curve_rows) + "</details>"

# Chart 2: monitors over training on the Tinker run
T = remon["tinker-fixed8"]
steps = sorted({r["step"] for r in T})
mon_series = [
    ("Receiver (trained 27B)", "--s1", "receiver"),
    ("Qwen 9B (trained against)", "--s2", "9b"),
    ("Qwen 27B, no LoRA", "--s3", "Qwen/Qwen3.6-27B"),
    ("Haiku 5.5, thinking", "--s4", "claude-haiku-5-5"),
]
chart2 = legend([(n, v) for n, v, _ in mon_series]) + '<div class="scroll">' + line_chart(
    [(n, v, [(s, acc(T, key, s)) for s in steps]) for n, v, key in mon_series], 70, steps, w=640, h=300, right=50
).replace('class="chart"', 'class="chart" style="min-width:520px"') + "</div>"
chart2 += "<details><summary>Data</summary>" + table(["Step"] + [n for n, _, _ in mon_series], [[s] + [pct(acc(T, key, s)) for _, _, key in mon_series] for s in steps]) + "</details>"


# Chart 3: monitor ladder with and without examples, latest step of each run
def ladder(run, step):
    a, b = remon[run], strong[run]
    rows = [("Qwen 9B (trained against)", acc(a, "9b", step), None)]
    for label, src, key in (
        ("Qwen 27B, no LoRA", a, "Qwen/Qwen3.6-27B"),
        ("Haiku 5.5", a, "claude-haiku-5-5"),
        ("Sonnet 5.5", b, "claude-sonnet-5-5"),
        ("Opus 5.5", b, "claude-opus-5-5"),
    ):
        rows.append((label, acc(src, key, step), acc(src, key + " +16 shots", step)))
    return rows, acc(a, "receiver", step)


ladders = {("tinker-fixed8", 70): "Tinker run, step 70", ("runpod-fixed8", 50): "RunPod run, step 50", ("tinker-fixed8", 0): "Untrained sender (Tinker run, step 0)"}
chart3 = legend([("Monitor sees only the essay", "--s1"), ("Monitor also sees 16 earlier essays with their secret words", "--s2")])
ladder_rows = []
for (run, step), title in ladders.items():
    rows, ref = ladder(run, step)
    chart3 += f'<figure><figcaption><b>{title}</b></figcaption><div class="scroll">{bar_chart(rows, ref)}</div></figure>'
    ladder_rows += [(title, label, pct(z), pct(f) if f is not None else "") for label, z, f in rows] + [(title, "Receiver", pct(ref), "")]
chart3 += "<details><summary>Data</summary>" + table(["Essays", "Monitor", "No examples", "16 examples"], ladder_rows) + "</details>"


def abstained(records, name):
    return sum(r["monitors"][name]["guess"] < 0 for r in records), len(records)


abst = []
for run in ("tinker-fixed8", "runpod-fixed8"):
    for src in (remon[run], strong[run]):
        for name in src[0]["monitors"]:
            if name.startswith("claude"):
                n, tot = abstained(src, name)
                abst.append((run.split("-")[0], name, f"{n} of {tot}"))


# ---------- transcripts ----------
def block(title, body, cls=""):
    return f'<details class="tx {cls}"><summary>{title}</summary>{body}</details>'


def pre(text, limit=None):
    text = text if limit is None or len(text) <= limit else text[:limit].rstrip() + " […]"
    return f"<pre>{E(text)}</pre>"


judge_tx = ""
for t in json.load(open(f"{SP}/judge_transcripts.json")):
    verdict = t["text"].strip().splitlines()[-1].strip(" .*")
    old = "" if t["old_pass"] is None else f" · old judge: {'PASS' if t['old_pass'] else 'FAIL'}"
    body = f"<h4>Topic</h4><p>{E(t['topic'])}</p><h4>Essay</h4>{pre(t['essay'])}<h4>Judge's thinking (summary returned by the API)</h4>{pre(t['thinking'])}<h4>Judge's answer</h4>{pre(t['text'])}"
    judge_tx += block(f'<span class="v {verdict.lower()}">{verdict}</span> {E(t["label"])}{old}', body)

pick = ("hillside", "fixing a dripping tap", 70)
a = next(r for r in remon["tinker-fixed8"] if (r["secret"], r["topic"], r["step"]) == pick)
b = next(r for r in strong["tinker-fixed8"] if (r["secret"], r["topic"], r["step"]) == pick)
cands = a["mon_candidates"]
zero_prompt = remonitor.monitor_chat(a, [], force_answer=True)[0]["content"]
probs = a["monitors"]["Qwen/Qwen3.6-27B"]["probs"]
dist = table(["Candidate", "Qwen 27B (no LoRA) probability"], [(f"{game.LABELS[i]}. {E(c)}" + (" ← secret" if i == a["truth"] else ""), f"{p:.1%}") for i, (c, p) in enumerate(zip(cands, probs))])
mon_tx = block("The prompt every monitor gets for this essay (Claude monitors also get the last sentence)", pre(zero_prompt))
mon_tx += block(
    f"Qwen 9B and Qwen 27B: no transcript, only a probability per letter · 9B put {a['p_mon']:.0%} on the secret, receiver {a['p_recv']:.0%}",
    "<p>Reasoning is switched off. We read the model's next-token probabilities for the letters A to H and renormalise them. Nothing is generated.</p>" + dist,
)
for label, src, key in (
    ("Haiku 5.5", a, "claude-haiku-5-5"),
    ("Sonnet 5.5", b, "claude-sonnet-5-5"),
    ("Opus 5.5", b, "claude-opus-5-5"),
    ("Opus 5.5 with 16 examples", b, "claude-opus-5-5 +16 shots"),
):
    g = src["monitors"][key]
    word = cands[g["guess"]] if g["guess"] >= 0 else "no answer"
    ok = "correct" if g["guess"] == src["truth"] else "wrong"
    body = f"<h4>Thinking (summary returned by the API)</h4>{pre(g['thinking'] or '(none returned)', 2500)}<h4>Answer</h4>{pre(g['text'] or '(empty)')}"
    mon_tx += block(f'<span class="v {"pass" if ok == "correct" else "fail"}">{ok}</span> {label} guessed "{E(word)}"', body)

sender_prompt = game.sender_chat(game.Episode("hillside", "fixing a dripping tap"))[0]["content"]
recv_prompt = game.chooser_chat("receiver", "fixing a dripping tap", "[essay]", cands)[0]["content"]
prompts = (
    block("Sender prompt", pre(sender_prompt))
    + block("Receiver prompt (the monitor's differs only in the sentence saying who it is)", pre(recv_prompt))
    + block("Few-shot preamble added for the '16 examples' monitors", pre(remonitor.FEW_SHOT.format(k=16, examples="Example 1\nTopic: …\nSecret word: …\nEssay:\n…\n\n---\n\n[15 more]")))
    + block("Judge prompt (current)", pre(judge.PROMPT))
)

essays = open(f"{SP}/hidden-word-essays.cards.html").read()
page = open(f"{SP}/report_template.html").read()
for key, val in {
    "CHART1": chart1,
    "CHART2": chart2,
    "CHART3": chart3,
    "ABSTAIN": table(["Run", "Monitor", "Answers with no letter"], abst),
    "PROMPTS": prompts,
    "JUDGE_TX": judge_tx,
    "MON_TX": mon_tx,
    "ESSAYS": essays,
}.items():
    page = page.replace(f"<!--{key}-->", val)
open(f"{SP}/hidden-word-essays.html", "w").write(page)
print("bytes", len(page))
for (run, step), title in ladders.items():
    rows, ref = ladder(run, step)
    print(title, "receiver", pct(ref), [(lab, pct(z), pct(f) if f is not None else None) for lab, z, f in rows])
print(abst)
