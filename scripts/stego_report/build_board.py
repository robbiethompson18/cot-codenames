"""Builds the live experiment board. Usage (cwd = repo): python scripts/stego_report/build_board.py <launch log dir> <out.html>

For each run it reads the training log straight off the pod (host and port come from the launch log). Once a pod has
deleted itself, it falls back to the run's metrics.jsonl on Hugging Face.
"""

import html
import json
import re
import subprocess
import sys
import time
from pathlib import Path

LOGS, OUT = Path(sys.argv[1]), sys.argv[2]
REPO = "robbiethompson2018/cot-codenames-stego"
KEY = str(Path.home() / ".runpod/ssh/RunPod-Key-Go")
E = lambda s: html.escape(str(s), quote=False)
pct = lambda v: f"{round(v * 100)}%"

# (run, steps, what changes, the question it answers)
RUNS = [
    ("repro-lenient-9b", 100, "this morning's recipe exactly: lenient referee · 9B monitor", "Does the recipe that worked still work on tonight's code?"),
    ("lenient-27b", 100, "lenient referee · 27B monitor", "Changes only the monitor. Is the same-size monitor alone enough to stop learning?"),
    ("strict-9b-ramp", 100, "strict referee · 9B monitor · monitor penalty off until step 15, full by 45", "Let it learn to communicate inside the strict referee first, then ask it to hide."),
    ("strict-9b-softfail", 100, "strict referee · 9B monitor · failed essay costs 0.2, not 1", "Is the size of the failure penalty what drives the sender to give up?"),
    ("fixed8-9b", 100, "8 fixed words · 9B monitor · new referee · no reasoning", "Does the recipe that worked this morning still learn once off-topic essays are banned?"),
    ("cot300-9b-both", 100, "sender reasons (300 tokens) · 9B monitor reads reasoning + essay", "Trained against a reasoning monitor: does the reasoning stop naming the word, and what replaces it?"),
    ("bootstrap-27b-from-runpod-fixed8", 100, "starts from this morning's 9B-trained checkpoint · 27B monitor", "Bootstrapping: can a sender that already has a code keep it against a same-size monitor?"),
    ("fixed8-strict", 100, "8 fixed words · 27B monitor · no reasoning", "Trained from scratch against a same-size monitor. Collapsed to chance by step 25."),
]
OFFLINE = [
    ("Cross-run decoding", "Does one run's receiver decode another run's essays? If yes, the code is not private to a LoRA.", "queued"),
    ("Paraphrase test", "Does the receiver still decode after Haiku rewrites the essay?", "queued"),
    ("Detection monitor", "Can a monitor tell trained essays from honest ones, yes or no?", "queued"),
    ("27B monitor with reasoning on", "Separates 'smaller model' from 'no reasoning' in the monitor's handicap.", "queued"),
    ("White-box first look", "Logit lens and the activation oracle on the sender at essay positions.", "queued"),
]


def fetch(run):
    """(log lines, source) from the pod, else from Hugging Face, else nothing."""
    launch = LOGS / f"launch-{run}.log"
    m = re.search(r"started on \S+ \((\S+):(\d+)\)", launch.read_text()) if launch.exists() else None
    if m:
        cmd = ["ssh", "-i", KEY, "-p", m[2], "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]
        cmd += [f"root@{m[1]}", f"tr '\\r' '\\n' < /workspace/{run}.log | grep -E '^EVAL|^\\{{|Traceback|Error' | tail -400"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
        if r.returncode == 0:
            return r.stdout.splitlines(), "pod"
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(REPO, f"runs/{run}/metrics.jsonl", force_download=True)
        lines = []
        for line in open(path):
            d = json.loads(line)
            lines.append(("EVAL " if d.pop("kind") == "eval" else "") + json.dumps(d))
        return lines, "hf"
    except Exception:
        return [], "none"


def parse(lines):
    evals, train, errors = [], [], []
    for line in lines:
        if line.startswith("EVAL "):
            evals.append(json.loads(line[5:]))
        elif line.startswith("{"):
            train.append(json.loads(line))
        elif "Traceback" in line or "Error" in line:
            errors.append(line[:160])
    return evals, train, errors


def curve(evals, split, xmax, w=250, h=150):
    pts = [e for e in evals if e["split"] == split]
    if not pts:
        return ""
    L, T, B, R = 30, 10, 22, 40
    pw, ph = w - L - R, h - T - B
    X = lambda s: L + pw * s / xmax
    Y = lambda v: T + ph * (1 - v)
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for g in (0, 0.5, 1):
        out.append(
            f'<line x1="{L}" x2="{L + pw}" y1="{Y(g):.1f}" y2="{Y(g):.1f}" class="grid"/><text x="{L - 5}" y="{Y(g) + 3.5:.1f}" class="tick" text-anchor="end">{round(g * 100)}%</text>'
        )
    out.append(f'<line x1="{L}" x2="{L + pw}" y1="{Y(0.125):.1f}" y2="{Y(0.125):.1f}" class="chance"/>')
    for s in (0, xmax // 2, xmax):
        out.append(f'<text x="{X(s):.1f}" y="{h - 6}" class="tick" text-anchor="middle">{s}</text>')
    for key, var, name in (("recv_correct", "--s1", "Receiver"), ("mon_correct", "--s2", "Monitor")):
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(e['step']):.1f},{Y(e[key]):.1f}" for i, e in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="var({var})" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for e in pts:
            out.append(
                f'<circle cx="{X(e["step"]):.1f}" cy="{Y(e[key]):.1f}" r="3.5" fill="var({var})" class="dot"><title>{name}, step {e["step"]}: {pct(e[key])}</title></circle>'
            )
        out.append(f'<text x="{X(pts[-1]["step"]) + 7:.1f}" y="{Y(pts[-1][key]) + 3.5:.1f}" class="endlabel">{pct(pts[-1][key])}</text>')
    return "".join(out) + "</svg>"


rows, panels = [], []
for run, steps, what, why in RUNS:
    lines, src = fetch(run)
    evals, train, errors = parse(lines)
    done = train[-1]["step"] + 1 if train else 0
    last = {s: next((e for e in reversed(evals) if e["split"] == s), None) for s in ("train", "heldout")}
    if errors and done < steps:
        status, cls = "crashed", "bad"
    elif done >= steps:
        status, cls = "finished", "good"
    elif train or evals:
        status, cls = f"step {done} of {steps}", "live"
    else:
        status, cls = ("setting up" if src != "none" or (LOGS / f"launch-{run}.log").exists() else "not started"), "wait"

    def cell(e):
        return f"{pct(e['recv_correct'])} / {pct(e['mon_correct'])}" if e else "—"

    ref = pct(last["train"]["judge_pass"]) if last["train"] else "—"
    at = f"step {last['train']['step']}" if last["train"] else ""
    secs = f"{sum(t['secs'] for t in train[-10:]) / len(train[-10:]):.0f}s" if train else "—"
    rows.append(
        f'<tr><td><b>{E(run)}</b><span class="sub">{E(what)}</span></td><td><span class="pill {cls}">{E(status)}</span></td>'
        f'<td>{cell(last["train"])}<span class="sub">{at}</span></td><td>{cell(last["heldout"])}</td><td>{ref}</td><td>{secs}</td></tr>'
    )
    charts = "".join(
        f'<div><span class="sub">{label}</span>{curve(evals, split, steps)}</div>'
        for split, label in (("train", "training words"), ("heldout", "held-out words"))
        if curve(evals, split, steps)
    )
    err = f'<p class="err">{E(errors[-1])}</p>' if errors else ""
    panels.append(
        f'<article class="run"><h3>{E(run)} <span class="pill {cls}">{E(status)}</span></h3><p class="why">{E(why)}</p>{err}<div class="curves">{charts or "<p class=sub>No evaluation yet.</p>"}</div></article>'
    )

summary_path = Path("runs/stego/offline/summary.json")
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
OFFLINE = [
    (n, q + (" Result: " + summary[n]["result"] if summary.get(n, {}).get("result") else ""), summary.get(n, {}).get("status", s))
    for n, q, s in OFFLINE
]
offline = "".join(
    f'<tr><td><b>{E(n)}</b><span class="sub">{E(q)}</span></td><td><span class="pill {"good" if s == "done" else "live" if s == "running" else "wait"}">{E(s)}</span></td></tr>'
    for n, q, s in OFFLINE
)
notes_path = Path(__file__).with_name("board_notes.html")
notes = notes_path.read_text() if notes_path.exists() else ""
stamp = time.strftime("%-I:%M %p Pacific, %-d %B %Y", time.localtime())
page = Path(__file__).with_name("board_template.html").read_text()
for key, val in {"STAMP": stamp, "ROWS": "".join(rows), "PANELS": "".join(panels), "OFFLINE": offline, "NOTES": notes}.items():
    page = page.replace(f"<!--{key}-->", val)
Path(OUT).write_text(page)
print(stamp, "|", "; ".join(re.sub(r"<[^>]+>", " ", r)[:90] for r in rows))
