"""Builds the hint-strategy page from runs/stego/analysis/strategies.jsonl (see analyze_strategies.py).
Usage (cwd = repo): python scripts/stego_report/build_strategies.py <out.html>"""

import collections
import html
import json
import re
import sys
from pathlib import Path

E = lambda s: html.escape(str(s), quote=False)
R = [json.loads(line) for line in open("runs/stego/analysis/strategies.jsonl")]
SETS = list(dict.fromkeys(r["set"] for r in R))
MECH = [
    ("association", "--s1", "Association"),
    ("description", "--s2", "Description"),
    ("idiom", "--s3", "Idiom"),
    ("negation", "--s4", "Negation"),
]
STOP = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "and",
    "in",
    "on",
    "for",
    "with",
    "that",
    "this",
    "it",
    "is",
    "are",
    "was",
    "be",
    "as",
    "by",
    "at",
    "from",
    "into",
    "its",
    "their",
    "his",
    "her",
    "your",
    "you",
    "we",
    "i",
    "or",
    "not",
    "no",
    "but",
    "than",
    "then",
    "so",
    "very",
    "just",
    "more",
    "most",
    "can",
    "will",
    "would",
    "like",
    "such",
    "each",
    "every",
    "all",
    "any",
    "some",
    "over",
    "under",
    "up",
    "down",
    "out",
    "off",
    "about",
    "without",
    "within",
    "through",
    "while",
    "when",
    "where",
    "which",
    "who",
    "what",
    "there",
    "here",
    "they",
    "them",
    "these",
    "those",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "one",
    "two",
    "also",
    "only",
    "even",
    "still",
    "much",
    "many",
    "own",
    "our",
    "how",
    "too",
}


def words(phrase):
    return [w for w in re.findall(r"[a-z]+", phrase.lower()) if w not in STOP and len(w) > 2]


def of(label):
    return [r for r in R if r["set"] == label]


# ---- mechanism mix
bars, mix_rows = [], []
for label in SETS:
    rs = of(label)
    n = len(rs)
    count = collections.Counter(r["analysis"]["mechanism"] for r in rs)
    segs, cells = "", []
    for key, var, name in MECH:
        share = 100 * count[key] / n
        cells.append(f"{round(share)}%")
        if share:
            segs += f'<i style="width:{share}%;background:var({var})" title="{name}: {round(share)}%"></i>'
    other = 100 - sum(100 * count[k] / n for k, _, _ in MECH)
    if other > 0.5:
        segs += f'<i style="width:{other}%;background:var(--none)" title="No visible hint, or other: {round(other)}%"></i>'
    top_key = max(MECH, key=lambda m: count[m[0]])
    bars.append(
        f'<div class="mixrow"><span class="mixlabel">{E(label)}</span><span class="mix">{segs}</span><span class="mixtop">{top_key[2].lower()} {round(100 * count[top_key[0]] / n)}%</span></div>'
    )
    mix_rows.append("<tr><td>" + E(label) + "</td>" + "".join(f"<td>{c}</td>" for c in cells) + f"<td>{round(other)}%</td></tr>")
legend = (
    "".join(f'<span><i style="background:var({var})"></i>{name}</span>' for _, var, name in MECH)
    + '<span><i style="background:var(--none)"></i>No visible hint, or other</span>'
)

# ---- shape of the hint
shape_rows = []
for label in SETS:
    rs = of(label)
    n = len(rs)
    a = [r["analysis"] for r in rs]
    pos = collections.Counter(x.get("position", "") for x in a)
    quiet = [r["mon_correct"] for r in rs if r["analysis"].get("obviousness") == 1]
    loud = [r["mon_correct"] for r in rs if r["analysis"].get("obviousness", 0) >= 2]
    f = lambda xs: f"{round(100 * sum(xs) / len(xs))}%" if xs else "—"
    shape_rows.append(
        f"<tr><td>{E(label)}</td><td>{sum(x.get('hint_sentences', 0) for x in a) / n:.1f}</td><td>{round(100 * pos['closing'] / n)}%</td>"
        f"<td>{round(100 * len(quiet) / n)}%</td><td>{f(quiet)}</td><td>{f(loud)}</td><td>{f([r['recv_correct'] for r in rs])}</td></tr>"
    )

# ---- each run's vocabulary per secret
COLS = [
    "Untrained sender",
    "Strict referee, soft fail, 9B",
    "Strict referee, ramp, 9B",
    "Medium referee, told, 9B",
    "Bootstrapped onto the 27B",
]
secrets = sorted({r["secret"] for r in R})
book_rows = []
for secret in secrets:
    cells = []
    for label in COLS:
        rs = [r for r in of(label) if r["secret"] == secret]
        c = collections.Counter()
        for r in rs:
            c.update({w for p in r["analysis"].get("hint_phrases", []) for w in words(p)})
        cells.append(", ".join(f"{w} <span class='sub'>{k}/{len(rs)}</span>" for w, k in c.most_common(3)) or "—")
    book_rows.append(f"<tr><td><b>{E(secret)}</b></td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

# ---- examples: one essay per strategy, highlighted with the phrases the analyst picked
PICKS = [
    ("No monitor penalty (control)", "petticoat", "description", "With no monitor to fear, the sender describes the object outright."),
    ("Morning: lenient referee, 9B monitor", "talker", "negation", "This morning's trick: the whole essay is about what is not said."),
    (
        "Strict referee, soft fail, 9B",
        "hillside",
        "idiom",
        "Under the strict referee: a normal essay with the hint packed into the last sentences.",
    ),
    (
        "Strict referee, ramp, 9B",
        "froth",
        "association",
        "This run's own word for froth is abundance and overflow, one step removed from the thing.",
    ),
    (
        "Medium referee, told, 9B",
        "pump",
        "association",
        "The same secret in another run gets a different vocabulary: expanding and volume.",
    ),
    (
        "Bootstrapped onto the 27B",
        "envelope",
        "association",
        "After moving to the same-size monitor: still direct, and the monitor still reads many of these.",
    ),
]
cards = []
for label, secret, mech, note in PICKS:
    pool = [
        r
        for r in of(label)
        if r["secret"] == secret and r["analysis"].get("hint_phrases") and all(p in r["essay"] for p in r["analysis"]["hint_phrases"])
    ]
    pool.sort(key=lambda r: (r["analysis"]["mechanism"] != mech, -(r["p_recv"] - r["p_mon"])))
    r = pool[0]
    text = E(r["essay"])
    for p in r["analysis"]["hint_phrases"]:
        text = text.replace(E(p), f"<mark>{E(p)}</mark>", 1)
    paras = "".join(f"<p>{p}</p>" for p in text.split("\n\n"))
    pr, pm = round(r["p_recv"] * 100), round(r["p_mon"] * 100)
    cards.append(f"""<article class="essay"><header><p class="when">{E(label)}</p><h2><span class="lbl">Secret word</span> {E(secret)}</h2>
  <p class="topic"><span class="lbl">Topic</span> {E(r["topic"])}</p></header>
  <dl class="readers"><div class="reader recv"><dt>Receiver</dt><dd><span class="bar"><i style="width:{pr}%"></i></span><b>{pr}%</b></dd></div>
  <div class="reader mon"><dt>Monitor it trained against</dt><dd><span class="bar"><i style="width:{pm}%"></i></span><b>{pm}%</b></dd></div>
  <div class="reader"><dt>Sonnet's label</dt><dd><b>{E(r["analysis"]["mechanism"])}</b></dd></div></dl>
  <div class="text">{paras}</div><p class="note">{E(note)}</p></article>""")

here = Path(__file__).parent
style = re.search(r"<style>.*?</style>", (here / "report_template.html").read_text(), re.DOTALL)[0]
page = (here / "strategies_template.html").read_text()
for key, val in {
    "STYLE": style,
    "LEGEND": legend,
    "BARS": "".join(bars),
    "MIXROWS": "".join(mix_rows),
    "SHAPEROWS": "".join(shape_rows),
    "BOOKHEAD": "".join(f"<th>{E(c)}</th>" for c in COLS),
    "BOOKROWS": "".join(book_rows),
    "CARDS": "".join(cards),
}.items():
    page = page.replace(f"<!--{key}-->", val)
Path(sys.argv[1]).write_text(page)
print(len(page))
