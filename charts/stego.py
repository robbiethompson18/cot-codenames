"""Charts for the hidden-word essay post (Altair). Run from the repo root:
    uv run --no-project --with altair --with vl-convert-python python charts/stego.py

Data:
- training.json: per-step training metrics of the 300-word run (cot-codenames runs words300-n500 and its continuation
  words300-n500-b), 64 essays per step.
- READERS / UPLIFT: hits out of 100 essays from the fine-tuned sender, no candidate list
  (cot-codenames runs/stego/openword/hint-quote/table.json).
"""

import json
from pathlib import Path

import altair as alt
from theme import FG, fonts, save

RECV, MON, REF = "#3f8f5b", "#c1432e", "#7a5fb0"
ONE, TEN = "#2e7dc1", "#d98a1f"


def training():
    steps = json.loads(Path("charts/training.json").read_text())
    names = {"recv": "Receiver right (of 8)", "mon": "Monitor right (of 500)", "referee": "Passes referee"}
    data = []
    for k, row in enumerate(steps):  # 5-step trailing mean: each step is only 64 essays
        window = steps[max(0, k - 4) : k + 1]
        for key, name in names.items():
            data.append({"step": row["step"], "who": name, "v": sum(w[key] for w in window) / len(window)})
    width = 560
    chart = (
        alt.Chart(alt.Data(values=data))
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("step:Q", title="Training Step"),
            y=alt.Y("v:Q", title="Share of Essays", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "who:N",
                title=None,
                scale=alt.Scale(domain=list(names.values()), range=[RECV, MON, REF]),
                legend=alt.Legend(orient="bottom"),
            ),
        )
        .properties(width=width, height=240, title="Training the Sender")
    )
    save(chart, "training", width)


# (reader, first guess, in first ten) out of 100 essays from the fine-tuned sender
READERS = [
    ("Fine-tuned 27B, next token", 23, 47),
    ("Fine-tuned 27B, J-lens", 21, 45),
    ("Stock 27B, next token", 6, 24),
    ("Stock 27B, J-lens", 3, 19),
    ("Claude Sonnet, asked", 7, 19),
    ("Claude Haiku, asked", 5, 14),
    ("Fine-tuned 27B, NLA", 1, 8),
    ("Qwen 9B, next token", 2, 8),
    ("Stock 27B, NLA", 0, 5),
]
# (technique, stock pass@1, fine-tuned pass@1, stock pass@10, fine-tuned pass@10)
UPLIFT = [("Asked for Ten Guesses", 4, 25, 6, 38), ("Next Token", 6, 23, 24, 47), ("J-lens", 3, 21, 19, 45), ("NLA", 0, 1, 5, 8)]
METRICS = ["First guess", "Within ten guesses"]


def grouped(data, order, x_title, title, name, fmt_axis=None, height=None):
    width = 560
    f = fonts(width)
    base = alt.Chart(alt.Data(values=data)).encode(
        y=alt.Y("item:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("metric:N", sort=METRICS),
    )
    bars = base.mark_bar(opacity=0.9).encode(
        x=alt.X("v:Q", title=x_title, axis=alt.Axis(format=fmt_axis) if fmt_axis else alt.Axis()),
        color=alt.Color("metric:N", title=None, scale=alt.Scale(domain=METRICS, range=[ONE, TEN]), legend=alt.Legend(orient="bottom")),
    )
    text = base.mark_text(align="left", dx=4, color=FG, fontSize=f["label"]).encode(x="v:Q", text="label:N")
    save((bars + text).properties(width=width, height=height or 30 * len(order), title=title), name, width)


def readers():
    data = [{"item": n, "metric": m, "v": v, "label": f"{v}%"} for n, one, ten in READERS for m, v in zip(METRICS, (one, ten))]
    grouped(
        data,
        [n for n, _, _ in READERS],
        "Essays Where the Reader Names the Hidden Word (%)",
        "Who Can Read the Fine-Tuned Sender",
        "readers",
    )


def uplift():
    order = [n for n, *_ in UPLIFT]
    absolute, relative = [], []
    for n, s1, f1, s10, f10 in UPLIFT:
        for m, s, f in zip(METRICS, (s1, s10), (f1, f10)):
            absolute.append({"item": n, "metric": m, "v": f - s, "label": f"+{f - s} ({s}% to {f}%)"})
            if s:
                relative.append({"item": n, "metric": m, "v": (f - s) / s, "label": f"+{(f - s) / s:.0%}"})
            else:  # 0 of 100 for the stock model: no ratio to report
                relative.append({"item": n, "metric": m, "v": 0, "label": f"n/a ({s}% to {f}%)"})
    grouped(absolute, order, "Percentage Points Gained", "Fine-Tuned Reader over Stock Reader, Absolute", "uplift-absolute", height=200)
    grouped(relative, order, "Relative Gain", "Fine-Tuned Reader over Stock Reader, Relative", "uplift-relative", fmt_axis="%", height=200)


training()
readers()
uplift()
