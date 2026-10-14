"""Charts for the hidden-word essay post (Altair). Run from the repo root:
    uv run --no-project --with altair --with vl-convert-python python charts/stego.py

Data:
- training.json: per-step training metrics of the whole lineage, 64 essays per step: cot-codenames run words300-n500
  steps 0-49, its continuation words300-n500-b steps 0-49 (overall 50-99), then open-vocab-27b steps 0-80 (overall
  100-180), which switched to no candidate lists and the stock 27B as monitor.
- READERS / UPLIFT: hits out of 100 essays from that run's step-75 sender, no candidate list
  (cot-codenames runs/stego/openword-ov75/topic-quote/table.json, readers told the essay's topic). The J-lens rows are
  layer 42, the layer the NLA reads.
- jlens-layers.json: J-lens hits at every layer on those 100 essays (runs/stego/openword-ov75/topic-quote/read.jsonl).
"""

import json
from pathlib import Path

import altair as alt  # ty: ignore[unresolved-import]  # run with --no-project --with altair, see the docstring
from theme import FG, fonts, save

RECV, MON, REF = "#3f8f5b", "#c1432e", "#7a5fb0"
ONE, TEN = "#2e7dc1", "#d98a1f"


def training():
    steps = json.loads(Path("charts/training.json").read_text())
    names = {"recv": "Receiver right", "mon": "Monitor right", "referee": "Passes referee"}
    data = []
    for phase in ("list", "open"):  # trailing means never reach across the switch, and the lines break there
        rows = [r for r in steps if r["phase"] == phase]
        for k, row in enumerate(rows):  # 5-step trailing mean: each step is only 64 essays
            window = rows[max(0, k - 4) : k + 1]
            for key, name in names.items():
                data.append({"step": row["step"], "phase": phase, "who": name, "v": sum(w[key] for w in window) / len(window)})
    width = 560
    f = fonts(width)
    switch = 100
    lines = (
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
            detail="phase:N",
        )
    )
    rule = alt.Chart(alt.Data(values=[{"step": switch}])).mark_rule(color=FG, strokeDash=[4, 4]).encode(x="step:Q")
    labels = (
        alt.Chart(
            alt.Data(
                values=[
                    {"step": switch / 2, "text": "Pick from 8; monitor: 9B, picks from 500"},
                    {"step": switch + 40, "text": "No list; monitor: stock 27B"},
                ]
            )
        )
        .mark_text(color=FG, fontSize=f["label"], dy=-8)
        .encode(x="step:Q", y=alt.value(0), text="text:N")
    )
    save((lines + rule + labels).properties(width=width, height=240, title="Training the Sender"), "training", width)


# (reader, first guess, in first ten) out of 100 essays from the fine-tuned sender
READERS = [
    ("Fine-tuned 27B, next token", 80, 99),
    ("Claude Sonnet, asked", 61, 84),
    ("Stock 27B, next token", 31, 72),
    ("Fine-tuned 27B, NLA", 37, 69),
    ("Claude Haiku, asked", 40, 62),
    ("Stock 27B, NLA", 13, 38),
    ("Fine-tuned 27B, J-lens", 20, 34),
    ("Qwen 9B, next token", 9, 34),
    ("Stock 27B, J-lens", 3, 9),
]
# (reader, first guess, in first ten) on the same 100 essays: Qwen models by their next token, Claude by asking for ten
ROLES = [
    ("Fine-tuned 27B (receiver)", 80, 99),
    ("Stock 27B (monitor)", 31, 72),
    ("Qwen 9B", 9, 34),
    ("Claude Haiku", 40, 62),
    ("Claude Sonnet", 61, 84),
]
# (technique, stock pass@1, fine-tuned pass@1, stock pass@10, fine-tuned pass@10)
UPLIFT = [("Asked for Ten Guesses", 21, 65, 30, 80), ("Next Token", 31, 80, 72, 99), ("J-lens", 3, 20, 9, 34), ("NLA", 13, 37, 38, 69)]
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


def roles():
    data = [{"item": n, "metric": m, "v": v, "label": f"{v}%"} for n, one, ten in ROLES for m, v in zip(METRICS, (one, ten))]
    grouped(data, [n for n, _, _ in ROLES], "Essays Where the Reader Names the Hidden Word (%)", "Receiver vs Monitors", "roles")


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


def layers():
    names = {"lora": "Fine-tuned 27B", "base": "Stock 27B"}
    data = [
        {"layer": r["layer"], "reader": names[r["reader"]], "metric": m, "v": r[key]}
        for r in json.loads(Path("charts/jlens-layers.json").read_text())
        for m, key in zip(METRICS, ("first", "ten"))
    ]
    nla = [  # the NLA only exists at layer 42
        {"layer": 42, "reader": names[rd], "metric": m, "v": v}
        for rd, hits in (("lora", (37, 69)), ("base", (13, 38)))
        for m, v in zip(METRICS, hits)
    ]
    width = 560
    color = alt.Color(
        "reader:N", title=None, scale=alt.Scale(domain=list(names.values()), range=[RECV, MON]), legend=alt.Legend(orient="bottom")
    )
    x = alt.X("layer:Q", title="Layer (of 63)", scale=alt.Scale(domain=[0, 63]))

    def panel(metric):
        y = alt.Y("v:Q", title="Essays Naming the Word (%)" if metric == METRICS[0] else None, scale=alt.Scale(domain=[0, 100]))
        lines = (
            alt.Chart(alt.Data(values=[d for d in data if d["metric"] == metric])).mark_line(strokeWidth=2.5).encode(x=x, y=y, color=color)
        )
        points = (
            alt.Chart(alt.Data(values=[d for d in nla if d["metric"] == metric]))
            .mark_point(size=110, filled=True, shape="diamond", stroke=FG, strokeWidth=1, opacity=1)
            .encode(x=x, y=y, color=color)
        )
        return (lines + points).properties(width=width // 2 - 45, height=220, title=alt.Title(metric, fontSize=fonts(width)["axis_title"]))

    chart = alt.hconcat(*[panel(m) for m in METRICS]).properties(title="J-Lens by Layer (Diamonds: NLA at Layer 42)")
    save(chart, "jlens-layers", width)


training()
readers()
roles()
layers()
uplift()
