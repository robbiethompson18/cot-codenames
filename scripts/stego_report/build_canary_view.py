"""Build the bare-essay viewer: per token, the secret's rank among its canaries, the J-lens's top words and the NLA's text.

    uv run python scripts/stego_report/build_canary_view.py <out-dir>
Writes <out-dir>/hidden-word-viewer.html and <out-dir>/data/<essay>.json (one file per viewer essay, fetched on demand).
"""

import json
import sys
from pathlib import Path

import numpy as np

from cot_codenames.stego import canary
from cot_codenames.stego.lens import read_jsonl

HERE = Path(__file__).parent
out = Path(sys.argv[1])
(out / "data").mkdir(parents=True, exist_ok=True)
meta = json.loads((canary.OUT / "meta.json").read_text())
words, token_layers = meta["words"], meta["token_layers"]
scores = np.load(canary.OUT / "scores.npz")
rows = canary.essays()
by_i = {r["i"]: r for r in rows}
view = {(v["i"], v["reader"]): v for v in read_jsonl(canary.OUT / "view.jsonl")}
notes: dict[tuple[int, str], dict[int, str]] = {}
for note in read_jsonl(canary.OUT / "nla-view.jsonl"):
    notes.setdefault((note["i"], note["reader"]), {})[note["t"]] = note["text"]

secret_at = np.array([words.index(r["secret"]) for r in rows])
other = (secret_at[:, None] != np.arange(len(words))[None, :]).astype(np.float32)[:, None, :]
index = []
for reader in ("base", "lora"):
    # the same centring as canary.score: each word's average over the essays of other secrets
    tok_mean = np.stack([scores[f"{r['i']}_{reader}_tok"].astype(np.float32).mean(1) for r in rows])
    tok_base = (tok_mean * other).sum(0) / other.sum(0)
    for i in sorted({i for i, rd in view if rd == reader}):
        r = by_i[i]
        x = scores[f"{i}_{reader}_tok"].astype(np.float32) - tok_base[:, None, :]
        rank = (x[:, :, canary.canaries(r, words)] > x[:, :, [words.index(r["secret"])]]).sum(2)  # [token layers, tokens]
        path = out / "data" / f"{i}.json"
        doc = json.loads(path.read_text()) if reader == "lora" else {k: r[k] for k in ("i", "sender", "secret", "topic", "essay")}
        doc.setdefault("spans", view[(i, reader)]["spans"])
        doc.setdefault("readers", {})[reader] = {
            "rank": rank.tolist(),
            "top": {layer: [w[:5] for w in per] for layer, per in view[(i, reader)]["top"].items()},
            "nla": [notes[(i, reader)][t][:450] for t in range(rank.shape[1])],
        }
        path.write_text(json.dumps(doc, separators=(",", ":")))
        if reader == "lora":
            at = token_layers.index(48)
            index.append({k: r[k] for k in ("i", "sender", "secret", "topic")} | {"hot": round(float((rank[at] == 0).mean()), 3)})

table = json.loads((canary.OUT / "table.json").read_text())
la, tk = table["layers"].index(48), token_layers.index(48)
summary = [
    {
        "reader": c["reader"],
        "sender": c["sender"],
        "best": c["essay_max_top1"][la],
        "auc": c["essay_max_auc"][la],
        "avg": c["essay_mean_top1"][la],
        "tokens": c["token_top1_share"][tk],
        "where": c["token_top1_by_decile"][tk],
    }
    for c in table["cells"]
]
page = (HERE / "canary_view_template.html").read_text()
page = page.replace("/*DATA*/", json.dumps({"essays": index, "summary": summary, "token_layers": token_layers}))
(out / "hidden-word-viewer.html").write_text(page)
print(len(index), "essays")
