"""Static single-file HTML dashboard for a stage's games.jsonl: summary table, every clue (for hand validation), and a
per-game view with the board, each turn's full spymaster + guesser transcripts (board words highlighted), and the exact
raw messages each agent saw.

uv run python -m cot_codenames.dashboard --stage 0   -> runs/stage-0/dashboard.html
"""

import argparse
import json
from pathlib import Path

from cot_codenames.game import load_games


def slim(game: dict) -> dict:
    """Drop `reasoning_details` (duplicates `reasoning`) to keep the HTML small."""
    for role in ("spymaster", "guesser"):
        game[role]["messages"] = [{k: v for k, v in m.items() if k != "reasoning_details"} for m in game[role]["messages"]]
    return game


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>cot-codenames stage __STAGE__</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--card:#fff;--line:#e7e5e4;--team:#16a34a;--teambg:#dcfce7;--neu:#a8a29e;--neubg:#f5f5f4;--bad:#dc2626}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;--card:#292524;--line:#44403c;--team:#4ade80;--teambg:#14532d;--neu:#a8a29e;--neubg:#3f3a36;--bad:#f87171}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0;padding:16px;max-width:1300px;margin:auto}
h1,h2,h3{margin:.6em 0 .3em}table{border-collapse:collapse;margin:8px 0}td,th{border-bottom:1px solid var(--line);padding:3px 10px;text-align:left;vertical-align:top}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:10px 0}
.muted{color:var(--muted)}.team{background:var(--teambg);color:var(--team);font-weight:600;border-radius:3px;padding:0 2px}
.neutral{background:var(--neubg);border-radius:3px;padding:0 2px;text-decoration:underline dotted}
.board{display:grid;grid-template-columns:repeat(5,1fr);gap:4px;max-width:700px}.cell{padding:6px;border-radius:5px;text-align:center;font-size:12px;border:1px solid var(--line)}
.cell.t{background:var(--teambg);color:var(--team)}.cell.n{background:var(--neubg)}.cell sup{color:var(--muted)}
pre{white-space:pre-wrap;word-break:break-word;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:8px;margin:4px 0;font-size:12.5px}
.clue{font-size:18px;font-weight:700}.bad{color:var(--bad)}details>summary{cursor:pointer;color:var(--muted)}
select{font-size:14px;padding:4px;max-width:100%}.cols{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:800px){.cols{grid-template-columns:1fr}}
</style></head><body>
<h1>cot-codenames · stage __STAGE__</h1>
<div id="summary"></div>
<details class="card"><summary>All clues (hand validation)</summary><div id="clues"></div></details>
<div class="card">Game: <select id="pick"></select> <span id="nav"></span></div>
<div id="game"></div>
<script>
const GAMES = __DATA__;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const cond = g => g.condition.thinking ? "cot" : "nocot";
const mean = xs => xs.length ? (xs.reduce((a,b)=>a+b,0)/xs.length) : NaN;
// Pre-validation games (before check_clue) sometimes used a board word as the clue; flag those.
const boardClue = (g, t) => [...g.team, ...g.neutral].flatMap(w => w.split(" ")).includes(String(t.clue).toUpperCase());
const cost = g =>g.calls.reduce((a,c)=>a+((c.usage||{}).cost||0),0);

function highlight(text, g) {
  const team = new Set(g.team), words = [...g.team, ...g.neutral].sort((a,b)=>b.length-a.length);
  const re = new RegExp("\\b(" + words.map(w=>w.replace(/[.*+?^${}()|[\]\\]/g,"\\$&")).join("|") + ")\\b", "gi");
  return esc(text).replace(re, m => `<span class="${team.has(m.toUpperCase())?"team":"neutral"}">${m}</span>`);
}

// Summary
const groups = {};
for (const g of GAMES) (groups[g.model+" · "+cond(g)] ??= []).push(g);
let h = "<table><tr><th>model · condition</th><th>games</th><th>finished ≤10</th><th>mean turns (finished)</th><th>mean found</th><th>errors</th><th>$/game</th></tr>";
for (const [k, gs] of Object.entries(groups).sort()) {
  const fin = gs.filter(g=>g.turns_to_finish);
  h += `<tr><td>${esc(k)}</td><td>${gs.length}</td><td>${fin.length}</td><td>${mean(fin.map(g=>g.turns_to_finish)).toFixed(1)}</td><td>${mean(gs.map(g=>g.found)).toFixed(1)}</td><td class="${gs.some(g=>g.error)?"bad":""}">${gs.filter(g=>g.error).length}</td><td>${mean(gs.map(cost)).toFixed(3)}</td></tr>`;
}
document.getElementById("summary").innerHTML = h + "</table>";

// All clues
let c = "<table><tr><th>game</th><th>turn</th><th>clue</th><th>N</th><th>guesses → result</th></tr>";
GAMES.forEach((g, i) => g.turns.forEach(t => {
  c += `<tr><td><a href="#${i}">${esc(g.id)}</a></td><td>${t.turn}</td><td><b class="${boardClue(g,t)?"bad":""}">${esc(t.clue)}</b>${boardClue(g,t)?" (board word)":""}</td><td>${t.number}</td><td>${t.reveals.map(r=>`<span class="${r.kind==="team"?"team":r.kind==="invalid"?"bad":"neutral"}">${esc(r.word)}</span>`).join(" ")}</td></tr>`;
}));
document.getElementById("clues").innerHTML = c + "</table>";

// Game picker
const pick = document.getElementById("pick");
GAMES.forEach((g, i) => pick.add(new Option(`${g.id} — ${g.turns_to_finish ? "done in "+g.turns_to_finish : "found "+g.found+"/9"}${g.error?" ERROR":""}`, i)));
pick.onchange = () => { location.hash = pick.value; };
window.onhashchange = () => show(+location.hash.slice(1) || 0);

function agentTurn(g, role, turn) {
  const msgs = g[role].messages;
  return g.calls.filter(c => c.role===role && c.turn===turn).map(c => {
    const m = msgs[c.n_in] || {}, tok = (c.usage?.completion_tokens_details||{}).reasoning_tokens;
    const calls = (m.tool_calls||[]).map(tc=>`${tc.function.name}(${tc.function.arguments})`).join("\n");
    return `<div class="muted">${esc(role)} call${c.attempt?" (retry "+c.attempt+")":""} · ${tok??"?"} reasoning tok · ${c.latency_s}s · ${esc(c.provider)}${c.failed_attempts?.length ? ` · <span class="bad" title="${esc(c.failed_attempts.join("\n"))}">${c.failed_attempts.length} API retries</span>` : ""}</div>`
      + (c.rejected ? `<div class="bad">rejected: ${esc(c.rejected)}</div>` : "")
      + (m.reasoning ? `<details open><summary>reasoning (${m.reasoning.length} chars)</summary><pre>${highlight(m.reasoning, g)}</pre></details>` : `<div class="muted">no reasoning</div>`)
      + (m.content ? `<div>visible content:</div><pre>${highlight(m.content, g)}</pre>` : "")
      + `<pre>${esc(calls) || '<span class="bad">no tool call</span>'}</pre>`;
  }).join("");
}

function show(i) {
  const g = GAMES[i]; if (!g) return; pick.value = i;
  document.getElementById("nav").innerHTML = `<a href="#${Math.max(0,i-1)}">← prev</a> · <a href="#${Math.min(GAMES.length-1,i+1)}">next →</a>`;
  const when = {}; g.turns.forEach(t => t.reveals.forEach(r => { if (r.kind!=="invalid") when[r.word] = t.turn; }));
  let h = `<div class="card"><h2>${esc(g.id)}</h2><div>${g.turns_to_finish ? "Finished in <b>"+g.turns_to_finish+"</b> turns" : "Found "+g.found+"/9 (not finished)"} · $${cost(g).toFixed(3)}${g.error?` · <span class="bad">${esc(g.error_kind ?? "error")}: ${esc(g.error)}</span>`:""}</div>`;
  h += `<p class="muted">Board (guesser's order). Green = team. Superscript = turn revealed.</p><div class="board">` + g.board_order.map(w =>
    `<div class="cell ${g.team.includes(w)?"t":"n"}">${esc(w)}${when[w]?`<sup> ${when[w]}</sup>`:""}</div>`).join("") + `</div></div>`;
  // Iterate over turns that made calls, not g.turns: a turn that ended in a protocol error has calls but no turn record.
  for (const n of [...new Set(g.calls.map(c => c.turn))]) {
    const t = g.turns.find(t => t.turn === n);
    h += `<div class="card"><h3>Turn ${n}: ${t ? `<span class="clue">${esc(t.clue)} ${t.number}</span>${boardClue(g,t)?' <span class="bad">(board word!)</span>':""}` : '<span class="bad">no valid move</span>'}</h3>
      <div>Guesses: ${t ? t.reveals.map(r=>`<span class="${r.kind==="team"?"team":r.kind==="invalid"?"bad":"neutral"}">${esc(r.word)}</span> (${r.kind})`).join(", ") || "none" : "—"}</div>
      <div class="cols"><div><h3>Spymaster</h3>${agentTurn(g,"spymaster",n)}</div><div><h3>Guesser</h3>${agentTurn(g,"guesser",n)}</div></div></div>`;
  }
  for (const role of ["spymaster","guesser"])
    h += `<details class="card"><summary>Raw messages: ${role} (exact list sent on the last call; call k saw the first n_in messages) + tools</summary><pre>${esc(JSON.stringify({tools:g[role].tools, messages:g[role].messages}, null, 1))}</pre></details>`;
  document.getElementById("game").innerHTML = h;
}
show(+location.hash.slice(1) || 0);
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    args = ap.parse_args()
    run_dir = Path(f"runs/stage-{args.stage}")
    games = [slim(g) for g in load_games(run_dir / "games.jsonl")]
    games.sort(key=lambda g: (g["model"], not g["condition"]["thinking"], g["seed"]))
    # "</" inside JSON would close the <script> tag early.
    data = json.dumps(games).replace("</", "<\\/")
    out = run_dir / "dashboard.html"
    out.write_text(PAGE.replace("__STAGE__", str(args.stage)).replace("__DATA__", data))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(games)} games)")


if __name__ == "__main__":
    main()
