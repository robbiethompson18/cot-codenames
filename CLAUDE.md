# cot-codenames

Codenames with a CoT monitor: can open-weight reasoning models play spymaster without leaking board
words in their chain of thought?

@~/.claude/personal-repo-rules.md

`AGENTS.md` at the repo root is a symlink to `CLAUDE.md` so Codex/other agents see the same
instructions. Do not replace it with a separate file.

## Stack

- Python: `uv` for deps/venv, `ruff` for lint + format (line length 140), `ty` for types. Run
  `uv run ruff check . && uv run ruff format . && uv run ty check` before shipping.
- Markdown: Prettier, 100 cols, `proseWrap: always`. Prettier is Markdown-only here.
- Tests: none until this repo is roughly > 50k LOC. Verify by running the code, not by adding a test
  suite. If a test would genuinely save time, ask first.
- Secrets/machine-specific env go in `.envrc.local` (gitignored), never `.envrc`.

## Docs

Durable lessons about this repo go in git:

- **One-line rules** → this file (`CLAUDE.md`), or `CLAUDE.local.md` for machine-specific
  (gitignored).
- **Longer reference docs** (5–300 lines) → `docs/*.md`, with a one-line index entry below.
- **Local-only docs** (not in git) → `docs/local/*.md`.

See `~/.claude/personal-repo-rules.md` (imported above) for the full convention.

Current docs:

- [Plan — current plan: steganography organism and the interp transfer grid; read before touching `src/cot_codenames/stego/`](plan.md)
  — game, reward, two training backends (RunPod, Tinker), spike rungs, stopping rule
- [Stego data layout — read before analysing a stego run or adding a new output](docs/stego-data.md)
  — what each run uploads to Hugging Face, rollout fields, known gaps
- [Codenames plan (superseded) — read only for the history behind stages 0–3](docs/codenames-plan.md)
  — stages 0–5 of the original Codenames build, plus the Taboo pivot
- [Stage 0 writeup — read before stage 1 or before changing models/providers](docs/stage-0.md) —
  harness checks (CoT carryover, no-CoT), provider gotchas, turns-to-finish CoT vs no-CoT
- [Stage 1 writeup — read before stage 2 or before changing the told prompt / no-CoT setup](docs/stage-1.md)
  — grep monitor, told vs not told (0/30 comply), tool_choice ablation
- [Stage 2 writeup (interim) — read before running or changing the LLM monitor or word lists](docs/stage-2.md)
  — Luna vs GLM, hits@10, 6.4k-noun list, why top-1 saturates
- [Run format — read before analyzing or changing games.jsonl / monitor.jsonl records](docs/run-format.md)
  — game and call fields, error kinds and replay policy, `n_in` prompt reconstruction
- [Lit check: CoT control, stego, secret-hiding games — read before designing a monitor or condition](docs/lit-cot-control.md)
  — 18 papers, closest prior work, novelty, design implications
- [Related work: why CoT control fails and what raised it — read before designing a stronger told condition or CoT fine-tuning](docs/related-work.md)
  — CoT-Control ("chromosome") details, ReasonIF, white-bear rebound, prompting/SFT elicitation
  numbers, grid-index hiding idea
- [Taboo / Person Do Thing — read before extending the one-call taboo/PDT sweep (supersedes Codenames plan)](docs/taboo.md)
  — setup, 11-arm results table, CoT fixes PDT rule-breaking, Claude-low = 0 thinking, failure modes

<!-- As docs are added under docs/, list them here, one per line: -->
<!-- - [Title — when to read](docs/foo.md) — short gloss -->

## Operator Notes

**Stego runs live on Hugging Face, not git:** `runs/stego/` is gitignored; checkpoints and rollouts
upload to `robbiethompson2018/cot-codenames-stego` via `--hf-repo`.

**Tinker is banned (Robbie, 2026-10-08): too expensive.** Do not train or sample through Tinker,
including for small tests. A 100-step run cost about $42 there against about $9 on a RunPod H200.
Train on RunPod (`scripts/stego_runpod.sh`). `backend_tinker.py` and the `--tinker` option of
`remonitor.py` stay in the repo unused; its gpt-oss reasoning path is unfinished and untested.

**Never call Anthropic models through OpenRouter:** Robbie has Anthropic credits, so use the
Anthropic API directly.

** Maximum parallelism always:** Just kick off requests to OpenRoute in parallel, up to 500. If we
get rate limited lmk.
