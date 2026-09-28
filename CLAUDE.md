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

- [Plan — staged build plan; read before starting any stage](plan.md) — stages 0–5, each with a
  writeup, graphs, and example games to check
- [Stage 0 writeup — read before stage 1 or before changing models/providers](docs/stage-0.md) —
  harness checks (CoT carryover, no-CoT), provider gotchas, turns-to-finish CoT vs no-CoT
- [Run format — read before analyzing or changing games.jsonl records](docs/run-format.md) — game
  and call fields, error kinds and replay policy, `n_in` prompt reconstruction
- [Lit check: CoT control, stego, secret-hiding games — read before designing a monitor or condition](docs/lit-cot-control.md)
  — 18 papers, closest prior work, novelty, design implications

<!-- As docs are added under docs/, list them here, one per line: -->
<!-- - [Title — when to read](docs/foo.md) — short gloss -->

## Operator Notes

** Maximum parallelism always:** Just kick off requests to OpenRoute in parallel, up to 500. If we
get rate limited lmk.
