# Run format: `runs/stage-N/games.jsonl`

One JSON line per played game, append-only (`run.py` appends under `flock`). `load_games()` returns
the last record per `id`. Read the raw file to see earlier failed attempts of a replayed game.

## Game record

- `id`: `model|cot|seed` or `model|nocot|seed`. The seed fixes the board, so boards are shared
  across models and conditions.
- `condition`: a dict, currently `{"thinking": bool}`. Stage 1 adds told/not-told here. Older
  records have a bare `thinking` bool; `load_games` fills in `condition` for them.
- `config`: provenance only, nothing enforces it.
  - `code_hash`: sha of `prompts.py` + `game.py` + `wordlist.txt` (+ `wordlist-nouns.txt` from stage
    2 on)
  - `slug`, `provider`: the OpenRouter model and pinned provider
  - `git_sha`, `git_dirty`
- Board and outcome: `team`, `neutral`, `board_order` (the guesser's order), `turns[]` (`clue`,
  `number`, `raw_guesses`, `reveals[]`), `turns_to_finish` (null if the game didn't finish),
  `found`.
- `error`, `error_kind`:
  - `protocol`: the model never made a valid move. This is a result: it isn't replayed and it counts
    in the stats.
  - `infra`: `client.chat` gave up. Replayed.
  - `crash`: a bug in our code. The traceback is kept along with the transcript so far. Replayed.
- `spymaster` / `guesser`: `{messages, tools}`. This is the full final message list, sent verbatim
  (including `reasoning_details`).
- `calls[]`: one entry per API call.
  - `role`, `turn`, `attempt` (the retry number within the turn)
  - `n_in`: call k's input was `messages[:n_in]` and its output is `messages[n_in]`, so prompts are
    never stored more than once
  - `generation_id`: OpenRouter's id; `/api/v1/generation?id=` gives native tokens and the provider
  - `served_model`, `usage`, `provider`, `latency_s`, `finish_reason`
  - `failed_attempts`: the error text of each API retry before this call succeeded
  - `rejected`: only present on rejected attempts. It says why: a `check_clue` reason, "malformed
    tool call" or "no tool call".

A turn is one clue plus its guesses. It can span several calls per role when an attempt is rejected.
Plots that are "per turn" sum over those calls.

## Word pool

`condition.wordlist` is only present when the pool isn't the default 400-word Codenames list.
Currently the only other value is `"nouns"` (`wordlist-nouns.txt`, see [stage-2.md](stage-2.md)),
and those ids get a `-nouns` suffix, e.g. `deepseek-v4.1-flash|cot-told-nouns|3`.

# Monitor format: `runs/stage-2/monitor.jsonl`

One line per LLM-monitor call on one game (`llm_monitor.py`), append-only. `load()` returns the last
record per `id`.

- `id`: `s{stage}|{game id}|{view}|{list|nolist}|{monitor model}`. `stage` is the stage whose
  `games.jsonl` holds the game.
- `view`: `cot` (the spymaster's reasoning, reply text and every `submit_clue` call) or `clues`
  (accepted clues only). `wordlist`: whether the board's word list was in the prompt.
- `words`: the 10 words, uppercase, most confident first. `hits[]`: whether each one is on the
  board. `win` = `hits[0]`. Team vs neutral isn't stored: `--report` gets it from the game.
- `error`: null, `"protocol"` (no valid `submit_words` after 3 attempts; not retried), or the infra
  error text (retried on the next run).
- `messages`, `calls[]`: the monitor's own transcript and per-call usage, in the same shape as a
  game's `calls[]`.
