# Run format: `runs/stage-N/games.jsonl`

One JSON line per played game, append-only (`run.py` appends under `flock`). `load_games()` returns
the last record per `id`. Read the raw file to see earlier failed attempts of a replayed game.

## Game record

- `id`: `model|cot|seed` or `model|nocot|seed`. The seed fixes the board, so boards are shared
  across models and conditions.
- `condition`: a dict, currently `{"thinking": bool}`. Stage 1 adds told/not-told here. Older
  records have a bare `thinking` bool; `load_games` fills in `condition` for them.
- `config`: provenance only, nothing enforces it.
  - `code_hash`: sha of `prompts.py` + `game.py` + `wordlist.txt`
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
