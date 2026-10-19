# cot-codenames

Research code on whether models can hide information from monitors. Closed as of 2026-10-14.

## Write-ups

- [Attempt to Make Qwen 3.6 27B do Steganography](https://robbiewmthompson.com/blog/hidden-word-essays/):
  Qwen3.6-27B is RL-trained to hide a secret word in an on-topic essay so a copy of itself can
  recover it; the post measures how often frozen monitors (stock Qwen, Claude, J-lens, NLA) can too.
  Source: [writeup-stego.md](writeup-stego.md).
- [Models are Awful at Controlling Their CoT](https://robbiewmthompson.com/blog/cot-codenames/): the
  earlier work, Codenames and Taboo with a chain-of-thought monitor. Source:
  [writeup.md](writeup.md).

## Layout

- `src/cot_codenames/stego/`: the steganography game, trainer (`train.py`), referee (`judge.py`),
  re-scoring with other monitors (`remonitor.py`) and a run dashboard (`dash.py`). Read
  [plan.md](plan.md) first; it opens with the project's final status.
- `charts/`: the blog post's charts and their data.
- `docs/`: stage writeups for the Codenames and Taboo work, the data layout of stego runs
  ([docs/stego-data.md](docs/stego-data.md)) and a literature check.
- The rest of `src/cot_codenames/` is the earlier Codenames and Taboo harness, apart from
  `basharena/`, which is a separate project.

Checkpoints and rollouts for the steganography runs are in a private Hugging Face repo,
`robbiethompson2018/cot-codenames-stego`.

## Running

```bash
uv sync --group train                                    # torch, transformers, peft
scripts/stego_runpod.sh fixed8 --pool-size 8             # one training run on its own RunPod pod
uv run python -m cot_codenames.stego.train --help        # all training options
```

`scripts/stego_runpod.sh` needs `HF_TOKEN` and `ANTHROPIC_API_KEY` in the environment (put them in
`.envrc.local`). The Tinker backend (`--backend tinker`) still exists but was dropped as too
expensive.
