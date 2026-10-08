# cot-codenames

Research code on whether models can hide information from monitors, and whether training against one
monitoring technique breaks the others.

- **Current work:** a steganography organism. A model learns to hide a secret word in an ordinary
  essay so that a copy of itself can recover it and a frozen monitor cannot. See [plan.md](plan.md);
  code is in `src/cot_codenames/stego/`.
- **Earlier work:** Codenames and Taboo with a chain-of-thought monitor. See
  [docs/codenames-plan.md](docs/codenames-plan.md), the stage writeups in `docs/`, and
  [docs/taboo.md](docs/taboo.md).

Checkpoints and rollouts for the steganography runs are on Hugging Face at
[robbiethompson2018/cot-codenames-stego](https://huggingface.co/robbiethompson2018/cot-codenames-stego).

```bash
uv sync --group train     # local / RunPod backend (torch, transformers, peft)
uv sync --group tinker    # Tinker backend
uv run python -m cot_codenames.stego.train --backend local --run fixed8 --pool-size 8
```
