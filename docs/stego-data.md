# Stego runs: where the data lives

Written 2026-10-08. Read this before analysing a stego run or adding a new kind of output.

## Primary store: Hugging Face

Everything a run produces goes to the private repo
[robbiethompson2018/cot-codenames-stego](https://huggingface.co/robbiethompson2018/cot-codenames-stego)
under `runs/<run>/`. `runs/stego/` in this repo is a gitignored local mirror with the same layout. A
pod's disk is wiped when the pod is deleted, so anything not uploaded is lost. `train.py` uploads at
every evaluation, and `scripts/stego_runpod.sh` uploads once more (with the log) before it deletes
the pod.

| File                | Contents                                                                                 |
| ------------------- | ---------------------------------------------------------------------------------------- |
| `config.json`       | Every CLI argument, `git_sha` of the code, start time                                    |
| `metrics.jsonl`     | One line per training step (`kind: train`) and per evaluation split (`kind: eval`)       |
| `rollouts.jsonl`    | One line per essay, training and evaluation (fields below)                               |
| `ckpt-NNNN/`        | Local backend: the PEFT adapter. Tinker backend: `tinker.json` with `tinker://` paths    |
| `train.log`         | Full stdout of the run                                                                   |
| `remonitor*.jsonl`  | Evaluation essays re-scored by other monitors, with their answers and thinking summaries |
| `runs/calibration/` | 64 honest essays, referee transcripts, and the referee prompt the calibration used       |
| `runs/offline/`     | Experiments that need no training (`offline.py`), one file per experiment plus a summary |

## Dashboard

`uv run python -m cot_codenames.stego.dash` serves <http://stego.localhost/> (port 7103): one row
per run with its models, referee, thinking setting, progress and latest results; click a row for
curves and the latest essays. It copies `config.json`, `metrics.jsonl` and the last evaluation
essays off each reachable pod on every refresh.

Each run folder also has `launch.json`, written by `scripts/stego_runpod.sh`: pod id and address,
the arguments, the git commit and a one-line purpose (`STEGO_NOTE=... scripts/stego_runpod.sh ...`).
Add `"gone": true` when a pod is deleted by hand and `"stopped": "<why>"` to label the row. Only
runs with a `launch.json` appear on the dashboard.

## Rollout fields

`step`, `split` (`train` or `heldout` words), `kind`, `secret`, `topic`, `essay`, `recv_candidates`,
`mon_candidates`, `recv_probs`, `mon_probs` (the full distribution over candidates), `p_recv`,
`p_mon` (probability on the secret), `recv_correct`, `mon_correct`, `judge_pass`, `judge_text` (the
referee's whole answer), `mon_text` (an API monitor's answer), `leaked`, `reward`.

The sender's prompt is not stored. It is a pure function of `secret` and `topic`
(`game.sender_chat`), so the `git_sha` in `config.json` is enough to rebuild it.

## Gaps

- Runs from before commit `42658d3` (`tinker-*`, `runpod-fixed8`) have no `judge_text`,
  `recv_probs`, `mon_probs` or `git_sha`, and used the first, lenient referee.
- Tinker checkpoints are paths on Tinker's servers. Nothing has been downloaded or converted to a
  PEFT adapter yet.
- A Claude monitor has no probabilities, so its `mon_probs` is one-hot on its guess.
