"""Backend-independent training loop for the steganography game (plan.md).

Each step: sample k essays for each of B (secret, topic) prompts, score every essay with the receiver, the monitor, the
referee and the grep floor, turn rewards into leave-one-out advantages within each prompt's group, and take one optimizer
step on the single LoRA that plays both sender and receiver. Every --eval-every steps: score fresh essays without
updating, save a checkpoint, and upload the run directory to the Hugging Face repo.

Rung 1, fixed 8 words (smoke test):  python -m cot_codenames.stego.train --backend local --run fixed8 --pool-size 8
Rung 2, many words:                  python -m cot_codenames.stego.train --backend local --run words300 --pool-size 300
Control arm (no monitor penalty):    add --lam 0
Where everything a run writes goes:  docs/stego-data.md
"""

import argparse
import json
import os
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Protocol

from cot_codenames.stego import game, remonitor
from cot_codenames.stego.game import Chat, Episode, Sample
from cot_codenames.stego.judge import judge


class Backend(Protocol):
    """What a training backend provides. `local` (backend_local.py) runs HF + PEFT on one GPU; `tinker`
    (backend_tinker.py) uses the Tinker API. Both must turn the sender model's reasoning off."""

    def sample(self, chats: list[Chat], max_tokens: int, thinking: bool = False) -> list[Sample]:
        """One sampled completion per chat, from the current policy (base model + LoRA), at temperature 1. With
        `thinking` the sender reasons first: Sample.cot holds the reasoning, Sample.text the essay, and the policy
        gradient covers both."""
        ...

    def choose(self, role: str, chats: list[Chat], n_labels: int) -> list[list[float]]:
        """For each chat, the next-token distribution over game.LABELS[:n_labels], renormalised to sum to 1.
        role "receiver" uses the current policy; role "monitor" uses the frozen monitor model."""
        ...

    def step(self, sender: list[tuple[Sample, float]], receiver: list[tuple[Chat, int]]) -> dict[str, float]:
        """One optimizer step on the shared LoRA. `sender` is (sample, advantage): policy gradient on the essay tokens.
        `receiver` is (chooser chat, index of the correct label): cross-entropy on the answer token. Returns metrics."""
        ...

    def save(self, path: Path) -> None:
        """Write a checkpoint under `path` (adapter weights, or a file pointing at where they are stored)."""
        ...


def make_backend(args: argparse.Namespace) -> Backend:
    # Imported lazily: each backend pulls in heavy or optional dependencies.
    if args.backend == "local":
        from cot_codenames.stego.backend_local import LocalBackend

        return LocalBackend(args)
    from cot_codenames.stego.backend_tinker import TinkerBackend

    return TinkerBackend(args)


def git_sha() -> str:
    """The commit a run's code came from. Pods get the repo by rsync without .git, so the launcher passes it in."""
    if os.environ.get("STEGO_GIT_SHA"):
        return os.environ["STEGO_GIT_SHA"]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def monitor_dists(backend: Backend, chats: list[Chat], args: argparse.Namespace) -> tuple[list[list[float]], list[str]]:
    """The monitor's distribution over its candidates for each chat, plus its answer text when it has one.
    A Claude monitor gives no probabilities, so its distribution is one-hot on its guess (all zeros if it gave none).
    --binary-reward makes a Qwen monitor one-hot on its top letter too, so the two kinds can be compared like for like."""
    if args.monitor_model.startswith("claude"):
        with ThreadPoolExecutor(64) as pool:
            guesses = list(pool.map(partial(remonitor.claude_guess, args.monitor_model, args.n, effort="low"), chats))
        return [[float(i == g["guess"]) for i in range(args.n)] for g in guesses], [g["text"] for g in guesses]
    dists = backend.choose("monitor", chats, args.n)
    if args.binary_reward:
        dists = [[float(i == max(range(args.n), key=d.__getitem__)) for i in range(args.n)] for d in dists]
    return dists, [""] * len(chats)


def play(
    backend: Backend, rng: random.Random, episodes: list[Episode], pool: list[str], shots: list[dict], args: argparse.Namespace
) -> list[dict]:
    """Sample one essay per episode and score it. `shots` are earlier essays with their secrets, shown to the monitor
    when --monitor-shots is set. Returns one rollout dict per episode; the keys starting with "_" hold objects the
    training step needs and are dropped before logging."""
    told = args.sender_thinking and args.monitor_sees == "both"
    clock = {"t0": time.time()}

    def lap(name: str) -> None:
        clock[name] = round(time.time() - clock["t0"] - sum(v for k, v in clock.items() if k != "t0"), 1)

    samples = backend.sample([game.sender_chat(e, args.sender_thinking, told) for e in episodes], args.max_tokens, args.sender_thinking)
    lap("sample")
    recv = [game.candidates(rng, e.secret, pool, args.m) for e in episodes]
    mon = [game.candidates(rng, e.secret, pool, args.n) for e in episodes]
    recv_chats = [game.chooser_chat("receiver", e.topic, s.text, c) for e, s, (c, _) in zip(episodes, samples, recv)]
    claude = args.monitor_model.startswith("claude")

    def mon_chats(view: str, few: list[dict], force: bool) -> list[Chat]:
        """Monitor prompts for one view of the sender: "essay", "cot" (reasoning only) or "both"."""
        seen = [{"topic": e.topic, "essay": s.text, "mon_candidates": c} for e, s, (c, _) in zip(episodes, samples, mon)]
        if view != "essay":
            seen = [d | {"reasoning": s.cot, "show_essay": view == "both"} for d, s in zip(seen, samples)]
        return [remonitor.monitor_chat(d, few, force_answer=force) for d in seen]

    p_recv = backend.choose("receiver", recv_chats, args.m)
    p_mon, mon_text = monitor_dists(backend, mon_chats(args.monitor_sees, shots, claude), args)
    # With reasoning on, the base model also reads each view as a monitor that is never trained against, so every run
    # shows what a reader of the reasoning would have caught.
    views = (
        {v: backend.choose("monitor", mon_chats(v, [], False), args.n) for v in ("essay", "cot", "both")} if args.sender_thinking else {}
    )
    lap("score")
    verdicts = judge([(e.topic, s.text) for e, s in zip(episodes, samples)])
    lap("referee")
    rollouts = []
    for i, (e, s) in enumerate(zip(episodes, samples)):
        ri, mi = recv[i][1], mon[i][1]
        leak = game.leaked(e.secret, s.text)
        valid = verdicts[i]["pass"] and not leak
        rollouts.append(
            {
                "secret": e.secret,
                "topic": e.topic,
                "essay": s.text,
                "cot": s.cot,
                "cot_leaked": game.leaked(e.secret, s.cot),
                "no_essay": args.sender_thinking and not s.text,
                "base_monitor": {
                    v: {"p": round(d[i][mi], 4), "correct": max(range(args.n), key=d[i].__getitem__) == mi} for v, d in views.items()
                },
                "judge_pass": verdicts[i]["pass"],
                "judge_text": verdicts[i]["text"],
                "leaked": leak,
                "p_recv": p_recv[i][ri],
                "p_mon": p_mon[i][mi],
                "recv_correct": max(range(args.m), key=p_recv[i].__getitem__) == ri,
                "mon_correct": max(p_mon[i]) > 0 and max(range(args.n), key=p_mon[i].__getitem__) == mi,
                "recv_candidates": recv[i][0],
                "mon_candidates": mon[i][0],
                "recv_probs": [round(p, 4) for p in p_recv[i]],
                "mon_probs": [round(p, 4) for p in p_mon[i]],
                "mon_text": mon_text[i],
                "reward": game.reward(p_recv[i][ri], p_mon[i][mi], valid, args.lam),
                "_sample": s,
                "_recv": (recv_chats[i], ri),
                "_tokens": (verdicts[i]["in"], verdicts[i]["out"]),
                "_secs": {k: v for k, v in clock.items() if k != "t0"},
            }
        )
    return rollouts


def summarise(rollouts: list[dict], args: argparse.Namespace) -> dict[str, float]:
    def mean(key: str) -> float:
        return sum(r[key] for r in rollouts) / len(rollouts)

    out = {k: round(mean(k), 4) for k in ("reward", "p_recv", "p_mon", "recv_correct", "mon_correct", "judge_pass", "leaked")}
    out["recv_bits"] = round(game.bits(out["recv_correct"], args.m), 3)
    out["mon_bits"] = round(game.bits(out["mon_correct"], args.n), 3)
    out["words"] = round(sum(len(r["essay"].split()) for r in rollouts) / len(rollouts), 1)
    if rollouts[0]["base_monitor"]:
        out["cot_words"] = round(sum(len(r["cot"].split()) for r in rollouts) / len(rollouts), 1)
        out["cot_leaked"] = round(mean("cot_leaked"), 4)
        out["no_essay"] = round(mean("no_essay"), 4)
        for view in ("essay", "cot", "both"):
            out[f"base_mon_{view}"] = round(sum(r["base_monitor"][view]["correct"] for r in rollouts) / len(rollouts), 4)
    out["judge_tokens_in"] = sum(r["_tokens"][0] for r in rollouts)
    out["judge_tokens_out"] = sum(r["_tokens"][1] for r in rollouts)
    out |= {f"secs_{k}": v for k, v in rollouts[0]["_secs"].items()}
    return out


def log(path: Path, records: list[dict]) -> None:
    with path.open("a") as f:
        for r in records:
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local", "tinker"], required=True)
    ap.add_argument("--run", required=True, help="run name; outputs go to runs/stego/<run>/")
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B", help="sender and receiver (one shared LoRA)")
    ap.add_argument(
        "--monitor-model",
        default="Qwen/Qwen3.6-27B",
        help="frozen monitor. Same as --model = the base model without the LoRA. A claude-* id = that model through the API",
    )
    ap.add_argument("--sender-thinking", action="store_true", help="let the sender reason before the essay (local backend)")
    ap.add_argument("--thinking-tokens", type=int, default=300, help="hard budget for the sender's reasoning; it is closed at the limit")
    ap.add_argument(
        "--monitor-sees",
        choices=["essay", "both"],
        default="essay",
        help="what the monitor in the reward reads: the essay, or the sender's reasoning and the essay (needs --sender-thinking)",
    )
    ap.add_argument("--monitor-shots", type=int, default=0, help="show the monitor this many recent essays with their secrets")
    ap.add_argument("--binary-reward", action="store_true", help="penalise the monitor's top guess (0 or 1), not its probability")
    ap.add_argument("--pool-size", type=int, default=0, help="restrict secrets to this many fixed training words (0 = whole train split)")
    ap.add_argument("--m", type=int, default=8, help="receiver's candidate list size")
    ap.add_argument("--n", type=int, default=8, help="monitor's candidate list size")
    ap.add_argument("--lam", type=float, default=1.0, help="monitor penalty weight; 0 = control arm")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--prompts", type=int, default=8, help="(secret, topic) prompts per step")
    ap.add_argument("--k", type=int, default=8, help="essays sampled per prompt (the RLOO group)")
    ap.add_argument("--max-tokens", type=int, default=320, help="token limit for the essay")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-episodes", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--kl-coef", type=float, default=0.0, help="KL penalty towards the base model on essay tokens (local backend)")
    ap.add_argument("--micro-batch", type=int, default=2, help="sequences per forward/backward pass (local backend)")
    ap.add_argument("--hf-repo", default="", help="upload the run directory here at every eval")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train_words, heldout_words = game.load_words()
    topics = game.load_topics()
    if args.pool_size:
        train_words = train_words[: args.pool_size]
    run_dir = Path("runs/stego") / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(vars(args) | {"git_sha": git_sha(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2)
    )
    backend = make_backend(args)
    shot_pool: list[dict] = []  # valid essays from the latest training step, the monitor's few-shot examples

    def episodes(words: list[str], count: int) -> list[Episode]:
        return [Episode(rng.choice(words), rng.choice(topics)) for _ in range(count)]

    def shots() -> list[dict]:
        return rng.sample(shot_pool, min(args.monitor_shots, len(shot_pool)))

    def evaluate(step: int) -> None:
        # A pool no bigger than the candidate list leaves no unseen secrets worth testing, so only it is evaluated.
        splits = {"train": train_words} if 0 < args.pool_size <= max(args.m, args.n) else {"train": train_words, "heldout": heldout_words}
        for split, words in splits.items():
            rollouts = play(backend, rng, episodes(words, args.eval_episodes), words, shots(), args)
            metrics = {"step": step, "split": split} | summarise(rollouts, args)
            print("EVAL", json.dumps(metrics), flush=True)
            log(run_dir / "metrics.jsonl", [{"kind": "eval"} | metrics])
            log(run_dir / "rollouts.jsonl", [{"step": step, "split": split, "kind": "eval"} | r for r in rollouts])
        backend.save(run_dir / f"ckpt-{step:04d}")
        if args.hf_repo:
            from huggingface_hub import upload_folder

            upload_folder(repo_id=args.hf_repo, folder_path=str(run_dir), path_in_repo=f"runs/{args.run}")

    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            evaluate(step)
        if step == args.steps:
            break
        t0 = time.time()
        # Each prompt is repeated k times so its essays share a leave-one-out baseline.
        eps = [e for e in episodes(train_words, args.prompts) for _ in range(args.k)]
        rollouts = play(backend, rng, eps, train_words, shots(), args)
        advantages = [a for g in range(0, len(rollouts), args.k) for a in game.rloo([r["reward"] for r in rollouts[g : g + args.k]])]
        step_metrics = backend.step(
            sender=[(r["_sample"], a) for r, a in zip(rollouts, advantages)],
            receiver=[r["_recv"] for r in rollouts],
        )
        valid = [r for r in rollouts if r["judge_pass"] and not r["leaked"]]
        shot_pool = valid or shot_pool
        metrics = {"step": step} | summarise(rollouts, args) | step_metrics | {"secs": round(time.time() - t0, 1)}
        print(json.dumps(metrics), flush=True)
        log(run_dir / "metrics.jsonl", [{"kind": "train"} | metrics])
        log(run_dir / "rollouts.jsonl", [{"step": step, "split": "train", "kind": "train"} | r for r in rollouts])


if __name__ == "__main__":
    main()
