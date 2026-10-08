"""Backend-independent training loop for the steganography game (plan.md).

Each step: sample k essays for each of B (secret, topic) prompts, score every essay with the receiver, the monitor, the
judge and the grep floor, turn rewards into leave-one-out advantages within each prompt's group, and take one optimizer
step on the single LoRA that plays both sender and receiver. Every --eval-every steps: score fresh essays without
updating, save a checkpoint, and upload the run directory to the Hugging Face repo.

Rung 1, fixed 8 words (smoke test):  python -m cot_codenames.stego.train --backend local --run fixed8 --pool-size 8
Rung 2, fresh 8 words:               python -m cot_codenames.stego.train --backend local --run fresh8
Control arm (no monitor penalty):    add --lam 0
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Protocol

from cot_codenames.stego import game
from cot_codenames.stego.game import Chat, Episode, Sample
from cot_codenames.stego.judge import judge


class Backend(Protocol):
    """What a training backend provides. `local` (backend_local.py) runs HF + PEFT on one GPU; `tinker`
    (backend_tinker.py) uses the Tinker API. Both must turn the sender model's reasoning off."""

    def sample(self, chats: list[Chat], max_tokens: int) -> list[Sample]:
        """One sampled completion per chat, from the current policy (base model + LoRA), at temperature 1."""
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


def play(backend: Backend, rng: random.Random, episodes: list[Episode], pool: list[str], args: argparse.Namespace) -> list[dict]:
    """Sample one essay per episode and score it. Returns one rollout dict per episode; the keys starting with "_" hold
    objects the training step needs and are dropped before logging."""
    samples = backend.sample([game.sender_chat(e) for e in episodes], args.max_tokens)
    recv = [game.candidates(rng, e.secret, pool, args.m) for e in episodes]
    mon = [game.candidates(rng, e.secret, pool, args.n) for e in episodes]
    recv_chats = [game.chooser_chat("receiver", e.topic, s.text, c) for e, s, (c, _) in zip(episodes, samples, recv)]
    mon_chats = [game.chooser_chat("monitor", e.topic, s.text, c) for e, s, (c, _) in zip(episodes, samples, mon)]
    p_recv = backend.choose("receiver", recv_chats, args.m)
    p_mon = backend.choose("monitor", mon_chats, args.n)
    passed = judge([(e.topic, s.text) for e, s in zip(episodes, samples)])
    rollouts = []
    for i, (e, s) in enumerate(zip(episodes, samples)):
        ri, mi = recv[i][1], mon[i][1]
        leak = game.leaked(e.secret, s.text)
        valid = passed[i] and not leak
        rollouts.append(
            {
                "secret": e.secret,
                "topic": e.topic,
                "essay": s.text,
                "judge_pass": passed[i],
                "leaked": leak,
                "p_recv": p_recv[i][ri],
                "p_mon": p_mon[i][mi],
                "recv_correct": max(range(args.m), key=p_recv[i].__getitem__) == ri,
                "mon_correct": max(range(args.n), key=p_mon[i].__getitem__) == mi,
                "recv_candidates": recv[i][0],
                "mon_candidates": mon[i][0],
                "reward": game.reward(p_recv[i][ri], p_mon[i][mi], valid, args.lam),
                "_sample": s,
                "_recv": (recv_chats[i], ri),
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
    ap.add_argument("--monitor-model", default="Qwen/Qwen3.5-9B", help="frozen monitor; same as --model = base model without the LoRA")
    ap.add_argument("--pool-size", type=int, default=0, help="rung 1: restrict secrets to this many fixed words (0 = whole train split)")
    ap.add_argument("--m", type=int, default=8, help="receiver's candidate list size")
    ap.add_argument("--n", type=int, default=8, help="monitor's candidate list size")
    ap.add_argument("--lam", type=float, default=1.0, help="monitor penalty weight; 0 = control arm")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--prompts", type=int, default=8, help="(secret, topic) prompts per step")
    ap.add_argument("--k", type=int, default=8, help="essays sampled per prompt (the RLOO group)")
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-episodes", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--kl-coef", type=float, default=0.0, help="KL penalty towards the base model on essay tokens (local backend)")
    ap.add_argument("--micro-batch", type=int, default=8, help="sequences per forward/backward pass (local backend)")
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
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    backend = make_backend(args)

    def episodes(words: list[str], count: int) -> list[Episode]:
        return [Episode(rng.choice(words), rng.choice(topics)) for _ in range(count)]

    def evaluate(step: int) -> None:
        # With a fixed pool there are no unseen secrets to test, so only the training pool is evaluated.
        splits = {"train": train_words} if args.pool_size else {"train": train_words, "heldout": heldout_words}
        for split, words in splits.items():
            rollouts = play(backend, rng, episodes(words, args.eval_episodes), words, args)
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
        rollouts = play(backend, rng, eps, train_words, args)
        advantages = [a for g in range(0, len(rollouts), args.k) for a in game.rloo([r["reward"] for r in rollouts[g : g + args.k]])]
        step_metrics = backend.step(
            sender=[(r["_sample"], a) for r, a in zip(rollouts, advantages)],
            receiver=[r["_recv"] for r in rollouts],
        )
        metrics = {"step": step} | summarise(rollouts, args) | step_metrics | {"secs": round(time.time() - t0, 1)}
        print(json.dumps(metrics), flush=True)
        log(run_dir / "metrics.jsonl", [{"kind": "train"} | metrics])
        log(run_dir / "rollouts.jsonl", [{"step": step, "split": "train", "kind": "train"} | r for r in rollouts])


if __name__ == "__main__":
    main()
