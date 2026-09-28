"""Run games in parallel, appending each finished game to runs/stage-N/games.jsonl. Resumable: skips game ids already there.

uv run python -m cot_codenames.run --stage 1 --games 10 [--first-seed 0] [--models kimi-k3 ...] [--conditions cot cot-told ...]
"""

import argparse
import fcntl
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cot_codenames.client import DEFAULT_MODELS, MODELS, THINKING_MANDATORY
from cot_codenames.game import load_games, play

CONDITIONS = {
    "cot": {"thinking": True, "told": False},
    "nocot": {"thinking": False, "told": False},
    "cot-told": {"thinking": True, "told": True},
    "nocot-told": {"thinking": False, "told": True},
    "cot-told-fs": {"thinking": True, "told": True, "fewshot": True},  # + worked examples of label-only reasoning
    "cot-told-fs-idx": {"thinking": True, "told": True, "fewshot": True, "indexed": True},  # + pre-labeled board
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--games", type=int, default=10, help="games per (model, condition); seeds 0..games-1, shared across models")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(MODELS))
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--out", help="default: runs/stage-N/games.jsonl")
    args = ap.parse_args()

    out = Path(args.out or f"runs/stage-{args.stage}/games.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Infra/crash errors get replayed (the failed record stays in the file; readers keep the last per id). Protocol
    # errors are the model's own failure, so they count as played: replaying would select them out of the stats.
    done = {g["id"] for g in load_games(out) if not g["error"] or g.get("error_kind") == "protocol"} if out.exists() else set()
    jobs = [
        (m, CONDITIONS[c], s)
        for m in args.models
        for c in args.conditions
        if not (m in THINKING_MANDATORY and not CONDITIONS[c]["thinking"])
        for s in range(args.first_seed, args.first_seed + args.games)
        if f"{m}|{c}|{s}" not in done
    ]
    print(f"{len(jobs)} games to play ({len(done)} already in {out})", flush=True)

    def one(job: tuple[str, dict, int]) -> None:
        g = play(*job)
        cost = sum((c["usage"] or {}).get("cost", 0) for c in g["calls"])
        with out.open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)  # other runner processes may append to the same file concurrently
            f.write(json.dumps(g) + "\n")
        err = g["error"] and f"{g['error_kind']}: {g['error']}"
        print(f"{g['id']:32} finished={g['turns_to_finish']} found={g['found']} err={err} ${cost:.3f}", flush=True)

    # Games are sequential inside (turn t needs turn t-1), so all the parallelism is across games. 500 = client pool size.
    with ThreadPoolExecutor(min(max(len(jobs), 1), 500)) as pool:
        list(pool.map(one, jobs))


if __name__ == "__main__":
    main()
