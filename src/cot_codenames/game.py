"""Single-team Codenames: one model plays spymaster and guesser. Each role is one long tool-calling episode, and we pass
each assistant message back verbatim (incl. `reasoning_details`) so every agent sees its own full CoT history."""

import hashlib
import json
import random
import re
import subprocess
import traceback
from collections.abc import Callable
from pathlib import Path

from cot_codenames import prompts as P
from cot_codenames.client import MODELS, chat, tool_choice

SRC = Path(__file__).parent
WORDLIST = [w for w in (SRC / "wordlist.txt").read_text().split("\n") if w]
# Provenance only (not enforced): hash of everything that changes what a game measures.
CODE_HASH = hashlib.sha256(b"".join((SRC / f).read_bytes() for f in ("prompts.py", "game.py", "wordlist.txt"))).hexdigest()[:12]
_git = lambda *a: subprocess.run(["git", *a], cwd=SRC, capture_output=True, text=True, check=False).stdout.strip()
GIT = {"git_sha": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain", "--", "."))}
# Clues are exactly [a-z]+: no capitals, digits, punctuation or unicode, so the clue string has no side channel beyond
# the word itself. Stego ablation idea: loosen this (allow capitals/digits/punctuation) and see if models start
# smuggling information through it.
CLUE_RE = re.compile(r"[a-z]+")


def cond_name(condition: dict) -> str:
    return ("cot" if condition["thinking"] else "nocot") + ("-told" if condition.get("told") else "")


def system_prompt(role: str, condition: dict) -> str:
    base, tool = (P.SPYMASTER_SYSTEM, "submit_clue") if role == "spymaster" else (P.GUESSER_SYSTEM, "submit_guesses")
    told = P.TOLD_SUFFIX if role == "spymaster" and condition.get("told") else ""
    return base + told + (P.cot_suffix(tool) if condition["thinking"] else P.NO_COT_SUFFIX)


def check_clue(args: dict, board: list[str]) -> str | None:
    """Why a submit_clue call is rejected, or None if it's accepted. Structural checks only; whether the clue is a real
    English word (and not a form of a board word) is still hand-validated."""
    clue, number = args.get("clue"), args.get("number")
    if not isinstance(clue, str) or not CLUE_RE.fullmatch(clue):
        return "the clue must be a single word of lowercase letters a-z only"
    if clue.upper() in {part for w in board for part in w.split()}:  # split: "YORK" is part of "NEW YORK"
        return f"{clue.upper()} is (part of) a word on the board"
    if not isinstance(number, int) or isinstance(number, bool) or number < 1:
        return "the number must be a positive integer"
    return None


class Agent:
    def __init__(self, role: str, model: str, thinking: bool, system: str, tool: dict, nudge: str, calls: list[dict]):
        self.role, self.model, self.thinking, self.tool, self.nudge, self.calls = role, model, thinking, tool, nudge, calls
        self.messages: list[dict] = [{"role": "system", "content": system}]
        self.pending_call_ids: list[str] = []

    def act(self, text: str, turn: int, check: Callable[[dict], str | None] = lambda _: None) -> dict | None:
        """Send `text` (as the tool result for our last call, or as a user message on the first turn); return parsed
        tool args, or None if the model never made a valid call. `check` returns a rejection reason for bad args."""
        if self.pending_call_ids:
            first, *rest = self.pending_call_ids
            self.messages.append({"role": "tool", "tool_call_id": first, "content": text})
            self.messages += [{"role": "tool", "tool_call_id": i, "content": P.EXTRA_CALL_IGNORED} for i in rest]
            self.pending_call_ids = []
        else:
            self.messages.append({"role": "user", "content": text})
        name = self.tool["function"]["name"]
        for attempt in range(3):
            n_in = len(self.messages)
            r = chat(self.model, self.messages, [self.tool], self.thinking)
            msg = r["message"]
            self.messages.append(msg)
            call = {"role": self.role, "turn": turn, "attempt": attempt, "n_in": n_in, "generation_id": r["id"], "served_model": r["model"]}
            call |= {k: r[k] for k in ("usage", "provider", "latency_s", "finish_reason", "failed_attempts")}
            self.calls.append(call)
            calls = msg.get("tool_calls") or []
            if not calls:
                call["rejected"] = "no tool call"
                self.messages.append({"role": "user", "content": self.nudge})
                continue
            self.pending_call_ids = [c["id"] for c in calls]
            c = calls[0]
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = None
            if c["function"]["name"] != name or not isinstance(args, dict):
                call["rejected"], feedback = "malformed tool call", self.nudge
            elif reason := check(args):
                call["rejected"], feedback = reason, P.rejected(reason, name)
            else:
                return args
            # Answer every call so the transcript stays valid, then let the model try again.
            self.messages += [{"role": "tool", "tool_call_id": i, "content": feedback} for i in self.pending_call_ids]
            self.pending_call_ids = []
        return None


def reveal(guesses: list[str], number: int, board: list[str], team: set[str], revealed: set[str]) -> tuple[list[dict], str]:
    """Apply up to `number` guesses in order; stop at the first neutral. Returns (reveals, human-readable results)."""
    reveals, parts = [], []
    for g in guesses[:number]:
        w = str(g).strip().upper()
        if w not in board or w in revealed:
            reveals.append({"word": w, "kind": "invalid"})
            parts.append(f"{w} (not an unrevealed board word, ignored)")
            continue
        revealed.add(w)
        kind = "team" if w in team else "neutral"
        reveals.append({"word": w, "kind": kind})
        parts.append(f"{w} (team, correct)" if kind == "team" else f"{w} (neutral, turn over)")
        if kind == "neutral":
            break
    return reveals, ", ".join(parts) or "(no guesses)"


def play(model: str, condition: dict, seed: int) -> dict:
    thinking = condition["thinking"]
    rng = random.Random(seed)
    board = rng.sample(WORDLIST, P.N_TEAM + P.N_NEUTRAL)
    team, neutral = board[: P.N_TEAM], board[P.N_TEAM :]
    board_order = rng.sample(board, len(board))  # guesser's view, so team words aren't listed first
    calls: list[dict] = []
    spy = Agent("spymaster", model, thinking, system_prompt("spymaster", condition), P.SUBMIT_CLUE, P.NUDGE_CLUE, calls)
    gus = Agent("guesser", model, thinking, system_prompt("guesser", condition), P.SUBMIT_GUESSES, P.NUDGE_GUESS, calls)
    slug, provider = MODELS[model]
    game = {
        "id": f"{model}|{cond_name(condition)}|{seed}",
        "model": model,
        "condition": condition,
        "seed": seed,
        "config": {"code_hash": CODE_HASH, "slug": slug, "provider": provider, "tool_choice": tool_choice(thinking)} | GIT,
        "team": team,
        "neutral": neutral,
        "board_order": board_order,
        "turns": [],
        "turns_to_finish": None,
        "error": None,
        # "protocol": the model never made a valid move (a result, not replayed). "infra": the API gave up. "crash": a
        # bug in our code. Both of the latter get replayed by run.py; the failed record stays in the file.
        "error_kind": None,
    }
    revealed: set[str] = set()
    spy_text = P.spymaster_start(team, neutral)
    results = ""
    try:
        for turn in range(1, P.MAX_TURNS + 1):
            clue_args = spy.act(spy_text, turn, lambda a: check_clue(a, board))
            if clue_args is None:
                game["error"], game["error_kind"] = f"no valid clue on turn {turn}", "protocol"
                break
            clue, number = clue_args["clue"], clue_args["number"]  # both validated by check_clue
            unrevealed = [w for w in board_order if w not in revealed]
            found = len(revealed & set(team))
            g_text = (
                P.guesser_start(board_order, clue, number)
                if turn == 1
                else P.guesser_update(results, unrevealed, found, turn, clue, number)
            )
            g_args = gus.act(g_text, turn)
            if g_args is None:
                game["error"], game["error_kind"] = f"no valid guesses on turn {turn}", "protocol"
                break
            raw = g_args.get("guesses", [])
            raw = raw if isinstance(raw, list) else [raw]
            reveals, results = reveal(raw, number, board, set(team), revealed)
            game["turns"].append({"turn": turn, "clue": clue, "number": number, "raw_guesses": raw, "reveals": reveals})
            if set(team) <= revealed:
                game["turns_to_finish"] = turn
                break
            spy_text = P.spymaster_update(
                results, [w for w in team if w not in revealed], [w for w in neutral if w not in revealed], turn + 1
            )
    except RuntimeError as e:  # client.chat gave up
        game["error"], game["error_kind"] = str(e)[-300:], "infra"
    except Exception:  # noqa: BLE001 -- anything else is our bug; keep the transcript so far instead of losing the game
        game["error"], game["error_kind"] = traceback.format_exc()[-1000:], "crash"
    game["found"] = len(revealed & set(team))
    game["spymaster"] = {"messages": spy.messages, "tools": [P.SUBMIT_CLUE]}
    game["guesser"] = {"messages": gus.messages, "tools": [P.SUBMIT_GUESSES]}
    game["calls"] = calls
    return game


def load_games(path: Path) -> list[dict]:
    """Last record per game id. The file itself is append-only and keeps every attempt: a game that hit an infra/crash
    error and was replayed appears more than once (read the file directly to count failures)."""
    games = list({g["id"]: g for g in map(json.loads, path.open())}.values())
    for g in games:  # pre-config records (stage-0 pilot) stored a bare `thinking` bool
        g.setdefault("condition", {"thinking": g.get("thinking")})
    return games
