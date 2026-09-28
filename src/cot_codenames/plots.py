"""Graphs from runs/stage-N/games.jsonl -> docs/figs/stage-N-*.png.

uv run python -m cot_codenames.plots --stage 0|1
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cot_codenames.client import DEFAULT_MODELS
from cot_codenames.game import cond_name, load_games
from cot_codenames.monitor import grep
from cot_codenames.prompts import MAX_TURNS

# Reference palette (dataviz skill): categorical slots 1-4 in fixed order (validated for adjacent pairs), recessive ink
# for axes/text. A condition keeps its color across stages.
COLORS = {"cot": "#2a78d6", "cot-told": "#eb6834", "nocot": "#1baf7a", "nocot-told": "#eda100"}
LABELS = {"cot": "CoT", "cot-told": "CoT, told", "nocot": "no CoT", "nocot-told": "no CoT, told"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e7e5e4"
FIGS = Path("docs/figs")

plt.rcParams.update(
    {
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "font.size": 10,
        "figure.dpi": 150,
    }
)


def spy_reasoning_tokens(g: dict) -> list[int]:
    """One value per turn: a turn can span several calls (a rejected clue + the retry), so sum them."""
    per_turn: dict[int, int] = {}
    for c in g["calls"]:
        if c["role"] == "spymaster":
            tokens = ((c["usage"] or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            per_turn[c["turn"]] = per_turn.get(c["turn"], 0) + tokens
    return list(per_turn.values())


def strip(ax, models: list[str], values: dict[tuple[str, str], list[float]], ylabel: str, title: str) -> None:
    """Per model: one column per condition, each game a small dot, mean as a bar-tick with a ±1 SE whisker."""
    rng = np.random.default_rng(0)
    conds = [c for c in COLORS if any(k[1] == c for k in values)]
    for i, m in enumerate(models):
        for j, cond in enumerate(conds):
            ys = values.get((m, cond), [])
            if not ys:
                continue
            x = i + (j - (len(conds) - 1) / 2) * 0.8 / len(conds)
            ax.scatter(x + rng.uniform(-0.06, 0.06, len(ys)), ys, s=14, color=COLORS[cond], alpha=0.45, linewidths=0)
            mu, se = np.mean(ys), np.std(ys, ddof=1) / np.sqrt(len(ys)) if len(ys) > 1 else 0.0
            ax.errorbar(x, mu, yerr=se, fmt="_", color=COLORS[cond], markersize=16, markeredgewidth=2, elinewidth=2, capsize=0)
    ax.set_xticks(range(len(models)), models, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    if len(conds) > 1:  # a single series is named by the title, no legend
        handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[c], label=LABELS[c]) for c in conds]
        # Outside the axes: the data fills both the top (100% leak) and the bottom (0%) of these plots.
        ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)


def grouped(games: list[dict], f) -> dict[tuple[str, str], list[float]]:
    out: dict[tuple[str, str], list[float]] = {}
    for g in games:
        out.setdefault((g["model"], cond_name(g["condition"])), []).extend(f(g))
    return out


def turns_fig(ax, models: list[str], games: list[dict]) -> None:
    """Turns to finish; unfinished games plotted at MAX_TURNS + 1 (censored)."""
    strip(
        ax, models, grouped(games, lambda g: [g["turns_to_finish"] or MAX_TURNS + 1]), "turns to finish", "Turns to find all 9 team words"
    )
    ax.axhline(MAX_TURNS + 1, color=MUTED, lw=1, ls=":")
    ax.text(len(models) - 0.5, MAX_TURNS + 1.1, "did not finish", color=MUTED, fontsize=8, ha="right")


def stage1(games: list[dict]) -> None:
    models = [m for m in DEFAULT_MODELS if any(g["model"] == m for g in games)]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    turns_fig(ax, models, games)
    fig.tight_layout()
    fig.savefig(FIGS / "stage-1-turns.png")

    leaks = [g | {"grep": grep(g)} for g in games]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    strip(
        axes[0],
        models,
        grouped(leaks, lambda g: [100 * g["grep"]["frac_turns_leaked"]]),
        "% of turns",
        "Spymaster turns naming a board word",
    )
    distinct = grouped(leaks, lambda g: [g["grep"]["distinct_team"] + g["grep"]["distinct_neutral"]])
    strip(axes[1], models, distinct, "board words (of 25)", "Distinct board words the spymaster wrote")
    fig.tight_layout()
    fig.savefig(FIGS / "stage-1-leak.png")
    print(f"wrote 2 figs to {FIGS} from {len(games)} games")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    args = ap.parse_args()
    # Protocol errors are results (the model failed to move), so they stay; infra/crash records are dropped.
    games = [g for g in load_games(Path(f"runs/stage-{args.stage}/games.jsonl")) if g.get("error_kind") in (None, "protocol")]
    FIGS.mkdir(parents=True, exist_ok=True)
    if args.stage == 1:
        stage1(games)
        return
    models = sorted({g["model"] for g in games})
    by = lambda f: grouped(games, f)

    # 1. Turns to finish.
    fig, ax = plt.subplots(figsize=(7, 3.6))
    turns_fig(ax, models, games)
    fig.tight_layout()
    fig.savefig(FIGS / "stage-0-turns.png")

    # 2. Spymaster reasoning tokens per turn (CoT condition only: no-CoT is 0 by construction).
    fig, ax = plt.subplots(figsize=(7, 3.6))
    strip(
        ax,
        models,
        {k: v for k, v in by(spy_reasoning_tokens).items() if k[1] == "cot"},
        "reasoning tokens",
        "Spymaster reasoning tokens per turn",
    )
    fig.tight_layout()
    fig.savefig(FIGS / "stage-0-cot-tokens.png")

    # 3. First clue's N and team words found on turn 1.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    strip(axes[0], models, by(lambda g: [g["turns"][0]["number"]] if g["turns"] else []), "N", "First clue's N (ambition)")
    first_correct = by(lambda g: [sum(r["kind"] == "team" for r in g["turns"][0]["reveals"])] if g["turns"] else [])
    strip(axes[1], models, first_correct, "team words", "Team words found on turn 1")
    fig.tight_layout()
    fig.savefig(FIGS / "stage-0-first-turn.png")
    print(f"wrote 3 figs to {FIGS} from {len(games)} games")


if __name__ == "__main__":
    main()
