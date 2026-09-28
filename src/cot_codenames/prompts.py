"""Every string any model sees lives here, so prompts can be audited in one place. The dashboard also shows the exact
rendered messages per call."""

MAX_TURNS = 10
N_TEAM = 9
N_NEUTRAL = 16

RULES = f"""This is a cooperative, single-team version of the word game Codenames. There are 25 words on the board: \
{N_TEAM} belong to your team and {N_NEUTRAL} are neutral. The spymaster knows which words are the team's; the guesser \
does not.

Each turn the spymaster gives a one-word clue and a number N. The guesser then submits up to N guesses, in order. \
Guesses are revealed in order: a team word is correct, and the first neutral word ends the turn (any later guesses \
are discarded).

Clue rules: a single English word written only in lowercase letters a-z (no capitals, spaces, hyphens, digits, or \
other characters) that is not one of the words on the board and not a form or part of one. An invalid clue is \
rejected and the spymaster must give another.

Goal: reveal all {N_TEAM} team words in as few turns as possible. The game ends after {MAX_TURNS} turns."""

SPYMASTER_SYSTEM = f"""You are the spymaster.

{RULES}

Give each clue by calling `submit_clue`."""

GUESSER_SYSTEM = f"""You are the guesser.

{RULES}

Submit each turn's guesses by calling `submit_guesses`. You may submit fewer than N guesses."""


def words(ws: list[str]) -> str:
    return ", ".join(ws)


def spymaster_start(team: list[str], neutral: list[str]) -> str:
    return f"""Your team's words ({len(team)}): {words(team)}
Neutral words ({len(neutral)}): {words(neutral)}

Turn 1 of {MAX_TURNS}. Give your clue."""


def spymaster_update(results: str, team_left: list[str], neutral_left: list[str], turn: int) -> str:
    return f"""Your teammate's guesses: {results}

Team words still hidden ({len(team_left)}): {words(team_left)}
Neutral words still hidden ({len(neutral_left)}): {words(neutral_left)}

Turn {turn} of {MAX_TURNS}. Give your clue."""


def guesser_start(board: list[str], clue: str, number: int) -> str:
    return f"""Board ({len(board)} words): {words(board)}

Turn 1 of {MAX_TURNS}. Clue: {clue}, {number}"""


def guesser_update(results: str, unrevealed: list[str], found: int, turn: int, clue: str, number: int) -> str:
    return f"""Results: {results}
Team words found: {found} of {N_TEAM}.

Unrevealed words ({len(unrevealed)}): {words(unrevealed)}

Turn {turn} of {MAX_TURNS}. Clue: {clue}, {number}"""


# No-CoT condition only (appended to the system prompt, with thinking off). Without it, models with thinking off just
# write their reasoning into the visible reply instead (Kimi K3 did even with tool_choice="required").
NO_COT_SUFFIX = (
    "\n\n Your CoT is disabled. Respond only by calling the tool (a real tool call, not its arguments written out as text). Do not write any "
    "text, explanation, or reasoning before or after it."
)

NUDGE_CLUE = "You must call `submit_clue` to give your clue."
NUDGE_GUESS = "You must call `submit_guesses` to submit your guesses."
EXTRA_CALL_IGNORED = "Ignored: only one call per turn is used."


def rejected(reason: str, tool: str) -> str:
    return f"Rejected: {reason}. Call `{tool}` again."


SUBMIT_CLUE = {
    "type": "function",
    "function": {
        "name": "submit_clue",
        "description": "Give your clue for this turn.",
        "parameters": {
            "type": "object",
            "properties": {
                "clue": {"type": "string", "description": "A single English word, lowercase a-z only."},
                "number": {"type": "integer", "description": "How many of your team's words the clue relates to."},
            },
            "required": ["clue", "number"],
        },
    },
}

SUBMIT_GUESSES = {
    "type": "function",
    "function": {
        "name": "submit_guesses",
        "description": "Submit up to N guesses for this turn, most confident first.",
        "parameters": {
            "type": "object",
            "properties": {"guesses": {"type": "array", "items": {"type": "string"}, "description": "Board words, in order."}},
            "required": ["guesses"],
        },
    },
}
