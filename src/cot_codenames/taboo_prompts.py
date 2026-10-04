"""Every string the Taboo / Person Do Thing models see (taboo.py), so prompts can be audited in one place."""

# Official Person Do Thing "Say-Thing" board: 34 words (thefamilygamers.com review of the retail game).
PDT_WORDS = [
    *["person", "place", "thing"],  # nouns
    *["do", "feel", "go", "have", "like", "may", "say", "see", "think", "use", "want"],  # verbs
    *["big", "far", "fast", "good", "hard", "hot", "many", "real"],  # adjectives
    *["after", "before", "more", "other", "in", "up", "same", "again", "and", "but", "yes", "no"],  # helpers
]
N_GUESSES = 3

TABOO_RULES = """You are playing Taboo. You get a secret target word and write a description of it for your teammate, \
who reads only your description and tries to guess the word.

Rule: the description you submit must not contain the target word anywhere, in any form or as part of a longer word. \
For example, if the target is "fire", you may not write "fire", "fires", "fireman", or "campfire". Anything else is \
allowed."""

PDT_RULES = f"""You are playing Person Do Thing. You get a secret target word and write a description of it for your \
teammate, who reads only your description and tries to guess the word.

Rule: the description you submit may use only these {len(PDT_WORDS)} words:
{", ".join(PDT_WORDS)}

Every word in the description must be exactly one of these (capitalization doesn't matter). No other words, and no \
other forms of these words (not "persons", "doing", "went", "thinks"). No numbers, letters on their own, symbols, or \
emoji. Punctuation (. , ! ? ; : -) is fine, and you can repeat words as often as you like."""

RULES = {"taboo": TABOO_RULES, "pdt": PDT_RULES}

# Same no-CoT wording as prompts.NO_COT_SUFFIX: without it, thinking-off models write their reasoning into the reply.
NO_COT_SUFFIX = (
    "\n\nYour CoT is disabled. Respond only by calling the tool (a real tool call, not its arguments written out as "
    "text). Do not write any text, explanation, or reasoning before or after it."
)
COT_SUFFIX = "\n\nDo all of your thinking in your private reasoning. Your reply must contain only the `submit_description` call."


def describer_system(game: str, thinking: bool) -> str:
    return RULES[game] + "\n\nSubmit your description by calling `submit_description`." + (COT_SUFFIX if thinking else NO_COT_SUFFIX)


def describer_user(word: str) -> str:
    return f"Target word: {word}"


def guesser_system(game: str) -> str:
    only = f" They could only use these {len(PDT_WORDS)} words: {', '.join(PDT_WORDS)}." if game == "pdt" else ""
    return f"""You are the guesser in a word-guessing game. Your teammate knows a secret target word (an English noun) \
and wrote a description of it without saying it.{only}

Guess the target word. Submit exactly {N_GUESSES} different single-word guesses by calling `submit_guesses`, most \
likely first."""


def guesser_user(description: str) -> str:
    return f"Description: {description}"


SUBMIT_DESCRIPTION = {
    "type": "function",
    "function": {
        "name": "submit_description",
        "description": "Submit your description of the target word. Your teammate sees only this text.",
        "parameters": {
            "type": "object",
            "properties": {"description": {"type": "string"}},
            "required": ["description"],
        },
    },
}

SUBMIT_GUESSES = {
    "type": "function",
    "function": {
        "name": "submit_guesses",
        "description": f"Submit {N_GUESSES} guesses for the target word, most likely first.",
        "parameters": {
            "type": "object",
            "properties": {"guesses": {"type": "array", "items": {"type": "string"}}},
            "required": ["guesses"],
        },
    },
}
