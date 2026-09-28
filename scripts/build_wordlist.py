"""Build src/cot_codenames/wordlist-nouns.txt: a ~6k-word list of common concrete English nouns, replacing the 400-word
Codenames list for the monitor-difficulty ablation. With 400 words, a clue-only monitor finds a board word ~50% of the
time and models recite the official Codenames list from memory; a big custom list shrinks both effects.

A word is kept if it is:
- in the Brysbaert et al. (2014) concreteness norms as a single word, rated >= MIN_CONCRETENESS (1-5 scale), which
  keeps BAT/MOON/FIRE and drops "implementation"
- common: wordfreq Zipf >= MIN_ZIPF (about once per 3M words)
- a lowercase WordNet noun lemma (drops proper names like "david" and function words), not a plural or a verb form
- not on the hand-written BLOCK list (sexual terms, slurs, bodily functions)

uv run python scripts/build_wordlist.py
"""

import csv
import io
import urllib.request
from pathlib import Path

import nltk
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

CONCRETENESS_URL = "https://raw.githubusercontent.com/ArtsEngine/concreteness/master/Concreteness_ratings_Brysbaert_et_al_BRM.txt"
MIN_CONCRETENESS = 3.5
MIN_ZIPF = 2.5
OUT = Path("src/cot_codenames/wordlist-nouns.txt")
# Parts of speech (SUBTLEX dominant POS) that are never board nouns. "Name" stays: SUBTLEX tags HAWK, KNIGHT, CRANE as
# names because of surnames.
SKIP_POS = {"Number", "Preposition", "Interjection", "Article", "Conjunction", "Determiner"}
BLOCK = set(
    """anus asshole bitch blowjob boner boob bondage cock crap crotch cum cunt defecation dick dildo douche dyke
    ejaculate ejaculation erection faggot fanny fart foreskin genitalia genitals hooker homosexual hymen jackass junkie leper
    lesbian lube masturbation menstruation muff nigger nipple nude orgasm orgy ovary panty pecker penetration penis pimp
    piss porn porno pornography potty poop prick prostitute puss pussy pussycat rape rapist rectum retard scrotum scum
    semen sex shit slave sperm tampon teat testicle threesome tit titty turd urine uterus vagina vomit vulva weenie
    whore womb""".split()  # noqa: SIM905 -- a word blob is easier to scan than a quoted list
)


def noun_lemma(w: str) -> bool:
    return any(lemma.name() == w for s in wn.synsets(w, wn.NOUN) for lemma in s.lemmas())


def inflected(w: str) -> bool:
    """Plurals (regular and irregular, e.g. MEN) and -ing/-ed verb forms."""
    if any(b != w for b in wn._morphy(w, wn.NOUN)):  # MEN -> MAN; plain morphy returns MEN since it's in WordNet too
        return True
    if w.endswith("s") and not w.endswith("ss"):
        bases = {w[:-1], w[:-2] if w.endswith("es") else "", w[:-3] + "y" if w.endswith("ies") else ""}
        if any(noun_lemma(b) for b in bases if b):
            return True
    return w.endswith(("ing", "ed")) and (wn.morphy(w, wn.VERB) or w) != w


def main() -> None:
    nltk.download("wordnet", quiet=True)
    rows = csv.DictReader(io.StringIO(urllib.request.urlopen(CONCRETENESS_URL).read().decode()), delimiter="\t")
    words = sorted(
        r["Word"].upper()
        for r in rows
        if r["Bigram"] == "0"
        and (w := r["Word"]).isalpha()
        and w.isascii()
        and w.islower()
        and len(w) >= 3
        and r["Dom_Pos"] not in SKIP_POS
        and float(r["Conc.M"]) >= MIN_CONCRETENESS
        and zipf_frequency(w, "en") >= MIN_ZIPF
        and w not in BLOCK
        and noun_lemma(w)
        and not inflected(w)
    )
    OUT.write_text("\n".join(words) + "\n")
    print(f"{len(words)} words -> {OUT}")


if __name__ == "__main__":
    main()
