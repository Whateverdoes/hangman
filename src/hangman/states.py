"""Generation of game states to train on.

The single most important property here: every state must be REACHABLE in a
real game. Two invariants the naive masked-LM setup violates:

1. Reveals are closed under letter identity. If one ``c`` is visible, every
   ``c`` is visible. Masking positions independently produces states like
   ``c_ca cola`` that the game can never present, wasting model capacity on
   physics that do not exist.
2. Hidden positions are known NOT to contain any already-guessed letter. That
   negative evidence is a large part of what a good player conditions on, so
   the guessed set is carried explicitly rather than inferred from the pattern.

Two generators, mixed. Letter-class masking gives broad coverage of the state
space; play simulation gives the on-policy distribution the deployed model will
actually meet. Simulation alone risks collapsing onto one policy's trajectory,
so we never iterate it — one round, blended with random masking.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

BLANK = "_"
LETTERS = "abcdefghijklmnopqrstuvwxyz"


@dataclass(slots=True)
class GameState:
    word: str
    pattern: str
    guessed: frozenset[str]
    wrongs: int

    @property
    def hidden_letters(self) -> set[str]:
        """Letters still to be found — the set-level prediction target."""
        return set(self.word) - self.guessed


def reveal(word: str, revealed: set[str]) -> str:
    return "".join(c if c in revealed else BLANK for c in word)


def letter_class_state(word: str, rng: random.Random, max_lives: int = 6) -> GameState:
    """Sample a reachable state by revealing a subset of DISTINCT letters."""
    distinct = sorted(set(word))
    # Number of distinct letters correctly guessed so far.
    n_hit = rng.randint(0, max(len(distinct) - 1, 0))
    hit = set(rng.sample(distinct, n_hit)) if n_hit else set()

    # Plus some wrong guesses: letters absent from the word.
    absent = [c for c in LETTERS if c not in set(word)]
    wrongs = rng.randint(0, max_lives - 1)
    miss = set(rng.sample(absent, min(wrongs, len(absent))))

    return GameState(
        word=word,
        pattern=reveal(word, hit),
        guessed=frozenset(hit | miss),
        wrongs=len(miss),
    )


def simulate_game(word, choose, max_lives: int = 6, record=None):
    """Play one game with ``choose(state) -> letter``; return (won, wrongs).

    ``record`` receives every pre-guess state, which is what makes this double
    as an on-policy state generator.
    """
    revealed: set[str] = set()
    guessed: set[str] = set()
    wrongs = 0
    target = set(word)

    while wrongs < max_lives and revealed != target:
        state = GameState(word, reveal(word, revealed), frozenset(guessed), wrongs)
        if record is not None:
            record.append(state)
        letter = choose(state)
        if letter is None or letter in guessed:
            break
        guessed.add(letter)
        if letter in target:
            revealed.add(letter)
        else:
            wrongs += 1

    return revealed == target, wrongs


def build_state_pool(words, choose, rng, sim_fraction=0.5, max_lives=6,
                     per_word_random=1, progress=None):
    """Mixed pool of reachable states across ``words``.

    ``sim_fraction`` of words contribute their full on-policy trajectory; the
    rest contribute ``per_word_random`` letter-class samples.
    """
    pool: list[GameState] = []
    for i, word in enumerate(words):
        if rng.random() < sim_fraction:
            simulate_game(word, choose, max_lives=max_lives, record=pool)
        else:
            for _ in range(per_word_random):
                pool.append(letter_class_state(word, rng, max_lives))
        if progress and (i + 1) % progress == 0:
            print(f"  states: {i+1}/{len(words)} words -> {len(pool)} states",
                  flush=True)
    return pool
