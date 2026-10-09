"""Leave-one-out oracle distillation: turning the teacher into training labels.

The student imitates a player, not a letter-frequency table. For every game
state we ask the teacher "what would an optimal player guess here?" and train
against that decision.

LEAVE-ONE-OUT IS LOAD-BEARING. For target word ``w`` the teacher's candidate
set is built from ``train \\ {w}``. If ``w`` stayed in, the teacher would hold
the answer, its "optimal" move would collapse into a lookup, and the student
would distill memorization rather than strategy — the precise leakage failure
this project has already been burned by once. With ``w`` removed the teacher
solves the same unseen-word problem the student meets at test time.

Because the teacher is denied the answer it sometimes has NO consistent
candidates, and its label is then noise. Each example therefore carries a
``trust`` weight; the trainer uses it to fade between imitating the teacher and
falling back to the always-valid set-level target.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .states import GameState, letter_class_state, simulate_game
from .teacher.corpus import TeacherCorpus
from .teacher.oracle import best_letter, trust


@dataclass(slots=True)
class Example:
    state: GameState
    letter: str | None
    trust: float


class TeacherPolicy:
    """Callable oracle over a vocabulary, with the target word held out."""

    def __init__(self, corpus: TeacherCorpus, tau: float = 50.0,
                 max_lives: int = 6, risk_floor: float = 0.0):
        self.corpus = corpus
        self.tau = tau
        self.max_lives = max_lives
        self.risk_floor = risk_floor

    def evaluate(self, state: GameState) -> tuple[str | None, float]:
        bucket, idx = self.corpus.candidates(
            state.pattern, state.guessed, exclude=state.word)
        n = 0 if idx is None else len(idx)
        letter = best_letter(
            bucket, idx, state.guessed,
            lives_left=self.max_lives - state.wrongs,
            max_lives=self.max_lives, risk_floor=self.risk_floor)
        return letter, trust(n, self.tau)

    def __call__(self, state: GameState) -> str | None:
        letter, _ = self.evaluate(state)
        if letter is None:
            # Teacher is blind here; fall back so the rollout keeps moving and
            # keeps producing states rather than stalling.
            for c in "esiarntolcdupmghbyfvkwzxqj":
                if c not in state.guessed:
                    return c
        return letter


def generate(words, policy: TeacherPolicy, rng: random.Random,
             sim_fraction: float = 0.5, per_word_random: int = 2,
             max_lives: int = 6, progress: int = 0) -> list[Example]:
    """Mixed on-policy / random-mask states, each labelled by the teacher."""
    examples: list[Example] = []
    for i, word in enumerate(words):
        states: list[GameState] = []
        if rng.random() < sim_fraction:
            simulate_game(word, policy, max_lives=max_lives, record=states)
        else:
            states = [letter_class_state(word, rng, max_lives)
                      for _ in range(per_word_random)]
        for state in states:
            letter, confidence = policy.evaluate(state)
            examples.append(Example(state, letter, confidence))
        if progress and (i + 1) % progress == 0:
            print(f"  {i+1}/{len(words)} words -> {len(examples)} examples",
                  flush=True)
    return examples


def load_examples(path: str, limit: int = 0) -> list[Example]:
    """Read back a TSV written by ``scripts/build_labels.py``."""
    examples: list[Example] = []
    with open(path, encoding="utf-8") as fh:
        header = next(fh, "")
        if not header.startswith("word\t"):
            raise ValueError(f"{path}: unexpected header {header!r}")
        for line in fh:
            word, pattern, guessed, wrongs, letter, confidence = \
                line.rstrip("\n").split("\t")
            examples.append(Example(
                state=GameState(word, pattern, frozenset(guessed), int(wrongs)),
                letter=letter or None,
                trust=float(confidence),
            ))
            if limit and len(examples) >= limit:
                break
    return examples
