"""Vocabulary loading and the train/validation split.

The provided ``train.txt`` is the ONLY permitted data source. Everything the
model sees derives from it, so that the organizers' re-training pass reproduces
our numbers exactly.

The validation slice is held out from *both* the student's training states and
the teacher's candidate vocabulary. It is the only honest proxy we have for the
firewalled private evaluation set, so nothing may leak into it.
"""

from __future__ import annotations

import os
import random
import re

DEFAULT_TRAIN = os.path.join("data", "train.txt")

_NON_LETTER = re.compile(r"[^a-z]")

# Word-length bands. Defined here rather than in evaluate.py because both
# evaluation (reporting) and training (length-weighted sampling) need them, and
# a second definition would be free to drift out of step with the first.
LENGTH_BANDS = [("1-5", 1, 5), ("6-8", 6, 8), ("9-12", 9, 12),
                ("13-17", 13, 17), ("18+", 18, 999)]


def band_of(length: int) -> str:
    """The LENGTH_BANDS name covering ``length``."""
    for name, lo, hi in LENGTH_BANDS:
        if lo <= length <= hi:
            return name
    raise ValueError(f"no band covers length {length}")


def normalize(raw: str) -> str:
    """Lowercase and drop anything outside ``a-z``.

    The measured ``train.txt`` is already pure ``a-z`` with no spaces or
    punctuation, so this is defensive rather than load-bearing. It exists so a
    differently-shaped ``test.txt`` cannot inject unexpected tokens.
    """
    return _NON_LETTER.sub("", raw.strip().lower())


def load_words(path: str = DEFAULT_TRAIN, min_len: int = 1) -> list[str]:
    """Deduplicated, order-stable vocabulary from a newline-delimited file."""
    seen: dict[str, None] = {}
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            word = normalize(line)
            if len(word) >= min_len:
                seen.setdefault(word, None)
    return list(seen)


def split_words(
    words: list[str], val_size: int = 5000, seed: int = 7
) -> tuple[list[str], list[str]]:
    """Deterministic disjoint split.

    Returns ``(train, val)``. Sorting before shuffling makes the split
    independent of the input file's line order, so the audit reproduces it even
    if the file is re-serialized.
    """
    pool = sorted(words)
    random.Random(seed).shuffle(pool)
    val_size = min(val_size, len(pool) // 2)
    return pool[val_size:], pool[:val_size]


def describe(words: list[str]) -> dict:
    """Corpus statistics for the README and for Step-0-style sanity checks."""
    lengths = [len(w) for w in words]
    alphabet = sorted({c for w in words for c in w})
    return {
        "count": len(words),
        "unique": len(set(words)),
        "alphabet_size": len(alphabet),
        "alphabet": "".join(alphabet),
        "min_len": min(lengths, default=0),
        "max_len": max(lengths, default=0),
        "mean_len": round(sum(lengths) / max(len(lengths), 1), 2),
    }


DEFAULT_TEST = os.path.join("data", "test.txt")


def assert_no_leakage(vocabulary, test_path: str = DEFAULT_TEST,
                      label: str = "training vocabulary") -> int:
    """Abort if any evaluation word appears in a set used for training.

    Called automatically wherever training data is assembled. It is cheap, it
    runs every time, and it fails loudly — the predecessor project shipped
    numbers inflated up to 3x by exactly this class of contamination, discovered
    only after the fact.

    Returns the size of the checked test set so callers can log that the check
    actually ran against something, rather than silently passing on a missing
    file.
    """
    if not os.path.exists(test_path):
        print(f"[leak-check] no {test_path}; nothing to check against")
        return 0

    held_out = set(load_words(test_path))
    overlap = held_out & set(vocabulary)
    if overlap:
        sample = sorted(overlap)[:10]
        raise SystemExit(
            f"LEAKAGE: {len(overlap)} evaluation words are present in the "
            f"{label}.\n  e.g. {sample}\n"
            f"Training must never see {test_path}. Refusing to continue."
        )
    print(f"[leak-check] OK — 0 of {len(held_out)} evaluation words appear in "
          f"the {label}")
    return len(held_out)
