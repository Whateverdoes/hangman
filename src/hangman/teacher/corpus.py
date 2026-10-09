"""Bucketed vocabulary index used ONLY by the offline distillation teacher.

Never imported by ``policy`` or ``submit`` — inference is pure-neural, and
``tests/test_pipeline.py`` enforces that mechanically.

Words are grouped by length into per-length ``uint8`` matrices (``a``=0 ..
``z``=25) so filtering a whole bucket is a handful of vectorized comparisons
rather than a Python loop over the vocabulary.
"""

from __future__ import annotations

import numpy as np

BLANK = "_"


class _Bucket:
    __slots__ = ("words", "mat")

    def __init__(self, words: list[str], mat: np.ndarray):
        self.words = words
        self.mat = mat


class TeacherCorpus:
    """Length-bucketed candidate index over the training vocabulary.

    Weights are uniform: a single provided dataset means every word carries the
    same prior mass, so candidate *mass* reduces to candidate *count*.
    """

    def __init__(self, words):
        by_len: dict[int, list[str]] = {}
        for word in words:
            by_len.setdefault(len(word), []).append(word)

        self.buckets: dict[int, _Bucket] = {}
        self.row_of: dict[str, tuple[int, int]] = {}
        for n, group in by_len.items():
            mat = (
                np.array([w.encode("ascii") for w in group], dtype=f"|S{n}")
                .view(np.uint8)
                .reshape(len(group), n)
                - 97
            )
            self.buckets[n] = _Bucket(group, mat)
            for row, w in enumerate(group):
                self.row_of[w] = (n, row)

    def candidates(self, pattern: str, guessed, exclude: str | None = None):
        """Row indices of vocabulary words consistent with the game state.

        ``pattern`` uses ``_`` for hidden positions. ``guessed`` must be the
        COMPLETE set of prior guesses: a hidden position is known not to hold
        any already-guessed letter, and that negative evidence is a large part
        of the filter's power. A partial ``guessed`` silently over-admits.

        ``exclude`` drops one word from the candidate set — this is the
        leave-one-out hook. Without it the teacher holds the answer and its
        "optimal" move degenerates into a lookup, which would distill
        memorization into the student.
        """
        bucket = self.buckets.get(len(pattern))
        if bucket is None:
            return None, None

        mask = np.ones(len(bucket.words), dtype=bool)
        codes = np.array([ord(g) - 97 for g in guessed], dtype=np.uint8)
        for i, ch in enumerate(pattern):
            col = bucket.mat[:, i]
            if ch == BLANK:
                if len(codes):
                    mask &= ~np.isin(col, codes)
            else:
                mask &= col == (ord(ch) - 97)
            if not mask.any():
                break

        if exclude is not None:
            located = self.row_of.get(exclude)
            if located is not None and located[0] == len(pattern):
                mask[located[1]] = False

        return bucket, np.nonzero(mask)[0]
