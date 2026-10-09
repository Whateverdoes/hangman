"""The distillation teacher: Bayes-optimal-ish play over the training vocabulary.

Offline only. Given a game state, it filters the vocabulary to the consistent
candidate set and picks the letter maximizing a blend of information gain and
hit probability. That decision — not "which letters are in this word" — is what
the student network is trained to imitate.

Deliberately 1-ply. Exact expectimax is provably better but explodes
combinatorially, and the organizers re-run this pipeline, so label-generation
cost is part of the reproduction budget we ship to them.
"""

from __future__ import annotations

import math

import numpy as np

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def trust(n_candidates: int, tau: float = 50.0) -> float:
    """Confidence in the candidate set, in [0, 1].

    ``n / (n + tau)`` — candidate evidence weighed against an out-of-vocabulary
    prior of pseudo-mass ``tau``. Under leave-one-out the teacher frequently has
    no consistent candidates at all; this scalar is what lets the trainer
    down-weight those states instead of learning from noise.

    ``tau -> 0`` recovers always-trust, ``tau -> inf`` never-trust.
    """
    if n_candidates <= 0:
        return 0.0
    return n_candidates / (n_candidates + tau)


MAX_CANDIDATES = 4000


def best_letter(bucket, idx, guessed, lives_left=6, max_lives=6,
                risk_floor=0.0, max_candidates=MAX_CANDIDATES):
    """Highest expected-value guess over the candidate set.

    Blends normalized information gain against hit probability, sliding toward
    hit probability as lives run out. At ``lives_left == 1`` the blend is pure
    hit probability: information has no value if the next miss ends the game.

    Very large candidate sets are subsampled to ``max_candidates``. Both terms
    are expectations over the candidate distribution, so a sample estimates the
    same quantity; on a 29k-word bucket the ranking is stable long before the
    full set is consulted, and this keeps label generation inside the budget we
    ask the organizers to re-run. Bucket order is already randomized by the
    shuffled train split, so a prefix slice is an unbiased sample and stays
    deterministic.
    """
    if bucket is None or idx is None or len(idx) == 0:
        return None

    if len(idx) > max_candidates:
        idx = idx[:max_candidates]
    mat = bucket.mat[idx]
    total = len(idx)
    span = max(max_lives - 1, 1)
    alpha = 1.0 - (1.0 - risk_floor) * (max(lives_left, 1) - 1) / span
    log_total = math.log2(total) if total > 1 else 0.0

    best, best_score = None, -1.0
    for li in range(26):
        letter = LETTERS[li]
        if letter in guessed:
            continue
        occ = mat == li
        hits = int(occ.any(axis=1).sum())
        if hits == 0:
            continue

        # Partition candidates by WHICH positions the letter occupies: two words
        # only stay mutually consistent if the reveal lands identically.
        occ = np.ascontiguousarray(occ)
        sig = occ.view(np.dtype((np.void, occ.shape[1]))).ravel()
        _, counts = np.unique(sig, return_counts=True)
        probs = counts / total
        entropy = float(-(probs * np.log2(probs)).sum())
        info = entropy / log_total if log_total > 0 else 0.0

        score = (1.0 - alpha) * info + alpha * (hits / total)
        if score > best_score:
            best, best_score = letter, score
    return best


def letter_presence(bucket, idx, guessed):
    """Posterior P(letter occurs among hidden positions) over the candidates.

    Returned as a 26-vector. Used as a soft distillation target alongside the
    teacher's hard letter choice.
    """
    out = np.zeros(26, dtype=np.float32)
    if bucket is None or idx is None or len(idx) == 0:
        return out
    mat = bucket.mat[idx]
    total = len(idx)
    for li in range(26):
        if LETTERS[li] in guessed:
            continue
        out[li] = float((mat == li).any(axis=1).sum()) / total
    return out
