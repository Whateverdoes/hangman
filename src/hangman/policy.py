"""Inference policy — PURE NEURAL.

This module must never import :mod:`hangman.teacher`. The oracle exists only to
generate training labels offline; the submitted player is the network alone.
``tests/test_pipeline.py`` enforces this mechanically, so the claim is
checkable by an auditor reading the test rather than trusting prose.
"""

from __future__ import annotations

import torch

from .encoding import encode_states
from .simulate import DEFAULT_HEAD, MAX_LIVES, letter_scores
from .tokenizer import LETTERS


class NeuralPolicy:
    """Single-state guessing, for debugging and interactive inspection.

    Batched play uses :mod:`hangman.simulate` instead; this is the readable
    reference implementation of the same decision rule.
    """

    def __init__(self, model, device="cpu", head=DEFAULT_HEAD):
        self.model = model.to(device).eval()
        self.device = device
        self.head = head

    @torch.no_grad()
    def scores(self, pattern: str, guessed) -> dict[str, float]:
        """Score a reachable board using the complete prior-guess set.

        Every successful guess is visible in the pattern, so the remaining
        guessed letters determine the wrong-guess count without extra input.
        """
        guessed = set(guessed)
        wrong_count = len(guessed - set(pattern))
        tokens, multihot, wrongs = encode_states(
            [pattern], [guessed], [wrong_count], device=self.device)
        revealed = torch.tensor([[c != "_" for c in pattern]],
                                dtype=torch.bool, device=self.device)
        out = letter_scores(self.model(tokens, multihot, wrongs),
                            revealed, self.head)[0]
        return {c: float(out[i]) for i, c in enumerate(LETTERS)}

    def guess(self, pattern: str, guessed) -> str | None:
        guessed = set(guessed)
        if "_" not in pattern or len(guessed - set(pattern)) >= MAX_LIVES:
            return None
        scores = self.scores(pattern, guessed)
        available = [c for c in LETTERS if c not in guessed]
        # Alphabetical ties match the batched simulator's torch.argmax.
        return max(available, key=scores.__getitem__) if available else None
