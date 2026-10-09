"""Conversion between game states and model tensors.

Shared by training and inference so the two can never drift apart — a mismatch
here is the classic source of a model that scores well offline and collapses in
the real loop.
"""

from __future__ import annotations

import torch

from .tokenizer import PAD, MASK, N_LETTERS, letter_to_token, letter_index

BLANK = "_"
IGNORE = -100


def encode_states(patterns, guessed_sets, wrongs, device=None):
    """``(tokens, guessed_multihot, wrongs)`` tensors for a batch of states.

    Sequences are right-padded to the batch maximum; ``PAD`` positions are
    excluded downstream by the key-padding mask and the pooling weights.
    """
    batch = len(patterns)
    width = max(len(p) for p in patterns)

    tokens = torch.full((batch, width), PAD, dtype=torch.long)
    multihot = torch.zeros(batch, N_LETTERS, dtype=torch.float32)
    for row, (pattern, guessed) in enumerate(zip(patterns, guessed_sets)):
        tokens[row, :len(pattern)] = torch.tensor(
            [MASK if c == BLANK else letter_to_token(c) for c in pattern],
            dtype=torch.long)
        for letter in guessed:
            multihot[row, letter_index(letter)] = 1.0

    wrongs = torch.as_tensor(list(wrongs), dtype=torch.long)
    if device is not None:
        tokens, multihot, wrongs = (t.to(device) for t in (tokens, multihot, wrongs))
    return tokens, multihot, wrongs


def encode_targets(states, teacher_letters, device=None):
    """Training targets aligned with :func:`encode_states`.

    - ``policy``: the teacher's chosen letter (``IGNORE`` where it had none).
    - ``presence``: multi-hot over letters still hidden — the set-level target,
      always well defined because it comes from the true word, not the teacher.
    - ``per_position``: the true letter at hidden positions, ``IGNORE`` at
      revealed and padded ones.
    """
    batch = len(states)
    width = max(len(s.pattern) for s in states)

    policy = torch.full((batch,), IGNORE, dtype=torch.long)
    presence = torch.zeros(batch, N_LETTERS, dtype=torch.float32)
    per_position = torch.full((batch, width), IGNORE, dtype=torch.long)

    for row, (state, letter) in enumerate(zip(states, teacher_letters)):
        if letter is not None:
            policy[row] = letter_index(letter)
        for hidden in state.hidden_letters:
            presence[row, letter_index(hidden)] = 1.0
        for col, (mask_char, true_char) in enumerate(zip(state.pattern, state.word)):
            if mask_char == BLANK:
                per_position[row, col] = letter_index(true_char)

    if device is not None:
        policy, presence, per_position = (
            t.to(device) for t in (policy, presence, per_position))
    return policy, presence, per_position
