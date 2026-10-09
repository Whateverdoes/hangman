"""Batched lockstep game simulation.

The submission needs ~250k games x ~10 guesses. Played one word at a time that
is ~2.5M single-sequence forward passes and will not finish inside a Kaggle
session. Here every game advances together: one batched forward per TURN, so
the cost scales with the *maximum* number of guesses rather than their sum.

Games are grouped by exact word length, which removes padding entirely — each
group is a dense rectangular tensor.
"""

from __future__ import annotations

import torch

from .tokenizer import MASK, PAD, letter_to_token

MAX_LIVES = 6
DEFAULT_HEAD = "perpos"


def letter_scores(out, revealed, head=DEFAULT_HEAD):
    """Per-letter "is it hidden here" scores from the model's heads.

    ``perpos`` aggregates the per-position distribution over the still-hidden
    slots: P(letter somewhere) = 1 - prod(1 - p_i). That product assumes
    positional independence, which is false inside a word — but the
    per-position head supplies the documented inference rule. Its supervision
    comes from the true character at each hidden position.

    ``presence`` is the set-level head, kept for ablation.
    """
    if head == "presence":
        return out["presence"]
    if head == "policy":
        return out["policy"]
    logp = torch.log_softmax(out["per_position"], dim=-1)
    hidden = (~revealed).unsqueeze(-1)
    log_miss = (torch.log1p(-logp.exp().clamp(max=1 - 1e-6)) * hidden).sum(1)
    return torch.log1p(-log_miss.exp().clamp(max=1 - 1e-6))


def _words_to_matrix(words):
    return torch.tensor([[ord(c) - 97 for c in w] for w in words],
                        dtype=torch.long)


@torch.no_grad()
def play_group(model, words, device="cpu", max_lives=MAX_LIVES,
               head=DEFAULT_HEAD):
    """Play a batch of equal-length words. Returns (guess_strings, won, wrongs).

    ``guess_strings[i]`` is the chronological sequence actually played, which is
    exactly the ``guessed_letters_string`` the submission format expects.
    """
    n, length = len(words), len(words[0])
    if any(len(w) != length for w in words):
        raise ValueError(
            "play_group requires equal-length words (it builds a dense "
            "rectangular batch). Use play_words, which groups by length.")
    letters = _words_to_matrix(words).to(device)

    revealed = torch.zeros(n, length, dtype=torch.bool, device=device)
    guessed = torch.zeros(n, 26, dtype=torch.bool, device=device)
    wrongs = torch.zeros(n, dtype=torch.long, device=device)
    active = torch.ones(n, dtype=torch.bool, device=device)
    sequences: list[list[str]] = [[] for _ in range(n)]

    # A letter token offset: MASK where hidden, real token where revealed.
    token_of_letter = letters + 2

    for _ in range(26):
        if not bool(active.any()):
            break

        tokens = torch.where(revealed, token_of_letter,
                             torch.full_like(letters, MASK))
        out = model(tokens, guessed.float(), wrongs.clamp(max=max_lives))
        logits = letter_scores(out, revealed, head)
        # Never re-guess: a repeat counts as a wrong guess under the rules.
        logits = logits.masked_fill(guessed, float("-inf"))
        pick = logits.argmax(dim=1)

        occurrences = letters == pick.unsqueeze(1)
        hit = occurrences.any(dim=1)

        revealed |= occurrences & active.unsqueeze(1)
        guessed[torch.arange(n, device=device), pick] |= active
        wrongs += (~hit & active).long()

        for i in torch.nonzero(active, as_tuple=False).flatten().tolist():
            sequences[i].append(chr(97 + int(pick[i])))

        solved = revealed.all(dim=1)
        active = active & ~solved & (wrongs < max_lives)

    won = revealed.all(dim=1)
    return ["".join(s) for s in sequences], won.cpu(), wrongs.cpu()


def play_words(model, words, device="cpu", batch_size=2048,
               max_lives=MAX_LIVES, head=DEFAULT_HEAD, progress=0):
    """Play an arbitrary word list, preserving input order in the output."""
    by_length: dict[int, list[int]] = {}
    for i, word in enumerate(words):
        by_length.setdefault(len(word), []).append(i)

    guesses = [""] * len(words)
    won = [False] * len(words)
    wrongs = [0] * len(words)

    done = 0
    for length in sorted(by_length):
        indices = by_length[length]
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            seqs, w, e = play_group(
                model, [words[i] for i in chunk], device=device,
                max_lives=max_lives, head=head)
            for slot, i in enumerate(chunk):
                guesses[i] = seqs[slot]
                won[i] = bool(w[slot])
                wrongs[i] = int(e[slot])
            previous, done = done, done + len(chunk)
            if progress and done // progress > previous // progress:
                print(f"  played {done}/{len(words)}", flush=True)

    return guesses, won, wrongs
