"""Training loop with configurable losses on a shared character encoder.

The documented recipe trains the per-position head on hidden characters and
uses that head at inference. The experimental policy head imitates an offline
teacher with trust-weighted cross-entropy; the presence head predicts the set
of hidden letters with binary cross-entropy. Their loss weights are disabled
in the documented recipe, but remain configurable for experiments.
"""

from __future__ import annotations

import math
import random
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import band_of, LENGTH_BANDS
from .encoding import encode_states, encode_targets, IGNORE
from .model import build_model, save_checkpoint
from .tokenizer import LETTERS


warnings.filterwarnings(
    "ignore", message=".*enable_nested_tensor is True.*")


def pick_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_batch(examples, device):
    states = [e.state for e in examples]
    tokens, guessed, wrongs = encode_states(
        [s.pattern for s in states], [s.guessed for s in states],
        [s.wrongs for s in states], device=device)
    policy, presence, per_position = encode_targets(
        states, [e.letter for e in examples], device=device)
    weights = torch.tensor([e.trust for e in examples],
                           dtype=torch.float32, device=device)
    return tokens, guessed, wrongs, policy, presence, per_position, weights


def compute_loss(out, policy, presence, per_position, weights,
                 w_policy=1.0, w_presence=1.0, w_perpos=0.5):
    labelled = policy != IGNORE
    if labelled.any():
        raw = F.cross_entropy(
            out["policy"][labelled], policy[labelled], reduction="none")
        # Trust-weighted mean: states the teacher could not judge contribute
        # nothing here, rather than contributing a wrong answer.
        scale = weights[labelled]
        policy_loss = (raw * scale).sum() / scale.sum().clamp(min=1e-6)
    else:
        policy_loss = out["policy"].sum() * 0.0

    presence_loss = F.binary_cross_entropy_with_logits(
        out["presence"], presence)

    perpos_loss = F.cross_entropy(
        out["per_position"].transpose(1, 2), per_position,
        ignore_index=IGNORE)

    total = (w_policy * policy_loss + w_presence * presence_loss
             + w_perpos * perpos_loss)
    return total, {
        "policy": float(policy_loss.detach()),
        "presence": float(presence_loss.detach()),
        "perpos": float(perpos_loss.detach()),
    }


def band_cum_weights(examples, band_weights):
    """Cumulative sampling weights by word-length band, or None for uniform.

    Weighting belongs HERE, in the loader, not in label generation: every label
    row carries its own word, so the sampling mix can be changed without paying
    for another ~22-minute build_labels.py pass. Bands absent from
    ``band_weights`` keep weight 1.0.
    """
    if not band_weights:
        return None
    unknown = set(band_weights) - {name for name, _, _ in LENGTH_BANDS}
    if unknown:
        raise ValueError(f"unknown length band(s): {sorted(unknown)}")
    cum, total = [], 0.0
    for ex in examples:
        total += band_weights.get(band_of(len(ex.state.word)), 1.0)
        cum.append(total)
    return cum


def train(examples, cfg, steps=4000, batch_size=256, lr=3e-4, warmup=200,
          seed=1, device=None, out="model.pt", log_every=100,
          w_policy=1.0, w_presence=1.0, w_perpos=0.5, band_weights=None):
    if not examples:
        raise ValueError(
            "no training examples — run scripts/build_labels.py first "
            "(its output is the --labels input to this stage)")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = pick_device(device)
    print(f"device: {device} | examples: {len(examples)} | cfg: {cfg}",
          flush=True)

    model = build_model(cfg).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M",
          flush=True)

    cum_weights = band_cum_weights(examples, band_weights)
    if cum_weights is not None:
        print(f"length-band sampling weights: {band_weights}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warm = torch.optim.lr_scheduler.LinearLR(opt, 0.05, 1.0, max(warmup, 1))
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps - warmup, 1))

    model.train()
    started = time.time()
    for step in range(1, steps + 1):
        batch = (rng.choices(examples, cum_weights=cum_weights, k=batch_size)
                 if cum_weights is not None else
                 [examples[rng.randrange(len(examples))]
                  for _ in range(batch_size)])
        tokens, guessed, wrongs, policy, presence, per_position, weights = \
            make_batch(batch, device)

        out_heads = model(tokens, guessed, wrongs)
        loss, parts = compute_loss(
            out_heads, policy, presence, per_position, weights,
            w_policy, w_presence, w_perpos)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        (warm if step <= warmup else cos).step()

        if step % log_every == 0:
            print(f"step {step:>5} | loss {float(loss.detach()):.4f} | "
                  f"policy {parts['policy']:.4f} presence {parts['presence']:.4f} "
                  f"perpos {parts['perpos']:.4f} | {time.time()-started:.0f}s",
                  flush=True)

    save_checkpoint(model, cfg, out)
    print("saved ->", out, flush=True)
    return model


def golden_word_check(model, device=None, max_mean_rank=6.0):
    """Hard gate: catch a catastrophically broken training run.

    Each probe hides exactly ONE letter class in a common word, so a single
    letter is correct and the check is unambiguous. We score with the same
    aggregation the deployed policy uses, and require the correct letter to
    rank near the top on average. Random guessing averages rank ~13.

    This is a smoke test for "did training actually happen", not a quality bar.
    An earlier version asserted a specific letter on the pattern 'c_mpute_',
    which hides TWO letters ('o' and 'r') and so has no unique answer — it
    rejected a trained model. The current probes hide exactly one letter class.
    """
    from .simulate import letter_scores
    from .tokenizer import LETTERS

    # Each probe hides one letter that occurs NOWHERE else in the word. If the
    # answer also appeared revealed, the state would be unreachable (reveals are
    # closed under letter identity) and the model would be right to reject it.
    probes = [("comp_ter", "u"), ("_anana", "b"), ("elep_ant", "h"),
              ("quest_on", "i"), ("_eyboard", "k"), ("mount_in", "a"),
              ("un_verse", "i"), ("penc_l", "i")]
    for pattern, answer in probes:
        assert answer not in pattern, (
            f"invalid probe {pattern!r}: '{answer}' is also revealed, so the "
            f"state is unreachable in a real game")

    device = pick_device(device)
    model = model.to(device).eval()
    ranks = []
    for pattern, answer in probes:
        guessed = {c for c in pattern if c != "_"}
        tokens, multihot, wrongs = encode_states(
            [pattern], [guessed], [0], device=device)
        revealed = torch.tensor(
            [[c != "_" for c in pattern]], dtype=torch.bool, device=device)
        with torch.no_grad():
            scores = letter_scores(model(tokens, multihot, wrongs),
                                   revealed)[0]
        ordered = sorted(
            ((float(scores[i]), c) for i, c in enumerate(LETTERS)
             if c not in guessed), reverse=True)
        ranks.append([c for _, c in ordered].index(answer) + 1)

    mean_rank = sum(ranks) / len(ranks)
    detail = ", ".join(f"{p}->#{r}" for (p, _), r in zip(probes, ranks))
    if mean_rank > max_mean_rank:
        raise AssertionError(
            f"golden-word check FAILED: mean rank {mean_rank:.1f} of the correct "
            f"letter across {len(probes)} unambiguous probes exceeds "
            f"{max_mean_rank} (random is ~13). Training is broken — do not "
            f"benchmark this checkpoint.\n  {detail}")
    print(f"golden-word check passed: mean rank {mean_rank:.1f} "
          f"(random ~13)\n  {detail}")
    return True
