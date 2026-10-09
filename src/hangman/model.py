"""Conditioned character Transformer and bidirectional LSTM baseline.

The encoder receives the revealed board, complete guessed-letter set, and
wrong-guess count. Three heads predict per-position letters, letter presence,
and an optional offline teacher's action. The documented recipe trains and
plays with the per-position head; the other heads support experiments.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from .tokenizer import PAD, VOCAB_SIZE, N_LETTERS

MAX_LEN = 40

# norm_first=True is deliberate (pre-LN trains more stably at this depth);
# torch emits a nested-tensor notice about it that only clutters logs.
warnings.filterwarnings("ignore", message=".*enable_nested_tensor is True.*")


class _Heads(nn.Module):
    """Policy / presence / per-position heads shared by both encoders."""

    def __init__(self, dim: int):
        super().__init__()
        self.policy = nn.Linear(dim, N_LETTERS)
        self.presence = nn.Linear(dim, N_LETTERS)
        self.per_position = nn.Linear(dim, N_LETTERS)

    def forward(self, hidden, keep):
        # Masked mean-pool: padding must not dilute the word representation.
        weights = keep.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1.0)
        return {
            "policy": self.policy(pooled),
            "presence": self.presence(pooled),
            "per_position": self.per_position(hidden),
        }


class _Conditioning(nn.Module):
    """Shared state conditioning: guessed-letter set and wrong-guess count.

    Either term can be disabled for ablation. Disabling is done by zeroing the
    contribution rather than dropping the module, so checkpoints stay
    structurally identical and remain loadable either way.
    """

    def __init__(self, dim: int, max_lives: int, cond_guessed: bool = True,
                 cond_wrongs: bool = True):
        super().__init__()
        self.guessed = nn.Linear(N_LETTERS, dim)
        self.wrongs = nn.Embedding(max_lives + 1, dim)
        self.cond_guessed = cond_guessed
        self.cond_wrongs = cond_wrongs

    def forward(self, guessed, wrongs):
        total = 0.0
        if self.cond_guessed:
            total = total + self.guessed(guessed)
        if self.cond_wrongs:
            total = total + self.wrongs(wrongs)
        if isinstance(total, float):
            return torch.zeros(guessed.size(0), 1, self.guessed.out_features,
                               device=guessed.device, dtype=self.guessed.weight.dtype)
        return total.unsqueeze(1)


class CharTransformer(nn.Module):
    def __init__(self, dim=256, heads=8, layers=6, ff=1024, dropout=0.1,
                 max_lives=6, cond_guessed=True, cond_wrongs=True, **_):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, dim, padding_idx=PAD)
        self.pos = nn.Embedding(MAX_LEN, dim)
        self.cond = _Conditioning(dim, max_lives, cond_guessed, cond_wrongs)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ff, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.heads = _Heads(dim)

    def forward(self, tokens, guessed, wrongs):
        n = tokens.size(1)
        if n > MAX_LEN:
            raise ValueError(f"sequence length {n} exceeds MAX_LEN={MAX_LEN}")
        pad = tokens == PAD
        h = self.emb(tokens) + self.pos.weight[:n].unsqueeze(0)
        h = h + self.cond(guessed, wrongs)
        h = self.encoder(h, src_key_padding_mask=pad)
        return self.heads(self.norm(h), ~pad)


class CharLSTM(nn.Module):
    """Bidirectional baseline, same interface — used for ablation/ensembling."""

    def __init__(self, dim=256, hidden=256, layers=2, dropout=0.1,
                 max_lives=6, cond_guessed=True, cond_wrongs=True, **_):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, dim, padding_idx=PAD)
        self.cond = _Conditioning(dim, max_lives, cond_guessed, cond_wrongs)
        self.lstm = nn.LSTM(dim, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.heads = _Heads(hidden * 2)

    def forward(self, tokens, guessed, wrongs):
        pad = tokens == PAD
        h = self.emb(tokens) + self.cond(guessed, wrongs)
        h, _ = self.lstm(h)
        return self.heads(h, ~pad)


ARCHITECTURES = {"transformer": CharTransformer, "lstm": CharLSTM}


def build_model(cfg: dict) -> nn.Module:
    """Construct from a config dict; ``cfg['arch']`` selects the family.

    Checkpoints store their own cfg, so any checkpoint loads without the caller
    needing to know which architecture produced it.
    """
    params = {k: v for k, v in cfg.items() if k != "arch"}
    return ARCHITECTURES[cfg.get("arch", "transformer")](**params)


def save_checkpoint(model, cfg, path):
    torch.save({"state": model.state_dict(), "cfg": cfg}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["cfg"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model.to(device), ckpt["cfg"]
