"""Stage 2: train the student on the distilled labels.

    .venv/bin/python scripts/train.py --fast    # ~3 min   smoke check
    .venv/bin/python scripts/train.py --full    # ~1-3h on a T4

Ends with a hard golden-word gate. If that raises, the checkpoint is broken and
must not be benchmarked — a silent device-specific training failure has bitten
this codebase before and reported healthy-looking loss curves throughout.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hangman.dataset import assert_no_leakage                  # noqa: E402
from hangman.distill import load_examples                     # noqa: E402
from hangman.train import train, golden_word_check            # noqa: E402


def parse_band_weights(spec: str) -> dict[str, float]:
    """``'6-8:2,1-5:1.5'`` -> ``{'6-8': 2.0, '1-5': 1.5}``. Empty -> uniform.

    Band names are validated against LENGTH_BANDS inside band_cum_weights, so a
    typo fails loudly at startup rather than silently training uniform.
    """
    if not spec.strip():
        return {}
    weights = {}
    for part in spec.split(","):
        band, _, value = part.partition(":")
        if not value:
            raise ValueError(f"expected 'band:weight', got {part!r}")
        weights[band.strip()] = float(value)
    return weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join("experiments", "labels.tsv"))
    ap.add_argument("--out", default="model.pt")
    ap.add_argument("--arch", choices=["transformer", "lstm"],
                    default="transformer")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--w-policy", type=float, default=1.0)
    ap.add_argument("--w-presence", type=float, default=1.0)
    ap.add_argument("--w-perpos", type=float, default=0.5)
    ap.add_argument("--no-cond-guessed", action="store_true",
                    help="ablation: remove guessed-letter conditioning")
    ap.add_argument("--no-cond-wrongs", action="store_true",
                    help="ablation: remove lives/wrong-guess conditioning")
    ap.add_argument("--band-weights", default="",
                    help="oversample word-length bands during training, e.g. "
                         "'6-8:2' or '1-5:1.5,6-8:2'. Unlisted bands stay at "
                         "1.0. Sampling-only; needs no relabelling run.")
    ap.add_argument("--skip-golden", action="store_true",
                    help="only for deliberately undertrained smoke runs")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    if args.fast:
        args.steps, args.dim, args.layers, args.heads, args.ff = 600, 128, 3, 4, 512
    if args.full:
        args.steps = 6000

    cfg = {"arch": args.arch, "dim": args.dim, "layers": args.layers,
           "dropout": args.dropout,
           "cond_guessed": not args.no_cond_guessed,
           "cond_wrongs": not args.no_cond_wrongs}
    if args.arch == "transformer":
        cfg.update(heads=args.heads, ff=args.ff)

    examples = load_examples(args.labels, limit=args.limit)
    print(f"loaded {len(examples)} examples from {args.labels}", flush=True)

    # Second gate, independent of the first: whatever produced these labels,
    # no example may be built from an evaluation word.
    assert_no_leakage({e.state.word for e in examples},
                      label="distilled label set")

    model = train(examples, cfg, steps=args.steps, batch_size=args.batch_size,
                  lr=args.lr, warmup=args.warmup, seed=args.seed,
                  device=args.device, out=args.out,
                  w_policy=args.w_policy, w_presence=args.w_presence,
                  w_perpos=args.w_perpos,
                  band_weights=parse_band_weights(args.band_weights))

    if not args.skip_golden:
        golden_word_check(model)


if __name__ == "__main__":
    main()
