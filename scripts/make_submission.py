"""Stage 4: play every test word and write submission.csv.

    .venv/bin/python scripts/make_submission.py --model model.pt \
        --test data/test.txt

Batched lockstep play: one forward pass per TURN across the whole batch, so
250k games cost roughly max-guesses passes rather than 2.5M.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hangman.model import load_checkpoint                     # noqa: E402
from hangman.submit import generate                           # noqa: E402
from hangman.train import pick_device                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--test", default=os.path.join("data", "test.txt"))
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--head", default="perpos",
                    choices=["perpos", "presence", "policy"])
    args = ap.parse_args()

    device = pick_device(args.device)
    model, _ = load_checkpoint(args.model, device=device)

    started = time.time()
    info = generate(model, args.test, out=args.out, device=device,
                    batch_size=args.batch_size, head=args.head,
                    progress=25000)
    print(f"\nwrote {info['rows']} rows -> {info['path']} "
          f"in {time.time()-started:.0f}s")
    print(f"local self-play win rate: {info['self_reported_win_rate']:.2%} "
          f"| total wrong {info['total_wrong']} (tie-break metric)")
    print("NOTE: this is measured on the public test words and is NOT an "
          "estimate of private-set performance. Trust scripts/evaluate.py.")


if __name__ == "__main__":
    main()
