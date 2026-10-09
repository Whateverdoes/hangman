"""Stage 3: score a checkpoint on the held-out validation split.

    .venv/bin/python scripts/evaluate.py --model model.pt --seeds 1 2 3

The validation words never entered the student's training states nor the
teacher's vocabulary, so this is the honest proxy for the private evaluation
set. Always compare configurations at identical --val-size and --seed, and
never promote a change whose gain sits inside the reported CI.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hangman.dataset import load_words, split_words           # noqa: E402
from hangman.evaluate import evaluate, report, log_result     # noqa: E402
from hangman.model import load_checkpoint                     # noqa: E402
from hangman.submit import load_test_words                    # noqa: E402
from hangman.train import pick_device                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--train", default=os.path.join("data", "train.txt"))
    ap.add_argument("--val-size", type=int, default=5000)
    ap.add_argument("--n", type=int, default=2000,
                    help="words to score from the validation split")
    ap.add_argument("--seed", type=int, default=7, help="split seed")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="score several validation samples (may overlap)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--head", default="perpos",
                    choices=["perpos", "presence", "policy"])
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--ledger", default=os.path.join("experiments", "results.jsonl"))
    ap.add_argument("--test", default=None,
                    help="score the FULL list at this path instead of the "
                         "held-out split (e.g. data/test.txt). Measurement "
                         "only; never select a configuration on it.")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, cfg = load_checkpoint(args.model, device=device)

    if args.test:
        # Scores the FULL list at --test, with the same CI and length bands the
        # held-out path reports, and writes one ledger row. This exists so the
        # public-test headline in the README is reproducible from the repo; the
        # previous number was produced ad hoc and left no trace anywhere.
        # load_test_words, not load_words: the latter deduplicates, which would
        # silently change n and break the row-order the word_id depends on.
        words = load_test_words(args.test)
        result = evaluate(model, words, device=device, head=args.head,
                          batch_size=args.batch_size, progress=25000)
        report(result, f"{os.path.basename(args.model)} on {args.test}")
        log_result(result, args.ledger, model=args.model, cfg=cfg,
                   head=args.head, sample_seed=None,
                   tag=args.tag or "full-test")
        print("\nMEASUREMENT ONLY. The public test words are not the private "
              "evaluation set — never promote a configuration on this number.")
        return

    _, val = split_words(load_words(args.train), args.val_size, args.seed)

    rates = []
    for sample_seed in (args.seeds or [args.seed]):
        import random
        words = random.Random(sample_seed).sample(val, min(args.n, len(val)))
        result = evaluate(model, words, device=device, head=args.head,
                          batch_size=args.batch_size)
        report(result, f"{os.path.basename(args.model)} seed={sample_seed}")
        log_result(result, args.ledger, model=args.model, cfg=cfg,
                   head=args.head, sample_seed=sample_seed, tag=args.tag)
        rates.append(result["win_rate"])

    if len(rates) > 1:
        mean = sum(rates) / len(rates)
        spread = max(rates) - min(rates)
        print(f"\nacross {len(rates)} seeds: mean {mean:.2%} | "
              f"spread {spread:.2%} | {[f'{r:.2%}' for r in rates]}")
        print("Report this spread in the README: the audit re-runs training, "
              "and an unstated variance reads as a discrepancy.")


if __name__ == "__main__":
    main()
