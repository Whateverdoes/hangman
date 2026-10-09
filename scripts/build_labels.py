"""Stage 1: generate leave-one-out teacher labels.

    .venv/bin/python scripts/build_labels.py --fast     # ~1 min,  smoke check
    .venv/bin/python scripts/build_labels.py --full     # ~15 min on 8 cores

Output is a plain TSV so the labels are inspectable and diffable rather than an
opaque blob — an auditor can read them without running our code.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hangman.dataset import (assert_no_leakage, load_words,   # noqa: E402
                             split_words)
from hangman.distill import TeacherPolicy, generate          # noqa: E402
from hangman.teacher.corpus import TeacherCorpus             # noqa: E402

_POLICY: TeacherPolicy | None = None
_ARGS = None


def _init(train_words, tau, risk_floor, max_lives):
    global _POLICY
    _POLICY = TeacherPolicy(TeacherCorpus(train_words), tau=tau,
                            max_lives=max_lives, risk_floor=risk_floor)


def _work(job):
    chunk, seed, sim_fraction, per_word_random = job
    rng = random.Random(seed)
    rows = []
    for ex in generate(chunk, _POLICY, rng, sim_fraction=sim_fraction,
                       per_word_random=per_word_random):
        rows.append((ex.state.word, ex.state.pattern,
                     "".join(sorted(ex.state.guessed)), ex.state.wrongs,
                     ex.letter or "", f"{ex.trust:.4f}"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.path.join("data", "train.txt"))
    ap.add_argument("--out", default=os.path.join("experiments", "labels.tsv"))
    ap.add_argument("--val-size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tau", type=float, default=50.0)
    ap.add_argument("--risk-floor", type=float, default=0.0)
    ap.add_argument("--max-lives", type=int, default=6)
    ap.add_argument("--sim-fraction", type=float, default=0.5)
    ap.add_argument("--per-word-random", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the number of training words (0 = all)")
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 2, 1))
    ap.add_argument("--fast", action="store_true",
                    help="8k words — reproduction smoke test")
    ap.add_argument("--full", action="store_true",
                    help="whole training split — the real run")
    args = ap.parse_args()

    if args.fast:
        args.limit = 8000
    if args.full:
        args.limit = 0

    words = load_words(args.train)
    train_words, val_words = split_words(words, args.val_size, args.seed)
    targets = train_words[:args.limit] if args.limit else train_words
    # Hard gate: the teacher's vocabulary must never contain an eval word.
    assert_no_leakage(train_words, label="teacher vocabulary")

    print(f"vocabulary {len(words)} | train {len(train_words)} | "
          f"val {len(val_words)} | labelling {len(targets)} words "
          f"on {args.workers} workers", flush=True)

    # The teacher's vocabulary is the TRAIN split only. Validation words are
    # invisible to it, exactly as the private test set will be.
    size = max(len(targets) // (args.workers * 8), 1)
    jobs = [(targets[i:i + size], args.seed + i, args.sim_fraction,
             args.per_word_random)
            for i in range(0, len(targets), size)]

    started = time.time()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    written = labelled = 0
    with open(args.out, "w") as fh:
        fh.write("word\tpattern\tguessed\twrongs\tletter\ttrust\n")
        with Pool(args.workers, initializer=_init,
                  initargs=(train_words, args.tau, args.risk_floor,
                            args.max_lives)) as pool:
            for done, rows in enumerate(pool.imap_unordered(_work, jobs), 1):
                for row in rows:
                    fh.write("\t".join(str(x) for x in row) + "\n")
                    written += 1
                    labelled += bool(row[4])
                if done % 10 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)} chunks | {written} examples "
                          f"| {time.time()-started:.0f}s", flush=True)

    print(f"\n{written} examples -> {args.out} in {time.time()-started:.0f}s")
    print(f"teacher produced a label on {labelled}/{written} "
          f"({labelled/max(written,1):.1%}); the rest are carried by the "
          f"set-level head via trust weighting")


if __name__ == "__main__":
    main()
