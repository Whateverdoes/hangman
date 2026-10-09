"""Stage 1b: teacher-free training states, generated in bulk.

    .venv/bin/python scripts/build_states.py --per-word 25

The shipped recipe is ``--w-perpos 1.0 --w-policy 0 --w-presence 0``, and the
per-position target is the true letter at each hidden position — read straight
off the word by ``encode_targets``. No oracle is consulted, so a training state
costs nothing but the RNG.

That matters because the teacher pass in ``build_labels.py`` takes ~22 minutes
and yields only 6.68 states per word, of which half the vocabulary contributes
just two. The model then draws 10.24M samples from 1.47M unique states and sees
each about seven times. This script lifts that ceiling without an oracle.

Output is byte-compatible with ``labels.tsv`` (``letter`` empty, ``trust`` 0.0)
so ``load_examples`` and ``scripts/train.py`` consume it unchanged. Use
``--include`` to copy an existing labels file through first, making the result a
strict superset — that keeps the on-policy states the teacher produced and adds
coverage around them, rather than swapping one state distribution for another.

Leakage discipline is identical to build_labels.py: states are built ONLY from
the training split, and the same hard gate aborts if an evaluation word ever
reaches it.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hangman.dataset import (assert_no_leakage, load_words,      # noqa: E402
                             split_words)
from hangman.states import letter_class_state                    # noqa: E402

HEADER = "word\tpattern\tguessed\twrongs\tletter\ttrust\n"

# The split seed build_labels.py and evaluate.py both use. States built under a
# different one are trained on words evaluate.py will hold out.
CANONICAL_SPLIT_SEED = 7


def states_for_word(word, rng, per_word, max_lives, attempts_factor=3):
    """Up to ``per_word`` DISTINCT reachable states for one word.

    A state's identity is (pattern, guessed) — not pattern alone, because the
    model conditions on the guessed set and two boards showing the same letters
    after different wrong guesses are genuinely different decisions.

    Short words are not the constraint most would expect: 'cat' still supports
    far more than 25 states at six lives, because the 23 absent letters make the
    miss-set space large. Saturation only bites when lives are scarce. The
    attempt budget bounds the work either way.
    """
    seen, out = set(), []
    for _ in range(per_word * attempts_factor):
        if len(out) >= per_word:
            break
        state = letter_class_state(word, rng, max_lives=max_lives)
        key = (state.pattern, "".join(sorted(state.guessed)))
        if key not in seen:
            seen.add(key)
            out.append(state)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.path.join("data", "train.txt"))
    ap.add_argument("--out", default=os.path.join("experiments", "states.tsv"))
    ap.add_argument("--per-word", type=int, default=25,
                    help="target distinct states per word; short words yield "
                         "fewer because their state space is genuinely smaller")
    ap.add_argument("--val-size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7,
                    help="SPLIT seed — must match build_labels.py and "
                         "evaluate.py (both default to 7). A different value "
                         "produces a different val split, which puts held-out "
                         "words into training and inflates every number.")
    ap.add_argument("--max-lives", type=int, default=6)
    ap.add_argument("--include", default=None,
                    help="copy an existing labels TSV through first, making "
                         "the output a strict superset of it")
    ap.add_argument("--progress", type=int, default=25000)
    args = ap.parse_args()

    words = load_words(args.train)
    train_words, val_words = split_words(words, args.val_size, args.seed)

    # Same hard gate as build_labels.py, for the same reason: a state built
    # from an evaluation word silently inflates every number downstream.
    assert_no_leakage(train_words, label="state vocabulary")

    # Second, independent gate. The one above only knows about test.txt, so it
    # cannot see the far likelier mistake: a --seed that disagrees with
    # evaluate.py, which quietly moves held-out words into training.
    _, canonical_val = split_words(words, args.val_size, CANONICAL_SPLIT_SEED)
    overlap = set(train_words) & set(canonical_val)
    if overlap:
        raise SystemExit(
            f"[split-check] ABORT: {len(overlap)} of these words are held out "
            f"under the canonical split seed {CANONICAL_SPLIT_SEED} "
            f"(e.g. {sorted(overlap)[:5]}). Re-run with "
            f"--seed {CANONICAL_SPLIT_SEED} --val-size {args.val_size}; "
            f"otherwise every held-out number this produces is inflated.")
    print(f"[split-check] OK — 0 of {len(canonical_val)} canonical held-out "
          f"words appear in the state vocabulary", flush=True)

    print(f"vocabulary {len(words)} | train {len(train_words)} | "
          f"val {len(val_words)} | up to {args.per_word} states/word",
          flush=True)

    rng = random.Random(args.seed)
    started = time.time()
    carried = written = 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(HEADER)

        if args.include:
            train_set = set(train_words)
            with open(args.include, encoding="utf-8") as src:
                header = next(src, "")
                if not header.startswith("word\t"):
                    raise ValueError(
                        f"{args.include}: unexpected header {header!r}")
                for line in src:
                    # Re-check every carried row: the included file was built
                    # by a separate run and its split is not ours to trust.
                    if line.split("\t", 1)[0] in train_set:
                        fh.write(line)
                        carried += 1
            print(f"carried {carried} rows from {args.include}", flush=True)

        for i, word in enumerate(train_words, 1):
            for state in states_for_word(word, rng, args.per_word,
                                         args.max_lives):
                fh.write(f"{state.word}\t{state.pattern}\t"
                         f"{''.join(sorted(state.guessed))}\t"
                         f"{state.wrongs}\t\t0.0\n")
                written += 1
            if args.progress and i % args.progress == 0:
                print(f"  {i}/{len(train_words)} words -> {written} states "
                      f"({time.time()-started:.0f}s)", flush=True)

    total = carried + written
    print(f"\nwrote {total} rows -> {args.out} in {time.time()-started:.0f}s")
    print(f"  {written} generated ({written/len(train_words):.2f} per word)"
          + (f" + {carried} carried" if carried else ""))
    print("No oracle was consulted. The per-position target is read from the "
          "word itself, so these states are as supervised as the teacher's.")


if __name__ == "__main__":
    main()
