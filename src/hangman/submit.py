"""Submission file generation and validation.

Schema required by the competition:

    word_id,guessed_letters_string
    0,etaonsrhld

``word_id`` is the zero-based line index of ``test.txt``. The simulation
terminates a word the instant it wins or reaches the 6th wrong guess, so the
recorded string contains only guesses that were actually played.

Uses the ``csv`` stdlib rather than pandas: one less dependency for the
organizers' re-run to satisfy, and the output is byte-identical.
"""

from __future__ import annotations

import csv

from .dataset import normalize
from .simulate import play_words

FIELDS = ["word_id", "guessed_letters_string"]


def load_test_words(path: str) -> list[str]:
    """Line-ordered test words. Order defines ``word_id`` and must be kept."""
    with open(path, encoding="utf-8") as fh:
        return [normalize(line) for line in fh if line.strip()]


def write_submission(guesses, path: str = "submission.csv") -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(FIELDS)
        for word_id, sequence in enumerate(guesses):
            writer.writerow([word_id, sequence])


def validate_submission(path: str, expected_rows: int) -> dict:
    """Fail loudly on a malformed file — a silent schema error scores zero."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))

    problems = []
    if not rows or rows[0] != FIELDS:
        problems.append(f"header must be {FIELDS}, got {rows[0] if rows else None}")

    body = rows[1:]
    if len(body) != expected_rows:
        problems.append(f"expected {expected_rows} rows, got {len(body)}")

    for i, row in enumerate(body):
        if len(row) != 2:
            problems.append(f"row {i}: expected 2 fields, got {len(row)}")
            break
        if row[0] != str(i):
            problems.append(f"row {i}: word_id is {row[0]}, expected {i}")
            break
        seq = row[1]
        if not all("a" <= c <= "z" for c in seq):
            problems.append(f"row {i}: non-lowercase-alpha guesses {seq!r}")
            break
        if not seq:
            problems.append(f"row {i}: empty guess string")
            break
        if len(set(seq)) != len(seq):
            problems.append(f"row {i}: repeated guess in {seq!r} (counts as a strike)")
            break

    if problems:
        raise AssertionError("invalid submission:\n  " + "\n  ".join(problems))
    return {"rows": len(body), "path": path, "valid": True}


def generate(model, test_path: str, out: str = "submission.csv",
             device: str = "cpu", batch_size: int = 2048,
             head: str = "perpos", progress: int = 0) -> dict:
    words = load_test_words(test_path)
    guesses, won, wrongs = play_words(
        model, words, device=device, batch_size=batch_size,
        head=head, progress=progress)
    write_submission(guesses, out)
    validate_submission(out, len(words))
    return {
        "rows": len(words),
        "self_reported_win_rate": round(sum(won) / max(len(words), 1), 4),
        "total_wrong": sum(wrongs),
        "path": out,
    }
