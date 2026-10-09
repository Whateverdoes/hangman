"""Evaluation: win rate with confidence intervals, plus the tie-break metric.

Two habits carried over deliberately from the predecessor project, both of
which were learned the hard way:

* **Always report a CI.** At n=400 the 95% interval is about +/-5pp, so a 2-3pp
  "improvement" is noise. A previous round of solver tuning produced three
  changes that all looked like small wins and were all statistically null.
* **Never evaluate on words the engine has seen.** The validation slice is held
  out of both the student's states and the teacher's vocabulary.

The competition ranks on win rate first and total wrong guesses second, so both
are reported. Note a loss always contributes exactly ``max_lives`` wrongs, which
means the tie-break is decided entirely by efficiency on games already won.
"""

from __future__ import annotations

import json
import math
import os
import time

from .dataset import LENGTH_BANDS
from .simulate import play_words


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — well behaved near 0 and 1, unlike normal-approx."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def evaluate(model, words, device="cpu", batch_size=2048, max_lives=6,
             head="perpos", progress=0) -> dict:
    started = time.time()
    guesses, won, wrongs = play_words(
        model, words, device=device, batch_size=batch_size,
        max_lives=max_lives, head=head, progress=progress)

    n = len(words)
    wins = sum(won)
    low, high = wilson_interval(wins, n)

    bands = {}
    for name, lo, hi in LENGTH_BANDS:
        idx = [i for i, w in enumerate(words) if lo <= len(w) <= hi]
        if idx:
            bands[name] = {
                "n": len(idx),
                "win_rate": round(sum(won[i] for i in idx) / len(idx), 4),
            }

    return {
        "n": n,
        "wins": wins,
        "win_rate": round(wins / n, 4),
        "ci95": [round(low, 4), round(high, 4)],
        "total_wrong": sum(wrongs),
        "mean_wrong": round(sum(wrongs) / n, 3),
        "mean_guesses": round(sum(len(g) for g in guesses) / n, 2),
        "by_length": bands,
        "seconds": round(time.time() - started, 1),
    }


def report(result: dict, label: str = "") -> None:
    lo, hi = result["ci95"]
    print(f"\n=== {label or 'eval'} (n={result['n']}, {result['seconds']}s) ===")
    print(f"  win rate     {result['win_rate']:.2%}  "
          f"[95% CI {lo:.2%} - {hi:.2%}]")
    print(f"  total wrong  {result['total_wrong']}  "
          f"(mean {result['mean_wrong']}/word)  <- tie-break metric")
    print(f"  mean guesses {result['mean_guesses']}")
    for name, band in result["by_length"].items():
        print(f"    len {name:>6}: {band['win_rate']:6.2%}  (n={band['n']})")


def log_result(result: dict, path: str, **meta) -> None:
    """Append one row to the experiment ledger (JSON Lines, append-only)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    row = {**meta, **result, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
