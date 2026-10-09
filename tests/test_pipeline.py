"""Correctness gates. Run with: .venv/bin/python tests/test_pipeline.py

These are deliberately about INVARIANTS rather than numbers: game-rule
compliance, reachable training states, submission schema, and the pure-neural
inference boundary. Numbers move between runs; these must never break.
"""

import ast
import pathlib
import random
import sys

import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hangman.dataset import (assert_no_leakage, band_of, normalize,  # noqa: E402
                             split_words)
from hangman.simulate import play_group, play_words               # noqa: E402
from hangman.states import letter_class_state, simulate_game      # noqa: E402
from hangman.submit import (validate_submission, write_submission)  # noqa: E402
from hangman.teacher.corpus import TeacherCorpus                  # noqa: E402
from hangman.teacher.oracle import best_letter, trust             # noqa: E402
from hangman.train import band_cum_weights                        # noqa: E402


class _Perfect(nn.Module):
    """Always names a present, unguessed letter — must win every game."""

    def __init__(self, words):
        super().__init__()
        self.words = words

    def forward(self, tokens, guessed, wrongs):
        out = torch.full((len(self.words), 26), -9.0)
        for i, w in enumerate(self.words):
            for c in w:
                if not guessed[i, ord(c) - 97]:
                    out[i, ord(c) - 97] = 1.0
        return {"presence": out}


# ---------------------------------------------------------------- data

def test_normalize_strips_non_letters():
    assert normalize("  Coca-Cola!\n") == "cocacola"
    assert normalize("A1B2") == "ab"


def test_split_is_disjoint_and_deterministic():
    words = [f"word{i}" for i in range(1000)]
    a1, b1 = split_words(words, val_size=100, seed=7)
    a2, b2 = split_words(words, val_size=100, seed=7)
    assert (a1, b1) == (a2, b2)
    assert not set(a1) & set(b1)
    assert len(b1) == 100


def test_band_weights_shift_the_sampled_length_mix():
    """Oversampling must actually change what the loader draws.

    A weight that silently did nothing would look exactly like a null result in
    the ablation table, which is the expensive kind of bug: it does not fail, it
    just quietly refutes a true hypothesis.
    """
    from types import SimpleNamespace

    def ex(word):
        return SimpleNamespace(state=SimpleNamespace(word=word))

    # Equal counts per band, so any shift is caused by the weights alone.
    examples = ([ex("abc")] * 200 +            # 1-5
                [ex("abcdefg")] * 200 +        # 6-8
                [ex("abcdefghij")] * 200)      # 9-12

    assert band_cum_weights(examples, None) is None
    assert band_cum_weights(examples, {}) is None

    cum = band_cum_weights(examples, {"6-8": 3.0})
    assert len(cum) == len(examples)
    assert cum[-1] == 200 * 1.0 + 200 * 3.0 + 200 * 1.0

    rng = random.Random(0)
    drawn = rng.choices(examples, cum_weights=cum, k=6000)
    share = sum(band_of(len(e.state.word)) == "6-8" for e in drawn) / len(drawn)
    # Expected 3/5 = 0.60 with weight 3 against two bands at 1.
    assert 0.55 < share < 0.65, share

    uniform = random.Random(0).choices(examples, k=6000)
    base = sum(band_of(len(e.state.word)) == "6-8" for e in uniform) / len(uniform)
    assert share > base + 0.15, (share, base)

    try:
        band_cum_weights(examples, {"7-9": 2.0})
    except ValueError as exc:
        assert "7-9" in str(exc)
    else:
        raise AssertionError("a misspelled band name must fail loudly")


def test_generated_states_are_distinct_and_reachable():
    """Bulk teacher-free states must obey the same rules as teacher states.

    These are generated without an oracle, so nothing downstream re-checks
    them; if they were unreachable the model would train on boards the game can
    never present, exactly as independent position masking would produce.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_states import states_for_word                      # noqa: E402

    rng = random.Random(0)
    for word in ["cat", "abidance", "unconscionable"]:
        states = states_for_word(word, rng, per_word=25, max_lives=6)
        keys = [(s.pattern, "".join(sorted(s.guessed))) for s in states]
        assert len(keys) == len(set(keys)), f"{word}: duplicate states"

        for s in states:
            shown = {c for c, p in zip(s.word, s.pattern) if p != "_"}
            # Reveals are closed under letter identity.
            for ch, mark in zip(s.word, s.pattern):
                assert (ch in shown) == (mark != "_"), s
            # Every revealed letter was guessed, and wrongs match the misses.
            assert shown <= set(s.guessed), s
            assert s.wrongs == len(set(s.guessed) - set(s.word)), s
            # A state the game would already have ended is not a state.
            assert s.wrongs < 6, s

    # Never over-delivers.
    assert len(states_for_word("abidance", random.Random(1),
                               per_word=4, max_lives=6)) == 4

    # Saturation: with one life the miss set is always empty, so 'cat' has only
    # its 2^3 - 1 = 7 reachable reveal subsets and cannot reach 25 however many
    # times it is sampled. Short words alone do NOT saturate at six lives --
    # the 23 absent letters make the guessed-set space large.
    short = states_for_word("cat", random.Random(1), per_word=25, max_lives=1)
    assert len(short) == 7, len(short)


def test_leak_guard_fires_on_contamination(tmp_path):
    """The guard must abort training, not warn. The predecessor project shipped
    numbers inflated up to 3x by exactly this contamination."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("alpha\nbeta\ngamma\n")

    assert_no_leakage({"delta", "epsilon"}, str(test_file)) == 3

    try:
        assert_no_leakage({"delta", "beta"}, str(test_file))
    except SystemExit as exc:
        assert "LEAKAGE" in str(exc) and "beta" in str(exc)
    else:
        raise AssertionError("guard did not fire on a contaminated vocabulary")


def test_leak_guard_reports_when_it_checked_nothing(tmp_path):
    """A missing eval file must return 0, so a silent pass is distinguishable
    from a real check in the logs."""
    assert assert_no_leakage({"alpha"}, str(tmp_path / "absent.txt")) == 0


# --------------------------------------------------------------- states

def test_states_are_reachable():
    """Reveals must be closed under letter identity, and hidden positions must
    never hold an already-guessed letter. Independent position masking — the
    naive MLM setup — violates both."""
    rng = random.Random(0)
    for word in ["banana", "discoverers", "aaa", "mississippi"]:
        for _ in range(20):
            s = letter_class_state(word, rng)
            for i, j in [(i, j) for i in range(len(word)) for j in range(len(word))]:
                if word[i] == word[j]:
                    assert (s.pattern[i] == "_") == (s.pattern[j] == "_")
            for pos, ch in enumerate(s.pattern):
                if ch == "_":
                    assert word[pos] not in s.guessed
            assert s.wrongs <= 6


# -------------------------------------------------------------- teacher

def test_leave_one_out_removes_the_answer():
    corpus = TeacherCorpus(["knish", "krish", "banana"])
    _, idx = corpus.candidates("knish", set("knish"))
    assert len(idx) == 1
    _, idx = corpus.candidates("knish", set("knish"), exclude="knish")
    assert len(idx) == 0, "LOO must deny the teacher the answer"


def test_trust_is_monotone_and_bounded():
    assert trust(0) == 0.0
    assert 0.0 < trust(10) < trust(1000) < 1.0


def test_teacher_never_repeats_a_guess():
    corpus = TeacherCorpus(["banana", "bandana", "cabana"])
    b, idx = corpus.candidates("______", set("xy"))
    letter = best_letter(b, idx, set("xy"))
    assert letter is not None and letter not in set("xy")


# ------------------------------------------------------------ game rules

def test_perfect_player_always_wins_with_no_wrongs():
    words = ["banana", "shrink"]
    seqs, won, wrongs = play_group(_Perfect(words), words, head="presence")
    assert won.all() and int(wrongs.sum()) == 0
    assert seqs[0] == "abn"


def test_games_respect_life_limit_and_never_repeat():
    class Fixed(nn.Module):
        def forward(self, tokens, guessed, wrongs):
            return {"presence": torch.zeros(tokens.size(0), 26)}

    words = ["discoverers", "glossoplasty", "knish"]
    seqs, won, wrongs = play_words(Fixed(), words, head="presence")
    for seq, wrong in zip(seqs, wrongs):
        assert len(set(seq)) == len(seq), "a repeated guess costs a life"
        assert wrong <= 6


def test_play_group_rejects_ragged_input():
    try:
        play_group(_Perfect(["ab"]), ["ab", "abc"], head="presence")
    except ValueError as exc:
        assert "equal-length" in str(exc)
    else:
        raise AssertionError("expected ValueError on ragged batch")


def test_default_head_uses_per_position_aggregation():
    """The measured-best head must be the default the policy actually plays."""
    import torch
    from hangman.simulate import DEFAULT_HEAD, letter_scores

    assert DEFAULT_HEAD == "perpos"
    # Two hidden slots that both strongly favour 'a' must beat a single slot
    # favouring 'z' — i.e. the aggregate accumulates evidence across positions.
    per_position = torch.full((1, 3, 26), -9.0)
    per_position[0, 0, 0] = per_position[0, 1, 0] = 2.0     # 'a' at two blanks
    per_position[0, 2, 25] = 3.0                            # 'z' at one blank
    revealed = torch.tensor([[False, False, False]])
    scores = letter_scores({"per_position": per_position}, revealed)[0]
    assert scores[0] > scores[25], "aggregation must combine across positions"


# ----------------------------------------------------------- submission

def test_reference_policy_conditions_on_wrongs_and_uses_perpos():
    from hangman.policy import NeuralPolicy

    class Observe(nn.Module):
        def forward(self, tokens, guessed, wrongs):
            assert wrongs.tolist() == [2]
            assert int(guessed.sum()) == 3
            out = torch.zeros(1, tokens.size(1), 26)
            out[:, :, 1] = 8.0  # 'b'; presence would incorrectly choose 'a'.
            return {"per_position": out, "presence": torch.zeros(1, 26)}

    policy = NeuralPolicy(Observe())
    assert policy.guess("a__a", set("axy")) == "b"


def test_reference_policy_stops_and_matches_batched_ties():
    from hangman.policy import NeuralPolicy

    class Fixed(nn.Module):
        def forward(self, tokens, guessed, wrongs):
            return {"presence": torch.zeros(tokens.size(0), 26)}

    policy = NeuralPolicy(Fixed(), head="presence")
    assert policy.guess("___", set()) == "a"
    assert policy.guess("___", {"a"}) == "b"
    assert policy.guess("cat", set("cat")) is None
    assert policy.guess("___", set("uvwxyz")) is None


def test_reference_policy_matches_batched_transformer_games():
    from hangman.model import CharTransformer
    from hangman.policy import NeuralPolicy

    # Compare complete trajectories through both public inference APIs.
    with torch.random.fork_rng():
        torch.manual_seed(5)
        model = CharTransformer(dim=16, heads=2, layers=1, ff=32).eval()
    words = ["banana", "cat", "mississippi"]
    batched, _, _ = play_words(model, words)
    policy = NeuralPolicy(model)
    for word, expected in zip(words, batched):
        guessed, played = set(), []
        while True:
            pattern = "".join(c if c in guessed else "_" for c in word)
            letter = policy.guess(pattern, guessed)
            if letter is None:
                break
            assert letter not in guessed
            guessed.add(letter)
            played.append(letter)
        assert "".join(played) == expected, (word, played, expected)


def test_submission_schema_roundtrip(tmp_path):
    path = tmp_path / "submission.csv"
    write_submission(["abc", "de"], str(path))
    assert validate_submission(str(path), 2)["valid"]


def test_submission_rejects_repeats_and_bad_counts(tmp_path):
    path = tmp_path / "bad.csv"
    write_submission(["aab"], str(path))
    for args, needle in [((str(path), 1), "repeated"), ((str(path), 5), "rows")]:
        try:
            validate_submission(*args)
        except AssertionError as exc:
            assert needle in str(exc)
        else:
            raise AssertionError(f"expected failure containing {needle!r}")


# ---------------------------------------------------- architecture boundary

def test_inference_modules_never_import_the_teacher():
    """The submitted player is the network alone. The oracle is offline-only
    training infrastructure, and this makes that claim machine-checkable."""
    for name in ["policy.py", "simulate.py", "submit.py", "model.py"]:
        source = (ROOT / "src" / "hangman" / name).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "teacher" not in node.module, f"{name} imports teacher"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "teacher" not in alias.name, f"{name} imports teacher"


# --------------------------------------------------------------- runner

def _main() -> int:
    """Zero-dependency runner: ``.venv/bin/python tests/test_pipeline.py``.

    Works under pytest too, but does not require it — the organizers' re-run
    should not need to install anything beyond requirements.txt.
    """
    import inspect
    import tempfile
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, fn in tests:
            try:
                if "tmp_path" in inspect.signature(fn).parameters:
                    fn(pathlib.Path(tmp))
                else:
                    fn()
                print(f"  PASS  {name}")
            except Exception:
                failed += 1
                print(f"  FAIL  {name}")
                traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
