"""Token vocabulary and index conventions.

TWO DISJOINT INDEX SPACES — do not mix them (historical bug: this exact
mix-up, on Apple Silicon MPS, trains garbage silently):

  input tokens:   0=PAD, 1=MASK, 2='a' .. 27='z', 28=SEP (any revealed
                  non-letter character class; train/val normalization may
                  reveal e.g. spaces)
  label targets:  0..25 for 'a'..'z', plus -100 for "ignore" in loss masks
"""

LETTERS = "abcdefghijklmnopqrstuvwxyz"
PAD, MASK, SEP = 0, 1, 28
VOCAB_SIZE = 29
N_LETTERS = 26


def letter_to_token(ch):
    return 2 + ord(ch) - 97


def token_to_letter(t):
    return chr(97 + t - 2)


def letter_index(ch):
    """0..25 offset used in multi-hot conditioning / target vectors."""
    return ord(ch) - 97
