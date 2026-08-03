"""Vocabulary and special tokens for the synthetic battery.

A single flat vocabulary of 22 ids is shared by every task: six special
tokens, then a 16-symbol content alphabet.  Everything here is provided;
students never modify this module.

Token map
---------
====  =======  ====================================================
id    token    role
====  =======  ====================================================
0     PAD      padding to batch-max length; never contributes to loss
1     BOS      begin sequence
2     EOS      end of the target
3     SEP      separates input from output in the single-stream framing
4     DELIM    marks "copy the next token" (selective copy)
5     QUERY    marks "the next token is the lookup key" (assoc. recall)
6-21  c0-c15   the 16-symbol payload alphabet, ordered by id for Sort
====  =======  ====================================================
"""

# --- special tokens ---------------------------------------------------------
PAD = 0
BOS = 1
EOS = 2
SEP = 3
DELIM = 4
QUERY = 5

# --- content alphabet -------------------------------------------------------
# Content ids run from the first non-special id up to VOCAB_SIZE.
N_SPECIAL = 6
N_CONTENT = 16
VOCAB_SIZE = N_SPECIAL + N_CONTENT  # 22

CONTENT = range(N_SPECIAL, VOCAB_SIZE)  # 6 .. 21 inclusive

_SPECIAL_NAMES = {
    PAD: "PAD",
    BOS: "BOS",
    EOS: "EOS",
    SEP: "SEP",
    DELIM: "DELIM",
    QUERY: "QUERY",
}


def is_content(token: int) -> bool:
    """Return ``True`` if *token* is a content symbol (``c0``..``c15``)."""
    return token in CONTENT


def decode_token(token: int) -> str:
    """Return a human-readable name for a single token id."""
    if token in _SPECIAL_NAMES:
        return _SPECIAL_NAMES[token]
    if token in CONTENT:
        return f"c{token - N_SPECIAL}"
    raise ValueError(f"token id {token} outside vocabulary [0, {VOCAB_SIZE})")


def decode(tokens: list[int]) -> list[str]:
    """Map a list of token ids to their human-readable names."""
    return [decode_token(t) for t in tokens]


def encode_token(name: str) -> int:
    """Map a token name (``"BOS"``, ``"c3"``) back to its id."""
    for tid, tname in _SPECIAL_NAMES.items():
        if tname == name:
            return tid
    if name.startswith("c"):
        idx = int(name[1:])
        if 0 <= idx < N_CONTENT:
            return N_SPECIAL + idx
    raise ValueError(f"unknown token name {name!r}")


def encode(names: list[str]) -> list[int]:
    """Map a list of token names to their ids."""
    return [encode_token(n) for n in names]
