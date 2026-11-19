"""Trim a byte-level BPE tokenizer to a subset of its tokens without changing kept text.

BPE builds every token from the 256 byte tokens by merges, applying at each step the
lowest-ranked merge between neighbours. Keep a set of tokens that holds both parts of every
merge producing a kept token, and keep only the merges among kept tokens, in their original
order. Then for any text whose original tokens are all kept, every merge the original applies
is still there and still the lowest-ranked one available, so the trimmed tokenizer applies the
same merges and returns the same tokens. Other text falls back to shorter tokens, at worst
single bytes.

Works on the parsed `tokenizer.json` of a Hugging Face byte-level BPE tokenizer.
"""

from collections.abc import Iterable


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's map from each byte to the printable character that stands for it in tokens."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    extra = (b for b in range(256) if b not in printable)
    return {b: chr(b) for b in printable} | {b: chr(256 + i) for i, b in enumerate(extra)}


BYTE_CHAR = _bytes_to_unicode()
CHAR_BYTE = {char: byte for byte, char in BYTE_CHAR.items()}


def token_bytes(token: str) -> bytes:
    """The raw bytes a vocabulary entry stands for."""
    return bytes(CHAR_BYTE[char] for char in token)


def byte_token(byte: int) -> str:
    return BYTE_CHAR[byte]


def script_tokens(vocab: Iterable[str], chars: Iterable[str]) -> set[str]:
    """Tokens whose bytes can occur inside text written only with `chars`.

    A token may start or end in the middle of a character's UTF-8 encoding, so its ends may
    be a tail or a head of an allowed character, with whole allowed characters between.
    Every token that tokenizing such text can produce is in this set.
    """
    encoded = [char.encode() for char in chars]
    whole = set(encoded)
    heads = {e[:i] for e in encoded for i in range(1, len(e))}
    tails = {e[i:] for e in encoded for i in range(1, len(e))}

    def fits(data: bytes) -> bool:
        if data in whole or data in heads or data in tails:
            return True
        for start in range(min(4, len(data))):  # a partial character first, then whole ones
            if start and data[:start] not in tails:
                continue
            if _whole_then_head(data[start:], whole, heads):
                return True
        return False

    return {token for token in vocab if fits(token_bytes(token))}


def _whole_then_head(data: bytes, whole: set[bytes], heads: set[bytes]) -> bool:
    """True when `data` is whole allowed characters, then optionally the head of one."""
    position = 0
    while position < len(data):
        for size in (1, 2, 3, 4):
            if data[position : position + size] in whole:
                position += size
                break
        else:
            return data[position:] in heads
    return True


def close_under_merges(keep: set[str], merges: list[list[str]]) -> set[str]:
    """Add, recursively, both parts of every merge that produces a kept token."""
    makers: dict[str, list[list[str]]] = {}
    for left, right in merges:
        makers.setdefault(left + right, []).append([left, right])
    closed, pending = set(keep), list(keep)
    while pending:
        for pair in makers.get(pending.pop(), ()):
            for part in pair:
                if part not in closed:
                    closed.add(part)
                    pending.append(part)
    return closed


def trim(spec: dict, keep: set[str]) -> tuple[dict, list[int]]:
    """A trimmed copy of a parsed tokenizer.json, and the old id of each new id.

    `keep` is closed under merges, and the 256 byte tokens and every added token (special or
    not) are kept, so any text still has a tokenization. Kept tokens are renumbered from 0 in
    their old order.
    """
    model = spec["model"]
    if model["type"] != "BPE" or not all(isinstance(m, list) for m in model["merges"]):
        raise ValueError("expected a BPE tokenizer.json with merges as pairs")
    added = {t["content"] for t in spec["added_tokens"]}
    keep = close_under_merges(keep | added | {byte_token(b) for b in range(256)}, model["merges"])
    tokens = {i: t for t, i in model["vocab"].items()} | {t["id"]: t["content"] for t in spec["added_tokens"]}
    missing = keep - set(tokens.values())
    if missing:
        raise ValueError(f"{len(missing)} kept tokens are not in the vocabulary, e.g. {sorted(missing)[:5]}")

    old_ids = sorted(i for i, t in tokens.items() if t in keep)
    new_id = {tokens[old]: new for new, old in enumerate(old_ids)}
    vocab = {t: i for t, i in new_id.items() if t in model["vocab"]}
    merges = [m for m in model["merges"] if m[0] in vocab and m[1] in vocab and m[0] + m[1] in vocab]
    new_added = [{**t, "id": new_id[t["content"]]} for t in spec["added_tokens"]]
    return {**spec, "added_tokens": new_added, "model": {**model, "vocab": vocab, "merges": merges}}, old_ids
