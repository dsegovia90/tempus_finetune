import json

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from tempus.trim import remap_token_ids
from tempus.vocab import close_under_merges, script_tokens, token_bytes, trim

CORPUS = ["dinner at 7 tonight", "lunch at noon, café?", "the dinner is at seven tonight 🍕"] * 5


@pytest.fixture(scope="module")
def spec() -> dict:
    """A small byte-level BPE tokenizer with one special token, as tokenizer.json."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=320, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=["<|end|>"]
    )
    tokenizer.train_from_iterator(CORPUS, trainer)
    return json.loads(tokenizer.to_str())


def encode(spec: dict, text: str) -> list[int]:
    return Tokenizer.from_str(json.dumps(spec)).encode(text).ids


def test_token_bytes_undo_the_byte_level_alphabet():
    assert token_bytes("Ġdinner") == b" dinner"
    assert token_bytes("Ã©") == "é".encode()


def test_kept_text_tokenizes_as_before_with_ids_renumbered_in_order(spec):
    keep = set(Tokenizer.from_str(json.dumps(spec)).encode("dinner at 7 tonight").tokens)
    trimmed, old_ids = trim(spec, keep)
    assert old_ids == sorted(old_ids) and len(old_ids) < len(spec["model"]["vocab"])
    new_id = {old: new for new, old in enumerate(old_ids)}
    for text in ("dinner at 7 tonight", "dinner at 7<|end|>", "at 7 tonight"):
        assert encode(trimmed, text) == [new_id[i] for i in encode(spec, text)]


def test_other_text_falls_back_to_kept_tokens_and_decodes_the_same(spec):
    trimmed, _ = trim(spec, set())
    backend = Tokenizer.from_str(json.dumps(trimmed))
    text = "lunch at noon, café? 🍕"
    ids = backend.encode(text).ids
    assert len(ids) == len(text.encode()) and backend.decode(ids) == text  # only byte tokens left


def test_trim_keeps_bytes_and_added_tokens_and_only_merges_among_kept(spec):
    trimmed, old_ids = trim(spec, set())
    vocab = trimmed["model"]["vocab"]
    assert len(vocab) == 257 and len(old_ids) == 257  # 256 bytes and <|end|>, which is in the vocab
    assert trimmed["added_tokens"][0]["id"] == vocab["<|end|>"]
    assert all(a in vocab and b in vocab and a + b in vocab for a, b in trimmed["model"]["merges"])


def test_closure_adds_the_parts_of_every_merge_making_a_kept_token():
    merges = [["a", "b"], ["b", "c"], ["ab", "c"], ["a", "bc"]]
    assert close_under_merges({"abc"}, merges) == {"abc", "ab", "bc", "a", "b", "c"}


def test_script_tokens_allow_partial_characters_only_from_allowed_ones(spec):
    vocab = {"a": 0, "Ġa": 1, "Ã": 2, "Ã©": 3, "ðŁ": 4, "Ġæ": 5}  # a, " a", é head, é, emoji head, CJK head
    assert script_tokens(vocab, [chr(c) for c in range(128)]) == {"a", "Ġa"}
    assert script_tokens(vocab, [chr(c) for c in range(128)] + ["é", "🍕"]) == {"a", "Ġa", "Ã", "Ã©", "ðŁ"}


def test_remap_token_ids_renumbers_ids_and_lists_at_any_depth():
    config = {"eos_token_id": 9, "bos_token_id": None, "text_config": {"pad_token_id": 8}, "vocab_size": 9}
    assert remap_token_ids(config, {9: 1, 8: 0}) == {
        "eos_token_id": 1, "bos_token_id": None, "text_config": {"pad_token_id": 0}, "vocab_size": 9
    }
    assert remap_token_ids({"eos_token_id": [9, 8]}, {9: 1, 8: 0}) == {"eos_token_id": [1, 0]}
