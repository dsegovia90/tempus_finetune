from tempus.candidates import find
from tempus.classifier import encode


class Words:
    """A stand-in tokenizer: one id per whitespace-separated word, numbered in order of first use."""

    cls_token_id, sep_token_id = 1, 2

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def __call__(self, text: str, add_special_tokens: bool):
        return {"input_ids": [self.vocab.setdefault(w, len(self.vocab) + 10) for w in text.split()]}


def test_each_candidate_starts_a_token_and_is_located():
    tokenizer = Words()
    message = "open 9 to 9 at room 3"
    ids, firsts = encode(tokenizer, message, find(message))
    words = {i: w for w, i in tokenizer.vocab.items()}
    assert [words.get(ids[i]) for i in firsts] == ["9", "9", "3"]
    assert ids[0] == 1 and ids[-1] == 2 and len(ids) == 2 + len(message.split())


def test_a_message_without_candidates_is_one_piece():
    ids, firsts = encode(Words(), "see you soon", [])
    assert len(ids) == 5 and firsts == []
