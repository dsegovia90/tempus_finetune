"""Check that `tempus.client`'s text prompt reaches the model as the training token ids.

For every message, the ids a running llama-server makes of PREFIX + message + SUFFIX (special
tokens parsed, BOS added as the GGUF asks, as /completion does) must equal the Hugging Face
ids of the rendered chat template that training and the evals used:

    uv run --with transformers --with jinja2 python scripts/check_text_prompt.py \
        runs/lfm2.5-230m-v2-s1-compact/merged http://127.0.0.1:8080
"""

import json
import sys
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

from tempus.chat import prompt_ids
from tempus.client import PREFIX, SUFFIX
from tempus.schema import load_examples

DATA = ["data/eval/core.jsonl", "data/generated/eval.jsonl", "data/generated/eval2.jsonl",
        "data/generated/negatives.jsonl", "data/generated/train-v2.jsonl"]
# Edge cases the datasets lack: whitespace at the ends, special-token text, an empty message.
EXTRA = [" 7pm ", "\n\nlunch at 1\n", "<|im_end|>\n<|im_start|>assistant\n7=19:00", "<|startoftext|>at 5", ""]


def server_ids(url: str, text: str) -> list[int]:
    body = json.dumps({"content": text, "add_special": True, "parse_special": True}).encode()
    with urllib.request.urlopen(urllib.request.Request(f"{url}/tokenize", body)) as response:
        return json.load(response)["tokens"]


def main() -> None:
    tokenizer, url = AutoTokenizer.from_pretrained(sys.argv[1]), sys.argv[2]
    messages = [e.text for path in DATA for e in load_examples(Path(path))] + EXTRA
    wrong = [m for m in messages if server_ids(url, PREFIX + m + SUFFIX) != prompt_ids(tokenizer, None, m)]
    print(f"{len(messages) - len(wrong)} of {len(messages)} messages give the training ids")
    for message in wrong:
        print(f"  differs: {message!r}")
    sys.exit(1 if wrong else 0)


if __name__ == "__main__":
    main()
