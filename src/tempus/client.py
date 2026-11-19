"""Read the clock times in one message from the LFM2.5-230M finetune behind `llama-server`.

Standard library only. The prompt is the chat template's text with the message inside, and
llama.cpp tokenizes it to exactly the token ids the model was trained on: it parses the special
tokens and adds the one BOS the GGUF asks for (`scripts/check_text_prompt.py` checks this
against the Hugging Face tokenizer). See docs/serving.md.
"""

import json
import urllib.request
from dataclasses import dataclass

from tempus.llama import parse as parse_response, request
from tempus.target import parse

URL = "http://127.0.0.1:8080"
# The template around one user message, no system prompt, BOS left out: the GGUF has
# add_bos_token set. The same text as web/prompts/lfm2.5-230m-v2.json.
PREFIX = "<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


@dataclass(frozen=True)
class Time:
    text: str  # the span in the message, e.g. "7"
    time: str  # 24-hour clock, e.g. "19:00"
    minute: int  # minute of the day, e.g. 1140


@dataclass(frozen=True)
class Result:
    times: list[Time]
    reply: str  # the model's raw reply
    malformed: bool  # off-format, cut at the length cap, or naming text not in the message


def extract(message: str, url: str = URL, timeout: float = 10.0) -> Result:
    """The times in `message`, greedy as in the evals. Raises on HTTP or connection errors."""
    body = json.dumps(request(PREFIX + message + SUFFIX, None)).encode()
    call = urllib.request.Request(f"{url}/completion", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(call, timeout=timeout) as response:
        reply, timings = parse_response(json.load(response))
    return read(message, reply, timings["stop"])


def read(message: str, reply: str, stop: str) -> Result:
    """Turn a reply into times. Anything doubtful gives no times, never a guessed one."""
    extractions = parse(reply, "compact") if stop != "limit" else None
    if extractions is None or not all(e.text in message for e in extractions):
        return Result([], reply, malformed=True)
    times = [Time(e.text, f"{e.minute // 60:02d}:{e.minute % 60:02d}", e.minute) for e in extractions]
    return Result(times, reply, malformed=False)
