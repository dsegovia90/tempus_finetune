"""Score a model served elsewhere behind an OpenAI-style chat endpoint.

For reference models this project does not run itself. Each message is one request: the
labelling spec as the system prompt, the message as the user turn, greedy decoding. Every
request is timed, so the latency includes the network and whatever the server does.
"""

import json
import statistics
import time
import urllib.request
from pathlib import Path

from tempus.chat import MAX_NEW_TOKENS
from tempus.evaluate import save, system_prompt
from tempus.schema import Example

WARMUP = 5  # untimed requests before the scored pass


def ask(url: str, model: str, system: str, message: str, key: str | None, extra: dict, timeout: float) -> str:
    """The model's reply to one message."""
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": message}],
        "temperature": 0,
        "max_tokens": MAX_NEW_TOKENS,
        **extra,
    }
    headers = {"Content-Type": "application/json"} | ({"Authorization": f"Bearer {key}"} if key else {})
    call = urllib.request.Request(f"{url.rstrip('/')}/v1/chat/completions", json.dumps(body).encode(), headers)
    with urllib.request.urlopen(call, timeout=timeout) as response:
        return json.load(response)["choices"][0]["message"]["content"] or ""


def evaluate(
    url: str,
    model: str,
    examples: list[Example],
    out: Path,
    fmt: str = "compact",
    key: str | None = None,
    extra: dict | None = None,
    record: dict | None = None,
    timeout: float = 120.0,
) -> dict:
    """Answer every message one request at a time, save results under `out`, return the summary.

    `extra` is merged into each request body (e.g. a switch that turns thinking off) and
    `record` into the summary (e.g. quant and device, which only the caller knows).
    """
    system, extra = system_prompt(fmt), extra or {}
    for example in examples[:WARMUP]:
        ask(url, model, system, example.text, key, extra, timeout)
    replies, seconds = [], []
    for example in examples:
        started = time.perf_counter()
        replies.append(ask(url, model, system, example.text, key, extra, timeout))
        seconds.append(time.perf_counter() - started)
    timed = sorted(seconds)
    run = {
        "model": model,
        "base_model": model,
        "kind": "base",
        "format": fmt,
        "with_spec": True,
        "engine": "http",
        "seconds": round(sum(seconds), 1),
        "latency_ms_p50": round(statistics.median(timed) * 1000, 1),
        "latency_ms_p95": round(timed[int(len(timed) * 0.95) - 1] * 1000, 1),
        "request_extra": extra,
        **(record or {}),
    }
    return save(run, examples, replies, out)
