"""The model-facing output formats.

The model writes a 24-hour clock string, never the minute number, so it is not
asked to do arithmetic. Two formats exist so their accuracy and speed can be compared:

    json     [{"text": "noon", "time": "12:00"}, {"text": "4pm", "time": "16:00"}]
    compact  noon=12:00
             4pm=16:00          (the word `none` when there is no time)

Datasets store the minute of the day; this module converts in both directions.
"""

import json
import re

from tempus.schema import Extraction

FORMATS = ("json", "compact")
NO_TIMES = "none"
CLOCK = re.compile(r"([01]\d|2[0-3]):([0-5]\d)")
FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def render(times: tuple[Extraction, ...], fmt: str = "json") -> str:
    """The exact string a model should output for these times."""
    pairs = [(t.text, f"{t.minute // 60:02d}:{t.minute % 60:02d}") for t in times]
    if fmt == "compact":
        return "\n".join(f"{text}={clock}" for text, clock in pairs) or NO_TIMES
    return json.dumps([{"text": text, "time": clock} for text, clock in pairs], ensure_ascii=False)


def parse(raw: str, fmt: str = "json") -> list[Extraction] | None:
    """Parse model output. Return None for anything off-format.

    One surrounding Markdown code fence is tolerated, because some base models
    always add one; `is_fenced` reports it so the habit stays visible.
    """
    fenced = FENCE.fullmatch(raw.strip())
    body = fenced[1] if fenced else raw
    pairs = _compact_pairs(body) if fmt == "compact" else _json_pairs(body)
    if pairs is None:
        return None
    extractions = []
    for text, clock in pairs:
        match = CLOCK.fullmatch(clock) if isinstance(clock, str) else None
        if not isinstance(text, str) or not text or match is None:
            return None
        extractions.append(Extraction(text, int(match[1]) * 60 + int(match[2])))
    return extractions


def is_fenced(raw: str) -> bool:
    """True when the output is wrapped in a Markdown code fence."""
    return raw.strip().startswith("```")


def _json_pairs(body: str) -> list[tuple] | None:
    try:
        items = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list):
        return None
    if not all(isinstance(item, dict) and set(item) == {"text", "time"} for item in items):
        return None
    return [(item["text"], item["time"]) for item in items]


def _compact_pairs(body: str) -> list[tuple] | None:
    body = body.strip()
    if body == NO_TIMES:
        return []
    lines = body.split("\n")
    if not all("=" in line for line in lines):
        return None
    return [tuple(line.rsplit("=", 1)) for line in lines]
