"""Rule-based baseline: Microsoft Recognizers-Text plus the glue that maps its output to the spec.

The library finds date/time entities and resolves them, listing both readings when am/pm is
unknown. The glue keeps only entities that hold a clock time, trims each to the shortest
clock expression, and picks one reading: a part-of-day word in the message decides am/pm
(spec rule 2), otherwise the fallback is am, with a bare 12 read as noon (rule 4).
Needs the `rules` dependency group: `uv sync --group rules`.
"""

import re
import statistics
import time
from functools import cache
from pathlib import Path

from tempus.evaluate import save
from tempus.schema import MINUTES_PER_DAY, Example, Extraction
from tempus.target import render

# Entity types that can hold a clock time; durations, dates and date ranges are dropped.
POINT_TYPES = {"datetimeV2.time", "datetimeV2.datetime"}
RANGE_TYPES = {"datetimeV2.timerange", "datetimeV2.datetimerange"}
# The shortest clock expression, as a whole token: not part of "1:1", "16:9", "11/12" or "5th".
CLOCK = re.compile(
    r"\b(?:noon|midnight|(?<![\d/:])\d{1,2}(?::\d{2})?(?:\s?(?:[ap]\.m\.?|[ap]m)(?![a-z]))?)"
    r"(?![\d/:]|st\b|nd\b|rd\b|th\b)",
    re.IGNORECASE,
)
# The library's part-of-day codes in a timex: morning, afternoon, evening, night.
PERIOD = re.compile(r"T(MO|AF|EV|NI)\b")
NOON = 12 * 60

Period = tuple[int, int, int]  # position in the message, first and last minute of the period


def extract(message: str) -> list[Extraction]:
    """Every clock time in the message, in order of appearance."""
    entities = [e for e in _model().parse(message) if e.type_name in POINT_TYPES | RANGE_TYPES]
    periods = [period for e in entities if (period := _period(message, e))]
    found: dict[int, Extraction] = {}  # by position; overlapping entities can share a token
    for entity in entities:
        # entity.text is lowercased; its offsets (end inclusive) index the original message.
        tokens = list(CLOCK.finditer(message, entity.start, entity.end + 1))
        if not tokens:
            continue
        values = (entity.resolution or {}).get("values") or []
        if entity.type_name in RANGE_TYPES and len(tokens) > 1:
            ends = [(tokens[0], _readings(values, "start")), (tokens[-1], _readings(values, "end"))]
        else:
            ends = [(tokens[-1], _readings(values, "value", "start", "end"))]
        for token, readings in ends:
            if readings and token.start() not in found:
                found[token.start()] = Extraction(token[0], _choose(readings, token.start(), periods))
    return [found[position] for position in sorted(found)]


def reply(message: str) -> str:
    """The baseline's answer in the compact output format."""
    return render(tuple(extract(message)), "compact")


def evaluate(examples: list[Example], out: Path) -> dict:
    """Answer every message, time each one, save results under `out`, return the summary."""
    _model()  # built once up front, so its setup is not timed as part of a message
    replies, seconds = [], []
    for example in examples:
        started = time.perf_counter()
        replies.append(reply(example.text))
        seconds.append(time.perf_counter() - started)
    timed = sorted(seconds)
    run = {
        "model": "recognizers-text",
        "engine": "rules",
        "quant": "none",
        "device": "cpu",
        "format": "compact",
        "with_spec": False,
        "seconds": round(sum(seconds), 1),
        "latency_ms_p50": round(statistics.median(timed) * 1000, 2),
        "latency_ms_p95": round(timed[int(len(timed) * 0.95) - 1] * 1000, 2),
    }
    return save(run, examples, replies, out)


@cache
def _model():
    """The English date/time model, built once (it compiles many regexes)."""
    from recognizers_date_time import DateTimeRecognizer  # optional dependency group
    from recognizers_text import Culture

    return DateTimeRecognizer(Culture.English).get_datetime_model()


def _readings(values: list[dict], *keys: str) -> set[int]:
    """Distinct minutes of the day the library offers under these keys."""
    clocks = {value[key][-8:-3] for value in values for key in keys if value.get(key)}
    return {int(clock[:2]) * 60 + int(clock[3:]) for clock in clocks}


def _period(message: str, entity) -> Period | None:
    """A part-of-day entity with no clock time in it ("tonight"), with the hours the library gives it."""
    values = (entity.resolution or {}).get("values") or []
    if not any(PERIOD.search(value.get("timex", "")) for value in values):
        return None
    if CLOCK.search(message, entity.start, entity.end + 1):
        return None
    starts, ends = _readings(values, "start"), _readings(values, "end")
    if len(starts) != 1 or len(ends) != 1:
        return None
    return entity.start, starts.pop(), ends.pop()


def _choose(readings: set[int], position: int, periods: list[Period]) -> int:
    """One reading: the only one, else the one nearest the closest part-of-day period,
    else am (a bare 12 is noon)."""
    if len(readings) == 1:
        return readings.pop()
    if periods:
        _, first, last = min(periods, key=lambda period: abs(period[0] - position))
        return min(sorted(readings), key=lambda reading: _distance(reading, first, last))
    return next((r for r in sorted(readings) if 60 <= r < NOON + 60), min(readings))


def _distance(minute: int, first: int, last: int) -> int:
    """Minutes from a time to a period on the 24-hour circle; 0 inside it."""
    if first <= minute <= last:
        return 0
    return min(min(abs(minute - edge), MINUTES_PER_DAY - abs(minute - edge)) for edge in (first, last))
