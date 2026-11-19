"""Candidate clock expressions in a message, and the reply assembled from their readings.

Code finds every expression that could be a clock time ("7", "9:30", "4 PM", "7:05 a.m.",
"15:00", "noon") with its character offsets, and reads its digits. Each candidate then needs
one decision, made by the classifier: not a time, am or pm. Some candidates are settled by
code alone. A marker ("4pm") and the words noon and midnight always make a clock time in the
labelled data, so code settles both questions. A 24-hour form (hour 0 or 13-24, or a
zero-padded hour) settles the reading, but "back in 20" and "lost 2-0" are not times, so
whether it is a time stays the classifier's call.

Classes: `none` (not a clock time), `am` (00:00-11:59), `pm` (12:00-23:59). So a bare 12 is
`pm` at noon and `am` at midnight, as "12pm" and "12am" are.
"""

import re
from dataclasses import dataclass

from tempus.schema import Example, Extraction

CLASSES = ("none", "am", "pm")
NOON = 12 * 60

CANDIDATE = re.compile(
    r"(?<!\w)(?P<word>noon|midnight)(?!\w)"
    r"|(?<![\w.,:/$£€#@'])"  # not inside a word, decimal, list of numbers, ratio, date or price
    r"(?P<hour>[0-9]{1,2})(?::(?P<minute>[0-5][0-9]))?"
    r"(?![0-9:/%]|[.,][0-9])"  # not followed by more digits, a ratio, a date, a percent or a decimal
    r"(?:(?P<marker>\s?[ap]\.m\.|\s?[ap]\.?m(?!\w))|(?![^\W\d_])|(?=ish(?!\w)))",  # a marker, or no letter after
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Candidate:
    start: int
    end: int
    text: str
    hour: int  # as written: 0-24 (12 for noon, 0 for midnight)
    minute: int
    fixed: int | None  # minute of the day when code settles am/pm, else None
    marked: bool  # an am/pm marker, noon or midnight: a clock time whatever the classifier says

    def reading(self, meridiem: str) -> int:
        """Minute of the day for `am` or `pm`; a settled candidate ignores the choice."""
        if self.fixed is not None:
            return self.fixed
        return (self.hour % 12 + (12 if meridiem == "pm" else 0)) * 60 + self.minute


def find(message: str) -> list[Candidate]:
    """Every candidate clock expression in the message, in order."""
    found = []
    for match in CANDIDATE.finditer(message):
        if match["word"]:
            hour, minute = (12, 0) if match["word"].lower() == "noon" else (0, 0)
            fixed, marked = hour * 60, True
        else:
            hour, minute = int(match["hour"]), int(match["minute"] or 0)
            if hour > 24 or (hour == 24 and minute):
                continue
            fixed = _settled(hour, minute, match["hour"].startswith("0"), match["marker"])
            marked = bool(match["marker"])
        found.append(Candidate(match.start(), match.end(), match[0], hour, minute, fixed, marked))
    return found


def _settled(hour: int, minute: int, padded: bool, marker: str | None) -> int | None:
    if hour == 0 or hour >= 13 or padded:
        return hour % 24 * 60 + minute
    if marker:
        return (hour % 12 + (12 if "p" in marker.lower() else 0)) * 60 + minute
    return None


def assemble(candidates: list[Candidate], classes: list[str]) -> tuple[Extraction, ...]:
    """The times in the message: each candidate classed `am` or `pm`, or marked, read as settled
    or as its class says."""
    return tuple(
        Extraction(c.text, c.reading(label))
        for c, label in zip(candidates, classes, strict=True)
        if label != "none" or c.marked
    )


def label(example: Example) -> tuple[list[Candidate], list[str], list[str]]:
    """Candidates of a labelled example, the class each should get, and every labelled time
    the candidates cannot reproduce (an extraction miss or a reading they cannot give)."""
    candidates = find(example.text)
    classes = ["none"] * len(candidates)
    problems = []
    cursor = 0
    for time in example.times:
        index = next(
            (i for i, c in enumerate(candidates) if c.start >= cursor and c.text == time.text), None
        )
        if index is None:
            problems.append(f"no candidate for {time.text!r}")
            continue
        candidate = candidates[index]
        meridiem = "am" if time.minute < NOON else "pm"
        if candidate.reading(meridiem) != time.minute:
            problems.append(f"{time.text!r} cannot read as {time.minute // 60:02d}:{time.minute % 60:02d}")
        classes[index] = meridiem
        cursor = candidate.end
    return candidates, classes, problems
