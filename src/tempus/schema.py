"""Task schema: a message and the clock times it contains."""

from dataclasses import asdict, dataclass
from pathlib import Path

from tempus.jsonl import read_jsonl

MINUTES_PER_DAY = 1440
SLICES = ("implicit", "cue", "explicit", "range", "fallback", "none")


@dataclass(frozen=True)
class Extraction:
    """One clock time: the exact span in the message and its minute of the day."""

    text: str
    minute: int


@dataclass(frozen=True)
class Example:
    id: str
    slice: str
    text: str
    times: tuple[Extraction, ...]

    def problems(self) -> list[str]:
        """Return every way this example breaks the labelling spec."""
        found = []
        if self.slice not in SLICES:
            found.append(f"unknown slice {self.slice!r}")
        if (self.slice == "none") != (not self.times):
            found.append("slice 'none' must have no times, other slices at least one")
        cursor = 0
        for time in self.times:
            if not 0 <= time.minute < MINUTES_PER_DAY:
                found.append(f"minute {time.minute} out of range")
            position = self.text.find(time.text, cursor)
            if position < 0:
                found.append(f"span {time.text!r} missing or out of order")
            else:
                cursor = position + len(time.text)
        return found

    def to_row(self) -> dict:
        return asdict(self)


def parse_times(rows: list[dict]) -> tuple[Extraction, ...]:
    return tuple(Extraction(row["text"], row["minute"]) for row in rows)


def load_examples(path: Path) -> list[Example]:
    """Read examples from a JSONL file."""
    return [
        Example(row["id"], row["slice"], row["text"], parse_times(row["times"]))
        for row in read_jsonl(path)
    ]
