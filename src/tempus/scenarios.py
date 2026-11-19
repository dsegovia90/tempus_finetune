"""Situations whose time of day is clear from world knowledge alone.

Each scenario is (description, earliest minute, latest minute). Train and eval use
different scenarios so the eval set measures generalisation, not recall. A small
hand-written set lives here; the bulk is in data/scenarios/<split>.jsonl.
"""

from pathlib import Path

from tempus.jsonl import read_jsonl

Scenario = tuple[str, int, int]
FOLDER = Path(__file__).parents[2] / "data" / "scenarios"


def _at(hour: int, minute: int = 0) -> int:
    return hour * 60 + minute


def _from_file(split: str) -> tuple[Scenario, ...]:
    """Scenarios drafted in bulk: {"description", "earliest": "HH:MM", "latest": "HH:MM"}."""
    path = FOLDER / f"{split}.jsonl"
    rows = read_jsonl(path) if path.exists() else []
    clock = lambda text: _at(*map(int, text.split(":")))  # noqa: E731
    return tuple((row["description"], clock(row["earliest"]), clock(row["latest"])) for row in rows)


TRAIN: tuple[Scenario, ...] = (
    ("breakfast plans", _at(7), _at(9, 30)),
    ("the daily standup", _at(9), _at(10, 30)),
    ("dropping kids at school before work", _at(7, 30), _at(8, 45)),
    ("lunch plans", _at(11, 30), _at(13, 30)),
    ("a dentist or doctor appointment during the workday", _at(9), _at(16, 30)),
    ("picking kids up from school", _at(14, 30), _at(15, 30)),
    ("a client meeting during office hours", _at(13), _at(17)),
    ("after-work drinks or happy hour", _at(17), _at(19)),
    ("dinner plans", _at(18), _at(21)),
    ("a movie, concert or show to go to after dinner", _at(19), _at(21, 30)),
    ("the kids' bedtime", _at(19), _at(20, 30)),
    ("the bar's closing time or last call", _at(1), _at(2, 30)),
    ("being paged or woken by an on-call incident overnight", _at(1), _at(4, 30)),
    ("a batch job that runs while everyone is asleep", _at(1), _at(4)),
    ("an alarm for the gym before work", _at(5), _at(6, 30)),
    ("a sunrise hike or run", _at(5), _at(6, 30)),
    ("a red-eye flight's departure", _at(22, 30), _at(23, 55)),
    ("kickoff of a midweek football match after work", _at(19), _at(20, 30)),
    ("when the office or shop opens for the day", _at(7), _at(10)),
    ("when the shop closes for the day", _at(17), _at(21)),
    ("staying up far too late debugging", _at(2), _at(4)),
    ("weekend brunch", _at(10), _at(11, 30)),
    ("daycare pickup at the end of the workday", _at(17), _at(18)),
    ("the commuter train to work", _at(6, 30), _at(8, 30)),
    *_from_file("train"),
)

EVAL: tuple[Scenario, ...] = (
    ("a wedding ceremony", _at(14), _at(17)),
    ("a farmers market opening", _at(7), _at(9)),
    ("karaoke with the team", _at(20), _at(22)),
    ("the start of a hospital night shift", _at(22), _at(23, 30)),
    ("a bakery shift that starts before dawn", _at(3), _at(5)),
    ("kids' football practice after school", _at(15, 30), _at(17)),
    ("a pub quiz", _at(19), _at(20, 30)),
    ("a taxi to the airport for a dawn flight", _at(4), _at(5, 30)),
    ("a lunch-and-learn session", _at(12), _at(13)),
    ("book club after work", _at(19), _at(20)),
    ("a physio appointment in the workday afternoon", _at(13), _at(16)),
    ("a baby's feed that wakes the parents", _at(2), _at(4)),
    ("coffee with a coworker before the first meeting", _at(8), _at(9, 30)),
    ("fireworks on new year's eve", _at(23), _at(23, 55)),
    *_from_file("eval"),
)

# Scenario pools for held-out splits; every other split draws from TRAIN.
HELD_OUT: dict[str, tuple[Scenario, ...]] = {"eval": EVAL, "eval2": _from_file("eval2")}
