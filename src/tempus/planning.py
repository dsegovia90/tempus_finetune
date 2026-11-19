"""Plan examples before any text exists, so every label is known by construction.

A plan fixes the slice, the times and their spans. A writer agent then turns the
plan into a chat message (see prompts/writer.md). Fallback plans need no writer.
"""

from dataclasses import asdict, dataclass
from random import Random

from tempus import scenarios, surface
from tempus.schema import Extraction, parse_times

MIX = {"implicit": 0.35, "cue": 0.12, "explicit": 0.15, "range": 0.12, "fallback": 0.06, "none": 0.20}
NEGATIVES = "negatives"  # an eval-only split of messages with no clock time, built to tempt a model
HARD = "train2"  # training data for false alarms: traps alone, and traps next to a real clock time
HARD_MIX = {"none": 0.53, "implicit": 0.22, "explicit": 0.12, "cue": 0.08, "range": 0.05}

STYLES = (
    "all lowercase apart from the required strings, minimal punctuation",
    "casual, with one typo",
    "casual, with an emoji",
    "starts with an @mention",
    "two or three short sentences",
    "polite and fully punctuated",
    "very short",
    "a reply in the middle of a thread",
)

AUDIENCES = (
    "to a coworker",
    "in a team channel",
    "to a friend",
    "in a family group chat",
    "to a partner",
    "to a roommate",
    "to a client or customer",
    "in a hobby or sports group chat",
)

# (cue phrases, earliest minute, latest minute)
DAYPARTS = (
    (("in the morning", "this morning", "tomorrow morning"), 300, 690),
    (("in the afternoon", "this afternoon", "tomorrow afternoon"), 780, 1050),
    (("tonight", "this evening", "in the evening"), 1080, 1290),
    (("tonight", "at night", "late tonight"), 1320, 1410),
    (("in the morning", "in the middle of the night"), 60, 210),
)

DISTRACTORS = (
    "a duration, such as how long a call or task takes",
    "a relative time, such as 'in 2 hours' or 'in a bit'",
    "only a vague part of the day, such as 'this afternoon' or 'tonight'",
    "a count of people, items or tests",
    "a room, floor or desk number",
    "a price or budget",
    "a software version or build number",
    "a ticket, order or port number",
    "a sports score or a ratio such as 16:9",
    "a calendar date or day of the month",
    "a percentage or metric",
    "a one-on-one meeting written as 1:1, with no time given",
    "no numbers at all",
)

# Numbers that look like times but are not. Each is a brief for the writer of a `negatives` message.
TRAPS = (
    "a short relative wait with a bare number and no unit, such as 'back in 5' or 'give me 10'",
    "a wait or delay in minutes or hours, such as 'gimme 10 min' or 'running 20 late'",
    "a duration written with a colon, such as a race or lap result ('finished in 3:45') or a video length",
    "a position inside a recording, such as 'skip to 2:30 in the video' or 'around 1:15 in the podcast'",
    "a ratio written with a colon, such as an aspect ratio (16:9, 4:3) or odds (3:1)",
    "a score written with a colon or a dash, such as 'we won 3:2' or 'lost 2-1'",
    "a one-on-one meeting written as 1:1, with no time given",
    "a bible verse or section reference, such as 'John 3:16' or 'section 4:2'",
    "a count of people, seats or items right after the word 'at' or 'for', such as 'a table for 6'",
    "a room, floor, gate, platform, bus line or desk number, such as 'meet at gate 12' or 'take the 9'",
    "a price or amount of money without a currency symbol next to it, such as 'they want 7 for it'",
    "a software version, build, ticket, order or port number",
    "a calendar date or day of the month, such as 'the 3rd', '7/11' or 'on the 12th'",
    "a year, quarter or week number, such as 'back in 2019', 'Q3' or 'week 9'",
    "an age, temperature, weight, distance or speed, such as 'she turns 7' or 'a 5k'",
    "a rating, percentage or multiplier, such as '9 out of 10', 'down 12%' or '10x'",
    "a phone extension, PIN, door code or other code",
    "an idiom with numbers that is not a schedule, such as '24/7' or 'a 180 on that decision'",
    "only a vague part of the day or a meal, such as 'after lunch', 'tonight' or 'end of day'",
    "two or three different non-time numbers in the same message",
)

# Trap kinds kept out of training, so the negatives eval can show whether a model learned the
# idea ("a number that is not a time of day") or only the kinds it was shown.
HELD_OUT_TRAPS = tuple(t for t in TRAPS if t.startswith(("a bible verse", "an age,", "a phone extension")))
TRAIN_TRAPS = tuple(t for t in TRAPS if t not in HELD_OUT_TRAPS)

RANGE_FORMS = ("{a} to {b}", "{a}-{b}", "from {a} to {b}", "between {a} and {b}")
FALLBACK_FORMS = (
    "{t}", "{t}?", "at {t}", "at {t}?", "around {t}", "at {t} then", "{t} then", "ok {t}",
    "{t} ok?", "by {t}", "before {t}", "after {t}", "{t} works", "say {t}", "{t} it is", "until {t}",
)


@dataclass(frozen=True)
class Plan:
    id: str
    slice: str
    brief: str  # what the writer must make the message about
    style: str
    must_include: tuple[str, ...]  # exact strings the message must contain
    times: tuple[Extraction, ...]
    text: str | None = None  # set when no writer is needed

    def to_row(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: dict) -> "Plan":
        return cls(
            row["id"], row["slice"], row["brief"], row["style"],
            tuple(row["must_include"]), parse_times(row["times"]), row.get("text"),
        )


def make_plans(count: int, split: str, seed: int = 0) -> list[Plan]:
    """Build a reproducible list of plans. Eval splits use their own held-out scenarios."""
    rng = Random(f"{split}-{seed}")
    pool = scenarios.HELD_OUT.get(split, scenarios.TRAIN)
    builders = {
        "implicit": lambda: _implicit(rng, pool),
        "cue": lambda: _cue(rng),
        "explicit": lambda: _explicit(rng),
        "range": lambda: _range(rng),
        "fallback": lambda: _fallback(rng),
        "none": lambda: _none(rng),
    }
    plans, ready_made = [], set()
    mix = {NEGATIVES: {"none": 1.0}, HARD: HARD_MIX}.get(split, MIX)
    if split == NEGATIVES:
        builders["none"] = lambda: _trap(rng, TRAPS)
    if split == HARD:
        builders = {name: _with_trap(build, rng) for name, build in builders.items()}
        builders["none"] = lambda: _trap(rng, TRAIN_TRAPS)
    for index, name in enumerate(rng.choices(list(mix), weights=list(mix.values()), k=count)):
        brief, spans, times, text = builders[name]()
        for _ in range(20):  # fallback messages come from few templates, so redraw repeats
            if text is None or text not in ready_made:
                break
            brief, spans, times, text = builders[name]()
        ready_made.add(text)
        if text is None:
            brief = f"Written {rng.choice(AUDIENCES)}. {brief}"
        plans.append(
            Plan(f"{split}-{index:05d}", name, brief, rng.choice(STYLES), spans, times, text)
        )
    return plans


def _pick_minute(rng: Random, low: int, high: int) -> int:
    """Pick a minute in [low, high], mostly on the hour or half hour."""
    step = rng.choices((60, 30, 15, 5), weights=(50, 30, 15, 5))[0]
    options = [m for m in range(low, high + 1) if m % step == 0]
    return rng.choice(options or list(range(low, high + 1, 5)))


def _implicit(rng: Random, pool: tuple[scenarios.Scenario, ...]):
    chosen = rng.sample(pool, k=2 if rng.random() < 0.15 else 1)
    # Earliest first, so a two-event message reads in the order the day happens.
    picked = sorted((_pick_minute(rng, low, high), description) for description, low, high in chosen)
    times = tuple(Extraction(surface.bare(minute), minute) for minute, _ in picked)
    topics = " and then ".join(description for _, description in picked)
    brief = (
        f"About {topics}. The situation alone must make the time of day obvious: "
        "no am/pm and no words like morning, afternoon, evening, tonight or night."
    )
    return brief, tuple(t.text for t in times), times, None


def _cue(rng: Random):
    phrases, low, high = rng.choice(DAYPARTS)
    minute = _pick_minute(rng, low, high)
    phrase = rng.choice(phrases)
    time = Extraction(surface.bare(minute), minute)
    brief = f'Any everyday or work topic. Use the words "{phrase}" so the reader knows which half of the day is meant.'
    return brief, (time.text, phrase), (time,), None


def _explicit(rng: Random):
    count = rng.choices((1, 2, 3), weights=(80, 15, 5))[0]
    minutes = rng.sample(range(0, 1440, 5), k=count)
    times = tuple(Extraction(surface.explicit(m, rng), m) for m in minutes)
    extra = rng.choice((
        "",
        " Add a timezone abbreviation right after one of the times.",
        " Also mention a duration or a count that is not a time.",
    ))
    return f"Any everyday or work topic.{extra}", tuple(t.text for t in times), times, None


def _range(rng: Random):
    kind = rng.random()
    if kind < 0.15:  # both ends carry their own marker
        start = _pick_minute(rng, 0, 1410)
        end = (start + rng.choice((60, 90, 120, 180, 240))) % 1440
        first, second = surface.explicit(start, rng), surface.explicit(end, rng)
    elif kind < 0.25:  # crosses noon: "11-1pm" means 11am to 1pm
        start, end = rng.choice((600, 630, 660, 690)), rng.choice((780, 810, 840))
        first, second = surface.bare(start), surface.with_meridiem(end, rng)
    else:  # one marker shared by both ends: "2 to 4pm"
        half = rng.choices((0, 720), weights=(35, 65))[0]
        offset = _pick_minute(rng, 60, 600)
        length = rng.choice([d for d in (30, 60, 90, 120, 180) if offset + d <= 690])
        start, end = half + offset, half + offset + length
        first, second = surface.bare(start), surface.with_meridiem(end, rng)
    phrase = rng.choice(RANGE_FORMS).format(a=first, b=second)
    times = (Extraction(first, start), Extraction(second, end))
    return "A time window for any everyday or work topic.", (phrase,), times, None


def _fallback(rng: Random):
    span_minute = _pick_minute(rng, 60, 779)  # 1:00 to 12:59 on a 12-hour clock
    span = surface.bare(span_minute)
    text = rng.choice(FALLBACK_FORMS).format(t=span)
    return "", (span,), (Extraction(span, surface.fallback_minute(span_minute)),), text


def _with_trap(build, rng: Random):
    """A plan with real clock times that must also carry a number which is not one."""
    def contrast():
        brief, spans, times, text = build()
        extra = (
            f" In the same message, also include {rng.choice(TRAIN_TRAPS)}. That number is not a clock "
            "time and a careful reader must be certain of it, while the planned times stay clear."
        )
        return brief + extra, spans, times, text
    return contrast


def _trap(rng: Random, traps: tuple[str, ...]):
    brief = (
        f"Any everyday or work topic. Contains no clock time at all. Include {rng.choice(traps)}. "
        "Write it the way people really do, so a careless reader could take the number for a time of day, "
        "while a careful reader is certain it is not one."
    )
    return brief, (), (), None


def _none(rng: Random):
    brief = f"Any everyday or work topic. Contains no clock time. Include {rng.choice(DISTRACTORS)}."
    return brief, (), (), None
