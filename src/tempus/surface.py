"""Render a minute of the day as the text a person would type."""

from random import Random

MERIDIEM_STYLES = ("{t}{m}", "{t} {m}", "{t}{M}", "{t} {M}", "{t} {d}")


def bare(minute: int) -> str:
    """12-hour clock with no am/pm: 900 -> '3', 1170 -> '7:30'."""
    hour, mins = divmod(minute, 60)
    hour12 = hour % 12 or 12
    return f"{hour12}:{mins:02d}" if mins else str(hour12)


def with_meridiem(minute: int, rng: Random) -> str:
    """12-hour clock with a marker: '3pm', '3 PM', '7:30 p.m.'."""
    marker = "am" if minute < 720 else "pm"
    return rng.choice(MERIDIEM_STYLES).format(
        t=bare(minute), m=marker, M=marker.upper(), d=f"{marker[0]}.m."
    )


def explicit(minute: int, rng: Random) -> str:
    """Any unambiguous form: a marker, an unambiguous 24-hour time, or noon/midnight."""
    if minute in (0, 720) and rng.random() < 0.5:
        return "noon" if minute else "midnight"
    hour, mins = divmod(minute, 60)
    # 10:00-12:59 in 24-hour form looks like a bare time, so it always gets a marker.
    if (hour < 10 or hour >= 13) and rng.random() < 0.3:
        return f"{hour:02d}:{mins:02d}"
    return with_meridiem(minute, rng)


def fallback_minute(span_minute: int) -> int:
    """Label for a bare time with no context: am, except 12:xx which is noon."""
    return span_minute % 720 + (720 if span_minute % 720 < 60 else 0)
