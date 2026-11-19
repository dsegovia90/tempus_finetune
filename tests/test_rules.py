import pytest

from tempus.rules import CLOCK, _choose, _readings, reply


def test_clock_token_is_the_shortest_whole_expression():
    text = "from 1:15 to 3:15 p.m. or 4pm, noon, midnight; not 1:1, 16:9, 11/12, the 5th or afternoon"
    assert [m[0] for m in CLOCK.finditer(text)] == ["1:15", "3:15 p.m.", "4pm", "noon", "midnight"]


def test_readings_come_from_times_and_datetimes():
    values = [{"value": "07:00:00"}, {"value": "2026-11-12 19:00:00"}, {"start": "08:30:00"}]
    assert _readings(values, "value") == {420, 1140}
    assert _readings(values, "start", "end") == {510}


def test_a_single_reading_is_kept():
    assert _choose({1140}, 0, []) == 1140


def test_fallback_is_am_and_bare_12_is_noon():
    assert _choose({420, 1140}, 0, []) == 420
    assert _choose({720, 0}, 0, []) == 720
    assert _choose({750, 30}, 0, []) == 750


def test_nearest_part_of_day_decides():
    afternoon, night = (20, 720, 960), (40, 1200, 1439)
    assert _choose({180, 900}, 10, [afternoon]) == 900
    assert _choose({120, 840}, 30, [night]) == 120  # "2 in the middle of the night"
    assert _choose({600, 1320}, 35, [afternoon, night]) == 1320


def test_reply_end_to_end():
    pytest.importorskip("recognizers_date_time")
    assert reply("free from noon to 4pm tomorrow") == "noon=12:00\n4pm=16:00"
    assert reply("can we do 2 to 4pm") == "2=14:00\n4pm=16:00"
    assert reply("call at 3 in the afternoon") == "3=15:00"
    assert reply("at 3?") == "3=03:00"
    assert reply("a 30 minute call, room 12") == "none"
    assert reply("our 1:1 got moved") == "none"
