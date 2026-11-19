import json
from pathlib import Path

import pytest

from tempus.candidates import assemble, find, label
from tempus.schema import Example, Extraction, load_examples

# Cases for the candidate finder, kept as data.
CASES = json.loads(Path("tests/candidates_cases.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[case["message"][:30] for case in CASES])
def test_shared_cases(case):
    found = [{"text": c.text, "start": c.start, "fixed": c.fixed, "marked": c.marked} for c in find(case["message"])]
    assert found == case["candidates"]


def test_spans_are_the_shortest_clock_expression():
    found = find("7:05 a.m. or 3 PM EST, 6ish, noon")
    assert [c.text for c in found] == ["7:05 a.m.", "3 PM", "6", "noon"]
    assert [c.start for c in found] == [0, 13, 23, 29]


def test_markers_settle_the_reading_and_bare_hours_do_not():
    am, pm, bare, padded = find("12am 12pm 12 07:30")
    assert (am.fixed, pm.fixed, bare.fixed, padded.fixed) == (0, 720, None, 450)
    assert bare.reading("pm") == 720 and bare.reading("am") == 0  # noon, or "12 at night"


def test_non_times_are_not_candidates():
    assert find("1:1 at room 3a, 16:9, 11/12, the 5th, 3.5, $20, 50%, 1,000, v2, 30 mins") == []


def test_assembly_keeps_times_in_order_and_marked_candidates_always():
    found = find("open 9 to 9, back in 5, till 4pm")
    times = assemble(found, ["am", "pm", "none", "none"])
    assert times == (Extraction("9", 540), Extraction("9", 1260), Extraction("4pm", 960))


def test_labels_follow_offsets_for_repeated_spans():
    example = Example("x", "range", "open 9 to 9", (Extraction("9", 540), Extraction("9", 1260)))
    _, classes, problems = label(example)
    assert classes == ["am", "pm"] and problems == []


def test_a_range_without_a_shared_marker():
    example = Example("x", "range", "11-1pm works", (Extraction("11", 660), Extraction("1pm", 780)))
    assert label(example)[1:] == (["am", "pm"], [])


@pytest.mark.parametrize("path", ["data/eval/core.jsonl", "data/generated/eval.jsonl", "data/generated/eval2.jsonl"])
def test_every_labelled_time_is_a_candidate(path):
    assert all(not label(example)[2] for example in load_examples(Path(path)))
