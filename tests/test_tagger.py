import pytest

from tempus import tagger
from tempus.schema import Example, Extraction

TEXT = "can we do 3 instead of the 30 minute slot at 9:30pm"
#       0         1         2         3         4         5
EXAMPLE = Example("a", "explicit", TEXT, (Extraction("3", 900), Extraction("9:30pm", 1290)))
# One token per word, with "9:30pm" in four pieces, between [CLS] and [SEP].
OFFSETS = [(0, 0), (0, 3), (4, 6), (7, 9), (10, 11), (12, 19), (20, 22), (23, 26), (27, 29), (30, 36),
           (37, 41), (42, 44), (45, 46), (46, 47), (47, 49), (49, 51), (0, 0)]


def test_spans_skip_occurrences_inside_longer_numbers():
    assert tagger.spans(EXAMPLE) == [(10, 11, 900), (45, 51, 1290)]
    assert tagger.spans(Example("b", "implicit", "30 people at 3", (Extraction("3", 900),))) == [(13, 14, 900)]


def test_labels_and_decode_round_trip():
    tags, hours, minutes = tagger.labels(EXAMPLE, OFFSETS)
    assert [i for i, t in enumerate(tags) if t == tagger.BEGIN] == [4, 12]
    assert [i for i, t in enumerate(tags) if t == tagger.INSIDE] == [13, 14, 15]
    assert (hours[4], minutes[4], hours[12], minutes[12]) == (15, 0, 21, 6)  # minute 30 is step 6
    filled = [h if h != tagger.IGNORE else 0 for h in hours], [m if m != tagger.IGNORE else 0 for m in minutes]
    assert tagger.decode(TEXT, OFFSETS, tags, *filled) == EXAMPLE.times


def test_labels_reject_a_span_that_splits_a_token():
    with pytest.raises(ValueError):
        tagger.labels(Example("c", "explicit", "at 7pm", (Extraction("7", 1140),)), [(0, 0), (0, 2), (3, 6), (0, 0)])


def test_labels_reject_a_minute_off_the_step():
    with pytest.raises(ValueError):
        tagger.labels(Example("d", "explicit", "at 9:17", (Extraction("9:17", 557),)), [(0, 0), (0, 2), (3, 7), (0, 0)])


def test_decode_ignores_an_inside_tag_with_no_begin():
    assert tagger.decode("at 7", [(0, 2), (3, 4)], [tagger.OUTSIDE, tagger.INSIDE], [0, 19], [0, 0]) == ()
