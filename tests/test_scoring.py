from tempus import target
from tempus.schema import Example, Extraction
from tempus.scoring import score, summarize

RANGE = Example(
    id="t1",
    slice="range",
    text="I'm free tomorrow from noon to 4pm",
    times=(Extraction("noon", 720), Extraction("4pm", 960)),
)
EMPTY = Example(id="t2", slice="none", text="call me in 2 hours", times=())
GOOD = '[{"text": "noon", "time": "12:00"}, {"text": "4pm", "time": "16:00"}]'


def test_render_and_parse_round_trip():
    assert target.render(RANGE.times) == GOOD
    assert tuple(target.parse(GOOD)) == RANGE.times
    assert target.render(()) == "[]" and target.parse("[]") == []


def test_parse_rejects_off_format():
    for raw in [
        "3pm",
        '{"text": "3pm", "time": "15:00"}',
        '[{"text": "3pm"}]',
        '[{"text": "3pm", "minute": 900}]',
        '[{"text": "3pm", "time": 900}]',
        '[{"text": "3pm", "time": "3:00"}]',
        '[{"text": "3pm", "time": "24:00"}]',
        '[{"text": "3pm", "time": "15:00", "tz": "EST"}]',
        f"Here you go:\n```json\n{GOOD}\n```",
    ]:
        assert target.parse(raw) is None, raw


def test_compact_format_round_trip():
    compact = "noon=12:00\n4pm=16:00"
    assert target.render(RANGE.times, "compact") == compact
    assert tuple(target.parse(compact, "compact")) == RANGE.times
    assert target.render((), "compact") == "none" and target.parse("none", "compact") == []
    assert score(RANGE, compact, "compact").exact
    for raw in ["", "noon 12:00", "noon=12", "=12:00", GOOD]:
        assert target.parse(raw, "compact") is None, raw


def test_exact_match():
    result = score(RANGE, GOOD)
    assert result.exact and result.valid and result.grounded
    assert (result.matched, result.predicted, result.expected) == (2, 2, 2)


def test_partial_match_and_wrong_time():
    result = score(RANGE, GOOD.replace("16:00", "04:00"))
    assert not result.exact
    assert (result.matched, result.predicted, result.expected) == (1, 2, 2)


def test_wrong_order_is_not_exact():
    swapped = '[{"text": "4pm", "time": "16:00"}, {"text": "noon", "time": "12:00"}]'
    result = score(RANGE, swapped)
    assert not result.exact and result.matched == 2


def test_ungrounded_span():
    result = score(RANGE, GOOD.replace("noon", "12pm"))
    assert result.valid and not result.grounded and not result.exact


def test_invalid_output_scores_zero():
    result = score(RANGE, "noon and 4pm")
    assert not result.valid and not result.fenced and result.matched == 0


def test_fenced_output_is_accepted_but_flagged():
    result = score(RANGE, f"```json\n{GOOD}\n```")
    assert result.valid and result.exact and result.fenced


def test_empty_label():
    assert score(EMPTY, "[]").exact
    assert not score(EMPTY, '[{"text": "2", "time": "02:00"}]').exact


def test_summarize_by_slice():
    scores = [score(RANGE, '[{"text": "noon", "time": "12:00"}]'), score(EMPTY, "```\n[]\n```")]
    summary = summarize([RANGE, EMPTY], scores)
    assert summary["range"]["exact"] == 0.0
    assert summary["range"]["recall"] == 0.5
    assert summary["none"]["fenced"] == 1.0
    assert summary["all"]["n"] == 2
    assert summary["all"]["exact"] == 0.5
