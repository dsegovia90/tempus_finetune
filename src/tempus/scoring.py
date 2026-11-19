"""Score raw model output against labelled examples."""

from collections import Counter, defaultdict
from dataclasses import dataclass

from tempus import target
from tempus.schema import Example


@dataclass(frozen=True)
class Score:
    valid: bool  # output is in the target format
    fenced: bool  # output was wrapped in a code fence (tolerated, but tracked)
    grounded: bool  # every predicted span is a substring of the message
    exact: bool  # predictions equal the label, in order
    matched: int  # predicted (text, minute) pairs that are in the label
    predicted: int
    expected: int


def score(example: Example, raw: str, fmt: str = "json") -> Score:
    predictions = target.parse(raw, fmt)
    expected = len(example.times)
    fenced = target.is_fenced(raw)
    if predictions is None:
        return Score(False, fenced, False, False, 0, 0, expected)
    matched = sum((Counter(predictions) & Counter(example.times)).values())
    return Score(
        valid=True,
        fenced=fenced,
        grounded=all(p.text in example.text for p in predictions),
        exact=tuple(predictions) == example.times,
        matched=matched,
        predicted=len(predictions),
        expected=expected,
    )


def summarize(examples: list[Example], scores: list[Score]) -> dict[str, dict[str, float]]:
    """Aggregate metrics per slice, plus an 'all' row."""
    groups: dict[str, list[Score]] = defaultdict(list)
    for example, result in zip(examples, scores, strict=True):
        groups[example.slice].append(result)
        groups["all"].append(result)
    return {name: _metrics(group) for name, group in groups.items()}


def _metrics(scores: list[Score]) -> dict[str, float]:
    count = len(scores)
    matched = sum(s.matched for s in scores)
    return {
        "n": count,
        "exact": sum(s.exact for s in scores) / count,
        "valid": sum(s.valid for s in scores) / count,
        "fenced": sum(s.fenced for s in scores) / count,
        "grounded": sum(s.grounded for s in scores) / count,
        "precision": _ratio(matched, sum(s.predicted for s in scores)),
        "recall": _ratio(matched, sum(s.expected for s in scores)),
    }


def _ratio(numerator: int, denominator: int) -> float:
    """Ratio that counts 'nothing to find, nothing found' as perfect."""
    return numerator / denominator if denominator else 1.0
