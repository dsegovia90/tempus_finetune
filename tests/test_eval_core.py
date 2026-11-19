from pathlib import Path

from tempus.schema import SLICES, load_examples

CORE = Path(__file__).parent.parent / "data" / "eval" / "core.jsonl"


def test_core_follows_the_spec():
    examples = load_examples(CORE)
    problems = {e.id: e.problems() for e in examples if e.problems()}
    assert not problems


def test_core_ids_are_unique_and_every_slice_is_covered():
    examples = load_examples(CORE)
    assert len({e.id for e in examples}) == len(examples)
    assert {e.slice for e in examples} == set(SLICES)
