"""Group a scored run's misses by kind, with examples.

Kinds, judged by the spans first: `extra mention` (it reported a span that is not a labelled
time), `missed mention`, `span differs` (it found the time but copied a different span,
e.g. "7pm" for "7"), `mixed`; with the right spans, `am/pm` (12 hours off) or `reading`.
An am/pm miss is shown with the example's slice: on `implicit` it is world knowledge, on
`range` the shared-marker logic, on `cue` a part-of-day word.

    python analyze.py --name needle3-20l-v2 --data data/eval/core.jsonl data/generated/eval.jsonl
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from tempus import target
from tempus.jsonl import read_jsonl
from tempus.schema import Example, Extraction, load_examples

SHOWN = 4


def kind(example: Example, predicted: list[Extraction]) -> str:
    want = [t.text for t in example.times]
    got = [p.text for p in predicted]
    if got == want:
        off = {(p.minute - t.minute) % 1440 for p, t in zip(predicted, example.times) if p != t}
        return f"am/pm ({example.slice})" if off == {720} else "reading"
    if Counter(got) > Counter(want) or (not want and got):
        return "extra mention"
    if Counter(got) < Counter(want):
        return "missed mention"
    if len(got) == len(want) and all(g in w or w in g for g, w in zip(got, want)):
        return "span differs"
    return "mixed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="results/<name>")
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    examples = {e.id: e for path in args.data for e in load_examples(path)}
    groups = defaultdict(list)
    for row in read_jsonl(Path("results") / args.name / "samples.jsonl"):
        if not row["exact"]:
            example = examples[row["id"]]
            groups[kind(example, target.parse(row["reply"], "compact") or [])].append((example, row["reply"]))
    total = sum(len(rows) for rows in groups.values())
    print(f"{args.name}: {total} misses of {len(examples)}")
    for name, rows in sorted(groups.items(), key=lambda item: -len(item[1])):
        print(f"\n{name}: {len(rows)}")
        for example, reply in rows[:SHOWN]:
            want = target.render(example.times, "compact").replace("\n", "  ")
            print(f"  {example.text!r}\n    want {want}   got {reply.replace(chr(10), '  ')}")


if __name__ == "__main__":
    main()
