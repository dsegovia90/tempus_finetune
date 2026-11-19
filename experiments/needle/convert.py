"""Write the training set in Needle's format and check the formulation's ceiling.

The ceiling is the exact-match score of perfect Needle answers sent back through `compact`:
anything under 100% is lost to the formulation before the model is involved.

    python convert.py --out needle-data/train-v2-reasoning.jsonl
"""

import argparse
from pathlib import Path

from formulation import answers, compact, training_row
from tempus.jsonl import write_jsonl
from tempus.schema import load_examples
from tempus.scoring import score

SETS = {
    "train-v2": ["data/generated/train-v2.jsonl"],
    "main": ["data/eval/core.jsonl", "data/generated/eval.jsonl"],
    "fresh": ["data/generated/eval2.jsonl"],
    "negatives": ["data/generated/negatives.jsonl"],
}


def ceiling(paths: list[str]) -> tuple[int, int, list[str]]:
    """Exact matches of the round trip, the count, and the ids that fail."""
    examples = [e for path in paths for e in load_examples(Path(path))]
    failed = [e.id for e in examples if not score(e, compact(answers(e)), "compact").exact]
    return len(examples) - len(failed), len(examples), failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="training JSONL to write")
    parser.add_argument("--no-reasoning", action="store_true", help="omit the reasoning line (the first run)")
    args = parser.parse_args()
    examples = load_examples(Path(SETS["train-v2"][0]))
    write_jsonl(args.out, (training_row(e, not args.no_reasoning) for e in examples))
    for name, paths in SETS.items():
        right, total, failed = ceiling(paths)
        print(f"{name:<10} round trip {right}/{total} = {right / total:.2%}  failures: {failed[:10]}")


if __name__ == "__main__":
    main()
