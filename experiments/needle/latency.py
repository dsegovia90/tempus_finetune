"""Time one archive on the CPU: the first 100 main-eval messages, one at a time, after warm-up.

One process per thread count (the engine reads its thread count when it starts).

    python latency.py --weights runs/s0/20l.cact --threads 1
"""

import argparse
import json
from pathlib import Path

from run import Needle, answer, latency
from tempus.schema import load_examples

MESSAGES = 100
MAIN = [Path("data/eval/core.jsonl"), Path("data/generated/eval.jsonl")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, help="tuned .cact archive; omit for the published base")
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    examples = [e for path in MAIN for e in load_examples(path)][:MESSAGES]
    _, seconds = answer(Needle(args.weights, args.threads), examples)
    print(json.dumps({"weights": str(args.weights), "threads": args.threads, **latency(seconds)}))


if __name__ == "__main__":
    main()
