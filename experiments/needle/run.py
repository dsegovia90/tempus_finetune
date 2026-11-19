"""Answer every message of an eval set with a Needle archive on the CPU, and score it.

Runs in the Needle venv with the project's `src` on PYTHONPATH (tempus's scoring code is
stdlib only). Writes results/<name>/ through tempus.evaluate.save, plus needle.jsonl with the
raw calls; `log.py` then records it in MLflow from the project venv.

    CACTUS_TELEMETRY=0 DO_NOT_TRACK=1 PYTHONPATH=src:experiments/needle \\
        path/to/needle-venv/bin/python experiments/needle/run.py \\
        --weights runs/s0/20l.cact --name needle3-20l-v2 --eval-set core+eval \\
        --data data/eval/core.jsonl data/generated/eval.jsonl
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import cactus
from cactus.agent import fetch

from formulation import TOOLS_JSON, compact
from tempus.evaluate import format_summary, save
from tempus.jsonl import write_jsonl
from tempus.schema import Example, load_examples

WARMUP = 5
MAX_NEW_TOKENS = 128


class Needle:
    """One loaded archive (the published base when `weights` is None), answering one message at a time."""

    def __init__(self, weights: Path | None, threads: int):
        cactus.set_threads(threads)  # read when the engine or the tuned-archive worker starts
        # auto_date=False: the default system turn states the current date and clock time,
        # which training never saw and which is itself a clock time.
        self.agent = cactus.Cactus(
            tools=TOOLS_JSON, weights=weights, auto_date=False, stateless=True
        )

    def respond(self, message: str) -> dict:
        """The engine's calls for one message: kept, and suppressed for low confidence."""
        response = self.agent.complete(message, max_new_tokens=MAX_NEW_TOKENS)
        return {key: response.get(key) for key in ("function_calls", "suppressed_calls", "confidence")}


def calls(response: dict) -> list[dict]:
    """The calls `cactus.extract` would read: the kept ones, else the suppressed ones."""
    return response["function_calls"] or response["suppressed_calls"] or []


def answer(needle: Needle, examples: list[Example]) -> tuple[list[dict], list[float]]:
    """The response to every message, and the seconds each took."""
    for example in examples[:WARMUP]:
        needle.respond(example.text)
    responses, seconds = [], []
    for example in examples:
        started = time.perf_counter()
        responses.append(needle.respond(example.text))
        seconds.append(time.perf_counter() - started)
    return responses, seconds


def latency(seconds: list[float]) -> dict:
    timed = sorted(seconds)
    return {
        "latency_ms_p50": round(statistics.median(timed) * 1000, 1),
        "latency_ms_p95": round(timed[int(len(timed) * 0.95) - 1] * 1000, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, help="tuned .cact archive; omit for the published base")
    parser.add_argument("--quant", default="W4", help="weight format of the archive, for the record")
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-set", required=True, help="e.g. core+eval, eval2, negatives")
    parser.add_argument("--name", required=True, help="results are written to results/<name>")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    examples = [e for path in args.data for e in load_examples(path)]
    responses, seconds = answer(Needle(args.weights, args.threads), examples)
    out = Path("results") / args.name
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "needle.jsonl", ({"id": e.id, **r} for e, r in zip(examples, responses)))
    weights = args.weights or Path(fetch.fetch_weights(3))
    run = {
        "model": str(args.weights or "Cactus-Compute/needle3"),
        "base_model": "Cactus-Compute/needle3",
        "train_run_id": None,
        "eval_set": args.eval_set,
        "engine": "cactus",
        "quant": args.quant,
        "device": "cpu",
        "threads": args.threads,
        "format": "compact",
        "kind": "finetune" if args.weights else "base",
        "with_spec": False,  # Needle takes a tool schema, never the labelling spec
        "seconds": round(sum(seconds), 1),
        **latency(seconds),
        "weights_gb": round(weights.stat().st_size / 1e9, 4),
        "cactus_version": cactus.__version__,
    }
    summary = save(run, examples, [compact(calls(r)) for r in responses], out)
    print(format_summary(summary))
    print(json.dumps({k: summary[k] for k in ("latency_ms_p50", "latency_ms_p95", "weights_gb")}))


if __name__ == "__main__":
    main()
