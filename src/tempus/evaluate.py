"""Score one model on a labelled set and save the results."""

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from tempus.jsonl import read_jsonl, write_jsonl
from tempus.llama import Options, Server
from tempus.schema import SLICES, Example
from tempus.scoring import score, summarize

PROMPTS = Path("prompts")
LATENCY_SAMPLE = 100  # messages timed one at a time


def system_prompt(fmt: str) -> str:
    """What a base model is told: the labelling rules, then the output format."""
    spec = (PROMPTS / "spec.md").read_text().strip()
    return f"{spec}\n\n{(PROMPTS / f'format-{fmt}.md').read_text().strip()}"


def training_record(model_id: str) -> dict:
    """What `train` saved next to a finetune (runs/<name>/run.json); empty for a published model."""
    record = Path(model_id).parent / "run.json"
    return json.loads(record.read_text()) if record.exists() else {}


def evaluate(
    model_id: str,
    examples: list[Example],
    out: Path,
    fmt: str = "json",
    with_spec: bool = True,
    llama: Options | None = None,
) -> dict:
    """Run the model, save results under `out`, return the summary.

    `with_spec` puts the labelling spec in the system prompt (for base models). Finetunes
    are scored with it off, on the bare message. With `llama` set, the GGUF file it names
    is served by llama.cpp; `model_id` still supplies the tokenizer and training record.
    """
    system = system_prompt(fmt) if with_spec else None
    messages = [e.text for e in examples]
    with _engine(model_id, llama, out) as engine:
        started = time.monotonic()
        replies = engine.generate(system, messages)
        seconds = round(time.monotonic() - started, 1)
        timed = sorted(engine.latencies(system, messages[:LATENCY_SAMPLE]))
        details = engine.details()
    trained = training_record(model_id)
    run = {
        "model": model_id,
        "base_model": trained.get("model", model_id),
        "train_run_id": trained.get("mlflow_run_id"),
        "train_experiment_id": trained.get("mlflow_experiment_id"),
        "format": fmt,
        "with_spec": with_spec,
        "seconds": seconds,
        "latency_ms_p50": round(statistics.median(timed) * 1000, 1),
        "latency_ms_p95": round(timed[int(len(timed) * 0.95) - 1] * 1000, 1),
        **details,
    }
    return save(run, examples, replies, out)


def _engine(model_id: str, llama: Options | None, out: Path):
    """The inference engine: transformers on the GPU, or llama.cpp serving a GGUF file."""
    if llama is None:
        from tempus.infer import Transformers  # imported here so the CLI works without torch

        return Transformers(model_id)
    return Server(model_id, llama, out / "llama-server.log")


def rescore(examples: list[Example], out: Path) -> dict:
    """Recompute scores from saved replies, after a change to the scoring rules."""
    old = json.loads((out / "summary.json").read_text())
    replies = {row["id"]: row["reply"] for row in read_jsonl(out / "samples.jsonl")}
    run = {key: value for key, value in old.items() if key not in ("examples", "metrics")}
    return save(run, examples, [replies[e.id] for e in examples], out)


def save(run: dict, examples: list[Example], replies: list[str], out: Path) -> dict:
    """Write per-sample results and a summary under `out`, return the summary."""
    fmt = run.get("format", "json")
    scores = [score(example, reply, fmt) for example, reply in zip(examples, replies, strict=True)]
    write_jsonl(
        out / "samples.jsonl",
        (
            {"id": e.id, "slice": e.slice, "text": e.text, "reply": r, **asdict(s)}
            for e, r, s in zip(examples, replies, scores, strict=True)
        ),
    )
    summary = {**run, "examples": len(examples), "metrics": summarize(examples, scores)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def format_summary(summary: dict) -> str:
    """A small table: one row per slice."""
    lines = [f"{summary['model']}  ({summary['examples']} examples, {summary['seconds']}s)"]
    if "engine" in summary:
        lines.append(f"  {summary['engine']}, {summary['quant']} on {summary['device']}")
    if "latency_ms_p50" in summary:
        lines.append(
            f"  latency per message: median {summary['latency_ms_p50']} ms, "
            f"95th percentile {summary['latency_ms_p95']} ms"
        )
    if "server_prompt_ms_p50" in summary:
        lines.append(
            f"  server median: prompt {summary['server_prompt_ms_p50']} ms, "
            f"generation {summary['server_generation_ms_p50']} ms"
        )
    lines.append(f"  {'slice':<9}{'n':>5}{'exact':>8}{'valid':>8}{'fenced':>8}{'recall':>8}")
    for name in ("all", *SLICES):
        row = summary["metrics"].get(name)
        if row:
            lines.append(
                f"  {name:<9}{row['n']:>5}{row['exact']:>8.1%}{row['valid']:>8.1%}"
                f"{row['fenced']:>8.1%}{row['recall']:>8.1%}"
            )
    return "\n".join(lines)
