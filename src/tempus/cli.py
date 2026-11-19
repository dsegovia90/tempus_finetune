"""Command line: plan and assemble dataset splits, train models, and score them.

Layout under data/:
  plans/<split>/plans.jsonl        every plan
  plans/<split>/batch-NNN.jsonl    plans that need a writer agent
  messages/<split>/batch-NNN.jsonl written by writer agents: {"id", "text"}
  checks/<split>/batch-NNN.jsonl   written by checker agents: {"id", "times": [{"text", "clock"}]}
  generated/<split>.jsonl          validated examples
  generated/<split>.rejects.jsonl  everything that failed a gate, with reasons

Training writes runs/<name>/adapter and runs/<name>/merged, named after configs/<name>.yaml.
GGUF exports for llama.cpp go in runs/<name>/gguf/<quant>.gguf.
"""

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from tempus.assemble import assemble
from tempus import client, rules, tracking
from tempus.evaluate import evaluate, format_summary, rescore
from tempus.jsonl import read_jsonl, write_jsonl
from tempus.llama import Options
from tempus.planning import Plan, make_plans
from tempus.schema import Example, Extraction, load_examples
from tempus.target import FORMATS
from tempus.train import load_config, train
from tempus.trim import trim_model

DATA = Path("data")
EVAL_SETS = [DATA / "eval" / "core.jsonl", DATA / "generated" / "eval.jsonl"]
ALL_SETS = [DATA / "generated" / "train.jsonl", *EVAL_SETS, DATA / "generated" / "eval2.jsonl"]


def main() -> None:
    parser = argparse.ArgumentParser(prog="tempus")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="write plans and writer batches for a split")
    plan.add_argument("--split", required=True)
    plan.add_argument("--count", type=int, required=True)
    plan.add_argument("--batch-size", type=int, default=200)
    plan.add_argument("--seed", type=int, default=0)
    plan.set_defaults(run=_plan)

    build = commands.add_parser("assemble", help="validate a split and write its examples")
    build.add_argument("--split", required=True)
    build.add_argument(
        "--hold-out", type=Path, nargs="*", default=[],
        help="example files whose messages must not reappear in this split (the eval sets)",
    )
    build.set_defaults(run=_assemble)

    training = commands.add_parser("train", help="finetune a model from a run config (needs a GPU)")
    training.add_argument("--config", type=Path, required=True, help="e.g. configs/<name>.yaml")
    training.set_defaults(run=_train)

    evaluation = commands.add_parser("eval", help="score a model on a labelled set (needs a GPU)")
    evaluation.add_argument("--model", required=True, help="Hugging Face ID or local path")
    evaluation.add_argument("--data", type=Path, nargs="+", default=EVAL_SETS)
    evaluation.add_argument("--name", required=True, help="results are written to results/<name>")
    evaluation.add_argument("--format", choices=FORMATS, default="compact", help="output format")
    evaluation.add_argument(
        "--no-spec", action="store_true", help="send only the message (use for finetunes)"
    )
    evaluation.add_argument("--engine", choices=("transformers", "llama"), default="transformers")
    evaluation.add_argument("--group", help="MLflow `group` tag, e.g. ladder or quantization")
    llama = evaluation.add_argument_group("llama engine (--model still gives the tokenizer)")
    llama.add_argument("--gguf", type=Path, help="e.g. runs/<name>/gguf/Q8_0.gguf")
    llama.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    llama.add_argument("--threads", type=int, default=4, help="CPU threads")
    llama.add_argument(
        "--grammar", action="store_true", help="constrain replies with a GBNF grammar (compact only)"
    )
    llama.add_argument("--llama-server", default="llama-server", help="path to the executable")
    evaluation.set_defaults(run=_eval)

    remote = commands.add_parser(
        "eval-remote", help="score a model behind an OpenAI-style chat endpoint, given the labelling spec"
    )
    remote.add_argument("--url", required=True, help="server root, e.g. http://127.0.0.1:8080")
    remote.add_argument("--model", required=True, help="model name the server expects")
    remote.add_argument("--data", type=Path, nargs="+", default=EVAL_SETS)
    remote.add_argument("--name", required=True, help="results are written to results/<name>")
    remote.add_argument("--key-env", help="environment variable holding the API key")
    remote.add_argument("--extra", type=json.loads, default={}, help="JSON merged into each request body")
    remote.add_argument("--quant", default="unknown", help="weight format, for the record")
    remote.add_argument("--device", default="gpu", help="where the server runs the model, for the record")
    remote.add_argument("--group", help="MLflow `group` tag")
    remote.set_defaults(run=_eval_remote)

    baseline = commands.add_parser(
        "rules", help="score the rule-based baseline (no model, no MLflow; needs --group rules)"
    )
    baseline.add_argument("--data", type=Path, nargs="+", default=EVAL_SETS)
    baseline.add_argument("--name", required=True, help="results are written to results/<name>")
    baseline.set_defaults(run=_rules)

    classifier_training = commands.add_parser(
        "classifier-train", help="train the candidate classifier (needs the classifier and gpu groups)"
    )
    classifier_training.add_argument("--model", required=True, help="Hugging Face ID of the encoder")
    classifier_training.add_argument("--name", required=True, help="written to runs/<name>")
    classifier_training.add_argument("--data", type=Path, nargs="+", default=[DATA / "generated" / "train.jsonl"])
    classifier_training.add_argument("--seed", type=int, default=1)
    classifier_training.add_argument("--epochs", type=int, default=10)
    classifier_training.add_argument("--learning-rate", type=float, default=1e-4)
    classifier_training.set_defaults(run=_classifier_train)

    classifier_scoring = commands.add_parser(
        "classifier-eval", help="score the candidate classifier on the CPU (needs the classifier group)"
    )
    classifier_scoring.add_argument("--model", type=Path, required=True, help="e.g. runs/<name>/model")
    classifier_scoring.add_argument("--onnx", type=Path, help="score this ONNX export instead of PyTorch")
    classifier_scoring.add_argument("--data", type=Path, nargs="+", default=EVAL_SETS)
    classifier_scoring.add_argument("--name", required=True, help="results are written to results/<name>")
    classifier_scoring.add_argument("--no-log", action="store_true", help="do not log to MLflow")
    classifier_scoring.set_defaults(run=_classifier_eval)

    tagger_training = commands.add_parser(
        "tagger-train", help="train the model-only encoder tagger (needs the classifier group)"
    )
    tagger_training.add_argument("--model", required=True, help="Hugging Face ID of the encoder")
    tagger_training.add_argument("--name", required=True, help="written to runs/<name>")
    tagger_training.add_argument("--data", type=Path, nargs="+", default=[DATA / "generated" / "train-v2.jsonl"])
    tagger_training.add_argument("--seed", type=int, default=1)
    tagger_training.add_argument("--epochs", type=int, default=10)
    tagger_training.add_argument("--learning-rate", type=float, default=1e-4)
    tagger_training.set_defaults(run=_tagger_train)

    tagger_scoring = commands.add_parser("tagger-eval", help="score the tagger on the CPU (needs the classifier group)")
    tagger_scoring.add_argument("--model", type=Path, required=True, help="e.g. runs/<name>/model")
    tagger_scoring.add_argument("--onnx", type=Path, help="score this ONNX export instead of PyTorch")
    tagger_scoring.add_argument("--gpu", action="store_true", help="run the PyTorch model on CUDA")
    tagger_scoring.add_argument("--data", type=Path, nargs="+", default=EVAL_SETS)
    tagger_scoring.add_argument("--name", required=True, help="results are written to results/<name>")
    tagger_scoring.set_defaults(run=_tagger_eval)

    tagger_export = commands.add_parser("tagger-export", help="export the tagger to ONNX: fp32, fp16 and int8")
    tagger_export.add_argument("--model", type=Path, required=True, help="e.g. runs/<name>/model")
    tagger_export.set_defaults(run=_tagger_export)

    classifier_export = commands.add_parser(
        "classifier-export", help="export the classifier to ONNX: fp32, fp16 and int8 (classifier group)"
    )
    classifier_export.add_argument("--model", type=Path, required=True, help="e.g. runs/<name>/model")
    classifier_export.set_defaults(run=_classifier_export)

    again = commands.add_parser("rescore", help="recompute scores from saved replies")
    again.add_argument("--data", type=Path, nargs="+", default=EVAL_SETS)
    again.add_argument("--name", required=True)
    again.set_defaults(run=_rescore)

    trimming = commands.add_parser(
        "trim", help="cut a merged model's vocabulary to English chat (needs the gpu group)"
    )
    trimming.add_argument("--model", type=Path, required=True, help="e.g. runs/<name>/merged")
    trimming.add_argument(
        "--data", type=Path, nargs="+", default=ALL_SETS,
        help="examples whose conversations must tokenize as before (their tokens are kept)",
    )
    trimming.add_argument("--out", type=Path, required=True, help="e.g. runs/<name>-trim/merged")
    trimming.set_defaults(run=_trim)

    extraction = commands.add_parser(
        "extract", help="ask a running llama-server for the times in one message (docs/serving.md)"
    )
    extraction.add_argument("message")
    extraction.add_argument("--url", default=client.URL, help="the llama-server's address")
    extraction.set_defaults(run=_extract)

    args = parser.parse_args()
    args.run(args)


def _plan(args: argparse.Namespace) -> None:
    plans = make_plans(args.count, args.split, args.seed)
    folder = DATA / "plans" / args.split
    write_jsonl(folder / "plans.jsonl", (p.to_row() for p in plans))
    to_write = [p for p in plans if p.text is None]
    for number, start in enumerate(range(0, len(to_write), args.batch_size)):
        batch = to_write[start : start + args.batch_size]
        write_jsonl(folder / f"batch-{number:03d}.jsonl", (p.to_row() for p in batch))
    print(f"{len(plans)} plans, {len(to_write)} need a writer, in {folder}")


def _assemble(args: argparse.Namespace) -> None:
    plans = [Plan.from_row(row) for row in read_jsonl(DATA / "plans" / args.split / "plans.jsonl")]
    messages = {row["id"]: row["text"] for row in _rows(DATA / "messages" / args.split)}
    checks = {row["id"]: _checked_times(row) for row in _rows(DATA / "checks" / args.split)}
    held_out = tuple(e.text for path in args.hold_out for e in load_examples(path))
    examples, rejects = assemble(plans, messages, checks, held_out)
    write_jsonl(DATA / "generated" / f"{args.split}.jsonl", (e.to_row() for e in examples))
    write_jsonl(DATA / "generated" / f"{args.split}.rejects.jsonl", rejects)
    kept, dropped = Counter(e.slice for e in examples), Counter(r["slice"] for r in rejects)
    print(f"kept {len(examples)} of {len(plans)}")
    for name in sorted(kept | dropped):
        print(f"  {name:<9} kept {kept[name]:>5}  rejected {dropped[name]:>5}")


def _train(args: argparse.Namespace) -> None:
    summary = train(load_config(args.config))
    print("\n".join(f"  {key:<17}{value}" for key, value in summary.items()))


def _eval(args: argparse.Namespace) -> None:
    out = Path("results") / args.name
    llama = _llama_options(args) if args.engine == "llama" else None
    summary = evaluate(args.model, _examples(args.data), out, args.format, not args.no_spec, llama)
    _report(args.name, summary, out, args.data, args.group)


def _eval_remote(args: argparse.Namespace) -> None:
    from tempus import remote

    out = Path("results") / args.name
    key = os.environ[args.key_env] if args.key_env else None
    record = {"quant": args.quant, "device": args.device}
    summary = remote.evaluate(args.url, args.model, _examples(args.data), out, key=key, extra=args.extra, record=record)
    _report(args.name, summary, out, args.data, args.group)


def _llama_options(args: argparse.Namespace) -> Options:
    if args.gguf is None:
        raise SystemExit("--engine llama needs --gguf")
    if args.grammar and args.format != "compact":
        raise SystemExit("--grammar is only defined for --format compact")
    return Options(args.gguf, args.device, args.threads, args.grammar, args.llama_server)


def _trim(args: argparse.Namespace) -> None:
    summary = trim_model(args.model, _examples(args.data), args.out)
    print("\n".join(f"  {key:<14}{value}" for key, value in summary.items()))


def _rules(args: argparse.Namespace) -> None:
    """Not logged to MLflow here: copy results/<name> to the tracking machine to log it."""
    print(format_summary(rules.evaluate(_examples(args.data), Path("results") / args.name)))


def _tagger_train(args: argparse.Namespace) -> None:
    from tempus import tagger

    summary = tagger.train(args.name, args.model, _examples(args.data), args.seed, args.epochs, args.learning_rate)
    print("\n".join(f"  {key:<20}{value}" for key, value in summary.items()))


def _tagger_eval(args: argparse.Namespace) -> None:
    """Not logged to MLflow here: copy results/<name> to the tracking machine to log it."""
    from tempus import tagger

    out = Path("results") / args.name
    print(format_summary(tagger.evaluate(args.model, _examples(args.data), out, args.onnx, args.gpu)))


def _tagger_export(args: argparse.Namespace) -> None:
    from tempus import tagger

    for path in tagger.export(args.model):
        print(f"  {path}  {path.stat().st_size / 2**20:.2f} MiB")


def _classifier_train(args: argparse.Namespace) -> None:
    from tempus.classifier_train import train as train_classifier

    summary = train_classifier(args.name, args.model, args.data, args.seed, args.epochs, args.learning_rate)
    print("\n".join(f"  {key:<20}{value}" for key, value in summary.items()))


def _classifier_eval(args: argparse.Namespace) -> None:
    from tempus import classifier

    out = Path("results") / args.name
    summary = classifier.evaluate(args.model, _examples(args.data), out, args.onnx)
    if args.no_log:
        print(format_summary(summary))
    else:
        _report(args.name, summary, out, args.data)
    print(f"  candidates right: {summary['candidates']}")


def _classifier_export(args: argparse.Namespace) -> None:
    from tempus.classifier_export import export

    for path in export(args.model):
        print(f"  {path}  {path.stat().st_size / 2**20:.2f} MiB")


def _rescore(args: argparse.Namespace) -> None:
    out = Path("results") / args.name
    _report(args.name, rescore(_examples(args.data), out), out, args.data)


def _extract(args: argparse.Namespace) -> None:
    print(json.dumps(asdict(client.extract(args.message, args.url)), ensure_ascii=False, indent=1))


def _examples(paths: list[Path]) -> list[Example]:
    return [example for path in paths for example in load_examples(path)]


def _report(name: str, summary: dict, out: Path, data: list[Path], group: str | None = None) -> None:
    eval_set = "+".join(path.stem for path in data)  # e.g. "core+eval", or "eval2"
    tracking.log_eval(name, summary, out, eval_set, group)
    print(format_summary(summary))


def _checked_times(row: dict) -> tuple[Extraction, ...]:
    """Checkers give a 24-hour clock string; the minute is computed here, not by the agent."""
    times = []
    for item in row["times"]:
        hour, minute = item["clock"].split(":")
        times.append(Extraction(item["text"], int(hour) * 60 + int(minute)))
    return tuple(times)


def _rows(folder: Path) -> list[dict]:
    return [row for path in sorted(folder.glob("*.jsonl")) for row in read_jsonl(path)]
