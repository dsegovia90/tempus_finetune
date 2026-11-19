"""A small encoder that classes every candidate clock expression in one forward pass.

The message is tokenized in pieces split at candidate edges, so every candidate starts on a
token of its own; the token-classification head's output at that first token gives the
candidate's class (see candidates.py). Code then assembles the compact reply.

Runs on the CPU with PyTorch (a saved run, runs/<name>/model) or ONNX Runtime (an exported
file). Needs the `classifier` dependency group.
"""

import statistics
import time
from collections import Counter
from pathlib import Path

from tempus import candidates
from tempus.candidates import CLASSES, Candidate
from tempus.evaluate import save, training_record
from tempus.schema import Example
from tempus.target import render

WARMUP = 5  # untimed messages before timing


def encode(tokenizer, message: str, found: list[Candidate]) -> tuple[list[int], list[int]]:
    """Token ids, with [CLS] and [SEP], and the position of each candidate's first token.

    A candidate's piece takes the spaces before it, as a byte-level BPE token would ("Ġ7").
    """
    pieces = []
    cursor = 0
    for candidate in found:
        start = len(message[cursor : candidate.start].rstrip()) + cursor
        pieces += [message[cursor:start], message[start : candidate.end]]
        cursor = candidate.end
    pieces.append(message[cursor:])
    ids, firsts = [tokenizer.cls_token_id], []
    for index, piece in enumerate(pieces):
        if index % 2:
            firsts.append(len(ids))
        ids += tokenizer(piece, add_special_tokens=False)["input_ids"] if piece.strip() else []
    return ids + [tokenizer.sep_token_id], firsts


class Classifier:
    """A trained classifier: PyTorch from `path`, or the ONNX file `onnx` with `path`'s tokenizer."""

    def __init__(self, path: Path, onnx: Path | None = None):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if onnx is None:
            import torch
            from transformers import AutoModelForTokenClassification

            torch.set_num_threads(4)
            self.model = AutoModelForTokenClassification.from_pretrained(path).eval()
            self._logits = self._torch_logits
        else:
            import onnxruntime

            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = 4
            self.session = onnxruntime.InferenceSession(str(onnx), options, providers=["CPUExecutionProvider"])
            self._logits = self._onnx_logits

    def classify(self, message: str) -> list[tuple[Candidate, list[float]]]:
        """Each candidate in the message with its class probabilities, in CLASSES order."""
        found = candidates.find(message)
        if not found:
            return []
        ids, firsts = encode(self.tokenizer, message, found)
        rows = _softmax([self._logits(ids)[i] for i in firsts])
        return list(zip(found, rows, strict=True))

    def reply(self, message: str) -> str:
        """The compact reply, as the finetuned LLM writes it."""
        scored = self.classify(message)
        classes = [CLASSES[max(range(len(CLASSES)), key=p.__getitem__)] for _, p in scored]
        return render(candidates.assemble([c for c, _ in scored], classes), "compact")

    def _torch_logits(self, ids: list[int]) -> list[list[float]]:
        import torch

        with torch.inference_mode():
            return self.model(input_ids=torch.tensor([ids])).logits[0].tolist()

    def _onnx_logits(self, ids: list[int]) -> list[list[float]]:
        import numpy as np

        feed = {"input_ids": np.array([ids], dtype=np.int64), "attention_mask": np.ones((1, len(ids)), dtype=np.int64)}
        return self.session.run(["logits"], feed)[0][0].tolist()


def evaluate(path: Path, examples: list[Example], out: Path, onnx: Path | None = None) -> dict:
    """Answer every message one at a time on the CPU, save results under `out`, return the summary."""
    classifier = Classifier(path, onnx)
    for example in examples[:WARMUP]:
        classifier.reply(example.text)
    replies, seconds = [], []
    for example in examples:
        started = time.perf_counter()
        replies.append(classifier.reply(example.text))
        seconds.append(time.perf_counter() - started)
    timed = sorted(seconds)
    trained = training_record(str(path))
    weights = onnx or path / "model.safetensors"
    run = {
        "model": str(onnx or path),
        "base_model": trained.get("model", str(path)),
        "train_run_id": trained.get("mlflow_run_id"),
        "train_experiment_id": trained.get("mlflow_experiment_id"),
        "engine": "classifier",
        "quant": onnx.stem if onnx else "fp32",
        "device": "cpu",
        "format": "compact",
        "with_spec": False,
        "seconds": round(sum(seconds), 1),
        "latency_ms_p50": round(statistics.median(timed) * 1000, 2),
        "latency_ms_p95": round(timed[int(len(timed) * 0.95) - 1] * 1000, 2),
        "parameters_millions": trained.get("parameters_millions"),
        "weights_gb": round(weights.stat().st_size / 1e9, 4),
        "candidates": candidate_accuracy(classifier, examples),
    }
    return save(run, examples, replies, out)


def candidate_accuracy(classifier: Classifier, examples: list[Example]) -> dict:
    """Share of candidates given their labelled class, overall and by labelled class."""
    right, total = Counter(), Counter()
    for example in examples:
        _, wanted, _ = candidates.label(example)
        for (_, probs), label in zip(classifier.classify(example.text), wanted, strict=True):
            total[label] += 1
            right[label] += CLASSES[probs.index(max(probs))] == label
    return {
        "all": round(sum(right.values()) / sum(total.values()), 4),
        **{label: round(right[label] / total[label], 4) for label in CLASSES if total[label]},
    }


def _softmax(rows: list[list[float]]) -> list[list[float]]:
    import math

    result = []
    for row in rows:
        top = max(row)
        exps = [math.exp(value - top) for value in row]
        result.append([value / sum(exps) for value in exps])
    return result
