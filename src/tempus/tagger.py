"""A small encoder that does the whole task itself: finds each clock time and reads it.

Unlike the candidate classifier (classifier.py), no code finds numbers or reads digits. The
encoder tags every token as outside a time, beginning one or inside one, and at each beginning
token predicts the hour (0-23) and the minute in 5-minute steps (0, 5, ... 55). Code only turns
the tags back into spans of the message and prints the reply, as it parses a generative model's
reply.

The 5-minute step is a chosen limit: every time in the training data falls on one, and a
classification head cannot name a value it never saw, so "9:17" is read as a nearby step.

Training labels come from the dataset: each labelled span is located in the message, in order.
Needs the `classifier` dependency group (torch, transformers).
"""

import json
import random
import statistics
import time
from pathlib import Path

from tempus.evaluate import save
from tempus.schema import Example, Extraction
from tempus.target import render
from tempus.train import RUNS, split

OUTSIDE, BEGIN, INSIDE = 0, 1, 2
IGNORE = -100  # hour and minute labels of tokens that do not begin a time
MINUTE_STEP = 5  # the minute head has one class per step
HEADS = ("tag", "hour", "minute")  # the model's three per-token outputs, in order
BATCH_SIZE = 32
WARMUP = 0.1  # share of all steps
WARMUP_MESSAGES = 5  # untimed messages before timing


def spans(example: Example) -> list[tuple[int, int, int]]:
    """(start, end, minute) of each labelled time, located in the message in order.

    A span must not sit inside a longer number ("3" is not found in "30"), so the search skips
    occurrences with a digit on either side.
    """
    found, cursor = [], 0
    for item in example.times:
        start = example.text.find(item.text, cursor)
        while start != -1 and (_digit(example.text, start - 1) or _digit(example.text, start + len(item.text))):
            start = example.text.find(item.text, start + 1)
        if start == -1:
            raise ValueError(f"{example.id}: {item.text!r} not found after position {cursor}")
        found.append((start, start + len(item.text), item.minute))
        cursor = start + len(item.text)
    return found


def _digit(text: str, index: int) -> bool:
    return 0 <= index < len(text) and text[index].isdigit()


def encode(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Token ids with [CLS] and [SEP], and each token's character range in the message."""
    encoded = tokenizer(text, return_offsets_mapping=True)
    return encoded["input_ids"], [tuple(pair) for pair in encoded["offset_mapping"]]


def labels(example: Example, offsets: list[tuple[int, int]]) -> tuple[list[int], list[int], list[int]]:
    """Per token: its tag, and the hour and minute where it begins a time (IGNORE elsewhere).

    Raises when a labelled span does not start and end on token edges, or its minute is not on
    a MINUTE_STEP, which the labels cannot express.
    """
    tags = [OUTSIDE] * len(offsets)
    hours, minutes = [IGNORE] * len(offsets), [IGNORE] * len(offsets)
    for start, end, minute in spans(example):
        inside = [i for i, (a, b) in enumerate(offsets) if a < b and a >= start and b <= end]
        if not inside or offsets[inside[0]][0] != start or offsets[inside[-1]][1] != end:
            raise ValueError(f"{example.id}: {example.text[start:end]!r} does not fall on token edges")
        if minute % MINUTE_STEP:
            raise ValueError(f"{example.id}: {example.text[start:end]!r} is not on a {MINUTE_STEP}-minute step")
        for i in inside:
            tags[i] = INSIDE
        tags[inside[0]], hours[inside[0]], minutes[inside[0]] = BEGIN, minute // 60, minute % 60 // MINUTE_STEP
    return tags, hours, minutes


def decode(text: str, offsets, tags, hours, minutes) -> tuple[Extraction, ...]:
    """The times the tags describe: a span runs from a BEGIN token over the INSIDE tokens after it."""
    times, open_span = [], None
    for (start, end), tag, hour, minute in zip(offsets, tags, hours, minutes, strict=True):
        if start == end:  # [CLS], [SEP]
            continue
        if tag == BEGIN:
            open_span = [start, end, hour * 60 + minute * MINUTE_STEP]
            times.append(open_span)
        elif tag == INSIDE and open_span:
            open_span[1] = end
        else:
            open_span = None
    return tuple(Extraction(text[a:b], minute) for a, b, minute in times)


def build(encoder_id: str):
    """The encoder with its three per-token heads."""
    import torch
    from transformers import AutoModel

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(encoder_id)
            width = self.encoder.config.hidden_size
            self.tag = torch.nn.Linear(width, 3)
            self.hour = torch.nn.Linear(width, 24)
            self.minute = torch.nn.Linear(width, 60 // MINUTE_STEP)

        def forward(self, input_ids, attention_mask):
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            return self.tag(hidden), self.hour(hidden), self.minute(hidden)

    return Model()


class Tagger:
    """A trained tagger answering one message at a time: PyTorch from `path` (runs/<name>/model)
    on `device` ("cpu" or "cuda"), or the ONNX file `onnx` on the CPU with `path`'s tokenizer."""

    def __init__(self, path: Path, onnx: Path | None = None, threads: int = 4, device: str = "cpu"):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if onnx is None:
            import torch

            torch.set_num_threads(threads)
            self.device = device
            self.model = load(path).to(device)
            self._heads = self._torch_heads
        else:
            import onnxruntime

            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = threads
            self.session = onnxruntime.InferenceSession(str(onnx), options, providers=["CPUExecutionProvider"])
            self._heads = self._onnx_heads

    def tokens(self, message: str) -> tuple[list[int], list[tuple[int, int]], list[int], list[int], list[int]]:
        """Token ids, their character ranges, and per token the tag, hour and minute class."""
        ids, offsets = encode(self.tokenizer, message)
        return (ids, offsets, *self._heads(ids))

    def reply(self, message: str) -> str:
        _, offsets, tags, hours, minutes = self.tokens(message)
        return render(decode(message, offsets, tags, hours, minutes), "compact")

    def _torch_heads(self, ids: list[int]) -> list[list[int]]:
        import torch

        with torch.inference_mode():
            batch = torch.tensor([ids], device=self.device)
            return [head[0].argmax(-1).tolist() for head in self.model(batch, torch.ones_like(batch))]

    def _onnx_heads(self, ids: list[int]) -> list[list[int]]:
        import numpy as np

        feed = {"input_ids": np.array([ids], dtype=np.int64), "attention_mask": np.ones((1, len(ids)), dtype=np.int64)}
        return [head[0].argmax(-1).tolist() for head in self.session.run(list(HEADS), feed)]


def load(path: Path):
    """The saved encoder and heads, ready for inference."""
    import torch

    model = build(str(path)).eval()
    model.load_state_dict(torch.load(path / "heads.pt"), strict=False)
    return model


def export(model_dir: Path) -> list[Path]:
    """Write runs/<name>/onnx/: `fp32.onnx`, `fp16.onnx` (weights stored in 16 bits) and `int8.onnx`
    (dynamic 8-bit quantization), each with the outputs `tag`, `hour` and `minute`."""
    import onnx
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    from tempus.classifier_export import OPSET, _half_weights

    out = model_dir.parent / "onnx"
    out.mkdir(exist_ok=True)
    fp32, fp16, int8 = out / "fp32.onnx", out / "fp16.onnx", out / "int8.onnx"
    sample = torch.ones((1, 8), dtype=torch.long)
    axes = {0: "batch", 1: "tokens"}
    torch.onnx.export(
        load(model_dir), (sample, sample), fp32, input_names=["input_ids", "attention_mask"],
        output_names=list(HEADS), dynamic_axes={name: axes for name in ("input_ids", "attention_mask", *HEADS)},
        opset_version=OPSET, dynamo=False,
    )
    onnx.save(_half_weights(onnx.load(fp32)), fp16)
    quantize_dynamic(fp32, int8, weight_type=QuantType.QUInt8)
    return [fp32, fp16, int8]


def predict(model, tokenizer, message: str, device: str) -> tuple[Extraction, ...]:
    import torch

    ids, offsets = encode(tokenizer, message)
    with torch.inference_mode():
        batch = torch.tensor([ids], device=device)
        tag, hour, minute = (head[0].argmax(-1).tolist() for head in model(batch, torch.ones_like(batch)))
    return decode(message, offsets, tag, hour, minute)


def train(name: str, encoder_id: str, data: list[Example], seed: int, epochs: int, learning_rate: float) -> dict:
    """Train, keep the best epoch by validation exact match, save it under runs/<name>, return a summary."""
    import torch
    from torch.nn.functional import cross_entropy
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    train_set, val_set = split(data, seed)
    tokenizer = AutoTokenizer.from_pretrained(encoder_id)
    model = build(encoder_id).to(device)
    rows, skipped = [], 0
    for example in train_set:
        ids, offsets = encode(tokenizer, example.text)
        try:
            rows.append((ids, *labels(example, offsets)))
        except ValueError:
            skipped += 1
    steps = epochs * -(-len(rows) // BATCH_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    schedule = get_linear_schedule_with_warmup(optimizer, int(steps * WARMUP), steps)
    pad = tokenizer.pad_token_id
    started, best, curve = time.monotonic(), (-1.0, None), []
    for epoch in range(epochs):
        model.train()
        random.Random(seed * 1000 + epoch).shuffle(rows)
        for at in range(0, len(rows), BATCH_SIZE):
            batch = rows[at : at + BATCH_SIZE]
            width = max(len(row[0]) for row in batch)
            column = lambda index, fill: torch.tensor(
                [row[index] + [fill] * (width - len(row[index])) for row in batch], device=device
            )
            ids = column(0, pad)
            tag, hour, minute = model(ids, (ids != pad).long())
            loss = (
                cross_entropy(tag.flatten(0, 1), column(1, IGNORE).flatten(), ignore_index=IGNORE)
                + cross_entropy(hour.flatten(0, 1), column(2, IGNORE).flatten(), ignore_index=IGNORE)
                + cross_entropy(minute.flatten(0, 1), column(3, IGNORE).flatten(), ignore_index=IGNORE)
            )
            loss.backward()
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
        model.eval()
        exact = sum(
            render(predict(model, tokenizer, e.text, device), "compact") == render(e.times, "compact")
            for e in val_set
        ) / len(val_set)
        curve.append(round(exact, 4))
        if exact > best[0]:
            best = (exact, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    out = RUNS / name / "model"
    model.encoder.save_pretrained(out)
    tokenizer.save_pretrained(out)
    torch.save({k: v for k, v in best[1].items() if not k.startswith("encoder.")}, out / "heads.pt")
    summary = {
        "name": name, "model": encoder_id, "seed": seed, "epochs": epochs, "learning_rate": learning_rate,
        "batch_size": BATCH_SIZE, "train_examples": len(rows), "skipped_examples": skipped,
        "val_examples": len(val_set), "val_exact": best[0], "val_exact_by_epoch": curve,
        "parameters_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "minutes": round((time.monotonic() - started) / 60, 2), "device": device,
    }
    (RUNS / name / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def evaluate(path: Path, examples: list[Example], out: Path, onnx: Path | None = None, gpu: bool = False) -> dict:
    """Answer every message one at a time, save results under `out`, return the summary.

    On the CPU unless `gpu`, which runs the PyTorch model on CUDA (not the ONNX file).
    """
    tagger = Tagger(path, onnx, device="cuda" if gpu else "cpu")
    for example in examples[:WARMUP_MESSAGES]:
        tagger.reply(example.text)
    replies, seconds = [], []
    for example in examples:
        started = time.perf_counter()
        replies.append(tagger.reply(example.text))
        seconds.append(time.perf_counter() - started)
    timed = sorted(seconds)
    trained = json.loads((path.parent / "run.json").read_text())
    run = {
        "model": str(onnx or path),
        "base_model": trained["model"],
        "kind": "finetune",
        "engine": "tagger",
        "quant": onnx.stem if onnx else "fp32",
        "device": "gpu" if gpu else "cpu",
        "format": "compact",
        "with_spec": False,
        "seconds": round(sum(seconds), 1),
        "latency_ms_p50": round(statistics.median(timed) * 1000, 2),
        "latency_ms_p95": round(timed[int(len(timed) * 0.95) - 1] * 1000, 2),
        "parameters_millions": trained["parameters_millions"],
    }
    return save(run, examples, replies, out)
