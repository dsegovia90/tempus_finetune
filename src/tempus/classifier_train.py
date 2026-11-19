"""Train the candidate classifier (classifier.py) on labelled examples.

Each candidate's class comes from the labels themselves (`candidates.label`), so no model is
needed to make the data. A validation split is carved as `train.split` does; the epoch with the
best exact match on it is kept. Writes runs/<name>/model and runs/<name>/run.json and logs to
the MLflow `train` experiment. Needs the `classifier` group, and mlflow (the `gpu` group).
"""

import json
import random
import time
from pathlib import Path

from tempus import candidates, tracking
from tempus.candidates import CLASSES
from tempus.classifier import encode
from tempus.schema import Example, load_examples
from tempus.target import render
from tempus.train import RUNS, split

BATCH_SIZE = 32
WARMUP = 0.1  # share of all steps
IGNORE = -100  # label of tokens that are not a candidate's first token


def train(name: str, model_id: str, data: list[Path], seed: int, epochs: int, learning_rate: float) -> dict:
    """Train, keep the best epoch by validation exact match, save it, return the run's summary."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = RUNS / name
    train_set, val_set = split([e for path in data for e in load_examples(path)], seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        num_labels=len(CLASSES),
        id2label=dict(enumerate(CLASSES)),
        label2id={label: i for i, label in enumerate(CLASSES)},
    ).to(device)
    rows = [_row(tokenizer, example) for example in train_set]
    steps = epochs * -(-len(rows) // BATCH_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    schedule = get_linear_schedule_with_warmup(optimizer, int(steps * WARMUP), steps)
    parameters = sum(p.numel() for p in model.parameters())
    params = {
        "model": model_id, "data": ",".join(map(str, data)), "seed": seed, "epochs": epochs,
        "learning_rate": learning_rate, "batch_size": BATCH_SIZE, "train_examples": len(rows),
        "val_examples": len(val_set), "parameters_millions": round(parameters / 1e6, 2),
    }
    started = time.monotonic()
    best = (-1.0, None)
    with tracking.train_run(name, params) as ids:
        for epoch in range(epochs):
            model.train()
            random.Random(seed * 1000 + epoch).shuffle(rows)
            for start in range(0, len(rows), BATCH_SIZE):
                ids_batch, labels = _batch(rows[start : start + BATCH_SIZE], tokenizer.pad_token_id, device)
                loss = model(input_ids=ids_batch, attention_mask=(ids_batch != tokenizer.pad_token_id).long(), labels=labels).loss
                loss.backward()
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
            exact = _val_exact(model, tokenizer, val_set, device)
            tracking.log_metrics({"train_loss": loss.item(), "val_exact": exact}, step=epoch + 1)
            if exact > best[0]:
                best = (exact, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        model.load_state_dict(best[1])
        model.save_pretrained(out / "model")
        tokenizer.save_pretrained(out / "model")
        summary = {**params, "val_exact": best[0], "minutes": round((time.monotonic() - started) / 60, 2), **ids}
        tracking.log_metrics({"best_val_exact": best[0]})
    (out / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _row(tokenizer, example: Example) -> tuple[list[int], list[int]]:
    """Token ids and per-token labels: each candidate's class on its first token."""
    found, classes, problems = candidates.label(example)
    if problems:
        raise ValueError(f"{example.id}: {problems}")
    ids, firsts = encode(tokenizer, example.text, found)
    labels = [IGNORE] * len(ids)
    for position, label in zip(firsts, classes, strict=True):
        labels[position] = CLASSES.index(label)
    return ids, labels


def _batch(rows: list[tuple[list[int], list[int]]], pad: int, device: str):
    import torch

    width = max(len(ids) for ids, _ in rows)
    ids = [row + [pad] * (width - len(row)) for row, _ in rows]
    labels = [row + [IGNORE] * (width - len(row)) for _, row in rows]
    return torch.tensor(ids, device=device), torch.tensor(labels, device=device)


def _val_exact(model, tokenizer, examples: list[Example], device: str) -> float:
    """Share of validation messages whose assembled reply equals the label."""
    import torch

    model.eval()
    right = 0
    with torch.inference_mode():
        for example in examples:
            found = candidates.find(example.text)
            classes = []
            if found:
                ids, firsts = encode(tokenizer, example.text, found)
                logits = model(input_ids=torch.tensor([ids], device=device)).logits[0]
                classes = [CLASSES[int(logits[i].argmax())] for i in firsts]
            reply = render(candidates.assemble(found, classes), "compact")
            right += reply == render(example.times, "compact")
    return right / len(examples)
