"""Finetune one model with Unsloth in bf16 LoRA and save the adapter and a merged model.

A run is described by a small YAML file under configs/; the run is named after the file.
Settings held constant across the model ladder are the constants below. Needs the `gpu`
dependency group.
"""

import json
import random
import time
from collections import Counter
from dataclasses import MISSING, dataclass, fields
from pathlib import Path

import yaml

from tempus import tracking
from tempus.chat import end_of_turn, prompt_ids
from tempus.schema import Example, load_examples
from tempus.target import render

RUNS = Path("runs")
LORA_RANK = 16
LORA_ALPHA = 16
BATCH_SIZE = 8
MAX_SEQ_LENGTH = 512
VAL_FRACTION = 0.1
WARMUP = 0.1  # share of all steps
EVALS_PER_RUN = 4  # each validation pass costs about 10 s on Qwen3.5 0.8B
LOG_EVERY = 10  # steps; logging every step cost about 6% of training time


@dataclass(frozen=True)
class RunConfig:
    name: str
    model: str  # Hugging Face ID
    loader: str  # Unsloth class: FastModel (Gemma) or FastLanguageModel (the others)
    data: Path  # examples; a validation split is carved from them
    format: str  # output format the model learns: json or compact (see target.py)
    learning_rate: float
    epochs: int
    seed: int
    target_modules: tuple[str, ...] = ()  # LoRA layer names; none: every linear layer (see lora_targets)


def load_config(path: Path) -> RunConfig:
    """Read a run config. Fields with a default are optional; `name` is the file name."""
    raw = {"name": path.stem, **yaml.safe_load(path.read_text())}
    required = {f.name for f in fields(RunConfig) if f.default is MISSING}
    known = {f.name for f in fields(RunConfig)}
    if not required <= set(raw) <= known:
        raise ValueError(f"{path}: expected keys {sorted(required)}, optionally {sorted(known - required)}, got {sorted(raw)}")
    return RunConfig(**{f.name: f.type(raw[f.name]) for f in fields(RunConfig) if f.name in raw})


def lora_targets(names: tuple[str, ...]) -> dict:
    """Unsloth's LoRA target arguments: the named layers in every block, or with none named every
    linear layer of the language model.

    Named layers are passed with every `finetune_*` filter on: with one off, Unsloth keeps only
    names under its attention and MLP module tags, which drops LFM2.5's conv blocks.
    """
    filters = {"finetune_language_layers": True, "finetune_attention_modules": True, "finetune_mlp_modules": True}
    if names:
        return {"target_modules": list(names), "finetune_vision_layers": True, **filters}
    return {"target_modules": None, "finetune_vision_layers": False, **filters}


def split(examples: list[Example], seed: int) -> tuple[list[Example], list[Example]]:
    """Shuffle deterministically and hold out VAL_FRACTION (at least one) for validation."""
    shuffled = random.Random(seed).sample(examples, len(examples))
    held_out = max(1, round(len(examples) * VAL_FRACTION))
    return shuffled[held_out:], shuffled[:held_out]


def token_row(tokenizer, example: Example, fmt: str, turn_end: str) -> dict:
    """One tokenized row: the exact inference prompt (`chat.prompt_ids`), then the reply and end of
    turn, which alone carry loss.

    Tokenized here, not by the trainer: it would split prompt from reply by the length of the
    templated prompt, which is wrong when the template renders a finished turn differently.
    """
    prompt = prompt_ids(tokenizer, None, example.text)
    reply = tokenizer(render(example.times, fmt) + turn_end, add_special_tokens=False)["input_ids"]
    return {"input_ids": prompt + reply, "completion_mask": [0] * len(prompt) + [1] * len(reply)}


def train(config: RunConfig) -> dict:
    """Train, log to MLflow, save under runs/<name>/, and return the run's summary."""
    import unsloth  # must be imported before transformers and trl
    import torch
    from datasets import Dataset
    from huggingface_hub import snapshot_download
    from trl import SFTConfig, SFTTrainer

    out = RUNS / config.name
    train_set, val_set = split(load_examples(config.data), config.seed)
    loader = getattr(unsloth, config.loader)
    model, tokenizer = loader.from_pretrained(
        # A local path: given a Hub ID, Unsloth may load, and merge onto, its own copy instead.
        snapshot_download(config.model),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_16bit=True,
        text_only=True,  # build only the language model of a multimodal checkpoint
        random_state=config.seed,
    )
    model = loader.get_peft_model(
        model, r=LORA_RANK, lora_alpha=LORA_ALPHA, random_state=config.seed, **lora_targets(config.target_modules)
    )
    lora_layers = dict(Counter(name.split(".")[-2] for name, _ in model.named_modules() if name.endswith("lora_A")))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA on {lora_layers}, {trainable:,} trainable parameters")
    turn_end = end_of_turn(tokenizer)
    train_rows, val_rows = (
        Dataset.from_list([token_row(tokenizer, e, config.format, turn_end) for e in examples])
        for examples in (train_set, val_set)
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_rows,
        eval_dataset=val_rows,
        callbacks=[tracking.metrics_callback()],
        args=SFTConfig(
            output_dir=str(out),  # nothing is written here: checkpoints are off
            completion_only_loss=True,
            max_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=1,  # Unsloth's default is 2
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=WARMUP,
            optim="adamw_torch",  # Unsloth's default keeps 8-bit optimizer state
            weight_decay=0.001,  # Unsloth's default, set so it cannot drift
            bf16=True,
            seed=config.seed,
            data_seed=config.seed,
            logging_steps=LOG_EVERY,
            eval_strategy="steps",
            eval_steps=1 / EVALS_PER_RUN,
            save_strategy="no",
            report_to="none",
        ),
    )
    sample = _check_loss_mask(trainer, train_set, config.format, turn_end)
    print(f"Longest row, loss on the part after the prompt: {sample!r}")

    params = {
        **vars(config),
        "lora_layers": lora_layers,
        "trainable_parameters": trainable,
        "train_examples": len(train_set),
        "val_examples": len(val_set),
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "batch_size": BATCH_SIZE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "warmup": WARMUP,
        "log_every": LOG_EVERY,
        "evals_per_run": EVALS_PER_RUN,
    }
    with tracking.train_run(config.name, params) as mlflow_ids:
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        trainer.train()
        losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
        summary = {
            "train_seconds": round(time.monotonic() - started, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "first_train_loss": losses[0],
            "final_train_loss": losses[-1],
            "final_val_loss": trainer.evaluate()["eval_loss"],
        }
        tracking.log_metrics(summary)
    record = {**params, **summary, **mlflow_ids, "data": str(config.data)}
    (out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    model.save_pretrained_merged(out / "merged", tokenizer, save_method="merged_16bit")
    return summary


def _check_loss_mask(trainer, examples: list[Example], fmt: str, turn_end: str) -> str:
    """Fail unless every training row is the inference prompt, token for token, without loss,
    then exactly the reply and the end of turn with loss, and the model stops at that end of turn.

    Returns the longest row as text.
    """
    tokenizer = trainer.processing_class
    stop = trainer.model.generation_config.eos_token_id
    # Qwen3.5 declares no generation eos; its tokenizer's eos is the end-of-turn token.
    stop = {*(stop if isinstance(stop, list) else [stop]), tokenizer.eos_token_id} - {None}
    texts = []
    for row, example in zip(trainer.train_dataset, examples, strict=True):
        batch = trainer.data_collator([row])
        ids, labels = batch["input_ids"][0].tolist(), batch["labels"][0].tolist()
        prompt, reply = prompt_ids(tokenizer, None, example.text), render(example.times, fmt)
        trained = tokenizer.decode(ids[len(prompt):])
        if ids[: len(prompt)] != prompt or set(labels[: len(prompt)]) != {-100}:
            raise RuntimeError(f"loss mask is wrong: the prompt of {example.id} differs from chat.prompt_ids")
        if labels[len(prompt):] != ids[len(prompt):] or trained != reply + turn_end or ids[-1] not in stop:
            raise RuntimeError(f"loss mask is wrong: {trained!r} is not {reply!r} + end of turn {turn_end!r}")
        texts.append(tokenizer.decode(ids))
    return max(texts, key=len)
