"""Run a chat model over messages with plain transformers and greedy decoding.

The same code scores base models and merged finetunes, so the two are compared
under identical inference settings. Needs the `gpu` dependency group.
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tempus.chat import MAX_NEW_TOKENS, chat_prompt

WARMUP_CALLS = 3


class Transformers:
    """A model loaded in bf16 on the GPU."""

    def __init__(self, model_id: str):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = "left"  # decoder-only models must be padded on the left to batch
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
        self.model, self.tokenizer = model.eval(), tokenizer

    def __enter__(self) -> "Transformers":
        return self

    def __exit__(self, *exc) -> None:
        pass

    def generate(self, system: str | None, messages: list[str], batch_size: int = 16) -> list[str]:
        """Return the model's raw reply to each message."""
        replies = []
        for start in range(0, len(messages), batch_size):
            prompts = [chat_prompt(self.tokenizer, system, m) for m in messages[start : start + batch_size]]
            batch = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            batch = batch.to(self.model.device)
            with torch.inference_mode():
                output = self.model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            new_tokens = output[:, batch["input_ids"].shape[1] :]
            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            replies += [r.strip() for r in decoded]
        return replies

    def latencies(self, system: str | None, messages: list[str]) -> list[float]:
        """Seconds to answer each message alone, end to end, as a live request would be served."""
        for message in messages[:WARMUP_CALLS]:  # the first calls pay one-off compilation costs
            self.generate(system, [message])
        seconds = []
        for message in messages:
            started = time.perf_counter()
            self.generate(system, [message])
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - started)
        return seconds

    def details(self) -> dict:
        """Engine settings and model size, plus the most GPU memory used so far in this process."""
        return {
            "engine": "transformers",
            "quant": "bf16",
            "device": "gpu",
            "parameters_millions": round(sum(p.numel() for p in self.model.parameters()) / 1e6, 1),
            "weights_gb": round(self.model.get_memory_footprint() / 2**30, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
