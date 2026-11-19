---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
language:
- en
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- time-extraction
---

# tempus: Qwen3.5 2B finetuned to read clock times from chat messages

A 1.9B-parameter model that reads one English chat message and lists every clock time in it,
deciding from context whether a bare "7" means morning or evening. It is the most accurate
model of the project, and too slow for a 100 ms budget on CPU. For CPU or browser use, see the
230M model: https://huggingface.co/dsego/tempus-lfm2.5-230m

Code, data and the full results: https://github.com/dsegovia90/tempus_finetune

| Message | Reply |
|---|---|
| `dinner at 7 or lunch at noon` | `7=19:00`<br>`noon=12:00` |
| `got paged at 2, been up ever since` | `2=02:00` |
| `skip to 2:30 in the video, room 3 is booked` | `none` |

## Files

| File | Size |
|---|---|
| `qwen3.5-2b-v2-Q8_0.gguf` | 2.1 GB |

Text only: the base model's vision component is not included.

## Scores

Exact match per message: every time found, with the right span and the right 24-hour reading.
This is the best of three training runs, chosen by these same evals; the three-run mean is in
the right-hand column.

| Eval | This file (Q8_0) | Three-run mean |
|---|---|---|
| Main (530 messages) | 97.5% | 97.0% |
| Main, time of day implied by the situation only | 92.7% | 91.7% |
| Fresh (378 messages from situations not in training) | 99.7% | 98.9% |
| False alarms on 1,021 tricky messages with no clock time | 2 (0.2%) | 0.1% |

Latency: a median of 56 ms per message on an RTX 3090, and 1,105 ms on 4 cores of a shared
Ryzen 3600.

## Use

With [llama.cpp](https://github.com/ggml-org/llama.cpp) (tested at tag b12292), start
`llama-server --model qwen3.5-2b-v2-Q8_0.gguf` and send the message as a single user turn with
**no system prompt** and thinking turned off. The raw prompt is:

```
<|im_start|>user
{message}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

The reply is one `text=HH:MM` line per clock time, where `text` is copied from the message, or
the single word `none`. Use greedy decoding (temperature 0).

## Limits

- English only.
- Times written with digits, plus "noon" and "midnight". Spelled-out times are not supported.
- Durations, relative times and vague periods are ignored by design.
- Training and eval data are synthetic, written and checked by language models with a small
  human spot-check.

## Training

LoRA (rank 16) on [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) with Unsloth,
3 epochs over 11,299 synthetic messages, merged and converted to GGUF. Config:
`configs/qwen3.5-2b-v2-s0-compact.yaml` in the repository.

## Licence

Apache 2.0, the same as the base model.
