---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
language:
- en
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- time-extraction
---

# tempus: Qwen3.5 0.8B finetuned to read clock times from chat messages

A 0.8B-parameter model that reads one English chat message and lists every clock time in it,
deciding from context whether a bare "7" means morning or evening. It is the middle option of
the project: more accurate on unfamiliar situations than the 230M model, a quarter of the size
of the 2B model, and small enough to run in a browser tab. It is too slow for a 100 ms budget
on CPU; for that, see https://huggingface.co/dsego/tempus-lfm2.5-230m

Code, data and the full results: https://github.com/dsegovia90/tempus_finetune

| Message | Reply |
|---|---|
| `dinner at 7 or lunch at noon` | `7=19:00`<br>`noon=12:00` |
| `got paged at 2, been up ever since` | `2=02:00` |
| `skip to 2:30 in the video, room 3 is booked` | `none` |

## Files

| File | Size | Use |
|---|---|---|
| `qwen3.5-0.8b-v2-Q4_K_M.gguf` | 542 MB | Recommended. Within one message of Q8_0 on all three evals |
| `qwen3.5-0.8b-v2-Q8_0.gguf` | 834 MB | 8-bit weights |
| `qwen3.5-0.8b-v2-trim-Q4_K_M.gguf` | 448 MB | The Q4_K_M model with its vocabulary cut to the tokens English chat needs, for a smaller browser download |

Text only: the base model's vision component is not included.

## Scores

Exact match per message: every time found, with the right span and the right 24-hour reading.
This is the best of three training runs, chosen by these same evals; the three-run mean is in
the right-hand column.

| Eval | Q4_K_M | Q8_0 | Three-run mean (Q8_0) |
|---|---|---|---|
| Main (530 messages) | 97.0% | 97.2% | 96.7% |
| Main, time of day implied by the situation only | 91.5% | 92.1% | 90.6% |
| Fresh (378 messages from situations not in training) | 98.4% | 98.4% | 98.1% |
| Fresh, time of day implied by the situation only | 95.5% | 95.5% | 94.8% |
| Wrong replies on 1,021 tricky messages with no clock time | 7 (0.7%) | 8 (0.8%) | 0.6% |
| CPU latency, median, 4 cores of a shared Ryzen 3600 | 340 ms | 421 ms | 412 ms |
| GPU latency, median, RTX 3090 | 39 ms | 40 ms | 41 ms |

About half of the wrong replies on the no-time messages are not valid times ("35=35:00" for
"it's 35 degrees out"), so a client that checks the reply, as the repository's does, discards
them.

The trimmed file scores 97.0% on the main eval, with every reply identical to the untrimmed
Q4_K_M file's, 98.1% on the fresh eval and 5 wrong replies (0.5%) on the no-time messages. It
differs from the untrimmed file on 5 of the other 1,399 messages; those two runs were also on
different processors (CPU and GPU), which flips near-ties by itself.

## Use

With [llama.cpp](https://github.com/ggml-org/llama.cpp) (tested at tag b12292), start
`llama-server --model qwen3.5-0.8b-v2-Q4_K_M.gguf` and send the message as a single user turn
with **no system prompt** and thinking turned off. The raw prompt is:

```
<|im_start|>user
{message}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

The reply is one `text=HH:MM` line per clock time, where `text` is copied from the message, or
the single word `none`. Use greedy decoding (temperature 0).

In a browser through llama.cpp's WebAssembly build, this model's saved prompt states are large
and fill the memory after about 80 replies; the repository's demo restarts the worker every 16
replies to avoid that.

## Limits

- English only.
- Times written with digits, plus "noon" and "midnight". Spelled-out times are not supported.
- Durations, relative times and vague periods are ignored by design.
- Training and eval data are synthetic, written and checked by language models with a small
  human spot-check.

## Training

LoRA (rank 16) on [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) with Unsloth,
3 epochs over 11,299 synthetic messages, merged and converted to GGUF. Config:
`configs/qwen3.5-0.8b-v2-s0-compact.yaml` in the repository.

## Licence

Apache 2.0, the same as the base model.
