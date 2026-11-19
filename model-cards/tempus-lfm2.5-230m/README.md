---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-230M
language:
- en
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- time-extraction
---

# tempus: LFM2.5-230M finetuned to read clock times from chat messages

A 230M-parameter model that reads one English chat message and lists every clock time in it,
deciding from context whether a bare "7" means morning or evening. It answers in about 100 ms
on a 4-core CPU with no GPU, and runs in a browser tab.

Code, data and the full results: https://github.com/dsegovia90/tempus_finetune

| Message | Reply |
|---|---|
| `dinner at 7 or lunch at noon` | `7=19:00`<br>`noon=12:00` |
| `got paged at 2, been up ever since` | `2=02:00` |
| `skip to 2:30 in the video, room 3 is booked` | `none` |

## Files

| File | Size | Use |
|---|---|---|
| `lfm2.5-230m-v2-Q8_0.gguf` | 247 MB | Recommended. Same scores as the unquantized model |
| `lfm2.5-230m-v2-Q4_K_M.gguf` | 153 MB | Smaller and faster, about a point less accurate on the main eval |

## Scores

Exact match per message: every time found, with the right span and the right 24-hour reading.
This is the best of three training runs, chosen by these same evals; the three-run mean is in
the right-hand column.

| Eval | Q8_0 | Q4_K_M | Three-run mean (Q8_0) |
|---|---|---|---|
| Main (530 messages) | 97.0% | 95.7% | 96.5% |
| Main, time of day implied by the situation only | 92.1% | 89.3% | 91.0% |
| Fresh (378 messages from situations not in training) | 97.4% | 97.4% | 96.8% |
| False alarms on 1,021 tricky messages with no clock time | 0 | 0 | 0.2% |
| CPU latency, median, 4 cores of a shared Ryzen 3600 | 99 ms | 72 ms | 98 ms (two runs) |

## Use

With [llama.cpp](https://github.com/ggml-org/llama.cpp) (tested at tag b12292):

```
llama-server --model lfm2.5-230m-v2-Q8_0.gguf --host 127.0.0.1 --port 8080 \
    --threads 4 --threads-batch 4 --n-gpu-layers 0 --device none \
    --parallel 1 --ctx-size 1024 --cache-ram 0 --no-webui
```

Send the message as a single user turn with **no system prompt**; the model was trained that
way. The raw prompt is:

```
<|im_start|>user
{message}<|im_end|>
<|im_start|>assistant
```

llama.cpp adds the start-of-text token in front of it.

The reply is one `text=HH:MM` line per clock time, where `text` is copied from the message, or
the single word `none`. Use greedy decoding (temperature 0). A small Python client that parses
the reply is in the repository (`tempus.client.extract`).

## Limits

- English only.
- Times written with digits, plus "noon" and "midnight". Spelled-out times ("dinner at seven")
  are not supported.
- Durations, relative times ("in 2 hours") and vague periods ("this afternoon") are ignored by
  design.
- Inferring morning or evening in unfamiliar situations is the weakest area (about 92%).
- It sometimes drops a time that a larger model keeps: for "dinner at 7? or lunch at noon" it
  returns only the 7, and for "skip to 2:30 in the video, dinner at 7?" it returns nothing.
  The 0.8B model gets both right: https://huggingface.co/dsego/tempus-qwen3.5-0.8b
- Training and eval data are synthetic, written and checked by language models with a small
  human spot-check.

## Training

LoRA (rank 16) on [LiquidAI/LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M) with
Unsloth, 3 epochs over 11,299 synthetic messages, merged and converted to GGUF. Config:
`configs/lfm2.5-230m-v2-s1-compact.yaml` in the repository.

## Licence

This is a derivative of LFM2.5-230M and is distributed under the
[LFM Open License v1.0](LICENSE), copyright Liquid AI, Inc. That licence permits commercial
use only for organisations with under $10 million in annual revenue, and the limit applies to
anyone using this finetune. Read the licence before using it commercially.
