# Serving the time reader on a CPU server

One finetuned model behind llama.cpp's `llama-server`, called over HTTP by a small Python client
that turns its reply into structured times.

## The model file

**LFM2.5-230M finetuned on training data v2, run s1, at Q8_0**: one 247 MB GGUF file,
`lfm2.5-230m-v2-Q8_0.gguf` from https://huggingface.co/dsego/tempus-lfm2.5-230m (or
`runs/lfm2.5-230m-v2-s1-compact/gguf/Q8_0.gguf` from your own training run). To download it:

```
uvx hf download dsego/tempus-lfm2.5-230m lfm2.5-230m-v2-Q8_0.gguf --local-dir .
```

- **Why run s1:** of the three training runs it is best on every eval set: 97.0% exact on the
  main eval, 97.4% on the fresh eval, and no false alarm on the 1,021 hard no-time messages.
- **Why Q8_0:** it gives the same scores as the unquantized bf16 file. Q4_K_M is faster (72 ms
  instead of 99 ms) but loses accuracy for this model (1.3 points on the main eval for this
  run, 0.4 for run s0); bf16 is twice the size for nothing.

## Install

llama.cpp, tested at tag **b12292**. Prebuilt binaries are on the release page
(`llama-b12292-bin-<os>-<arch>.tar.gz` at https://github.com/ggml-org/llama.cpp/releases/tag/b12292;
the macOS arm64 build was used for the checks below), or build `llama-server` from source at
that tag. Nothing else: no GPU, no Python on the server side.

## Start the server

```
llama-server --model lfm2.5-230m-v2-Q8_0.gguf --host 127.0.0.1 --port 8080 \
    --threads 4 --threads-batch 4 --n-gpu-layers 0 --device none \
    --parallel 1 --ctx-size 1024 --cache-ram 0 --no-webui
```

- `--threads 4 --threads-batch 4`: the latency below needs four cores (237 ms on one, 130 ms on
  two). Use the number of physical cores, at least four.
- `--n-gpu-layers 0 --device none`: CPU only, even where a GPU exists.
- `--parallel 1 --ctx-size 1024`: one request at a time; concurrent requests queue. Prompts are
  20 to 40 tokens and replies at most 160.
- `--cache-ram 0`: by default llama-server keeps every finished prompt's state in a RAM cache of
  up to 8 GiB, which the client never reuses (it sends `cache_prompt: false`). On a Mac, the
  server's memory grew from 0.6 to 1.6 GB over 5,800 requests with the default and stayed at
  0.6 GB with this flag; replies and latency were identical.
- `--no-webui`, `--host 127.0.0.1`: an internal service, not a public endpoint.

These are the settings of the eval harness (`server_command` in `src/tempus/llama.py`), plus
`--cache-ram 0`.

## Call it

```python
from tempus.client import extract

result = extract("dinner at 7 or lunch at noon")  # url="http://127.0.0.1:8080" by default
result.times      # [Time(text='7', time='19:00', minute=1140), Time(text='noon', time='12:00', minute=720)]
result.malformed  # False
result.reply      # '7=19:00\nnoon=12:00', the model's raw reply
```

`times` is empty when the message has no time (`reply == "none"`, `malformed` False) and also
when the reply cannot be trusted: off-format, cut at the 160-token cap, or naming text that is
not in the message. Then `malformed` is True and `reply` shows what came back; no time is ever
guessed from such a reply. HTTP errors, connection errors and timeouts raise
(`urllib.error.URLError`, `TimeoutError`), so a server outage is never read as "no times". In
1,929 eval replies on the Mac, none was malformed.

From a terminal: `uv run tempus extract "dinner at 7?"` (`--url` for another server) prints the
result as JSON.

**Runtime dependencies:** Python 3.11+ and its standard library. `tempus.client` imports only
`tempus.llama` (the request body and response parsing the evals use) and `tempus.target` (the
reply parser), both standard-library only; installing the `tempus` package also pulls in
`pyyaml`, which the client does not import. No tokenizer is needed.

### The prompt is the training prompt

The client sends the model's chat template as text, `<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n`,
with no system prompt, and lets llama.cpp tokenize it: special tokens are parsed and the GGUF
(`add_bos_token = true`) adds the one BOS, `<|startoftext|>`. The template inserts the message
unchanged (no trimming), so the text is exactly the template's output minus the leading BOS.

`scripts/check_text_prompt.py` compares, for every message, the ids llama-server makes of this
text with the Hugging Face tokenizer's ids of the rendered template (what training and the evals
used). With b12292 they were identical for all 13,233 messages: the main, fresh and hard-negative
eval sets, the whole v2 training set, and edge cases (surrounding whitespace, special-token text,
an empty message). The replies agreed as well: sending the text or the ids gave the same reply
and the same prompt length on all 1,929 eval messages, on a Mac CPU. Against the reference replies
from llama.cpp on a CUDA GPU, 1,927 of 1,929 were identical; the two others are am/pm near-ties
on the fresh set that flip with the backend's arithmetic.

## Measured

Latency per message, Q8_0, four threads, on 4 virtual cores of a shared Ryzen 3600: **99 ms
median, 202 ms at the 95th percentile** for this file (96 ms and 169 ms for run s0, which has
the same architecture and size). With fewer threads, measured on the earlier version 1 finetune
of the same model: 237 ms on one thread, 130 ms on two.
On an Apple M3 Pro (four threads, client to reply) 33 ms median, 66 ms p95.

| | Main eval (530) | Fresh eval (378) | False alarms, hard set (1,021) |
|---|---|---|---|
| Three-run mean, all / implicit | 96.5% / 91.0% | 96.8% / 92.0% | 0.2% |
| Run s1 (this file), all / implicit | 97.0% / 92.1% | 97.4% / 93.3% | 0.0% |

Exact match: every time in the message found, with the right text and minute, and nothing else.

## Known limits

- English only; digit times ("7", "9:30", "5pm", "14:00") plus "noon" and "midnight".
  Spelled-out times are out of scope: "dinner at seven", "half past six" and "five thirty" came
  back as no time.
- The weakest case is an implicit am/pm ("dinner at 7"): about 91 to 92% exact.
- Messages are tokenized with special tokens parsed, as in training, so text such as
  `<|im_end|>` inside a message acts as a control token. Any time the model then reports must
  still appear in the message, or the result is flagged malformed.
- Training messages are at most 277 characters (median 82). Longer ones are untested for
  accuracy. Past about 1,000 tokens (roughly 4,500 characters) a message no longer fits the
  1,024-token context: llama-server answers HTTP 400 and the client raises.

## Licence

The base model, `LiquidAI/LFM2.5-230M`, is under the **LFM Open License v1.0**
(https://docs.liquid.ai/lfm/help/model-license). A finetune may be redistributed if the licence
and its notices are kept with it. Commercial use is allowed only for organisations with under
$10 million annual revenue, and that condition passes to everyone who downloads the finetune.
