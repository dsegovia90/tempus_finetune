# Browser demo and eval

A static page that runs a time reader in the browser, and an eval mode that checks its replies
against reference replies. Two engines, picked in the model drop-down:

- a finetuned LLM as a GGUF in [wllama](https://github.com/ngxson/wllama), checked against the
  native llama.cpp replies: LFM2.5-230M at Q8_0 (the default, and the model packaged for a CPU
  server in `docs/serving.md`) or at Q4_K_M (153 MB instead of 247 MB, about a point less
  accurate on the main eval), or Qwen3.5 0.8B at Q4_K_M (542 MB, more accurate on unfamiliar
  situations and slower), also offered with a trimmed vocabulary (448 MB);
- the tagger (`src/tempus/tagger.py`), which is not generative: a BERT-mini encoder (11M
  parameters, 11 MB) in ONNX Runtime Web tags every token as outside a time, beginning one or
  continuing one, and gives the hour and 5-minute step where a time begins. Code only turns
  the tags into spans of the message. Checked against the Python pipeline's replies and token
  ids with the same ONNX file.

```
uv run python web/serve.py                  # then open http://127.0.0.1:8000/
uv run python web/serve.py --no-isolation   # same, without COOP/COEP: one thread only
```

Query switches: `?eval=1` (run the eval), `?limit=N` (first N eval messages), `?cpu=1` (no WebGPU; the default on Android and in Firefox, where `?cpu=0` opts back in),
`?threads=N`, `?fresh=1` (drop the cached model first, to time a cold load), `?download=1` (download without asking), `?model=NAME` (a key
of `MODELS` in `engines/index.js`: `lfm2.5-230m-v2-Q8_0` (default), `lfm2.5-230m-v2-Q4_K_M`, `qwen3.5-0.8b-v2-Q4_K_M`,
`qwen3.5-0.8b-v2-trim-Q4_K_M`, `tagger-int8`). The eval result is in `window.__evalResult` and printed as
JSON on the page.

## Where the model files come from, and hosting

The page asks before it downloads a model, and never asks again for one the browser already
holds (`?download=1` skips the question, for scripted runs).

For each model, `engines/source.js` looks for a copy in `models/` beside the page and uses it
if there is one; otherwise it downloads the published file from Hugging Face
(`https://huggingface.co/<repo>/resolve/main/<file>`, which allows downloads from other
sites). So a clone needs no model files to run, and a local `models/` folder overrides the
published ones while developing.

To host the demo, copy this folder, without `models/`, to any static file host. Nothing runs
on the server. wllama's multi-threading needs the page to be cross-origin isolated, which
normally takes two response headers; `isolate.js` installs a service worker that adds them on
hosts that cannot (the page reloads once on a first visit). Where the host already sends the
headers, as `serve.py` does, the service worker is not installed.

## Files

- `index.html`, `style.css`, `app.js`: the page. `eval.js`: eval mode. `serve.py`: local server
  with COOP/COEP headers.
- `engines/`: one module per engine with the same interface (`load`, `describe`, `complete`,
  `evalSet`, `check`; see `engines/index.js`, which lists the drop-down's models). `gguf.js`:
  wllama with the native-equivalent completion settings, for any GGUF model; `index.js` gives each
  its file, prompt template, reference replies and worker restart interval. `tagger.js`: ONNX
  Runtime Web, and the decoding of tags into spans.
- `tagger/wordpiece.js`: a port of BERT's uncased WordPiece tokenizer that also gives each
  token's character range in the message, as the Hugging Face tokenizer's offsets do. The eval
  checks that every message's token ids equal Python's.
- `prompts/<model>.json`: the chat template around one user message (the text the page sends,
  and the native token ids). `eval/<model>-<quant>.json`: the 530 eval messages, their full
  native prompt token ids and the native replies of that GGUF. Both come from `export_eval.py`,
  which also checks that every eval prompt, as the page sends it, tokenizes to the native ids:

  ```
  uv run --with transformers --with jinja2 python web/export_eval.py \
      runs/lfm2.5-230m-v2-s1-compact/merged lfm2.5-230m-v2 \
      Q8_0=results/lfm2.5-230m-v2-s1-compact/samples.jsonl \
      Q4_K_M=results/lfm2.5-230m-v2-s1-compact-cpu-Q4_K_M/samples.jsonl
  uv run --with transformers --with jinja2 python web/export_eval.py \
      runs/qwen3.5-0.8b-v2-s0-compact/merged qwen3.5-0.8b-v2 \
      Q4_K_M=results/qwen3.5-0.8b-v2-s0-compact-gpu-Q4_K_M-main/samples.jsonl
  ```

- `models/`: optional local copies of the model files, git-ignored; without them the page
  downloads the published ones (see above). The layout is `models/lfm2.5-230m-v2-Q8_0.gguf`,
  `models/lfm2.5-230m-v2-Q4_K_M.gguf`, the two Qwen files and `models/tagger/{int8.onnx,vocab.txt}`.
- The LFM2.5-230M files are the finetune on training data v2, run s1 (of three, the best on the
  main eval, the fresh eval and the hard negatives): `runs/lfm2.5-230m-v2-s1-compact/gguf/` from
  your own training run, or https://huggingface.co/dsego/tempus-lfm2.5-230m. The tokenizer files
  for `export_eval.py` are that run's `merged/*.json` and `merged/chat_template.jinja`. The base
  model, `LiquidAI/LFM2.5-230M`, is under the
  [LFM Open License v1.0](https://docs.liquid.ai/lfm/help/model-license): the finetune may be
  redistributed with the licence and its notices kept, and commercial use is allowed only for
  organisations under $10 million annual revenue, a condition that binds everyone who downloads it.
- The Qwen3.5 0.8B files are the finetune on training data v2, run s0:
  `runs/qwen3.5-0.8b-v2-s0-compact/gguf/Q4_K_M.gguf` from your own training run, or
  https://huggingface.co/dsego/tempus-qwen3.5-0.8b (Apache 2.0). The trimmed file is the same
  model with its vocabulary cut from 248,320 to 142,841 tokens by `tempus trim` and converted
  with `scripts/convert_trimmed.py` (448 MB instead of 542 MB); every eval prompt tokenizes to
  the same length, so `eval/qwen3.5-0.8b-v2-Q4_K_M.json` serves both files.
- The tagger files (https://huggingface.co/dsego/tempus-bert-mini-tagger) and
  `eval/tagger-int8.json` (the 530 eval messages with the Python pipeline's replies and token
  ids) come from:

  ```
  uv run tempus tagger-export --model runs/tag-bert-mini-v2-s2/model
  uv run --group classifier python web/export_tagger.py runs/tag-bert-mini-v2-s2
  ```

- `vendor/onnxruntime-web/`: `onnxruntime-web` **1.31.1** from npm (`dist/ort.wasm.min.mjs`,
  `dist/ort-wasm-simd-threaded.mjs`, `dist/ort-wasm-simd-threaded.wasm`: the WASM-only build, no
  WebGPU; MIT licence, `LICENSE` from the onnxruntime repo at v1.31.1). The two `.mjs` files
  are stored here as `.js`, unchanged, because some static hosts serve `.mjs` with a content
  type that browsers refuse for modules. The `.wasm` is 14.8 MB (3.8 MB gzipped), more than
  the 8-bit model itself.
- `vendor/wllama/`: `@wllama/wllama` **3.13.0** from npm (`esm/index.js` and `esm/wasm/wllama.wasm`, stored here as
  `index.js` and `wasm/wllama.wasm`;
  MIT licence), built on llama.cpp `b12211-7d7f028`. `vendor/wllama-compat/`:
  `@wllama/wllama-compat` **3.13.0** (`wasm/wllama.js`, `wasm/wllama.wasm`, MIT), the Asyncify
  build wllama needs on browsers without JSPI or Memory64 (Safari). Both `wllama.wasm` files
  contain compiled llama.cpp (MIT), whose licence is `LICENSE-llama.cpp` beside each. ONNX
  Runtime's own third-party notices are in its repository
  (https://github.com/microsoft/onnxruntime/blob/v1.31.1/ThirdPartyNotices.txt).

## Prompt fidelity

The native client sends token ids from the Hugging Face tokenizer, with no system prompt. wllama
3.13 accepts only text prompts (token ids are rejected) and has no tokenize call, so the page
sends the template text from `prompts/<model>.json` with the message inside, and llama.cpp
tokenizes it, parsing special tokens and adding BOS when the GGUF's `add_bos_token` is set.

The LFM2.5 template text starts with `<|startoftext|>` (id 1) and the GGUF has
`add_bos_token = true`, so sending the templated text would give two start tokens.
`export_eval.py` drops the leading one from the text and records `add_bos: true`; llama.cpp adds
the one BOS. A model whose GGUF has `add_bos_token = false` (Qwen) gets `add_bos: false` and its
template text unchanged.

`gguf.js` refuses to start if the GGUF's `add_bos_token` differs from the template's `add_bos`.
Replies stop at llama.cpp's end-of-generation tokens from the GGUF, as natively. The eval checks, for every message, that the prompt length
llama.cpp reports equals the length of the native token ids, that the reply ended at such a
token (`finish_reason` `stop`, not the 160-token cap), and that the reply string is identical
to the native one.

The Q8_0 references come from llama.cpp on a CUDA GPU, and Chrome (Apple M3 Pro) matches all
530 replies on WebGPU (all 15 layers offloaded) and on six CPU threads, with and without worker
restarts: median 76 ms and p95 149 ms on WebGPU, 144 ms and 256 ms on CPU. The 247 MB file loads
in 1.5 s from the local server and 0.4 to 0.5 s from the browser cache.

The Q4_K_M references come from llama.cpp on an x86 CPU. Chrome matches 529 of 530 replies on
WebGPU and 528 on CPU (median 83 ms and 117 ms). The differences are near-ties that flip with
the backend's arithmetic, not prompt differences: every prompt length matches. An earlier model
showed the same thing between native llama.cpp on a GPU and on a Mac CPU.

## Known wllama limitations

llama.cpp's server keeps each finished prompt's state in a RAM cache of up to 8 GiB
(`--cache-ram`), which wllama 3.13 does not let you lower, so the wasm heap grows with every
reply. `gguf.js` therefore restarts the wllama worker every `restartEvery` replies, which takes
about a second in Chrome because the weights are already in the browser cache. LFM2.5-230M's
states are about 0.6 MB each (page memory grew from 0.81 to 1.09 GB on CPU and from 0.59 to
0.91 GB on WebGPU between 30 and 530 replies without restarts), so it restarts every 256, to
bound memory in a long session. Models with larger states need a much shorter interval: an
earlier Qwen3.5 0.8B finetune, whose hybrid layers store tens of MiB per state, ran out of heap
after about 80 replies and needed a restart every 16.
Requests use `cache_prompt: false`, as the native client does, and the eval matches native
replies across these restarts.

In Playwright's WebKit (the compat build, as Safari would use), the page freezes right after
that restart. Replies before the first restart match native. Real Safari is untested.
