# Finetuning Needle 3 locally: learnings

Notes from one side experiment (November 2026) that finetuned Cactus Compute's Needle 3 on
about 11,300 examples, with `cactus-compute` 4.2.0. Written as a starting
point for anyone finetuning Needle to call a specific set of tools.

Each point is marked **[measured]** (we ran it), **[docs]** (from the vendor's finetuning
guide, https://cactuscompute.com/blog/finetuning-needle, not verified by us) or
**[untested]**.

## What Needle is good and bad at

- **Built for:** picking a tool and filling its arguments, where every argument value is
  either a span copied from the input or a choice from an enum. **[docs]**
- **Cannot be relied on to:** write a value that is not in the input (a computed number, a
  normalised date, a rewritten string). Design the schema so code does that step afterwards.
  **[docs, and our schema followed it]**
- **Weak at:** judgements that need world knowledge. In our task 82% of its errors (74 of 90)
  were of that kind, while span copying and structure were mostly right. **[measured]**
- **Implication for a tool-calling task:** routing to one of N tools and copying arguments is
  the model's home ground, which our task was not. Expect the hard cases to be the ones where
  choosing the tool or an enum value needs knowledge the message does not state.

## Results we got, for scale

| Setting | Accuracy (exact match) | False calls on no-call inputs |
|---|---|---|
| Published model, untrained | 21.1% | not run |
| Vendor defaults, no reasoning line | 53.6% | 63.4% |
| Vendor defaults, with reasoning line | 79.6% | 7.7% |
| Rank 32, 6 epochs, lr 3e-4, with reasoning line (mean of 3 seeds) | 90.6% | 1.5% |

All **[measured]**, 20 layers, 4-bit export, scored through the engine on CPU. Seed spread at
the best setting: 0.5 points standard deviation overall.

## Install

```
uv venv --python 3.12
uv pip install "cactus-compute[train,gpu]==4.2.0"     # NVIDIA
uv pip install "cactus-compute[train,metal]==4.2.0"   # Apple Silicon [docs, untested]
```

- **NVIDIA:** pulled JAX 0.12 with CUDA 12 wheels; worked on an RTX 3090 with no system
  changes. **[measured]**
- **Apple Silicon:** supported through the `jax-metal` plugin, which pins JAX to 0.4.38 or
  older. The guide quotes 0.71 s per step on an M5 Max against 2.90 s on its CPU. **[docs]**
- **Our speed for comparison:** about 0.87 s per step on the 3090 (batch 16, max length 512).
  The guide does not state the shape behind its M5 Max figure, so the two are not directly
  comparable; they are in the same range.
- **Keep it in its own virtual environment.** The JAX stack conflicts with a PyTorch/CUDA 13
  project environment.

## Privacy and network behaviour

Set these for every command **[measured]**:

```
export CACTUS_TELEMETRY=0 DO_NOT_TRACK=1
export CACTUS_NEEDLE_WEIGHTS=/path/to/needle3.cact     # stops a config re-fetch on each load
```

- The package sends usage events unless one of the first two is set (`cactus/_telemetry.py`).
- `cactus build` re-downloads the base archive each time, and the tokenizer is fetched from
  the Hugging Face repo's main branch, not a pinned revision. Neither sends your data.
- Local finetuning needs no account and no API key. The hosted route
  (`cactus platform finetune`) needs `CACTUS_API_KEY` and uploads your data. **[docs]**
- The engine (`libcactus.so`) is a closed binary; only the Python package is on GitHub.
- Licence of the model repository: unmodified Apache 2.0. **[measured]**

## Training data format

One JSON object per line **[measured]**:

```json
{"query": "dinner at 7, home by 10:30pm",
 "tools": [{"name": "clock_times", "description": "...", "parameters": {"type": "object", "properties": {...}, "required": [...]}}],
 "answers": [{"name": "clock_times", "arguments": {"times": [{"mention": "7", "meridiem": "pm"}]}}],
 "reasoning": "'7' -> pm"}
```

- **`tools`:** the full tool list, repeated on every row.
- **`answers`:** the calls a perfect model makes. An input that should trigger no call gets
  `"answers": []`.
- **`reasoning`:** one short line. The guide calls it optional. Treat it as required; see the
  next section.
- **Check the ceiling first.** Before training, convert every labelled example to this format
  and back with your own code and confirm 100% round-trips. Ours did, on all four datasets.
  If it does not, the schema cannot express your labels and no training will fix that.

## The reasoning line is not optional in practice

- **What happens:** the engine always writes a reasoning line before the call, even for a
  model trained without one. A model trained without it then sees a context it never saw in
  training. **[measured]**
- **Evidence:** the same adapter gave about 15% false calls through the vendor's JAX path
  (which goes straight to the call) and about 63% through the engine, on a 40-example sample.
- **Fix:** add a templated line to every training row, including a fixed phrase for no-call
  rows (ours: `no clock time`). That alone took accuracy from 53.6% to 79.6%.
- **Consequence:** always score through the engine you will deploy, never through the
  training framework's own generation.

## Training

```
cactus finetune data.jsonl --checkpoint needle3.safetensors \
    --epochs 6 --lr 3e-4 --lora-rank 32 --seed 0 \
    --checkpoint-dir runs/x --out runs/x/adapter.safetensors
cactus build needle3.safetensors --lora runs/x/adapter.safetensors --layers 20 --out runs/x/20l.cact
```

- **What it trains:** LoRA on the attention projections with quantisation-aware numerics
  (4-bit weights, 8-bit activations) matching the export. **[measured]**
- **Settings that worked:** rank 32, 6 epochs, learning rate 3e-4. The guide suggests rank 32
  for grounding-heavy tasks. CLI defaults in 4.2.0 were rank 16, 3 epochs, 1e-4; the guide's
  own example uses 10 epochs. **[measured / docs]**
- **Cost on an RTX 3090:** 23 minutes for 3 epochs, 40 minutes for rank 32 with 6 epochs,
  about 12.5 GB of GPU memory, for 11,300 rows at batch 16. **[measured]**
- **Sequence length matters for speed.** The guide reports examples of 245 tokens training six
  times faster padded to 256 than to 1024 on CPU. Our runs used 512. Set the maximum length
  just above your longest row. **[docs; check `cactus finetune --help` for the flag]**
- **Validation:** the CLI holds out 10% of your file itself, chosen by seed.
- **Trap: the closing scoring pass.** After the last epoch `cactus finetune` generates an
  answer for every validation row before it writes the adapter. That took 20 to 45 minutes,
  there is no flag to skip it, and stopping during it loses the adapter. We lost one run this
  way. `finetune_unscored.py` in this folder calls the vendor's `finetune_local` with that
  pass turned off. **[measured]**

## Export limits of the local route

- **4-bit only.** A tuned 20-layer archive is 63 MB, larger than the published 35 MB 2-bit
  base. **[measured]**
- **No confidence head.** `confidence` is `None` for locally tuned archives, so you cannot
  threshold on it. **[measured / docs]**
- **Layer slicing does not survive local finetuning.** `cactus build --layers 8` produced an
  archive that scored about 40% with about 50% false calls. Use 20 layers. **[measured]**
- **The hosted route differs:** full-model training at every depth, 2-bit export, a trained
  confidence head, and your data mixed with Needle's original dataset. We did not use it.
  **[docs]**

## Inference

```python
import cactus

cactus.set_threads(4)                      # before the agent is built; later calls are ignored
agent = cactus.Cactus(tools=TOOLS_JSON, weights="runs/x/20l.cact", auto_date=False, stateless=True)
response = agent.complete(message, max_new_tokens=128)
calls = response["function_calls"] or response["suppressed_calls"] or []
```

- **`auto_date=False`:** the default adds a system turn with the current date and clock time,
  which training never saw. **[measured]**
- **`suppressed_calls`:** the engine may move calls there for low confidence; `cactus.extract`
  reads the kept ones and falls back to the suppressed ones, and so did we.
- **Threads:** tuned archives run in a child process that reads the thread count at start.
  Use one process per thread count when timing.
- **CPU only.** The engine has no GPU path.
- **Latency of the tuned 20-layer archive** on 4 virtual cores of a shared Ryzen 3600: median
  157 ms, 95th percentile 229 ms; on 1 thread 351 ms and 492 ms. Short chat messages, one
  tool. A longer tool list means a longer prompt, so expect more. **[measured]**
- **Time it with nothing else running.** Our first latency figures were taken while training
  ran on the same machine and were useless.

## Suggested order of work

1. Write the tool schemas so every argument is a copied span or an enum.
2. Write the converter both ways and run the ceiling check on your labelled data.
3. Build the eval sets before training: normal cases, a set from situations not in training,
   and a set of inputs that must trigger no call. In our project the no-call set exposed a
   13% to 26% false-call problem that the main eval hid.
4. Score the untrained published model through the engine as the baseline.
5. Train with reasoning lines at rank 32, 6 epochs, lr 3e-4, using `finetune_unscored.py`.
6. Score through the engine, group the misses by kind, and only then tune settings or data.
7. Run two more seeds on the best setting before trusting a difference under about 2 points.

## Not tested

- More than one tool. Our schema had a single tool, so tool selection among several is
  unmeasured here.
- Training on Apple Silicon, and the hosted platform route.
- Multi-turn inputs and tool results fed back to the model.

## Files in this folder

- `formulation.py`: the schema, training rows, and conversion both ways.
- `convert.py`: writes the training file and runs the ceiling check.
- `train.py`: wraps finetune and build; records time, memory and loss.
- `finetune_unscored.py`: finetune without the closing scoring pass.
- `run.py`, `latency.py`: scoring through the engine and clean CPU timing.
- `analyze.py`: groups misses by kind.

`formulation.py`, `convert.py`, `run.py` and `analyze.py` import this project's scoring code;
`train.py` and `finetune_unscored.py` stand alone.
