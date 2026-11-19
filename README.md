# tempus

Small models finetuned to read clock times out of chat messages, including the judgement of
whether a bare "7" means morning or evening. The smallest generative one answers in about
100 ms (median) on a CPU-only server and runs in a browser tab.

Three are small generative language models; one is a tiny encoder that is not generative,
included as an alternative to weigh them against. Each reads one English message and gives one
`text=HH:MM` line per clock time, or `none`:

| Message | Reply |
|---|---|
| `dinner at 7 or lunch at noon` | `7=19:00`<br>`noon=12:00` |
| `got paged at 2, been up ever since` | `2=02:00` |
| `skip to 2:30 in the video, room 3 is booked` | `none` |

## Results at a glance

All four models were trained on the same 11,299 synthetic messages (data version 2). Each row
is the mean of three training runs.

| Model | Kind | Parameters | File | Main eval | Fresh eval | False alarms | CPU latency | GPU latency | Base licence |
|---|---|---|---|---|---|---|---|---|---|
| LFM2.5-230M finetune | Generative | 230M | 247 MB | 96.5% | 96.8% | 0.2% | 98 ms | 16 ms | LFM Open License v1.0 |
| Qwen3.5 0.8B finetune | Generative | 0.8B | 834 MB | 96.7% | 98.1% | 0.6% | 412 ms | 41 ms | Apache 2.0 |
| Qwen3.5 2B finetune | Generative | 1.9B | 2.1 GB | 97.0% | 98.9% | 0.1% | about 1,100 ms | 56 ms | Apache 2.0 |
| BERT-mini tagger | Not generative | 11M | 11 MB | 93.4% | 93.6% | 1.9% | about 2 ms | about 3 ms | Apache 2.0 |

Reference points on the main eval, none of them finetuned:

| Reference | Main eval | Latency per message |
|---|---|---|
| Rule-based parser, no model (Recognizers-Text) | 73.0% | not measured |
| Qwen3.8-Flash-Next, a large general model, given the labelling rules | 94.3% | 1.5 s on the GPU |
| Claude Opus 5.5, given the labelling rules | 100% | not measured |

- Both large models got the full labelling rules as instructions, which the finetuned models
  never see.
- Qwen3.8-Flash-Next ran locally, one message per request with thinking off, from a 4-bit
  quantized file on the same RTX 3090. The model is larger than the card's memory, so this is
  not its best-case speed. Many of its misses are about the output format, not the time: for
  "i can do 6-8pm" it gives "6-8pm" as the text of both times.
- Opus ran as an agent reading all 530 messages from a file, so it has no per-message latency
  to compare. It is also from the model family that wrote and checked the generated messages
  (Claude), so its 100% shows the labels are consistent with that family's reading; it is not
  an independent ceiling.

- **Main eval, fresh eval:** exact match per message (every time found, with the right span and
  the right 24-hour reading). The fresh eval uses situations that never appear in training.
- **False alarms:** the share of 1,021 deliberately tricky messages with no clock time ("a
  table for 5", "finished in 3:45") for which the model reported one anyway.
- **CPU latency:** median per message through llama.cpp on 4 virtual cores of a shared Ryzen
  3600. Unlike the accuracy columns, this is not always three runs: the LFM2.5-230M figure is
  the mean of two runs (96 and 99 ms) and the Qwen3.5 2B figure is one run. The 95th
  percentile is about twice the median (202 ms for the published LFM2.5-230M file).
- **GPU latency:** median per message through llama.cpp on an RTX 3090, one message at a time.
- **Kind:** the generative models write the `text=HH:MM` lines as text. The tagger writes
  nothing: it labels each token of the message as outside a time, beginning one or continuing
  one, and gives the hour and minute where a time begins. Every model does the whole job
  itself; no code finds numbers or reads digits for any of them.
- Generative models were scored as GGUF files at Q8_0 (8-bit weights; "quantization" is storing
  weights in fewer bits), and the File column is that file. Smaller files are published too:
  Qwen3.5 0.8B at 542 MB scores within one message of its 834 MB file (see Published files
  below). The
  tagger was scored at full precision. Its CPU figure is the 8-bit ONNX
  file through ONNX Runtime; the same file takes 9.5 ms on a Raspberry Pi 4. Its GPU figure is
  the PyTorch model on the RTX 3090 (2.7 ms, against 3.6 ms for PyTorch on the CPU): a model
  this small gains nothing from a GPU, and the 8-bit file on a CPU is the fastest way to run it.

No single model meets every target we set (98% overall and 95% on the hard "implicit" slice of
the fresh eval, under 100 ms on CPU, under 2% false alarms). LFM2.5-230M meets the speed and
false-alarm targets and is one to three points short on accuracy. Qwen3.5 2B meets accuracy and
false alarms and is about eleven times too slow on CPU. Qwen3.5 0.8B sits between them: it
meets false alarms, reaches 98.1% overall and 94.8% on the implicit slice, and is four times
too slow on CPU.

The tagger is not held to those targets; it is there to show what a non-generative model
gives and gives up. It is twenty times smaller and about fifty times faster than the
smallest generative model, and its output cannot be malformed. It is three points less
accurate overall and seven to ten points behind where the time of day has to be inferred from
the situation, because an 11M-parameter encoder knows far less about daily life. It also
cannot copy what it has not seen: it reads minutes in 5-minute steps, so "9:17" comes out
as 09:15, where the generative models copy the digits.

### Published files

Each published file comes from the best of its model's three runs, so its scores sit a little
above the means in the table above. Every file here is on Hugging Face; the demo column says
which ones the browser demo offers.

| File | Size | Main eval | Fresh eval | False alarms | CPU latency | In the demo |
|---|---|---|---|---|---|---|
| LFM2.5-230M, Q8_0 | 247 MB | 97.0% | 97.4% | 0% | 99 ms | Yes, the default |
| LFM2.5-230M, Q4_K_M | 153 MB | 95.7% | 97.4% | 0% | 72 ms | Yes |
| Qwen3.5 0.8B, Q8_0 | 834 MB | 97.2% | 98.4% | 0.8% | 421 ms | No |
| Qwen3.5 0.8B, Q4_K_M | 542 MB | 97.0% | 98.4% | 0.7% | 340 ms | Yes |
| Qwen3.5 0.8B, Q4_K_M, trimmed vocabulary | 448 MB | 97.0% | 98.1% | 0.5% | 300 ms | Yes |
| Qwen3.5 2B, Q8_0 | 2.1 GB | 97.5% | 99.7% | 0.2% | 1,105 ms | No, too large for a browser tab |
| BERT-mini tagger, 8-bit | 11 MB | 93.8% | 94.2% | 1.7% | 1.8 ms | Yes |
| BERT-mini tagger, 16-bit | 21 MB | 93.8% | 94.7% | 1.7% | 1.8 ms | No |

Q4_K_M and Q8_0 are two precisions of the same weights (about 4.5 and 8 bits each). The trimmed
file is the Q4_K_M model with its vocabulary cut to the tokens English chat needs
(`tempus trim`). Scores can
differ by one message between processors, because near-ties flip with their arithmetic.

## Try it

Trained weights and the dataset are on Hugging Face:

- [dsego/tempus-lfm2.5-230m](https://huggingface.co/dsego/tempus-lfm2.5-230m)
- [dsego/tempus-qwen3.5-0.8b](https://huggingface.co/dsego/tempus-qwen3.5-0.8b)
- [dsego/tempus-qwen3.5-2b](https://huggingface.co/dsego/tempus-qwen3.5-2b)
- [dsego/tempus-bert-mini-tagger](https://huggingface.co/dsego/tempus-bert-mini-tagger)
- [dsego/tempus-clock-times](https://huggingface.co/datasets/dsego/tempus-clock-times) (the dataset)

**In a browser.**

```
uv run python web/serve.py
```

and open http://localhost:8000/. The page asks before downloading a model from Hugging Face
(the default is 247 MB, cached after the first load) and then runs it on your device with
llama.cpp compiled to WebAssembly. The folder is a static site: see
[web/README.md](web/README.md) for hosting it.

**On a CPU server.** With [llama.cpp](https://github.com/ggml-org/llama.cpp) installed
(tested at tag b12292):

```
llama-server --model lfm2.5-230m-v2-Q8_0.gguf --host 127.0.0.1 --port 8080 \
    --threads 4 --threads-batch 4 --n-gpu-layers 0 --device none \
    --parallel 1 --ctx-size 1024 --cache-ram 0 --no-webui

uv run tempus extract "dinner at 7?"
```

Details, measurements and limits are in [docs/serving.md](docs/serving.md).

**From Python** (standard library only at runtime):

```python
from tempus.client import extract

result = extract("dinner at 7 or lunch at noon")
result.times      # [Time(text='7', time='19:00', minute=1140), Time(text='noon', time='12:00', minute=720)]
result.malformed  # False; True means the reply could not be trusted, and no time is guessed
```

## How it works

- **The task.** Every clock time in a message, as the exact span copied from it plus its
  24-hour reading. The labelling rules are in [prompts/spec.md](prompts/spec.md). Slices of the
  eval sets name the kind of message; "implicit" means the time of day must be inferred from
  the situation ("dinner at 7").
- **Synthetic data, generated from the intent.** Code plans each example first (the time, the
  situation, the exact span), so the label exists before any text does
  ([src/tempus/planning.py](src/tempus/planning.py)). LLM agents then write a chat message for
  each plan and, separately and without seeing the plan, read the message back; an example is
  kept only if the blind reading matches ([prompts/](prompts/),
  [src/tempus/assemble.py](src/tempus/assemble.py)).
- **Three eval sets**, none seen in training: the main set (530 messages, 130 of them
  hand-written), a fresh set from different situations (378), and the hard negatives (1,021
  messages with no clock time).
- **Training.** LoRA finetuning (small trainable add-on matrices on a frozen base model) with
  Unsloth; each run is a short YAML file in [configs/](configs/)
  ([src/tempus/train.py](src/tempus/train.py)). The model sees only the message, with no system
  prompt.
- **Serving and scoring.** Finetunes are converted to GGUF, llama.cpp's model file format, and
  scored through `llama-server` ([src/tempus/llama.py](src/tempus/llama.py)).
- **The tagger** is a BERT-mini encoder with three small outputs on every token: a tag
  (outside, begins or continues a time), the hour and the 5-minute step. Its training labels
  come straight from the dataset's spans, it trains in about 4 minutes on a laptop, and it
  runs through ONNX Runtime ([src/tempus/tagger.py](src/tempus/tagger.py)).

**Also tried.** [Needle 3](https://cactuscompute.com/needle) (Cactus Compute, Apache 2.0), a
tiny tool-calling model with its own engine: at best 90.6% on the main eval, 1.5% false
alarms and 157 ms on CPU in a 63 MB file, so it was beaten by LFM2.5-230M on accuracy, false
alarms and speed, though not on size,
and was behind the BERT-mini tagger too. Scripts and notes are in
[experiments/needle/](experiments/needle/). A first non-generative design, a classifier that
only chose "not a time, am or pm" for numbers that a regular expression had already found and
code then read, scored 92.3% and 91.6% with 1.6% false alarms. The tagger replaced it: doing
the whole job itself, the same encoder is one to two points more accurate. That classifier is
still in the code ([src/tempus/classifier.py](src/tempus/classifier.py)) and on
[Hugging Face](https://huggingface.co/dsego/tempus-bert-mini). Earlier screening runs of other small models
(Gemma 3 270M, LFM2.5-350M and 1.2B, Granite 4.0 350M, MiniCPM5-1B) are in
[docs/results/](docs/results/).

## Reproducing it

Requirements: [uv](https://docs.astral.sh/uv/), a CUDA GPU for training (24 GB is ample), and
llama.cpp for conversion and scoring.

```
uv run tempus plan --split train --count 8000 --batch-size 200    # plans for writer agents
uv run tempus assemble --split train --hold-out data/eval/core.jsonl data/generated/eval.jsonl
uv run tempus train --config configs/lfm2.5-230m-v2-s1-compact.yaml
uv run tempus eval --model runs/<name>/merged --no-spec --engine llama \
    --gguf runs/<name>/gguf/Q8_0.gguf --device cpu --name <name>
uv run tempus tagger-train --model google/bert_uncased_L-4_H-256_A-4 --name <name> --epochs 20
uv run tempus tagger-eval --model runs/<name>/model --name <name>
uv run tempus rules --name rules        # the rule-based baseline
uv run tempus trim --help               # vocabulary trimming for a smaller browser file
```

Training, the tagger and the rule baseline need the `gpu`, `classifier` and `rules`
dependency groups (`uv sync --group gpu`).

In the repository: the code, run configs, prompts, plans, every generated dataset
([data/](data/), described in [data/README.md](data/README.md)), the browser demo, and a
settings-and-metrics summary of the scored model runs ([docs/results/](docs/results/)). Not in
it: trained weights (they are on Hugging Face) and per-message replies, apart from the demo's
reference replies in `web/eval/` and the Opus reference in `data/review/`.
The step where agents write and check messages was orchestrated outside this code, from the
prompts in [prompts/](prompts/); there is no single command for it.

## Limits

- Trained and evaluated in English only. Some of the finetunes also read times in other
  languages in informal probes, a carry-over from their multilingual base models; that is
  unmeasured and outside the scope of this project.
- Times written with digits, plus "noon" and "midnight". Spelled-out times ("dinner at seven",
  "half past three") are not supported.
- LFM2.5-230M sometimes drops a time that a larger model keeps. Two examples the published
  file gets wrong and both Qwen models get right: "dinner at 7? or lunch at noon" (it returns
  only the 7) and "skip to 2:30 in the video, dinner at 7?" (it returns nothing, wary of the
  video timestamp).
- Inferring am or pm for unfamiliar situations is the weakest area: on the fresh eval's implicit
  slice, LFM2.5-230M scores 92.0%, Qwen3.5 0.8B 94.8%, Qwen3.5 2B 97.0% and the tagger 82.1%.
- The tagger reads minutes in 5-minute steps, because every time in the training data falls
  on one. The generative models read any minute.
- Training does not repeat exactly from a fixed seed for some models; single runs differ by
  about a point, which is why the figures above are three-run means.
- 13 training messages are near-duplicates of an eval message once case and punctuation are
  ignored, mostly very short ones such as a bare "3" or "running 20 late, start without me".
  There are no exact duplicates, and the eval sets use situations that training does not.
- The training data and the eval labels were written and checked by models from one family
  (Claude), with only a small human spot-check. Agreement between them shows the labels are consistent,
  not that every one matches a person's reading.

## Licences

| What | Licence | What it means for you |
|---|---|---|
| Code, and everything else not listed below (prompts, configs, cards, results) | MIT OR Apache-2.0 | |
| Datasets in `data/`, and the eval messages repeated in `web/eval/` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ([data/LICENSE.md](data/LICENSE.md)) | Use, change and redistribute, commercially too, with credit and a note of any changes |
| Qwen3.5 finetunes | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) ([base model](https://huggingface.co/Qwen/Qwen3.5-2B)) | No extra conditions |
| BERT-mini tagger (and the earlier classifier) | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) ([base model](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4)) | No extra conditions |
| LFM2.5 finetunes | [LFM Open License v1.0](https://docs.liquid.ai/lfm/help/model-license) | Finetunes may be redistributed with the licence and notices kept. Commercial use is licensed only for organisations under $10 million annual revenue, and that applies to anyone using a finetune |

## Mistakes and lessons

- **The first training data taught false alarms.** Trained on version 1, every model reported
  a clock time for 13% to 26% of the tricky no-time messages, and the main eval did not show
  it because it had few such messages. A dedicated eval of 1,021 traps exposed it; adding
  about 3,300 trap messages to training (half of them with a real time beside the trap)
  brought it to 0.1% to 1.6%. Lesson: build an eval for the error you fear most before
  trusting an accuracy figure.
- **One trap label was wrong.** "A 9 to 5 job" was planned as a no-time trap, but the
  hand-written eval labels it as two times. The contradiction was caught only after
  generation, and those messages were removed from eval and training.
- **The first output format made the model do arithmetic.** Replies were minutes since
  midnight; a checking agent slipped on the sum (1295 for 21:35), so the format became
  24-hour clock strings, which need none.
- **The engine mattered more than the model for speed.** The same finetune ran about ten times
  faster through llama.cpp than through Transformers' `generate`. Early latency figures
  compared engines, not models.
- **Training code silently skipped parts of two models.** One chat template added an empty
  thinking block that broke the loss mask, and LoRA's default layer names missed LFM's
  convolution blocks. Both now fail a check before training starts
  ([src/tempus/train.py](src/tempus/train.py)).
- **A bigger model is not a free win.** Qwen3.5 2B is more accurate on unfamiliar situations
  but about eleven times slower on CPU, and no model we tried met every target.
- **Quantization is nearly free down to 4 bits, then falls off a cliff.** Q8_0 matched full
  precision, and Q4_K_M cost between nothing and about a point. On the two models we swept,
  3-bit files lost 0.6 and 2.5 points and 2-bit files collapsed (73% and 55%). Measure per
  model: the loss is not the same for each.

Built with [Unsloth](https://unsloth.ai), [llama.cpp](https://github.com/ggml-org/llama.cpp),
[wllama](https://github.com/ngxson/wllama), [ONNX Runtime Web](https://onnxruntime.ai),
[MLflow](https://mlflow.org) and
[Recognizers-Text](https://github.com/microsoft/Recognizers-Text) (for the rule-based baseline).
