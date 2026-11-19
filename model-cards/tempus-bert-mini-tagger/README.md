---
license: apache-2.0
base_model: google/bert_uncased_L-4_H-256_A-4
language:
- en
pipeline_tag: token-classification
library_name: onnx
tags:
- onnx
- time-extraction
---

# tempus: BERT-mini tagger for clock times in chat messages

An 11M-parameter encoder that reads one English chat message and finds every clock time in it,
deciding from context whether a bare "7" means morning or evening. It is not a generative
model: it writes no text. It is 11 MB and answers in a few milliseconds on a CPU.

It is the non-generative alternative in a project that otherwise finetunes small language
models for the same task. Code, data and the full comparison:
https://github.com/dsegovia90/tempus_finetune

## How it works

The model labels every token of the message, in one pass:

- a tag: outside a time, begins a time, or continues one;
- at each token that begins a time, the hour (0-23) and the minute in 5-minute steps.

```
see      outside
you      outside
at       outside
8        begins      hour 20, minute 00
##pm     continues
=> 8pm=20:00
```

A span runs from a "begins" token over the "continues" tokens after it, and is cut from the
original message. No code finds numbers or reads digits; the model does both. The decoding is
about 20 lines: `decode` in `src/tempus/tagger.py` (Python) and `web/engines/tagger.js`
(JavaScript) in the repository.

## Files

| File | Size | Use |
|---|---|---|
| `int8.onnx` | 11 MB | Recommended, 8-bit weights |
| `fp16.onnx` | 21 MB | 16-bit weights, same arithmetic as full precision |
| `vocab.txt` | 0.2 MB | WordPiece vocabulary (BERT uncased) |

Inputs: `input_ids`, `attention_mask` (with `[CLS]` and `[SEP]`). Outputs, one row per token:
`tag` (3 classes: outside, begins, continues), `hour` (24 classes) and `minute` (12 classes;
multiply by 5).

## Scores

Exact match per message: every time found, with the right span and the right 24-hour reading.
This is the best of three training runs, chosen by these same evals.

| Eval | `int8.onnx` | `fp16.onnx` | Three-run mean (full precision) |
|---|---|---|---|
| Main (530 messages) | 93.8% | 93.8% | 93.4% |
| Main, time of day implied by the situation only | 84.7% | 84.7% | 83.8% |
| Fresh (378 messages from situations not in training) | 94.2% | 94.7% | 93.6% |
| Fresh, time of day implied by the situation only | 83.6% | 85.1% | 82.1% |
| False alarms on 1,021 tricky messages with no clock time | 1.7% | 1.7% | 1.9% |

For comparison, the 230M generative model of the same project scores 96.5% and 96.8% on the
two evals with 0.2% false alarms, at about 100 ms per message on a CPU:
https://huggingface.co/dsego/tempus-lfm2.5-230m

Latency per message for `int8.onnx`, including tokenizing (median): 1.8 ms on 4 cores of a
shared Ryzen 3600, 0.5 ms on an Apple M3 Pro, about 4 ms in a browser tab on that Mac, and on a
Raspberry Pi 4, 9.5 ms with four threads and 21 ms with one.

## Limits

- **5-minute steps.** Every time in the training data falls on a 5-minute mark, and a
  classification output cannot name a value it never saw. "meet at 9:17" is read as 09:15. A
  generative model copies the digits and does not have this limit.
- **Digits only**, plus "noon" and "midnight". Spelled-out times ("dinner at seven") were not
  in the training data; the model may tag one but will not read it correctly.
- **Weak where the time of day needs knowledge of daily life** (about 84% on those messages,
  against 91% and more for the generative models). An 11M-parameter encoder knows much less
  about the world.
- English only. Durations, relative times and vague periods are ignored by design.
- Training and eval data are synthetic, written and checked by language models with a small
  human spot-check.

## Training

[google/bert_uncased_L-4_H-256_A-4](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4)
with three linear outputs on every token, 20 epochs over 11,299 synthetic messages, the epoch
with the best validation score kept. About 4 minutes on a laptop. `tempus tagger-train` in the
repository.

## Licence

Apache 2.0, the same as the base model.
