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

# tempus: BERT-mini classifier for clock times in chat messages

An 11M-parameter classifier that decides, for every number in an English chat message that
could be a clock time, whether it is not a time, a morning time or an afternoon/evening time.
It is 11 MB and answers in about 1 ms on a server CPU and 9 ms on a Raspberry Pi 4.

It is not a standalone model: code finds the candidate numbers and reads their digits, and
this model only labels each candidate. That code is in the repository
(`src/tempus/candidates.py` and `src/tempus/classifier.py`):
https://github.com/dsegovia90/tempus_finetune

## Files

| File | Size | Use |
|---|---|---|
| `int8.onnx` | 11 MB | Recommended, 8-bit weights |
| `fp16.onnx` | 21 MB | 16-bit weights, same arithmetic as full precision |
| `vocab.txt` | 0.2 MB | WordPiece vocabulary (lowercased) |

Inputs: `input_ids`, `attention_mask`. Output: `logits`, one row per token over the classes
none, am, pm. The class of a candidate is read at its first token.

## Scores

Exact match per message after the surrounding code assembles the reply. This is the best of
three training runs, chosen by these same evals.

| Eval | `int8.onnx` | Three-run mean (full precision) |
|---|---|---|
| Main (530 messages) | 92.8% | 92.3% |
| Fresh (378 messages from situations not in training) | 92.6% | 91.6% |
| False alarms on 1,021 tricky messages with no clock time | 1.8% | 1.6% |

On messages where the time of day is implied only by the situation ("dinner at 7"), the
three-run mean is 79.5% on the main eval and 76.6% on the fresh eval. That is its weak point:
it knows far less about daily life than a language model. The 230M generative model scores
about 92% there: https://huggingface.co/dsego/tempus-lfm2.5-230m

Latency per message, including candidate search and tokenizing: about 1.4 ms median on 4 cores
of a shared Ryzen 3600; 9 ms on a Raspberry Pi 4 (4 threads), 18 ms on one thread; about 4 ms
in a browser tab on an Apple Silicon Mac.

## Limits

- English only; times written with digits, plus "noon" and "midnight".
- Weak on unfamiliar situations, as above.
- Training and eval data are synthetic, written and checked by language models with a small
  human spot-check.

## Training

Token classification on
[google/bert_uncased_L-4_H-256_A-4](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4),
11,299 synthetic messages, exported to ONNX.

## Licence

Apache 2.0, the same as the base model.
