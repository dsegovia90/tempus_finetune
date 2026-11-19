---
license: cc-by-4.0
language:
- en
task_categories:
- token-classification
- text-generation
tags:
- synthetic
- time-extraction
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: eval_core
    path: eval_core.jsonl
  - split: eval_generated
    path: eval_generated.jsonl
  - split: eval_fresh
    path: eval_fresh.jsonl
  - split: eval_negatives
    path: eval_negatives.jsonl
---

# tempus: clock times in chat messages

13,228 short English chat messages, each labelled with every clock time it contains: the exact
span copied from the message and its reading as a minute of the day (0 to 1439). The label
includes the judgement a reader makes from context: "dinner at 7" is 19:00, "got paged at 2"
is 02:00, and "a table for 5" contains no time at all.

Code, the plans every message was written from, and trained models:
https://github.com/dsegovia90/tempus_finetune

```json
{"id": "core-001", "slice": "implicit", "text": "dinner at 7? that new ramen place", "times": [{"text": "7", "minute": 1140}]}
```

## Splits

| Split | Messages | What it is |
|---|---|---|
| `train` | 11,299 | Training data, version 2: 7,973 general messages plus 3,326 built around numbers that look like times but are not |
| `eval_core` | 130 | Written by hand |
| `eval_generated` | 400 | Generated, from situations held out of training. `eval_core` plus this split is the "main eval" of the project (530) |
| `eval_fresh` | 378 | Generated later from a second set of held-out situations |
| `eval_negatives` | 1,021 | No clock time in any of them; each is built to tempt a model ("finished in 3:45", "John 3:16", "gate 12") |

## Fields

- `id`: unique within the dataset.
- `slice`: the kind of message. `implicit` (time of day follows from the situation only), `cue`
  (a word such as "tonight" settles it), `explicit` ("3pm", "15:00"), `range` ("2 to 4pm"),
  `fallback` (a bare number with no context at all, read as morning by convention), `none` (no
  clock time).
- `text`: the message.
- `times`: a list of `{text, minute}`; empty when there is no clock time.

## Labelling rules, in short

- Clock times only. Durations, relative times ("in 2 hours"), dates and vague periods ("this
  afternoon") are not labelled.
- Times written with digits, plus "noon" and "midnight".
- A bare 12 is noon. A bare number with no context at all is read as morning.
- The full rules are in `prompts/spec.md` in the repository.

## How it was made

Everything except `eval_core` is synthetic. Code first planned each example (the situation,
the time, the exact span), so the label existed before any text. A language model then wrote a
chat message for the plan, and a second model, without seeing the plan, read the message back.
An example was kept only if that blind reading matched the plan.

## Limits

- Written and checked by language models from one family, with a small human spot-check.
  Agreement between them shows the labels are consistent, not that every one matches a
  person's reading.
- English only, and chat-style messages only.
- Three kinds of trap in `eval_negatives` (verse references, ages and measurements, codes and
  extensions) were deliberately kept out of `train`, to test whether a model learned the idea
  or only the kinds it was shown.

## Licence

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Copyright (c) 2026 Daniel Segovia.
