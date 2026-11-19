# Data

Everything here is under CC BY 4.0 (see `LICENSE.md`).

## The datasets

- `eval/core.jsonl`: 130 hand-written eval messages.
- `generated/eval.jsonl`: 400 generated eval messages. With `core.jsonl` this is the main eval
  (530).
- `generated/eval2.jsonl`: the fresh eval (378), from a second set of held-out situations.
- `generated/negatives.jsonl`: the hard negatives (1,021 messages with no clock time).
  `negatives.dropped-9-to-5.jsonl` holds 23 messages removed from it because their label
  contradicted the hand-written eval.
- `generated/train.jsonl`: training data version 1 (7,973 messages).
- `generated/train2.jsonl`: the second round (3,326 messages built around numbers that are
  not times).
- `generated/train-v2.jsonl`: training data version 2 (11,299), which every published model
  was trained on. It is `train.jsonl` followed by `train2.jsonl`.
- `generated/pilot.jsonl` and `pilot2.jsonl`: two small trial rounds made while the prompts
  were being written. They are not used for training or scoring.
- `generated/*.rejects.jsonl`: generated messages that failed a check and were left out, with
  the reason.

Each line is `{id, slice, text, times: [{text, minute}]}`; `minute` is the minute of the day.

## How they were made

- `scenarios/`: the situations ("a dentist appointment") used for training, the main eval and
  the fresh eval. The three lists do not overlap.
- `plans/<split>/`: what code planned for each example before any text existed: the slice, the
  times and the exact spans (`tempus plan`).
- `messages/<split>/`: the chat message a language model wrote for each plan.
- `checks/<split>/`: what a second model read from each message without seeing the plan.
- `tempus assemble` joins the three and keeps an example only if the blind reading matches
  the plan.

## Review files

- `review/eval-messages.jsonl`: the 530 main-eval messages without labels, as given to a large
  model.
- `review/ceiling-opus.jsonl`: Claude Opus 5.5's replies to them (the 100% reference in the
  main README).
- `review/train-sample.md`: a sample of training messages laid out for a person to spot-check.
