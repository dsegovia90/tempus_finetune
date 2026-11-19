# Checker instructions

You label chat messages for a dataset. Each line of the messages file is
`{"id": ..., "text": ...}`. For each message, extract the clock times by following
`prompts/spec.md` exactly.

Rules:

1. Judge each message only from its own text. Do not open any other project file: no plans,
   no existing datasets, no other batches.
2. Give your honest reading. If a message is ambiguous, label the most likely reading; do not
   try to guess what its author intended.
3. `text` must be copied exactly from the message.

Output: one JSON object per line, `{"id": "<message id>", "times": [{"text": ..., "clock":
"HH:MM"}]}`, one line for every message, in the same order, and nothing else in the file.
`clock` is the time on a 24-hour clock ("15:00", "07:30", "00:00"); the minute of the day is
computed from it by code, so do not do that arithmetic yourself.
