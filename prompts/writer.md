# Writer instructions

You write realistic chat messages (Slack, Teams, a group chat) for a dataset. Each line of the
plans file is one plan. For each plan, write exactly one message.

A plan has:

- `brief`: what the message is about and any constraint on how the time is conveyed.
- `style`: how the message should look.
- `must_include`: strings that must appear in the message exactly as given, character for
  character. Do not change their case, spacing or punctuation.
- `times`: the clock times the message contains and the minute of the day each one means
  (minutes since midnight: 900 is 15:00). A reader must arrive at exactly these times.

Rules:

1. The message contains the planned times and no other clock time. If `times` is empty, it
   contains no clock time at all.
2. Each time's `text` appears exactly as given, and the times appear in the planned order.
3. Do not add am/pm, a 24-hour time, or a part-of-day word unless it is in `must_include`.
   When the brief says the situation must make the time obvious, make the situation
   unmistakable: a stranger reading only this message should have no doubt.
4. Digits elsewhere in the message are fine when they are clearly not times (the brief may ask
   for one), but never put another number where it could be read as a clock time. Prefer
   forms that cannot be times: "room 4B", "3 out of 87 tests", "v3.8.2", "build 4521".
5. Sound like real people: varied topics, names, tone and length. Mostly one or two sentences.
   No two messages should share an opening or a template.
6. Read `prompts/spec.md` first so you know how a labeller will read your message.

Output: one JSON object per line, `{"id": "<plan id>", "text": "<message>"}`, one line for
every plan in the file, in the same order, and nothing else in the file.
