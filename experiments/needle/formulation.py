"""The task as one Needle extraction call, and back.

Needle fills a tool call whose values are copied spans of the message or picks from an enum;
it is not built to write a value that is not in the text, such as "19:00". So the model copies
each clock expression and picks its half of the day, and code reads the digits, as it does for
the classifier (tempus.candidates):

    "dinner at 7, home by 10:30pm"
    -> [{"name": "clock_times", "arguments": {"times": [
           {"mention": "7", "meridiem": "pm"}, {"mention": "10:30pm", "meridiem": "pm"}]}}]

`am` is 00:00-11:59 and `pm` 12:00-23:59, so a bare 12 is `pm` at noon and `am` at midnight.
A message with no clock time gets no call at all, as the vendor's guide asks for off-topic input.

The engine always writes a reasoning line (a `<think>` block) before the call, so training rows
carry one too, in the guide's style: "'7' -> pm; '10:30pm' -> pm", or "no clock time". A model
trained without it meets an unfamiliar context at inference.
"""

import json

from tempus import candidates, target
from tempus.candidates import NOON
from tempus.schema import Example, Extraction

TOOL = {
    "name": "clock_times",
    "description": "Every clock time of day in the message, in order of appearance.",
    "parameters": {
        "type": "object",
        "properties": {
            "times": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "mention": {
                            "type": "string",
                            "description": 'The clock expression copied from the message: "7", "9:30", "4pm", "noon".',
                        },
                        "meridiem": {
                            "type": "string",
                            "enum": ["am", "pm"],
                            "description": "am for 00:00-11:59, pm for 12:00-23:59.",
                        },
                    },
                    "required": ["mention", "meridiem"],
                },
            }
        },
        "required": ["times"],
    },
}
TOOLS_JSON = json.dumps([TOOL], separators=(",", ":"))


def answers(example: Example) -> list[dict]:
    """The calls a perfect model makes for this example."""
    if not example.times:
        return []
    times = [
        {"mention": t.text, "meridiem": "am" if t.minute < NOON else "pm"} for t in example.times
    ]
    return [{"name": TOOL["name"], "arguments": {"times": times}}]


def reasoning(example: Example) -> str:
    """The short line the model writes before its call."""
    pairs = [f"'{t.text}' -> {'am' if t.minute < NOON else 'pm'}" for t in example.times]
    return "; ".join(pairs) or "no clock time"


def training_row(example: Example, with_reasoning: bool = True) -> dict:
    """One line of `cactus finetune` input."""
    row = {"query": example.text, "tools": [TOOL], "answers": answers(example)}
    return {**row, "reasoning": reasoning(example)} if with_reasoning else row


def compact(calls: list[dict]) -> str:
    """Our compact reply for Needle's calls: each mention read by its digits and meridiem.

    A mention with no clock expression in it is dropped; the reply then misses or matches
    on its own merits.
    """
    times = []
    for call in calls:
        for item in (call.get("arguments") or {}).get("times") or []:
            mention = str(item.get("mention", ""))
            found = candidates.find(mention)
            if found:
                times.append(Extraction(mention, found[0].reading(item.get("meridiem"))))
    return target.render(tuple(times), "compact")
