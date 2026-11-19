"""The prompt every engine sends: one user message in the model's own chat template.

Kept free of torch so the llama.cpp client can build exactly the same tokens.
"""

MAX_NEW_TOKENS = 160  # far above the longest label; a reply this long is already wrong


def chat_prompt(tokenizer, system: str | None, message: str) -> str:
    """The templated prompt, generation prompt added, thinking mode off.

    Base models get the labelling spec as `system`. Finetunes get None: the task is
    in their weights, so they see only the message.
    """
    preamble = [{"role": "system", "content": system}] if system else []
    return tokenizer.apply_chat_template(
        [*preamble, {"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def prompt_ids(tokenizer, system: str | None, message: str) -> list[int]:
    """The prompt's token ids. The template already holds BOS, so none is added."""
    return tokenizer(chat_prompt(tokenizer, system, message), add_special_tokens=False)["input_ids"]


def end_of_turn(tokenizer) -> str:
    """What the template writes right after an assistant reply, trailing whitespace dropped.

    Taken from a rendered conversation, since only the reply's end is the same there: some
    templates (MiniCPM5) leave out the empty thinking block that the generation prompt adds.
    """
    marker = "REPLY_MARKER"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": marker}],
        tokenize=False,
        enable_thinking=False,
    )
    return text.split(marker, 1)[1].rstrip()
