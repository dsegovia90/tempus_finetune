"""Export a GGUF model's prompt template and eval references, as token ids, for the browser page.

    uv run --with transformers --with jinja2 python web/export_eval.py \
        runs/lfm2.5-230m-v2-s1-compact/merged lfm2.5-230m-v2 \
        Q8_0=results/lfm2.5-230m-v2-s1-compact/samples.jsonl

Writes web/prompts/NAME.json (the chat template around one user message, as the text the page
sends and as the native ids) and, per QUANT=SAMPLES pair, web/eval/NAME-QUANT.json (each eval
message with its full native prompt ids and the native llama.cpp reply). Native ids come from
`tempus.chat.prompt_ids`, the function the native client uses.

wllama takes text only and llama.cpp adds BOS itself when the GGUF says so. A template that
starts with BOS (LFM2.5) therefore loses it from the text and gets `add_bos: true`; the page
checks that the GGUF agrees. Every eval prompt is checked to tokenize, as the page will send it,
to exactly the native ids.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from tempus.chat import MAX_NEW_TOKENS, chat_prompt, prompt_ids
from tempus.jsonl import read_jsonl

WEB = Path(__file__).parent
SLOT = "\x00MESSAGE\x00"  # stands in for the message while the template is split


def template(tokenizer) -> dict:
    """The text before and after the message, without a leading BOS, and their native ids."""
    prefix, suffix = chat_prompt(tokenizer, None, SLOT).split(SLOT)
    add_bos = tokenizer.bos_token is not None and prefix.startswith(tokenizer.bos_token)
    encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]
    return {
        "prefix": prefix.removeprefix(tokenizer.bos_token) if add_bos else prefix,
        "suffix": suffix,
        "add_bos": add_bos,
        "prefix_ids": encode(prefix),
        "suffix_ids": encode(suffix),
        "bos_token_id": tokenizer.bos_token_id if add_bos else None,
        "eos_token": tokenizer.eos_token,
        "max_new_tokens": MAX_NEW_TOKENS,
    }


def browser_ids(tokenizer, prompt: dict, message: str) -> list[int]:
    """The ids of the text the page sends, with the BOS llama.cpp adds."""
    ids = tokenizer(prompt["prefix"] + message + prompt["suffix"], add_special_tokens=False)["input_ids"]
    return [prompt["bos_token_id"], *ids] if prompt["add_bos"] else ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tokenizer", type=Path, help="directory with the model's tokenizer files")
    parser.add_argument("name", help="model name: writes prompts/NAME.json and eval/NAME-QUANT.json")
    parser.add_argument("samples", nargs="+", metavar="QUANT=SAMPLES", help="samples.jsonl of a native llama.cpp run")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prompt = template(tokenizer)
    (WEB / "prompts").mkdir(exist_ok=True)
    (WEB / "prompts" / f"{args.name}.json").write_text(json.dumps(prompt, ensure_ascii=False, indent=1) + "\n")
    print(f"prompts/{args.name}.json: add_bos {prompt['add_bos']}, prefix {prompt['prefix']!r}, suffix {prompt['suffix']!r}")

    for pair in args.samples:
        quant, samples = pair.split("=", 1)
        items = []
        for sample in read_jsonl(Path(samples)):
            ids = prompt_ids(tokenizer, None, sample["text"])
            assert ids == browser_ids(tokenizer, prompt, sample["text"]), sample["id"]
            assert ids.count(tokenizer.bos_token_id) == (1 if prompt["add_bos"] else 0), sample["id"]
            items.append({"id": sample["id"], "text": sample["text"], "expected": sample["reply"], "ids": ids})
        out = WEB / "eval" / f"{args.name}-{quant}.json"
        out.write_text(json.dumps({"reference": f"native llama.cpp ({samples})", "items": items}, ensure_ascii=False) + "\n")
        print(f"{out.relative_to(WEB)}: {len(items)} messages, browser text tokenizes to the native ids")


if __name__ == "__main__":
    main()
