"""Trim a merged finetune's vocabulary to the tokens English chat can produce.

Kept: every token that text written with `CHARS` (ASCII, Latin letters, general punctuation,
currency signs, emoji and other symbols) can tokenize into, every token of the given
conversations (training and eval messages with their replies), the 256 byte tokens and all
added tokens. Any message written with `CHARS` then tokenizes exactly as before, up to the id
renumbering (see `vocab`). The tokenizer is rewritten, the embedding rows (tied with the output
layer) are cut to the kept ids in their new order, and a normal model folder is saved with
`vocab_map.json`, the old id of each new id. Needs torch and safetensors (the gpu group).
"""

import json
import shutil
from pathlib import Path

from tempus.chat import chat_prompt
from tempus.schema import Example
from tempus.target import render
from tempus.vocab import script_tokens, trim

CHAR_RANGES = (
    (0x00, 0x250),  # ASCII, Latin-1, Latin Extended-A and -B
    (0x300, 0x370),  # combining accents (text NFC cannot compose)
    (0x1E00, 0x1F00),  # Latin Extended Additional (Vietnamese and Welsh names)
    (0x2000, 0x2070),  # general punctuation: curly quotes, dashes, ellipsis, zero-width joiner
    (0x20A0, 0x20C1),  # currency signs
    (0x2100, 0x2150),  # letterlike symbols (™, ℹ)
    (0x2190, 0x2200),  # arrows
    (0x2300, 0x2400),  # technical symbols (⌚, ⏰)
    (0x25A0, 0x27C0),  # shapes, dingbats, miscellaneous symbols (☀, ✅, ❤)
    (0x2B00, 0x2C00),  # more arrows and stars (⭐)
    (0x3030, 0x3031), (0x303D, 0x303E), (0x3297, 0x3298), (0x3299, 0x329A),  # emoji outside the blocks
    (0xFE00, 0xFE10),  # variation selectors (❤️)
    (0x1D400, 0x1D800),  # mathematical letters, used as fancy fonts
    (0x1F000, 0x1FB00),  # emoji, flags, skin tones
    (0xE0020, 0xE0080),  # tag characters (subdivision flags)
)
CHARS = [chr(c) for start, end in CHAR_RANGES for c in range(start, end)]
WEIGHT_FILES = ("*.safetensors", "model.safetensors.index.json")
REWRITTEN = ("tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json")


def conversations(tokenizer, examples: list[Example]) -> list[str]:
    """Each message as the model sees it in training: prompt, compact reply, end of turn."""
    return [
        chat_prompt(tokenizer, None, e.text) + render(e.times, "compact") + tokenizer.eos_token
        for e in examples
    ]


def keep_set(spec: dict, texts: list[str]) -> set[str]:
    """The script tokens and every token of `texts` (`trim` adds bytes and added tokens)."""
    from tokenizers import Tokenizer

    backend = Tokenizer.from_str(json.dumps(spec))
    seen = {t for encoding in backend.encode_batch(texts, add_special_tokens=False) for t in encoding.tokens}
    return script_tokens(spec["model"]["vocab"], CHARS) | seen


def trim_model(model: Path, examples: list[Example], out: Path) -> dict:
    """Write the trimmed model folder to `out` and check the examples tokenize as before."""
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    spec = json.loads((model / "tokenizer.json").read_text())
    texts = conversations(tokenizer, examples)
    trimmed, old_ids = trim(spec, keep_set(spec, texts))
    new_id = {old: new for new, old in enumerate(old_ids)}

    config = json.loads((model / "config.json").read_text())
    text_config = config.get("text_config", config)
    rows = text_config["vocab_size"]
    tensors = {}
    for path in sorted(model.glob("*.safetensors")):
        tensors |= load_file(path)
    sliced = [name for name, tensor in tensors.items() if tensor.ndim == 2 and tensor.shape[0] == rows]
    if not sliced or not all(name.endswith(("embed_tokens.weight", "lm_head.weight")) for name in sliced):
        raise ValueError(f"expected the embedding (and maybe an output layer) with {rows} rows, got {sliced}")
    for name in sliced:
        tensors[name] = tensors[name][old_ids].contiguous()
    text_config["vocab_size"] = len(old_ids)

    out.mkdir(parents=True, exist_ok=True)
    for path in model.iterdir():
        if path.name not in REWRITTEN and not any(path.match(p) for p in WEIGHT_FILES):
            shutil.copy2(path, out / path.name)
    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
    _write_json(out / "tokenizer.json", trimmed)
    _write_json(out / "config.json", remap_token_ids(config, new_id))
    _write_json(out / "generation_config.json", remap_token_ids(_read_json(model / "generation_config.json"), new_id))
    tokenizer_config = _read_json(model / "tokenizer_config.json")
    tokenizer_config["added_tokens_decoder"] = {
        str(new_id[int(old)]): token for old, token in tokenizer_config["added_tokens_decoder"].items()
    }
    _write_json(out / "tokenizer_config.json", tokenizer_config)
    _write_json(out / "vocab_map.json", old_ids)

    changed = check_tokenization(tokenizer, AutoTokenizer.from_pretrained(out), new_id, texts)
    if changed:
        raise RuntimeError(f"{len(changed)} conversations tokenize differently, e.g. {changed[0]!r}")
    return {"vocab_before": rows, "vocab_after": len(old_ids), "sliced": sliced, "conversations": len(texts)}


def check_tokenization(old, new, new_id: dict[int, int], texts: list[str]) -> list[str]:
    """The texts whose new ids are not their old ids mapped to the new numbering."""
    encode = lambda tokenizer, text: tokenizer(text, add_special_tokens=False)["input_ids"]
    return [t for t in texts if encode(new, t) != [new_id.get(i) for i in encode(old, t)]]


def remap_token_ids(config: dict, new_id: dict[int, int]) -> dict:
    """Renumber every `*_token_id` value (an id or a list of ids), at any depth."""
    def renumber(key: str, value):
        if isinstance(value, dict):
            return {k: renumber(k, v) for k, v in value.items()}
        if not key.endswith("_token_id") or value is None:
            return value
        return [new_id[v] for v in value] if isinstance(value, list) else new_id[value]

    return renumber("", config)


def _read_json(path: Path):
    return json.loads(path.read_text())


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
