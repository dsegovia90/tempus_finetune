"""Copy a trained tagger's ONNX exports into the page, with the Python pipeline's eval replies.

    uv run --group classifier python web/export_tagger.py runs/<name>

Needs `tempus tagger-export --model runs/<name>/model` first. Writes web/models/tagger/
(<variant>.onnx and vocab.txt) and web/eval/tagger-<variant>.json: each of the 530 eval messages
with the reply and token ids the Python pipeline gives with that same ONNX file.
"""

import argparse
import json
import shutil
from pathlib import Path

from tempus.schema import load_examples
from tempus.tagger import Tagger

WEB = Path(__file__).parent
VARIANTS = ("int8",)  # the export the page offers
EVAL_SETS = [Path("data/eval/core.jsonl"), Path("data/generated/eval.jsonl")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="runs/<name>, with model/ and onnx/")
    args = parser.parse_args()

    out = WEB / "models" / "tagger"
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = json.loads((args.run / "model" / "tokenizer.json").read_text())
    vocab = sorted(tokenizer["model"]["vocab"].items(), key=lambda item: item[1])
    assert [i for _, i in vocab] == list(range(len(vocab)))
    (out / "vocab.txt").write_text("\n".join(token for token, _ in vocab))

    examples = [e for path in EVAL_SETS for e in load_examples(path)]
    for variant in VARIANTS:
        onnx = args.run / "onnx" / f"{variant}.onnx"
        shutil.copyfile(onnx, out / f"{variant}.onnx")
        tagger = Tagger(args.run / "model", onnx)
        items = [
            {"id": e.id, "text": e.text, "expected": tagger.reply(e.text), "ids": tagger.tokens(e.text)[0]}
            for e in examples
        ]
        reference = {"reference": f"the Python pipeline ({onnx}, ONNX Runtime)", "items": items}
        (WEB / "eval" / f"tagger-{variant}.json").write_text(json.dumps(reference, ensure_ascii=False) + "\n")
        print(f"{variant}: {onnx.stat().st_size / 2**20:.2f} MiB, {len(items)} reference replies")


if __name__ == "__main__":
    main()
