"""Convert a vocabulary-trimmed Qwen3.5 model folder to GGUF with llama.cpp's converter.

The converter names the pre-tokenizer by hashing the token ids of a test string, and refuses
hashes it does not know. Trimming drops tokens of that string (Khmer, Chinese, Bulgarian), so
the hash changes, while the pre-tokenizer itself (NFC, then Qwen3.5's split regex) does not.
This names it "qwen35" by hand and runs the converter unchanged. Check the result by comparing
`llama-tokenize` ids with the Hugging Face ids. Run with the converter's Python:

    path/to/converter-venv/bin/python scripts/convert_trimmed.py path/to/llama.cpp \
        runs/<name>-trim/merged --outtype bf16 --outfile runs/<name>-trim/gguf/bf16.gguf
"""

import sys
from pathlib import Path


def main() -> None:
    llama_cpp, converter_args = Path(sys.argv[1]).expanduser(), sys.argv[2:]
    sys.path.insert(0, str(llama_cpp))
    import convert_hf_to_gguf
    from conversion.base import TextModel

    TextModel.get_vocab_base_pre = lambda self, tokenizer: "qwen35"
    sys.argv = [convert_hf_to_gguf.__file__, *converter_args]
    convert_hf_to_gguf.main()


if __name__ == "__main__":
    main()
