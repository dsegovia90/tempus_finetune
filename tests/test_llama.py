from pathlib import Path

from tempus.chat import MAX_NEW_TOKENS, chat_prompt, end_of_turn, prompt_ids
from tempus.llama import COMPACT_GRAMMAR, Options, parse, request, server_command


class FakeTokenizer:
    """Records what the chat template is asked for; ids are character codes."""

    def apply_chat_template(self, conversation, **kwargs):
        self.kwargs = kwargs
        return "<bos>" + "".join(f"<{m['role']}>{m['content']}" for m in conversation) + "<model>"

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) for c in text]}


def test_finetune_prompt_is_the_bare_message_with_thinking_off():
    tokenizer = FakeTokenizer()
    assert chat_prompt(tokenizer, None, "dinner at 7?") == "<bos><user>dinner at 7?<model>"
    assert tokenizer.kwargs == {
        "tokenize": False, "add_generation_prompt": True, "enable_thinking": False
    }
    assert chat_prompt(tokenizer, "spec", "hi").startswith("<bos><system>spec<user>hi")


def test_prompt_ids_tokenize_the_template_without_adding_bos():
    assert prompt_ids(FakeTokenizer(), None, "7") == [ord(c) for c in "<bos><user>7<model>"]


def test_end_of_turn_is_what_follows_a_reply_without_trailing_whitespace():
    class Template(FakeTokenizer):
        def apply_chat_template(self, conversation, **kwargs):
            return "".join(f"<{m['role']}>{m['content']}<|im_end|>\n" for m in conversation)

    assert end_of_turn(Template()) == "<|im_end|>"


def test_request_is_greedy_capped_and_uncached():
    body = request([2, 105], None)
    assert body == {
        "prompt": [2, 105], "n_predict": MAX_NEW_TOKENS, "temperature": 0, "top_k": 1,
        "cache_prompt": False,
    }
    assert request([2], COMPACT_GRAMMAR)["grammar"] == COMPACT_GRAMMAR


def test_parse_strips_the_reply_and_keeps_server_timings():
    response = {
        "content": "7=19:00\n",
        "stop_type": "eos",
        "timings": {"prompt_n": 12, "prompt_ms": 1.5, "predicted_n": 6, "predicted_ms": 4.2},
    }
    assert parse(response) == ("7=19:00", {
        "prompt_ms": 1.5, "generation_ms": 4.2, "prompt_tokens": 12, "generated_tokens": 6,
        "stop": "eos",
    })


def test_cpu_server_offloads_nothing_and_gpu_offloads_everything():
    options = Options(Path("runs/m/gguf/Q8_0.gguf"), "cpu", 4, False, "llama-server")
    cpu = server_command(options, 8081)
    assert cpu[cpu.index("--n-gpu-layers") + 1] == "0" and cpu[cpu.index("--device") + 1] == "none"
    assert cpu[cpu.index("--threads") + 1] == "4"
    gpu = server_command(Options(options.gguf, "gpu", 4, False, "llama-server"), 8081)
    assert gpu[gpu.index("--n-gpu-layers") + 1] == "all" and "--device" not in gpu


def test_grammar_allows_compact_lines_or_none():
    rules = dict(line.split(" ::= ") for line in COMPACT_GRAMMAR.strip().splitlines())
    assert set(rules) == {"root", "line", "hour", "minute"}
    assert rules["root"] == '"none" | line ("\\n" line)*'
    assert rules["line"] == '[^=\\n]+ "=" hour ":" minute'
    assert rules["hour"] == '[01] [0-9] | "2" [0-3]' and rules["minute"] == "[0-5] [0-9]"
