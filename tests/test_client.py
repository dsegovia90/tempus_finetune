import io
import json
import urllib.error
from pathlib import Path

import pytest

from tempus import client
from tempus.chat import MAX_NEW_TOKENS
from tempus.client import PREFIX, SUFFIX, Result, Time, extract, read

WEB_PROMPT = Path(__file__).parents[1] / "web" / "prompts" / "lfm2.5-230m-v2.json"


def fake_server(monkeypatch, content: str, stop: str = "eos") -> list:
    """Answer every /completion with `content`; return the list of requests sent."""
    sent = []

    def urlopen(call, timeout):
        sent.append((call.full_url, json.loads(call.data), timeout))
        timings = {"prompt_n": 15, "prompt_ms": 30.0, "predicted_n": 6, "predicted_ms": 40.0}
        return io.BytesIO(json.dumps({"content": content, "stop_type": stop, "timings": timings}).encode())

    monkeypatch.setattr(client.urllib.request, "urlopen", urlopen)
    return sent


def test_extract_sends_the_training_prompt_greedily_and_returns_times(monkeypatch):
    sent = fake_server(monkeypatch, "7=19:00\nnoon=12:00\n")
    result = extract("dinner at 7? or lunch at noon", url="http://box:9000", timeout=2.0)
    assert result == Result([Time("7", "19:00", 1140), Time("noon", "12:00", 720)], "7=19:00\nnoon=12:00", False)
    assert sent == [("http://box:9000/completion", {
        "prompt": "<|im_start|>user\ndinner at 7? or lunch at noon<|im_end|>\n<|im_start|>assistant\n",
        "n_predict": MAX_NEW_TOKENS, "temperature": 0, "top_k": 1, "cache_prompt": False,
    }, 2.0)]


def test_none_is_no_times_and_not_malformed():
    assert read("give me 10", "none", "eos") == Result([], "none", False)


@pytest.mark.parametrize("reply", ["7pm", "7=7pm", "7=24:00", "", "none\n7=19:00", "sure! 7=19:00"])
def test_an_off_format_reply_gives_no_times_and_is_flagged(reply):
    assert read("dinner at 7", reply, "eos") == Result([], reply, True)


def test_a_reply_cut_at_the_cap_is_flagged_even_if_it_parses():
    assert read("at 7", "7=19:00", "limit") == Result([], "7=19:00", True)


def test_a_time_whose_text_is_not_in_the_message_is_flagged():
    assert read("dinner at seven", "7=19:00", "eos") == Result([], "7=19:00", True)


def test_server_errors_raise_instead_of_reading_as_no_times(monkeypatch):
    def urlopen(call, timeout):
        raise urllib.error.HTTPError(call.full_url, 400, "context exceeded", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr(client.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError):
        extract("at 7")


def test_template_text_is_the_exported_one_with_bos_left_to_llama_cpp():
    prompt = json.loads(WEB_PROMPT.read_text())
    assert (prompt["prefix"], prompt["suffix"]) == (PREFIX, SUFFIX)
    assert prompt["add_bos"] and prompt["prefix_ids"][0] == prompt["bos_token_id"] == 1
