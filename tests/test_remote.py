import io
import json

from tempus import remote
from tempus.schema import Example, Extraction


def test_evaluate_sends_spec_and_message_and_scores_replies(monkeypatch, tmp_path):
    sent = []

    def urlopen(call, timeout):
        body = json.loads(call.data)
        sent.append((call.full_url, dict(call.header_items()), body))
        reply = "7=19:00" if "dinner" in body["messages"][1]["content"] else "none"
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": reply}}]}).encode())

    monkeypatch.setattr(remote.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(remote, "system_prompt", lambda fmt: "RULES")
    examples = [
        Example("a", "implicit", "dinner at 7?", (Extraction("7", 1140),)),
        Example("b", "none", "room 3 is free", ()),
    ]
    summary = remote.evaluate(
        "http://box:8080/", "big", examples, tmp_path, key="k", extra={"think": False}, record={"quant": "Q4"}
    )
    url, headers, body = sent[-1]
    assert url == "http://box:8080/v1/chat/completions"
    assert headers["Authorization"] == "Bearer k"
    assert body["messages"][0] == {"role": "system", "content": "RULES"}
    assert body["temperature"] == 0 and body["think"] is False
    assert len(sent) == len(examples) * 2  # warm-up pass, then the scored pass
    assert summary["metrics"]["all"]["exact"] == 1.0
    assert summary["kind"] == "base" and summary["with_spec"] and summary["quant"] == "Q4"
    assert (tmp_path / "samples.jsonl").exists()
