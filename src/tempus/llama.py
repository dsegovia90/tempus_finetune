"""Run a GGUF model through llama.cpp's `llama-server`, one message at a time.

The prompt is templated and tokenized here with the model's Hugging Face tokenizer,
exactly as `infer` does, and sent as token ids to the raw `/completion` endpoint. The
server then cannot change the prompt: no second chat template, no extra BOS.
"""

import http.client
import json
import socket
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from tempus.chat import MAX_NEW_TOKENS, prompt_ids

WARMUP_CALLS = 3
STARTUP_SECONDS = 120
CONTEXT = 1024  # tokens; prompts are a few dozen, replies at most MAX_NEW_TOKENS

# Compact replies only: `text=HH:MM` lines, or the single word `none`.
COMPACT_GRAMMAR = r"""root ::= "none" | line ("\n" line)*
line ::= [^=\n]+ "=" hour ":" minute
hour ::= [01] [0-9] | "2" [0-3]
minute ::= [0-5] [0-9]
"""


@dataclass(frozen=True)
class Options:
    gguf: Path  # its stem names the quantization: bf16, Q8_0, Q4_K_M
    device: str  # gpu (every layer offloaded) or cpu (no offload at all)
    threads: int  # CPU threads
    grammar: bool  # constrain compact replies with COMPACT_GRAMMAR
    binary: str  # the llama-server executable


def server_command(options: Options, port: int) -> list[str]:
    """The llama-server command line: one slot, small context, no web UI."""
    command = [
        options.binary, "--model", str(options.gguf), "--host", "127.0.0.1", "--port", str(port),
        "--parallel", "1", "--ctx-size", str(CONTEXT), "--no-webui",
        "--threads", str(options.threads), "--threads-batch", str(options.threads),
    ]
    if options.device == "gpu":
        return command + ["--n-gpu-layers", "all"]
    return command + ["--n-gpu-layers", "0", "--device", "none"]


def request(prompt: list[int] | str, grammar: str | None) -> dict:
    """A /completion body: greedy, capped like `infer`, stopping at the model's end of turn.

    `prompt` is token ids, or text that llama.cpp tokenizes itself (see `tempus.client`).
    """
    body = {
        "prompt": prompt,
        "n_predict": MAX_NEW_TOKENS,
        "temperature": 0,
        "top_k": 1,
        "cache_prompt": False,  # every message is processed from scratch, as with transformers
    }
    if grammar:
        body["grammar"] = grammar
    return body


def parse(response: dict) -> tuple[str, dict]:
    """The reply text and the server's own timings for one completion."""
    timings = response["timings"]
    return response["content"].strip(), {
        "prompt_ms": timings["prompt_ms"],
        "generation_ms": timings["predicted_ms"],
        "prompt_tokens": timings["prompt_n"],
        "generated_tokens": timings["predicted_n"],
        "stop": response["stop_type"],
    }


class Server:
    """A llama-server process for one GGUF file, reached over one persistent connection."""

    def __init__(self, model_id: str, options: Options, log: Path):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.options, self.log = options, log
        self.grammar = COMPACT_GRAMMAR if options.grammar else None
        self.timings: list[dict] = []  # from the timed calls of the last `latencies`
        self.truncated = 0  # replies of the last `generate` cut off at MAX_NEW_TOKENS

    def __enter__(self) -> "Server":
        port = _free_port()
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("w") as log:
            self.process = subprocess.Popen(
                server_command(self.options, port), stdout=log, stderr=subprocess.STDOUT
            )
        self.connection = http.client.HTTPConnection("127.0.0.1", port)
        try:
            self._wait_until_ready()
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.connection.close()
        self.process.terminate()
        self.process.wait()

    def complete(self, system: str | None, message: str) -> tuple[str, dict]:
        ids = prompt_ids(self.tokenizer, system, message)
        return parse(self._post("/completion", request(ids, self.grammar)))

    def generate(self, system: str | None, messages: list[str]) -> list[str]:
        """Return the model's raw reply to each message."""
        results = [self.complete(system, message) for message in messages]
        self.truncated = sum(timings["stop"] == "limit" for _, timings in results)
        return [reply for reply, _ in results]

    def latencies(self, system: str | None, messages: list[str]) -> list[float]:
        """Seconds to answer each message alone, measured at the client, after warmup calls."""
        for message in messages[:WARMUP_CALLS]:
            self.complete(system, message)
        seconds, self.timings = [], []
        for message in messages:
            if self.connection.sock is None:  # the server closes a connection after 100 requests
                self.connection.connect()  # reopen it here, so no timed call pays for the handshake
            started = time.perf_counter()
            _, timings = self.complete(system, message)
            seconds.append(time.perf_counter() - started)
            self.timings.append(timings)
        return seconds

    def details(self) -> dict:
        """Engine settings, model size, and the server's medians over the timed calls."""
        meta = self._get("/v1/models")["data"][0]["meta"]
        details = {
            "engine": "llama.cpp",
            "gguf": str(self.options.gguf),
            "quant": self.options.gguf.stem,
            "device": self.options.device,
            "grammar": self.options.grammar,
            "parameters_millions": round(meta["n_params"] / 1e6, 1),
            "weights_gb": round(self.options.gguf.stat().st_size / 2**30, 3),
            "truncated": self.truncated,
        }
        for key in ("prompt_ms", "generation_ms", "prompt_tokens", "generated_tokens"):
            details[f"server_{key}_p50"] = statistics.median(t[key] for t in self.timings)
        return details

    def _post(self, path: str, body: dict) -> dict:
        self.connection.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        return self._read()

    def _get(self, path: str) -> dict:
        self.connection.request("GET", path)
        return self._read()

    def _read(self) -> dict:
        response = self.connection.getresponse()
        body = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(f"llama-server answered {response.status}: {body}")
        return body

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + STARTUP_SECONDS
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited, see {self.log}")
            try:
                self._get("/health")
                return
            except (ConnectionError, RuntimeError):
                self.connection.close()
                time.sleep(0.2)
        raise TimeoutError(f"llama-server not ready after {STARTUP_SECONDS}s, see {self.log}")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
