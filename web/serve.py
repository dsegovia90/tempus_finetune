"""Serve web/ locally with the headers wllama needs for multi-threading.

    uv run python web/serve.py                  # http://127.0.0.1:8000, cross-origin isolated
    uv run python web/serve.py --no-isolation   # without COOP/COEP: wllama falls back to one thread
"""

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ISOLATION = {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
}


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".json": "application/json",
        ".wasm": "application/wasm",
        ".gguf": "application/octet-stream",
    }
    isolated = True

    def end_headers(self) -> None:
        if self.isolated:
            for name, value in ISOLATION.items():
                self.send_header(name, value)
        self.send_header("Cache-Control", "no-cache")  # the model itself is cached by wllama (OPFS)
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-isolation", action="store_true", help="omit the COOP/COEP headers")
    args = parser.parse_args()

    Handler.isolated = not args.no_isolation
    handler = functools.partial(Handler, directory=str(Path(__file__).parent))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        mode = "cross-origin isolated" if Handler.isolated else "NOT cross-origin isolated"
        print(f"Serving web/ at http://127.0.0.1:{args.port}/ ({mode})", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
