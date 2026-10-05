"""serve/llamacpp_fake_server.py - a stand-in for llama.cpp's llama-server, for serve/test_llamacpp.py (no model).

    python serve/llamacpp_fake_server.py --port 8199 [any other llama-server flags, ignored]

GET /health answers {"status": "ok"}.  POST /completion streams the token ids in FAKE_TOKENS (comma-separated, one
SSE chunk each, FAKE_DELAY_S apart) and then the final chunk with stop_type and timings, as llama-server does with
"stream" and "return_tokens" on: stop_type "limit" when n_predict cut the list short, else FAKE_STOP_TYPE ("eos").
FAKE_DIE=1 exits in the middle of the first answer.  With FAKE_LOG set, every request body and the command line are
appended to that file, one JSON line each.
"""
from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def log(event: dict):
    path = os.environ.get("FAKE_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            self._json({"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        log({"body": body})
        tokens = [int(t) for t in os.environ.get("FAKE_TOKENS", "9419,1017,248046").split(",")]
        n = int(body.get("n_predict", len(tokens)))
        stop_type = "limit" if n < len(tokens) else os.environ.get("FAKE_STOP_TYPE", "eos")
        tokens = tokens[:n]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i, t in enumerate(tokens):
            if os.environ.get("FAKE_DIE") and i == 1:
                os._exit(3)
            time.sleep(float(os.environ.get("FAKE_DELAY_S", "0")))
            try:
                self._event({"content": "", "tokens": [t], "stop": False})
            except OSError:                              # the client closed the connection: llama-server stops too
                log({"aborted_after": i})
                return
        self._event({"content": "", "tokens": [], "stop": True, "stop_type": stop_type,
                     "timings": {"cache_n": 6, "prompt_n": len(body["prompt"]) - 6, "prompt_ms": 120.5,
                                 "predicted_n": len(tokens), "predicted_ms": 80.25}})

    def _event(self, obj):
        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.flush()

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    if "--version" in sys.argv:                          # as llama-server prints it (to stderr, after its log line)
        print("version: 0.4.1-dev (build 1, commit 3cf0325)", file=sys.stderr)
        sys.exit(0)
    log({"argv": sys.argv[1:]})
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[sys.argv.index("--port") + 1])), Handler).serve_forever()
