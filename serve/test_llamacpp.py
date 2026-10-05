"""serve/test_llamacpp.py - LlamaCppEngine (llama.cpp's llama-server as the engine, setup-mac.sh) against the fake
llama-server in serve/llamacpp_fake_server.py (no model, no GPU).

    python -m unittest serve.test_llamacpp -v
"""
from __future__ import annotations

import base64
import json
import os
import stat
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, EngineDied, LlamaCppEngine, LlamaCppVision, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FAKE = ROOT / "serve" / "llamacpp_fake_server.py"


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="strata-llamacpp-test-"))
        self.exe = self.dir / "llama-server"            # the engine runs an executable, as with the real one
        self.exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
        self.exe.chmod(self.exe.stat().st_mode | stat.S_IEXEC)
        self.log = self.dir / "fake.jsonl"
        self.env = {**os.environ, "FAKE_LOG": str(self.log)}
        self.engines = []

    def tearDown(self):
        for e in self.engines:
            e.close()

    def engine(self, **env):
        e = LlamaCppEngine(str(self.exe), "model.gguf", 4096, args=["--threads", "4"], env={**self.env, **env})
        self.engines.append(e)
        return e

    def events(self, key):
        return [x[key] for x in map(json.loads, self.log.read_text().splitlines()) if key in x]


class Engine(Base):
    def test_starts_llama_server_on_loopback(self):
        e = self.engine()
        self.assertTrue(e.alive())
        argv = self.events("argv")[0]
        for flag, value in (("-m", "model.gguf"), ("-c", "4096"), ("-np", "1"), ("-ngl", "99"), ("-fa", "on"),
                            ("--host", "127.0.0.1"), ("--threads", "4")):
            self.assertEqual(argv[argv.index(flag) + 1], value, flag)
        self.assertIn("--no-webui", argv)
        self.assertEqual(self.events("media_marker")[0], "<__media__>")   # llama-server's own is random per start
        self.assertEqual(e.max_context, 4096)
        self.assertEqual(e.info["version"], "llama.cpp 0.4.1-dev (3cf0325)")

    def test_streams_ids_and_records_timings(self):
        e = self.engine(FAKE_TOKENS="5,6,248046")
        out = list(e.generate(list(range(10)), 100, {}, threading.Event()))
        self.assertEqual(out, [5, 6, 248046])
        body = self.events("body")[0]
        self.assertEqual(body["prompt"], list(range(10)))
        self.assertEqual(body["n_predict"], 100)
        self.assertTrue(body["stream"] and body["return_tokens"] and body["cache_prompt"])
        self.assertEqual(body["temperature"], 0.0)        # no temperature means greedy, as with the Strata engine
        self.assertEqual(e.last, {"generated": 3, "prompt_tokens": 10, "prompt_ms": 120.5, "decode_ms": 80.25,
                                  "finish": "stop", "reused": 6})

    def test_consumer_stopping_at_end_of_turn_still_gets_timings(self):
        # the Service stops reading at <|im_end|>; the final chunk right after it must still fill `last`
        e = self.engine(FAKE_TOKENS="5,248046")
        for t in e.generate([1, 2, 3, 4, 5, 6, 7], 100, {}, threading.Event()):
            if t == 248046:
                break
        self.assertEqual(e.last["finish"], "stop")
        self.assertEqual(e.last["prompt_ms"], 120.5)

    def test_draft_counts_are_kept(self):
        # --spec-type ngram-simple: llama-server's draft_n / draft_n_accepted become the timings' draft fields
        e = self.engine(FAKE_DRAFTS="1")
        list(e.generate([1] * 8, 100, {}, threading.Event()))
        self.assertEqual((e.last["drafts_offered"], e.last["drafts_accepted"]), (4, 3))

    def test_limit_is_length(self):
        e = self.engine(FAKE_TOKENS="5,6,7,8")
        self.assertEqual(list(e.generate([1] * 8, 2, {}, threading.Event())), [5, 6])
        self.assertEqual(e.last["finish"], "length")

    def test_sampling_is_passed_through(self):
        e = self.engine()
        list(e.generate([1] * 8, 10, {"temperature": 0.7, "top_p": 0.9, "top_k": 20, "min_p": 0.05, "seed": 7,
                                      "repetition_penalty": 1.1, "frequency_penalty": 0.2, "presence_penalty": 0.3,
                                      "penalty_last_n": 32}, threading.Event()))
        body = self.events("body")[0]
        self.assertEqual({k: body[k] for k in ("temperature", "top_p", "top_k", "min_p", "seed", "repeat_penalty",
                                               "frequency_penalty", "presence_penalty", "repeat_last_n")},
                         {"temperature": 0.7, "top_p": 0.9, "top_k": 20, "min_p": 0.05, "seed": 7,
                          "repeat_penalty": 1.1, "frequency_penalty": 0.2, "presence_penalty": 0.3,
                          "repeat_last_n": 32})

    def test_cancel_closes_the_request(self):
        e = self.engine(FAKE_TOKENS=",".join(["5"] * 50), FAKE_DELAY_S="0.05")
        cancel, got = threading.Event(), []
        for t in e.generate([1] * 8, 100, {}, cancel):
            if t is not None:
                got.append(t)
                cancel.set()
        self.assertLess(len(got), 5)
        deadline = time.time() + 5
        while not self.events("aborted_after") and time.time() < deadline:
            time.sleep(0.05)
        self.assertTrue(self.events("aborted_after"), "llama-server was not told to stop")

    def test_death_is_engine_died_and_restart_recovers(self):
        e = self.engine(FAKE_DIE="1")
        with self.assertRaises(EngineDied):
            list(e.generate([1] * 8, 10, {}, threading.Event()))
        self.assertFalse(e.alive())
        e.spawn[-1].pop("FAKE_DIE")
        e.restart()
        self.assertTrue(e.alive())
        self.assertEqual(list(e.generate([1] * 8, 10, {}, threading.Event())), [9419, 1017, 248046])

    def test_unload_then_restart(self):
        e = self.engine()
        e.unload()
        self.assertFalse(e.alive())
        self.assertTrue(e.unloaded)
        e.restart()
        self.assertTrue(e.alive())
        self.assertFalse(e.unloaded)

    def test_images_go_as_prompt_string_and_base64(self):
        # Service joins each image's file into one: one base64 line per image, in prompt order
        tok, e = ByteTokenizer(), self.engine()
        ids = tok.encode("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is it?<|im_end|>\n"
                         "<|vision_start|><|image_pad|><|vision_end|>", parse_special=True)
        emb = self.dir / "req.sve"
        emb.write_text("QUFB\nQkJC\n")
        self.assertEqual(list(e.generate(ids, 10, {}, threading.Event(), embeddings=str(emb))), [9419, 1017, 248046])
        body = self.events("body")[0]
        self.assertEqual(body["prompt"], {"prompt_string": "<|im_start|>user\n<__media__>What is it?<|im_end|>\n"
                                                           "<__media__>", "multimodal_data": ["QUFB", "QkJC"]})

    def test_images_and_markers_must_match(self):
        tok, e = ByteTokenizer(), self.engine()
        emb = self.dir / "req.sve"
        emb.write_text("QUFB\nQkJC\n")
        ids = tok.encode("<|vision_start|><|image_pad|><|vision_end|>hi", parse_special=True)
        with self.assertRaises(ValueError):
            list(e.generate(ids, 10, {}, threading.Event(), embeddings=str(emb)))


class OverHttp(Base):
    def test_chat_completion_through_the_service(self):
        tok = ByteTokenizer()
        ids = tok.encode("</think>\n\nHi there") + tok.encode("<|im_end|>", parse_special=True)
        e = self.engine(FAKE_TOKENS=",".join(map(str, ids)))
        svc = Service(e, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                                         data=json.dumps({"model": "m", "messages": [{"role": "user",
                                                                                      "content": "hi"}]}).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                b = json.loads(r.read())
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertEqual(b["choices"][0]["message"]["content"], "Hi there")
        self.assertEqual(b["choices"][0]["finish_reason"], "stop")
        self.assertEqual(b["timings"]["cache_n"], 6)


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==")


class Images(Base):
    def test_vision_keeps_the_image_as_one_base64_line(self):
        v = LlamaCppVision()
        path, n = v.encode("data:image/png;base64," + base64.b64encode(PNG).decode())
        self.assertEqual(n, 1)                           # llama-server counts the image's tokens itself
        self.assertEqual(Path(path).read_text(), base64.b64encode(PNG).decode() + "\n")
        self.assertEqual(v.encode("data:image/png;base64," + base64.b64encode(PNG).decode())[0], path)
        self.assertTrue(v.alive())
        v.close()

    def test_chat_with_an_image_through_the_service(self):
        tok = ByteTokenizer()
        ids = tok.encode("</think>\n\nA dot") + tok.encode("<|im_end|>", parse_special=True)
        e = self.engine(FAKE_TOKENS=",".join(map(str, ids)))
        svc = Service(e, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=LlamaCppVision())
        httpd = serve(svc, port=0)
        url = "data:image/png;base64," + base64.b64encode(PNG).decode()
        msg = {"role": "user", "content": [{"type": "text", "text": "What is it?"},
                                           {"type": "image_url", "image_url": {"url": url}}]}
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                                         data=json.dumps({"model": "m", "messages": [msg]}).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                b = json.loads(r.read())
        finally:
            httpd.shutdown()
            httpd.server_close()
            svc.vision.close()
        self.assertEqual(b["choices"][0]["message"]["content"], "A dot")
        prompt = self.events("body")[0]["prompt"]
        self.assertEqual(prompt["multimodal_data"], [base64.b64encode(PNG).decode()])
        self.assertEqual(prompt["prompt_string"].count("<__media__>"), 1)
        self.assertNotIn("<|image_pad|>", prompt["prompt_string"])


class Config(unittest.TestCase):
    def test_engine_from_config(self):
        from serve.server import llamacpp_engine_from_config
        cfg = {"engine": "llamacpp", "exe": "bin/llama-server", "model": "m.gguf", "max_context": 8192,
               "args": ["--threads", "8"], "cwd": "/opt/strata", "log": "/tmp/x.log"}
        with mock.patch("serve.server.LlamaCppEngine") as cls:
            llamacpp_engine_from_config(cfg, env={"A": "1"})
        cls.assert_called_once_with("/opt/strata/bin/llama-server", "/opt/strata/m.gguf", 8192, args=["--threads", "8"],
                                    log="/tmp/x.log", env={"A": "1"})

    def test_mmproj_is_passed_to_llama_server(self):
        from serve.server import llamacpp_engine_from_config
        cfg = {"exe": "s", "model": "m.gguf", "mmproj": "v.gguf", "cwd": "/opt/strata", "args": ["--threads", "8"]}
        with mock.patch("serve.server.LlamaCppEngine") as cls:
            llamacpp_engine_from_config(cfg)
        self.assertEqual(cls.call_args.kwargs["args"], ["--threads", "8", "--mmproj", "/opt/strata/v.gguf"])


IOREG = b'''+-o AGXAcceleratorG13X  <class AGXAcceleratorG13X>
    {
      "model" = "Apple M1 Max"
      "gpu-core-count" = 32
      "PerformanceStatistics" = {"In use system memory"=400179200,"Device Utilization %"=96,"Alloc system memory"=43872518144}
    }
'''


class AppleTelemetry(unittest.TestCase):
    """The Mac's GPU and CPU in the web app's About and Monitor tabs (ioreg / sysctl, no sudo)."""

    def run_(self, out):
        return mock.patch("serve.telemetry.subprocess.run", return_value=mock.Mock(returncode=0, stdout=out))

    def test_gpu_from_ioreg(self):
        from serve import telemetry
        with self.run_(IOREG), mock.patch.object(telemetry.sys, "platform", "darwin"):
            g = telemetry.gpu_reader(0)
            self.assertTrue(g.ok())
            self.assertEqual(g.name(), "Apple M1 Max (32-core GPU)")
            r = g.read()
        self.assertEqual(r["util"], 96)
        self.assertEqual(r["mem_used"], 43872518144)

    def test_no_accelerator_is_not_ok(self):
        from serve import telemetry
        with self.run_(b""), mock.patch.object(telemetry.sys, "platform", "darwin"):
            self.assertFalse(telemetry.gpu_reader(0).ok())

    def test_cpu_name_from_sysctl(self):
        from serve import telemetry
        with self.run_(b"Apple M1 Max\n"), mock.patch.object(telemetry.sys, "platform", "darwin"), \
                mock.patch.object(telemetry.os, "name", "posix"):
            self.assertEqual(telemetry._cpu_name(), "Apple M1 Max")


if __name__ == "__main__":
    unittest.main()
