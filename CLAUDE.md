# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Read `AGENTS.md` first: it covers what Strata is, how to install it for a user (`docs/AI_SETUP.md`), and the doc
style rules (plain words, measured numbers with the hardware they were measured on, no claims without a measurement).

## Commands

Python tests use `unittest` and need no GPU, model or network (they run against `MockEngine` or fakes):

```sh
python -m unittest serve.test_server -v                       # one server test module
python -m unittest serve.test_server.MaxTokens -v             # one test class
python tools/test_setup_choices.py                            # setup.py tests: tools/test_setup_<name>.py
python -m unittest discover -s serve -p 'test_*.py'           # all server tests
```

Run the API server without a GPU (scripted engine), or with the real engine:

```sh
python -m serve.server --engine mock --port 8095
python -m serve.server --engine strata --config strata.json --port 8080
```

Engine (C++20 / CUDA / HIP, CMake + Ninja; `setup.py` drives the same build through `cmake_build`):

```sh
cmake -S . -B build -G Ninja -DSTRATA_ENABLE_CUDA=ON          # NVIDIA; sm_75+ (sm_60/70 via STRATA_EXPERIMENTAL_SM60)
cmake -S . -B build-hip -G Ninja -DSTRATA_ENABLE_HIP=ON       # AMD wave32 RDNA; see docs/AMD_HIP.md for the full line
cmake --build build --target strata
ctest --test-dir build --output-on-failure -R <name>          # one C++ test
```

Without CUDA/HIP only the CPU-side tools build (`strata-gguf`, `strata-dequant`, `strata-plan`, a few tests).
`STRATA_BUILD_TESTS` defaults ON only when `tests/CMakeLists.txt` and `bench/micro/` exist; this checkout has
neither, so the `*_parity` targets from `bench/micro` are not available. Other backends are separate, mutually
exclusive options: `STRATA_HIP_GFX906` (MI50, via the CUDA sources + `include/strata/platform/hip_compat/`) and
`STRATA_ENABLE_SYCL` (Intel Arc, its own project in `sycl/`). The vision encoder is a separate CMake project in
`tools/vision` (target `strata-vision`).

## Architecture

Three layers, each talking to the next through a narrow boundary:

1. **`setup.py`** (started by `START-HERE.bat` / `setup.sh`; `UPDATE.bat` / `update.sh` update only): checks the PC,
   picks model size by RAM, installs `.venv` from the pinned `requirements.txt`, downloads a prebuilt engine or
   compiles one, downloads the GGUF model, packs it for Strata (`tools/strata_pack.py`, `tools/mtp_pack.py`, ...),
   writes a JSON config and `run-<model>.bat/.sh`, then starts the server. Its tests in `tools/test_setup_*.py`
   stub out the hardware probes; `tools/test_setup_golden.json` pins generated configs.
2. **`serve/server.py`**: a stdlib `ThreadingHTTPServer` exposing OpenAI (`/v1/chat/completions`, Responses API in
   `serve/responses.py`) and Anthropic (`/v1/messages`) endpoints plus the web app (`serve/web/`). The engine
   boundary is `Engine.generate(prompt_ids, max_new, sampling, cancel) -> token ids`. `StrataEngine` keeps one
   `strata --serve` process resident and speaks a line protocol over stdin/stdout (`GEN`, `GENI` for image
   embeddings, `STOP`, `VRAM`, `QUIT`); `MockEngine` makes every API path testable without a GPU. Chat templating
   and output parsing (think blocks, tool calls) live in `serve/frontend.py` + `serve/chat_template.jinja`; MCP
   tool hosting in `serve/mcp.py`; JSON-schema output in `serve/structured.py`; config loading in
   `serve/runconfig.py`. Requests that exceed the context are rejected with 400, never truncated. One resident
   sequence at a time behind a FIFO (batching: `docs/BATCHING.md`).
   `tools/strata_mcp.py` is a separate MCP server that lets AI assistants install/start/stop Strata.
3. **The engine** (`src/`, `include/strata/`): `src/program/generate.cpp` is the `strata` executable and its
   `--serve` loop. The MoE model's 24,576 experts are split across the machine: GPU holds dense layers, KV cache
   and an adaptive **expert cache** (`src/core/expert_cache.cpp`) filling remaining VRAM; all experts sit pinned in
   RAM and the CPU computes cache misses in place, concurrently with the GPU (`src/kernels/cpu`, ggml-cpu for
   i-quants from `third_party/ggml`); a 28.8 GB n-gram table is read from SSD (`src/ngram`). Decoding is
   speculative: the model's MTP layer drafts up to 3 tokens and prompt lookup up to 5 (`src/spec/`), verified in
   one pass. Prompt processing (`src/prefill/`) runs in chunks up to 8,192 tokens, streaming the next layer's
   experts over PCIe. Conversation state is cached between turns (`src/core/conversation_*`). Kernels in
   `src/kernels/cuda` are compiled as HIP for AMD through the compat headers; `*_parity.cpp` files check GPU
   kernels against CPU references.

Full explanation and every measured number: `docs/DETAILS.md`, `docs/HOW_IT_WORKS.md`, `docs/paper/Strata-Paper.pdf`.

## Apple Silicon (this fork)

The CUDA/HIP engine does not run on macOS. On a Mac, `./setup-mac.sh` builds llama.cpp with Metal (at setup.py's
`LLAMA_CPP_COMMIT`, into `third_party/llama.cpp`), downloads IQ2_XS into `models/IQ2_XS/`, extracts the tokenizer, and
writes `strata-mac.json` + `run-mac.sh`. The server then runs `--engine llamacpp`: `LlamaCppEngine` in
`serve/server.py` keeps `llama-server` as a child on a free loopback port and sends it token ids over `/completion`
(`return_tokens`, `cache_prompt`), so templating, parsing, the APIs and the web app stay Strata's. No images, batch
slots or VRAM reserve on this engine. Tests: `python -m unittest serve.test_llamacpp` (fake server:
`serve/llamacpp_fake_server.py`). Use `.venv/bin/python`: macOS's own `python3` is 3.9, too old for the tools.

## Conventions

- The server must never listen beyond `127.0.0.1` without `--api-key`.
- Comments cite issue/PR numbers (`#533`) and the measurement behind a choice; keep that habit when changing code.
- `README.md` has six translations (`README.<lang>.md`); a README change needs them updated too.
