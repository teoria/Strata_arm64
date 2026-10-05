#!/bin/sh
# Strata for Apple Silicon Macs: llama.cpp's Metal engine behind Strata's own server (serve/server.py --engine
# llamacpp), so the web app, the OpenAI/Anthropic APIs and the MCP tools are the same as on Windows and Linux.
# The first run builds llama.cpp, downloads Qwen3.8-Flash-Next IQ2_XS (68 GB) and starts it; later runs just start it.
# Needs Xcode's command line tools (xcode-select --install), cmake (brew install cmake) and Python 3.10+.
#
#   ./setup-mac.sh                 build, download, start on http://127.0.0.1:8080
#   ./setup-mac.sh --no-start      everything but the start
#   STRATA_CONTEXT=65536 STRATA_PORT=8081 ./setup-mac.sh     (read only when strata-mac.json is first written)
set -eu
cd "$(dirname "$0")"
[ "$(uname -s)/$(uname -m)" = Darwin/arm64 ] || { echo "setup-mac.sh is for Apple Silicon Macs"; exit 1; }

Q=IQ2_XS
REPO=ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
# the same pins as setup.py (#214): llama.cpp's commit and the model repository's revision
COMMIT=$(sed -n 's/^LLAMA_CPP_COMMIT = "\([0-9a-f]*\)"/\1/p' setup.py)
REV=$(sed -n "s|^ *\"$REPO\": \"\([0-9a-f]*\)\".*|\1|p" setup.py)
# SHA-256 of the two shards at that revision (Hugging Face's LFS ids)
SHA1=92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7
SHA2=316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113

# 1. Python 3.10+ in .venv, with the pinned packages
if [ ! -x .venv/bin/python ]; then
  PY=""
  for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v $c >/dev/null 2>&1 && $c -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
      PY=$c; break
    fi
  done
  [ -n "$PY" ] || { echo "Python 3.10+ is needed: brew install python@3.12"; exit 1; }
  $PY -m venv .venv
  .venv/bin/python -m pip install -q -r requirements.txt
fi

# 2. llama.cpp with Metal, at the pinned commit
LLAMA=third_party/llama.cpp
if [ ! -x $LLAMA/build/bin/llama-server ]; then
  command -v cmake >/dev/null 2>&1 || { echo "cmake is needed: brew install cmake"; exit 1; }
  if [ ! -d $LLAMA/.git ]; then
    git init -q $LLAMA
    git -C $LLAMA remote add origin https://github.com/ggml-org/llama.cpp
  fi
  git -C $LLAMA fetch -q --depth 1 origin "$COMMIT"
  git -C $LLAMA checkout -q FETCH_HEAD
  cmake -S $LLAMA -B $LLAMA/build -DGGML_METAL=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build $LLAMA/build -j "$(sysctl -n hw.ncpu)" --target llama-server
fi

# 3. the model, resumable, checked once (68 GB takes a minute or two to hash)
DIR=models/$Q
mkdir -p $DIR
for i in 1 2; do
  f=Qwen3.8-Flash-Next-GSQ-RCO-$Q-0000$i-of-00002.gguf
  [ -f $DIR/$f.done ] && continue
  echo "downloading $f ..."
  curl -fL --retry 20 --retry-all-errors -C - -o $DIR/$f "https://huggingface.co/$REPO/resolve/$REV/$Q/$f"
  [ $i = 1 ] && want=$SHA1 || want=$SHA2
  got=$(shasum -a 256 $DIR/$f | cut -d' ' -f1)
  if [ "$got" != "$want" ]; then
    echo "$f is damaged (SHA-256 $got); it was deleted - run setup-mac.sh again"; rm -f $DIR/$f; exit 1
  fi
  touch $DIR/$f.done
done
MODEL=$DIR/Qwen3.8-Flash-Next-GSQ-RCO-$Q-00001-of-00002.gguf

# 4. the tokenizer and chat template, from the same file (the server's token ids are llama.cpp's)
[ -f $DIR/tokenizer/vocab.json ] || .venv/bin/python tools/strata_tokenizer.py --gguf $MODEL --out $DIR

# 5. the config and the start script (an existing config is kept: edit it to change the context or port)
if [ ! -f strata-mac.json ]; then
  cat > strata-mac.json <<EOF
{
  "engine": "llamacpp",
  "exe": "$LLAMA/build/bin/llama-server",
  "model": "$MODEL",
  "max_context": ${STRATA_CONTEXT:-32768},
  "tokenizer": "$DIR/tokenizer",
  "model_name": "qwen3.8-flash-next",
  "log": "strata-mac.log",
  "port": ${STRATA_PORT:-8080}
}
EOF
fi
PORT=$(.venv/bin/python -c 'import json; print(json.load(open("strata-mac.json")).get("port", 8080))')
cat > run-mac.sh <<EOF
#!/bin/sh
cd "\$(dirname "\$0")"
exec .venv/bin/python -m serve.server --engine llamacpp --config strata-mac.json --port $PORT "\$@"
EOF
chmod +x run-mac.sh

[ "${1:-}" = "--no-start" ] && { echo "ready: ./run-mac.sh starts it on http://127.0.0.1:$PORT"; exit 0; }
exec ./run-mac.sh
