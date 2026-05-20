#!/usr/bin/env sh
set -euo pipefail

/home/p/ai/inference/backends/llama.cpp/build/bin/llama-server \
--threads 16 \
--n-gpu-layers 999 \
--host 0.0.0.0 \
--port 9999 \
--no-mmap \
--ctx_size 131072 \
--parallel 1 \
--batch-size 2048 \
--ubatch-size 512 \
--flash-attn 'on' \
--model /home/p/ml/models/gguf/gpt-oss-20b-mxfp4.gguf \
--alias gpt-oss-20b \
--jinja \
--seed 69421337 \
--top_k 0 \
--top_p 1.0 \
--min_p 0.0 \
--temp 0.0 \
--samplers 'top_k;top_p;min_p;temperature' \
--no-cache-prompt \
--cache-ram 0 \
--chat-template-kwargs '{"reasoning_effort": "medium"}'
