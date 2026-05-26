#!/bin/bash
set -euo pipefail

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1
export FAKETIME="2026-05-25 00:00:00"

/home/p/ml/backends/llama.cpp/build/bin/llama-server \
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
--model /home/p/ml/models/gguf/ggml-org--gpt-oss-20b-mxfp4.gguf \
--alias gpt-oss-20b \
--jinja \
--top_k 0 \
--top_p 1.0 \
--min_p 0.0 \
--temp 1.0 \
--samplers 'top_k;top_p;min_p;temperature' \
--no-cache-prompt \
--ctx-checkpoints 0 \
--cache-ram 0 \
--chat-template-kwargs '{"reasoning_effort": "medium"}' \
--chat-template-file /home/p/ml/backends/chat_template_gpt_oss_fixed_tools.jinja
