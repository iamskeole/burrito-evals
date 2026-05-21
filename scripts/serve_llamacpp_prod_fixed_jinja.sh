#!/bin/bash
set -euo pipefail

/home/p/ml/backends/llama.cpp/build/bin/llama-server \
--threads 16 \
--n-gpu-layers 999 \
--host 0.0.0.0 \
--port 9999 \
--ctx_size 131072 \
--parallel 1 \
--flash-attn 'on' \
--model /home/p/ml/models/gguf/ggml-org--gpt-oss-20b-mxfp4.gguf \
--alias gpt-oss-20b \
--jinja \
--ubatch-size 2048 \
--batch-size 512 \
--top_k 64 \
--top_p 1.0 \
--min_p 0.0 \
--temp 1.0 \
--samplers 'top_k;top_p;min_p;temperature' \
--chat-template-kwargs '{"reasoning_effort": "medium"}' \
--chat-template-file /home/p/ml/backends/chat_template_gpt_oss_fixed_tools.jinja
