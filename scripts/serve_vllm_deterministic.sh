#!/usr/bin/env bash
set -euo pipefail

export VLLM_TORCH_COMPILE_LEVEL=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1

source /home/p/ai/inference/backends/vllm/.venv/bin/activate

vllm serve openai/gpt-oss-20b \
    --host 0.0.0.0 \
    --port 9999 \
    --served-model-name "openai/gpt-oss-20b" "openai-gpt-oss-20b-chat" "openai-gpt-oss-20b-responses" \
    --max-model-len 131072 \
    --tool-call-parser openai \
    --reasoning-parser openai_gptoss \
    --enable-auto-tool-choice \
    --quantization mxfp4 \
    --max-num-batched-tokens 32 \
    --max-num-seqs 1 \
    --max-cudagraph-capture-size 1 \
    --gpu-memory-utilization 0.90 \
    --seed 69421337 \
    --disable-custom-all-reduce \
    --no-enable-prefix-caching \
    --generation-config vllm \
    --override-generation-config '{"top_k": 0, "top_p": 1.0,  "min_p": 0.0, "temperature": 0.0}' \
    --enable-log-requests
