#/bin/bash

export MODEL_PATH=Qwen/Qwen3.5-27B-GPTQ-Int4

sglang serve \
    --model-path ${MODEL_PATH} \
    --tp 2 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --host 0.0.0.0 --port 3000 \
    --log-level warning
