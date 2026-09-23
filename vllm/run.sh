#/bin/bash

export MODEL_PATH=/home/nictheboy/models/Qwen3.8-27B-INT4

source .venv/bin/activate

# ---------------------------------------------------------------------------
# 上下文扩展：256K -> 512K（YaRN, factor=2.0）
#
# 模型原生 max_position_embeddings=262144 (256K)。该模型是 hybrid 架构
# （64 层中 48 层线性注意力 + 16 层全注意力），只有 16 层带 RoPE，且
# rope_theta=1e7、partial_rotary_factor=0.25（只有 25% 维度参与旋转），
# 外推友好性远高于普通 dense 模型。
#
# YaRN 是本模型在此 vLLM 版本下唯一验证可用的扩展方式：vLLM 0.30 的
# get_rope() "yarn" 分支同时支持 mrope_section + mrope_interleaved
# （本模型是 VLM，用 interleaved mrope），"ntk"/"dynamic" 分支不支持。
#
# factor=2.0 是外推甜点（512K）；factor=3/4 (768K/1M) 数字上可行但
# 长上下文质量退化明显，需要时把下面两个数字同步改即可。
#
# rope_parameters 必须给全量（覆盖是整键替换）：原有 4 个键 + yarn 3 个键。
# ---------------------------------------------------------------------------
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
ROPE_OVERRIDES='{
  "text_config": {
    "rope_parameters": {
      "rope_type": "yarn",
      "factor": 2.0,
      "original_max_position_embeddings": 262144,
      "rope_theta": 10000000,
      "partial_rotary_factor": 0.25,
      "mrope_section": [11, 11, 10],
      "mrope_interleaved": true
    }
  }
}'

vllm serve ${MODEL_PATH} \
    --served-model-name Qwen3.8-27B-INT4 \
    --tensor-parallel-size 2 \
    --max-model-len 524288 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.94 \
    --hf-overrides "$ROPE_OVERRIDES" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --host 0.0.0.0 \
    --port 3000
