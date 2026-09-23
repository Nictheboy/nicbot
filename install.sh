#/bin/bash

# Please ensure libnuma1 is installed on your system.
set -e

rm -rf .venv
uv venv --python 3.12
source .venv/bin/activate

VLLM_VERSION=0.30.0

uv pip install --prerelease=allow vllm==${VLLM_VERSION}
