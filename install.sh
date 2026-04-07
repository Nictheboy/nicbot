#/bin/bash

uv venv
source .venv/bin/activate

TORCH_VERSION=2.9.1
SGLANG_VERSION=0.5.10
SGLANG_KERNEL_VERSION=0.4.1

uv pip install --prerelease=allow torch==${TORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
uv pip install --prerelease=allow sglang==${SGLANG_VERSION}
uv pip install --prerelease=allow "https://github.com/sgl-project/whl/releases/download/v${SGLANG_KERNEL_VERSION}/sglang_kernel-${SGLANG_KERNEL_VERSION}+cu130-cp310-abi3-manylinux2014_x86_64.whl"
