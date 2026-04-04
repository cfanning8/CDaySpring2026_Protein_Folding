# OpenFold3 / AlphaFold3-class inference image for protein smoke (Linux + NVIDIA only).
# Build:  docker build -t openfold3-smoke docker/openfold3-smoke
# Run:    docker run --rm -it --gpus all -v %CD%:/workspace -w /workspace openfold3-smoke bash
# Then:   setup_openfold   # download weights (once)
#         python -u scripts/protein/run_smoke_pipeline.py --skip-structure-figures
#
# Notes:
# - Pip openfold3 pulls heavy deps; DS_* disables DeepSpeed JIT on incompatible hosts.
# - Match PyTorch CUDA to your NVIDIA driver (adjust cu124 index if needed).

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV DS_BUILD_OPS=0
ENV DS_BUILD_AIO=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv git wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip \
    && pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install openfold3

WORKDIR /workspace
