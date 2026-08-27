# ONNX-based DeOldify Video colorizer for RunPod Serverless.
# No fastai, no wandb. Just torch + onnxruntime.
#
# Model: we wget the ONNX file from the instant-high public GitHub
# release at build time. Saves 122MB from the repo and avoids GitHub's
# 100MB file size limit.

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps — ffmpeg required for extract/recompose
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Python deps (torch already in base image)
RUN pip install --no-cache-dir \
        "onnxruntime-gpu==1.18.0" \
        "opencv-python-headless>=4.8" \
        "Pillow>=10.0" \
        "numpy<2" \
        "requests>=2.31" \
        "runpod>=1.6"

# Bake the ONNX model (~122 MB) — downloaded at build time from public mirror
RUN mkdir -p /app/models && \
    wget -q \
        "https://github.com/instant-high/deoldify-onnx/releases/download/deoldify-onnx/deoldify_fp16.onnx" \
        -O /app/models/deoldify_fp16.onnx && \
    ls -lh /app/models/

# Bake the handler
WORKDIR /app
COPY src/handler.py /app/handler.py
COPY src/rp_schemas.py /app/rp_schemas.py

# Entrypoint
CMD ["python3", "-u", "/app/handler.py"]
