# ONNX-based DeOldify Video colorizer — Docker Hub version.
# Uses public pytorch image as base, no GHCR needed.

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# System deps
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

# Bake the ONNX model (~122 MB) at build time
# Using HuggingFace mirror that doesn't require auth
RUN mkdir -p /app/models && \
    wget -q \
        "https://github.com/instant-high/deoldify-onnx/releases/download/deoldify-onnx/deoldify_fp16.onnx" \
        -O /app/models/deoldify_fp16.onnx

WORKDIR /app
COPY src/handler.py /app/handler.py
COPY src/rp_schemas.py /app/rp_schemas.py

CMD ["python3", "-u", "/app/handler.py"]
