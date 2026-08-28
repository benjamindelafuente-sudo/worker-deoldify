# Worker DeOldify video - con ENTRYPOINT explícito + wrapper para ver logs

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/models && \
    curl -fL -o /app/models/deoldify_fp16.onnx \
        "https://github.com/instant-high/deoldify-onnx/releases/download/deoldify-onnx/deoldify_fp16.onnx" && \
    ls -la /app/models/

RUN pip install --no-cache-dir \
        onnxruntime-gpu==1.18.0 \
        opencv-python-headless \
        Pillow \
        numpy \
        requests \
        runpod

# Wrapper script que loguea cada paso
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

WORKDIR /app
COPY src/handler.py /app/handler.py
COPY src/rp_schemas.py /app/rp_schemas.py

# Reset entrypoint y usar wrapper
ENTRYPOINT ["/app/start.sh"]
