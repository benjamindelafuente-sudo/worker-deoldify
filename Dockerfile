# Worker DeOldify video - versión minimalista
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*

# Descargar modelo al directorio del handler
RUN mkdir -p /app/models && \
    curl -fL -o /app/models/deoldify_fp16.onnx \
        "https://github.com/instant-high/deoldify-onnx/releases/download/deoldify-onnx/deoldify_fp16.onnx" && \
    ls -la /app/models/

# Instalar deps
RUN pip install --no-cache-dir \
        onnxruntime-gpu==1.18.0 \
        opencv-python-headless \
        Pillow \
        numpy \
        requests \
        runpod

WORKDIR /app
COPY src/handler.py /app/handler.py
COPY src/rp_schemas.py /app/rp_schemas.py

# Asegurar handler.py es ejecutable
RUN chmod +x /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
