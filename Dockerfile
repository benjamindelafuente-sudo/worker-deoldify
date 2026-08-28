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
        curl \
    && rm -rf /var/lib/apt/lists/*

# Bake the ONNX model (~122 MB) at build time.
# Use curl -fL (no wget): -L follows redirects (GitHub releases use signed URLs),
# -f fails clean on HTTP error so build fails fast if URL is bad.
RUN mkdir -p /app/models && \
    curl -fL --retry 3 --retry-delay 2 \
        "https://github.com/instant-high/deoldify-onnx/releases/download/deoldify-onnx/deoldify_fp16.onnx" \
        -o /app/models/deoldify_fp16.onnx && \
    ls -lh /app/models/

# Verify model file is valid ONNX (not truncated / 404 HTML page)
RUN python3 -c "import onnx; m = onnx.load('/app/models/deoldify_fp16.onnx'); print(f'OK: {len(m.graph.node)} nodes')"

# Python deps (torch already in base image)
RUN pip install --no-cache-dir \
        "onnxruntime-gpu==1.18.0" \
        "opencv-python-headless>=4.8" \
        "Pillow>=10.0" \
        "numpy<2" \
        "requests>=2.31" \
        "runpod>=1.6"

WORKDIR /app
COPY src/handler.py /app/handler.py
COPY src/rp_schemas.py /app/rp_schemas.py

# Verify handler imports cleanly (catches missing deps early)
RUN python3 -c "import sys; sys.path.insert(0, '/app'); import handler; print('handler imports OK')"

CMD ["python3", "-u", "/app/handler.py"]
